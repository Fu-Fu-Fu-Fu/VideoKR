import datetime
import json
import os
import re
import sys
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import datasets
import numpy as np
import yaml
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks._task_utils.judge_utils import (
    evaluation_mode,
    extract_answer_content,
    get_judge_server,
    normalize_answer_letter,
    rule_correct_multiple_choice,
    use_vlm_judge,
)

# Initialize the LLM judge server
API_TYPE = os.getenv("API_TYPE", "azure")
MODEL_VERSION = os.getenv("MODEL_VERSION", "gpt-4o")
server = get_judge_server(API_TYPE, MODEL_VERSION)

VIDEO_TYPE = ["short", "medium", "long"]
CATEGORIES = ["Knowledge", "Film & Television", "Sports Competition", "Artistic Performance", "Life Record", "Multilingual"]

SUB_CATEGORIES = [
    "Humanity & History",
    "Literature & Art",
    "Biology & Medicine",
    "Finance & Commerce",
    "Astronomy",
    "Geography",
    "Law",
    "Life Tip",
    "Technology",
    "Animation",
    "Movie & TV Show",
    "Documentary",
    "News Report",
    "Esports",
    "Basketball",
    "Football",
    "Athletics",
    "Other Sports",
    "Stage Play",
    "Magic Show",
    "Variety Show",
    "Acrobatics",
    "Handicraft",
    "Food",
    "Fashion",
    "Daily Life",
    "Travel",
    "Pet & Animal",
    "Exercise",
    "Multilingual",
]

TASK_CATEGORIES = [
    "Temporal Perception",
    "Spatial Perception",
    "Attribute Perception",
    "Action Recognition",
    "Object Recognition",
    "OCR Problems",
    "Counting Problem",
    "Temporal Reasoning",
    "Spatial Reasoning",
    "Action Reasoning",
    "Object Reasoning",
    "Information Synopsis",
]


hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)
with open(Path(__file__).parent / "videomme.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)
cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]


def parse_subtitle_time(time_str):
    h, m, s_ms = time_str.split(":")
    s, ms = s_ms.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def load_subtitles(subtitle_path):
    subtitles = {}
    with open(subtitle_path, "r", encoding="utf-8") as file:
        content = file.read().split("\n\n")
        for section in content:
            if section.strip():
                lines = section.split("\n")
                if len(lines) >= 3:
                    time_range = lines[1].split(" --> ")
                    start_time = parse_subtitle_time(time_range[0])
                    end_time = parse_subtitle_time(time_range[1])
                    text = " ".join(line for line in lines[2:])
                    subtitles[(start_time, end_time)] = text
    return subtitles


def convert_time_to_frame(time_in_seconds, fps):
    return int(time_in_seconds * fps)


def extract_subtitles(video_path, subtitle_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    subtitles = load_subtitles(subtitle_path)

    subtitle_frames = []
    for (start_time, end_time), text in subtitles.items():
        start_frame = convert_time_to_frame(start_time, fps)
        end_frame = convert_time_to_frame(end_time, fps)
        subtitle_frames.append((start_frame, end_frame, text))

    return subtitle_frames, total_frame


def videmme_process_docs_base(dataset: datasets.Dataset, type: str) -> datasets.Dataset:
    return dataset.filter(lambda x: x["duration"] == type)


videomme_process_docs_long = partial(videmme_process_docs_base, type="long")


def videomme_doc_to_visual(doc):
    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_path = doc["videoID"] + ".mp4"
    video_path = os.path.join(cache_dir, "data", video_path)
    if os.path.exists(video_path):
        video_path = video_path
    elif os.path.exists(video_path.replace("mp4", "MP4")):
        video_path = video_path.replace("mp4", "MP4")
    elif os.path.exists(video_path.replace("mp4", "mkv")):
        video_path = video_path.replace("mp4", "mkv")
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path]

def videomme_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    question = doc["question"]
    option = "\n".join([f"{opt}" for i, opt in enumerate(doc["options"])])
    question = question + "\n" + option
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    full_prompt = f"{pre_prompt}{question}\n{post_prompt}"
    print(f"Full Prompt: {full_prompt}")
    return full_prompt


def videomme_doc_to_text_subtitle(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_path = doc["videoID"] + ".mp4"
    video_path = os.path.join(cache_dir, "data", video_path)
    subtitle_path = os.path.join(cache_dir, "subtitle", doc["videoID"] + ".srt")
    video_path = os.path.join(cache_dir, video_path)
    if os.path.exists(subtitle_path):  # Denote have subtitle
        subtitle = open(subtitle_path).readlines()
    else:
        subtitle = ""
    subtitles_prompt = "This video's subtitles are listed below: \n"
    if subtitle == "":
        subtitle = "No subtitles available"
    else:
        subtitle_text = "\n".join(line.strip() for line in subtitle if line.strip())
        if "gemini_api_flag" in lmms_eval_specific_kwargs:  # specific for gemini_api
            if lmms_eval_specific_kwargs["gemini_api_flag"] == "full subtitle":
                textlist = []
                for ele in subtitle:
                    pattern = r'<font color="white" size=".72c">(.*?)</font>'
                    matches = re.findall(pattern, ele)
                    if matches:
                        textlist.append(matches[0])
                subtitle_text = "\n".join(textlist)
        else:
            if "frame_num" in lmms_eval_specific_kwargs:
                frame_num = lmms_eval_specific_kwargs["frame_num"]
                subtitle_by_frame, total_frame = extract_subtitles(video_path, subtitle_path)
                if frame_num == -1:
                    frame_num = total_frame
                uniform_sampled_frames = np.linspace(0, total_frame - 1, frame_num, dtype=int).tolist()

                subtitle_by_frame_idx = []
                for frame_idx in uniform_sampled_frames:
                    for idx, title in enumerate(subtitle_by_frame):
                        if frame_idx < title[1] and frame_idx >= title[0]:
                            subtitle_by_frame_idx.append(idx)
                subtitle_by_frame_idx = list(set(subtitle_by_frame_idx))

                textlist = []
                for idx in subtitle_by_frame_idx:
                    pattern = r'<font color="white" size=".72c">(.*?)</font>'
                    raw_text = re.findall(pattern, subtitle_by_frame[idx][2])
                    try:
                        textlist.append(raw_text[0])
                    except:
                        continue
                subtitle_text = "\n".join(textlist)
        subtitle = subtitle_text

    question = doc["question"]
    option = "\n".join([f"{opt}" for i, opt in enumerate(doc["options"])])
    question = question + "\n" + option
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    full_prompt = f"{pre_prompt}{subtitles_prompt}{subtitle}\n{question}\n{post_prompt}"
    return full_prompt


def extract_characters_regex(s):
    s = s.strip()

    # 1. 先找 <answer>C</answer>
    match = re.search(r"<answer>\s*([ABCD])\s*</answer>", s)
    if match:
        return match.group(1)   # 这里用 group(1)，因为用了捕获组 ([ABCD])

    # 2. 其他情况保持和原始代码一致
    print("兜底提取答案字母...")
    blocks = re.findall(r"<answer[^>]*>(.*?)</answer>", s, flags=re.I | re.S)
    if blocks:
        inner = " ".join(blocks)

        # 去掉常见前缀（只对 inner 做清理）
        answer_prefixes = [
            "The answer is",
            "The correct answer is",
            "Answer:",
            "Option:",
            "the final answer is"
        ]
        for prefix in answer_prefixes:
            inner = inner.replace(prefix, "")

        # 在 <answer> 文本中匹配独立出现的 A–J（避免匹配到单词中间）
        m2 = re.search(r"\b([A-J])\b", inner, flags=re.I)
        if m2:
            return m2.group(1).upper()

    # 3) 没有 <answer> 或其中没有字母，返回空串
    return ""

    # matches = re.search(r"[ABCD]", s)
    # if matches is None:
    #     return ""
    # return matches[0]


matrices = []

for i in VIDEO_TYPE:
    for j in CATEGORIES:
        for k in SUB_CATEGORIES:
            for l in TASK_CATEGORIES:
                matrices.append(f"{i}_{j}_{k}_{l}")


def videomme_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case videomme score), value: metric value
    """
    pred = results[0]
    print(f"Raw model prediction: {pred}")
    judge_pred = extract_answer_content(pred)

    # 构建完整问题(包含选项)
    question = doc["question"]
    options = doc["options"]
    option_text = "\n".join([f"{opt}" for opt in options])
    full_question = f"{question}\n{option_text}"
    custom_prompt = """You are evaluating a multiple-choice video question answer.
The model's prediction may contain reasoning or explanation, but you should focus on the final answer letter.
Question:
{question}
Ground Truth Answer:
{answer}
Model Prediction:
{prediction}
Evaluation rules:
- Score 1 if the prediction contains the correct answer letter (A-J)
- The answer can appear anywhere in the response
- Ignore formatting, capitalization, or extra text
- Score 0 if the answer is incorrect or cannot be determined

Return only "1" or "0" with no additional text or formatting."""
    # 获取正确答案
    gt_answer = doc["answer"]
    print(f"Ground Truth Answer: {gt_answer}")
    if use_vlm_judge() and server is not None:
        try:
            # 使用 LLM Judge 进行二元评估
            result = server.evaluate_binary(
                question=full_question,
                answer=gt_answer,
                prediction=judge_pred,
                output_format="0/1",
                custom_prompt=custom_prompt
            )
            print(f"Judge raw response: {result}")
            if result["success"]:
                judge_score = result["result"]  # 直接使用整数
                correct = (judge_score == 1)
                print(f"Judge evaluation: {'Correct' if correct else 'Wrong'}")
            else:
                eval_logger.error(f"Judge evaluation failed: {result.get('raw_response', 'Unknown error')}")
                correct = False

        except Exception as e:
            eval_logger.error(f"Error getting judge response: {e}")
            correct = False
        parsed_pred = gt_answer if correct else "WRONG"
    else:
        parsed_pred = normalize_answer_letter(judge_pred, choices="ABCDEFGHIJ") or judge_pred
        correct = rule_correct_multiple_choice(judge_pred, gt_answer, choices="ABCDEFGHIJ")

    category = doc["domain"]
    sub_category = doc["sub_category"]
    task_category = doc["task_type"]
    data_dict = {
        "question_id": doc["question_id"],
        "duration": doc["duration"],
        "category": category,
        "sub_category": sub_category,
        "task_category": task_category,
        "pred_answer": parsed_pred,  # 保存完整预测
        "answer": gt_answer,
        "correct": int(correct),  # 添加 correct 字段
        "evaluation_mode": evaluation_mode(),
    }

    return {f"videomme_perception_score": data_dict}

    # pred_ans = extract_characters_regex(pred)
    # # gt_ans = doc["answer"].lower().strip().replace(".", "")
    # print(f"Extracted answer: {pred_ans}")
    # category = doc["domain"]
    # sub_category = doc["sub_category"]
    # task_category = doc["task_type"]
    # data_dict = {"question_id": doc["question_id"], "duration": doc["duration"], "category": category, "sub_category": sub_category, "task_category": task_category, "pred_answer": pred_ans, "answer": doc["answer"]}

    # # return {f"videomme_perception_score": data_dict for metric in matrices}
    # return {f"videomme_perception_score": data_dict}


def videomme_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    category2score = {}

    for video_type in VIDEO_TYPE:
        for category in CATEGORIES:
            for sub_category in SUB_CATEGORIES:
                for task_category in TASK_CATEGORIES:
                    key = f"{video_type}_{category}_{sub_category}_{task_category}"
                    category2score[key] = {"correct": 0, "answered": 0}

    for result in results:
        video_type = result["duration"]
        category = result["category"]
        sub_category = result["sub_category"]
        task_category = result["task_category"]
        key = f"{video_type}_{category}_{sub_category}_{task_category}"
        category2score[key]["answered"] += 1
        category2score[key]["correct"] += int(result.get("correct", result["pred_answer"] == result["answer"]))

    for video_type in VIDEO_TYPE:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if video_type in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        eval_logger.info(f"Evaluation on video Type: {video_type}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

    for category in CATEGORIES:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if category in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        eval_logger.info(f"Evaluation on Categories: {category}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

    for sub_cate in SUB_CATEGORIES:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if sub_cate in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        eval_logger.info(f"Evaluation on Video Sub Categories: {sub_cate}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

    for task_cate in TASK_CATEGORIES:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if task_cate in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        eval_logger.info(f"Evaluation on Task Categories: {task_cate}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

    total_correct = 0
    total_answered = 0
    for k, v in category2score.items():
        total_correct += v["correct"]
        total_answered += v["answered"]
    eval_logger.info(f"Evaluation mode: {evaluation_mode()}")
    eval_logger.info(f"Overall Performance: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")
    return 100 * total_correct / total_answered if total_answered > 0 else 0
