import datetime
import json
import os
import re
import string
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import PIL
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

DATA_LIST = {
    "object_interaction": "star/Charades_segment",
    "action_sequence": "star/Charades_segment",
    "action_prediction": "star/Charades_segment",
    "action_localization": "sta/sta_video_segment",
    "moving_count": "clevrer/video_validation",
    "fine_grained_pose": "nturgbd_convert",
    "character_order": "perception/videos",
    "object_shuffle": "perception/videos",
    "egocentric_navigation": "vlnqa",
    "moving_direction": "clevrer/video_validation",
    "episodic_reasoning": "tvqa/video_fps3_hq_segment",
    "fine_grained_action": "Moments_in_Time_Raw/videos",
    "scene_transition": "scene_qa/video",
    "state_change": "perception/videos",
    "moving_attribute": "clevrer/video_validation",
    "action_antonym": "ssv2_video_mp4",
    "unexpected_action": "FunQA_test/test",
    "counterfactual_inference": "clevrer/video_validation",
    "object_existence": "clevrer/video_validation",
    "action_count": "perception/videos",
}

MVBENCH_PROMPT = (
    "Please answer this question based on the visual content.\n"
    "Provide your thinking process between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags.\n"
    "Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer>...</answer> tags."
)

# Initialize the LLM judge server  
API_TYPE = os.getenv("API_TYPE", "azure")  
MODEL_VERSION = os.getenv("MODEL_VERSION", "gpt-4o")  
server = get_judge_server(API_TYPE, MODEL_VERSION)

hf_home = os.getenv("HF_HOME", "~/.cache/huggingface")
base_cache_dir = os.path.expanduser(hf_home)

with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)


cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]


def mvbench_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    cache_dir = os.path.join(base_cache_dir, cache_name)
    dataset_folder = DATA_LIST[lmms_eval_specific_kwargs["sub_task"]]
    video_path = os.path.join(cache_dir, dataset_folder, doc["video"])
    if os.path.exists(video_path):
        video_path = video_path
    elif os.path.basename(dataset_folder) in ["clevrer", "star"]:
        alternative_video_path = os.path.join(cache_dir, "data0613", dataset_folder, doc["video"])
        if os.path.exists(alternative_video_path):
            video_path = alternative_video_path
        else:
            eval_logger.error(f"Video path: {video_path} does not exist, please check.")
    else:
        eval_logger.error(f"Video path: {video_path} does not exist, please check.")
    return [video_path]


def mvbench_frames_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    cache_dir = os.path.join(base_cache_dir, cache_name)
    dataset_folder = DATA_LIST[lmms_eval_specific_kwargs["sub_task"]]
    video_path = os.path.join(cache_dir, dataset_folder, doc["video"])
    if os.path.exists(video_path):
        video_path = video_path
    elif os.path.basename(dataset_folder) in ["clevrer", "star"]:
        alternative_video_path = os.path.join(cache_dir, "data0613", dataset_folder, doc["video"])
        if os.path.exists(alternative_video_path):
            video_path = alternative_video_path
        else:
            eval_logger.error(f"Video path: {video_path} does not exist, please check.")
    else:
        eval_logger.error(f"Video path: {video_path} does not exist, please check.")

    frame_path_list = [os.path.join(video_path, f) for f in os.listdir(video_path) if f.endswith(".jpg") or f.endswith(".png")]
    frame_image_list = [PIL.Image.open(frame_path).convert("RGB") for frame_path in frame_path_list]
    return frame_image_list

def mvbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    option_prompt = ""
    option_list = doc["candidates"]
    option_letters = string.ascii_uppercase
    for char_index, option in enumerate(option_list):
        option_letter = option_letters[char_index]
        option_prompt += f"({option_letter}) {option}\n"
    full_text = f"Question:{doc['question']}\nOption:\n{option_prompt}{MVBENCH_PROMPT}"
    print("Full text prompt for MVBench:", full_text)
    return full_text


def mcq_acc(answer, pred):
    periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
    commaStrip = re.compile("(\d)(\,)(\d)")
    punct = [";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!"]

    def processPunctuation(inText):
        outText = inText
        for p in punct:
            if (p + " " in inText or " " + p in inText) or (re.search(commaStrip, inText) != None):
                outText = outText.replace(p, "")
            else:
                outText = outText.replace(p, " ")
        outText = periodStrip.sub("", outText, re.UNICODE)
        return outText

    def process(answer):
        option_regex = re.compile(r"^([A-E])\.\s*(.+)$", re.IGNORECASE)
        match = option_regex.match(answer.strip())

        if match:
            # If matched, return the option letter in uppercase
            return match.group(1).upper()
        else:
            # If no match, process the answer as before
            answer = answer.replace("\n", " ")
            answer = answer.replace("\t", " ")
            answer = answer.strip()
            answer = processPunctuation(answer)
            answer = answer.strip("'")
            answer = answer.strip('"')
            answer = answer.strip(")")
            answer = answer.strip("(")
            answer = answer.strip().lower()

            # Try to find any single letter (A-E) in the processed answer
            letter_match = re.search(r"\b([A-E])\b", answer, re.IGNORECASE)
            if letter_match:
                return letter_match.group(1).upper()

            return answer

    pred = process(pred)
    answer = process(answer)

    if pred == answer:
        score = 1
    else:
        score = 0

    return score


def mvbench_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case mvbench_perception_score), value: metric value
    """
    pred = results[0]
    print(pred)
    pred = extract_answer_content(pred)

    # pred = results[0]

    # Calculate the ground truth option letter
    # 构建问题文本(包含选项)  
    question = doc["question"]  
    option_prompt = ""  
    option_list = doc["candidates"]  
    option_letters = string.ascii_uppercase  
    for char_index, option in enumerate(option_list):  
        option_letter = option_letters[char_index]  
        option_prompt += f"{option_letter}. {option}\n"  
      
    full_question = f"{question}\n{option_prompt}"  
      
    # 计算正确答案的选项字母  
    gt_option_letter = None  
    for i, candidate in enumerate(doc["candidates"]):  
        if candidate == doc["answer"]:  
            gt_option_letter = option_letters[i]  
            break  
    print("Ground truth option letter:", gt_option_letter)
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
    if use_vlm_judge() and server is not None:
        try:
            # 使用 LLM Judge 进行二元评估
            result = server.evaluate_binary(
                question=full_question,
                answer=gt_option_letter,
                prediction=pred,
                output_format="0/1",
                custom_prompt=custom_prompt
                # 不使用 custom_prompt,让系统使用默认模板
            )
            print("Judge result:", result)
            if result["success"]:
                judge_score = result["result"]  # 直接使用整数
                correct = (judge_score == 1)
                print(f"Judge evaluation score: {judge_score}, correct: {correct}")
            else:
                eval_logger.error(f"Judge evaluation failed: {result.get('raw_response', 'Unknown error')}")
                correct = False

        except Exception as e:
            eval_logger.error(f"Error getting judge response: {e}")
            correct = False
        pred_answer = pred
    else:
        pred_answer = normalize_answer_letter(pred, choices="ABCDE") or pred
        correct = rule_correct_multiple_choice(pred, gt_option_letter, choices="ABCDE")
      
    data_dict = {  
        "pred_answer": pred_answer,
        "gt_answer": gt_option_letter,   
        "score": int(correct),
        "evaluation_mode": evaluation_mode(),
    }  
  
    return {"mvbench_accuracy": data_dict}
    # option_letters = string.ascii_uppercase
    # gt_option_letter = None
    # for i, candidate in enumerate(doc["candidates"]):
    #     if candidate == doc["answer"]:
    #         gt_option_letter = option_letters[i]
    #         break

    # # Calculate the score using mcq_acc function
    # score = mcq_acc(gt_option_letter, pred)

    # data_dict = {"pred_answer": pred, "gt_answer": gt_option_letter, "score": score}

    # return {"mvbench_accuracy": data_dict}


def mvbench_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    total_answered = 0
    total_correct = 0
    for result in results:
        if result["pred_answer"] != "":
            total_answered += 1
            total_correct += result["score"]

    eval_logger.info(f"Evaluation mode: {evaluation_mode()}")
    return 100 * total_correct / total_answered if total_answered > 0 else 0
