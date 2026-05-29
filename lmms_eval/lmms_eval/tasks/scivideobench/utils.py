import os
import random
import re
from pathlib import Path
from collections import defaultdict

from datetime import timedelta

from loguru import logger as eval_logger
from lmms_eval.tasks._task_utils.judge_utils import (
    evaluation_mode,
    extract_answer_content,
    get_judge_server,
    normalize_answer_letter,
    rule_correct_multiple_choice,
    use_vlm_judge,
)

# 初始化 Azure judge server
API_TYPE = os.getenv("API_TYPE", "azure")
MODEL_VERSION = os.getenv("MODEL_VERSION", "gpt-4o")
server = get_judge_server(API_TYPE, MODEL_VERSION)

import yaml

hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)
with open(Path(__file__).parent / "scivideobench.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)
cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]



def convert_time_to_frames_in_question(question, total_duration_sec, total_frames):
    def parse_time(tstr):
        mm, ss = map(int, tstr.split(":"))
        return timedelta(minutes=mm, seconds=ss).total_seconds()

    # Convert total duration (float in seconds) to mm:ss string
    total_minutes = int(total_duration_sec) // 60
    total_seconds = int(total_duration_sec) % 60
    total_duration_str = f"{total_minutes:02d}:{total_seconds:02d}"
    total_sec = parse_time(total_duration_str)

    # Replace timestamp ranges like 06:15–06:36 or 6:15-6:36
    def replace_range(match):
        start_time, end_time = match.group(1), match.group(2)
        start_sec = parse_time(start_time)
        end_sec = parse_time(end_time)
        start_frame = round((start_sec / total_sec) * total_frames)
        end_frame = round((end_sec / total_sec) * total_frames)
        return f"frame {start_frame} to frame {end_frame}"

    question = re.sub(r'(\d{1,2}:\d{2})\s*[-–]\s*(\d{1,2}:\d{2})', replace_range, question)

    # Replace standalone timestamps like 06:15
    def replace_single(match):
        time_str = match.group(1)
        time_sec = parse_time(time_str)
        frame = round((time_sec / total_sec) * total_frames)
        return f"frame {frame}"

    question = re.sub(r'(?<!frame )(\d{1,2}:\d{2})(?!\s*[-–])', replace_single, question)

    return question

def scivideobench_doc_to_visual(doc):
    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_path = os.path.join(cache_dir, f"jove_{doc['video_id']}.mp4")
    if os.path.exists(video_path):
        video_path = video_path
    elif os.path.exists(video_path.replace("mp4", "MP4")):
        video_path = video_path.replace("mp4", "MP4")
    elif os.path.exists(video_path.replace("mp4", "mkv")):
        video_path = video_path.replace("mp4", "mkv")
    else:
        raise FileNotFoundError(f"Video not found: {video_path}")
    return [video_path]


def format_options(opts):
    if isinstance(opts, dict):
        # ensure consistent A..Z order
        keys = sorted(opts.keys())
        return "\n".join(f"{k}. {opts[k]}" for k in keys)
    elif isinstance(opts, list):
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return "\n".join(f"{letters[i]}. {opt}" for i, opt in enumerate(opts))
    else:
        raise TypeError(f"Unsupported options type: {type(opts)}")


def scivideobench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt  = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    duration = doc["video_duration"]
    converted_question = convert_time_to_frames_in_question(doc['question'], duration, int(duration))

    options_text = format_options(doc["options"])  # ← fix here
    input_text = f"{pre_prompt}{converted_question}\n{options_text}\n{post_prompt}"
    print(f"Input text: {input_text}")
    return input_text


def scivideobench_doc_to_choice(doc):
    return ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

def scivideobench_doc_to_target(doc):
    return ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"].index(doc["ground_truth"])

def extract_answer_letter(s):
    """
    Extract letter from answer (A, B, C, D, E, F, G, H, I, J)

    """
    s = s.strip()
    # 1. 先找 <answer>C</answer>
    match = re.search(r"<answer>\s*([ABCDEFGHIJ])\s*</answer>", s)
    if match:
        return match.group(1)   # 这里用 group(1)，因为用了捕获组 ([ABCD])

    # 2. 其他情况保持和原始代码一致
    # 2) 兜底：只取出所有 <answer>...</answer> 的内容，合并后在其中搜索
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

#     }

#     return {
#         "scivideobench_acc": data_dict
#     }
def scivideobench_process_results(doc, results):
    prediction = results[0].strip() if isinstance(results, list) else results.strip()
    print(prediction)
    judge_prediction = extract_answer_content(prediction)
    # 构建问题文本(包含选项)
    question = doc["question"]
    options_text = format_options(doc["options"])
    full_question = f"{question}\n{options_text}"

    # 正确答案
    answer = doc["answer"].strip()
    print("Correct answer:", answer)
    # 自定义评判提示
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
                answer=answer,
                prediction=judge_prediction,
                output_format="0/1",
                custom_prompt=custom_prompt
            )
            print("Judge response:", result)
            if result["success"]:
                judge_score = result["result"]
                correct = (judge_score == 1)
                print(f"Judge evaluation score: {judge_score}, correct: {correct}")
            else:
                eval_logger.error(f"Judge evaluation failed: {result.get('raw_response', 'Unknown error')}")
                correct = False

        except Exception as e:
            eval_logger.error(f"Error getting judge response: {e}")
            correct = False
        pred_answer = judge_prediction
    else:
        pred_answer = normalize_answer_letter(judge_prediction, choices="ABCDEFGHIJ") or judge_prediction
        correct = rule_correct_multiple_choice(judge_prediction, answer, choices="ABCDEFGHIJ")

    data_dict = {
        "id": doc["video_id"],
        "question_type": doc["question_type"],
        "category": doc.get("category", "UNKNOWN"),
        "pred_answer": pred_answer,
        "answer": answer,
        "correct": correct,
        "evaluation_mode": evaluation_mode(),
        "raw_output": results[0] if isinstance(results, list) else results
    }

    return {
        "scivideobench_acc": data_dict
    }


def scivideobench_aggregate_results(results):
    # Buckets
    by_qtype = defaultdict(lambda: {"correct": 0, "total": 0})
    by_category = defaultdict(lambda: {"correct": 0, "total": 0})

    total_correct = 0
    total_examples = 0

    # Accumulate
    for r in results:
        qtype = r.get("question_type", "UNKNOWN")
        cat = r.get("category", "UNKNOWN")
        is_correct = 1 if r.get("correct", False) else 0

        by_qtype[qtype]["correct"] += is_correct
        by_qtype[qtype]["total"] += 1

        by_category[cat]["correct"] += is_correct
        by_category[cat]["total"] += 1

        total_correct += is_correct
        total_examples += 1

    # Build printable summaries
    def summarize(bucket):
        out = {}
        for k, v in bucket.items():
            acc = (v["correct"] / v["total"]) if v["total"] > 0 else 0.0
            out[k] = {"num": v["total"], "acc": round(acc * 100, 2)}
        return out

    printable_qtype = summarize(by_qtype)
    printable_category = summarize(by_category)
    overall_acc = round((total_correct / total_examples) * 100, 2) if total_examples else 0.0

    # Pretty print
    print("\nSciVideoBench Evaluation Results:")
    print(f"Evaluation mode: {evaluation_mode()}")
    print(f"Overall Accuracy: {overall_acc}%")

    print("\nStatistics by Question Type:")
    for k, v in printable_qtype.items():
        print(f"{k}: {v['acc']}% (samples: {v['num']})")

    print("\nStatistics by Discipline:")
    for k, v in printable_category.items():
        print(f"{k}: {v['acc']}% (samples: {v['num']})")

    return {
        "overall_acc": overall_acc,
        "evaluation_mode": evaluation_mode(),
        "by_question_type": printable_qtype,
        "by_category": printable_category
    }
