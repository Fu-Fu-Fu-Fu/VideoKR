import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import decord
import numpy as np
import torch
import yaml
from decord import VideoReader, cpu
from loguru import logger as eval_logger
from PIL import Image

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks._task_utils.judge_utils import (
    evaluation_mode,
    extract_answer_content,
    get_judge_server,
    use_vlm_judge,
)

# Initialize LLM judge server (same pattern as videokr_eval)
API_TYPE = os.getenv("API_TYPE", "azure")
MODEL_VERSION = os.getenv("MODEL_VERSION", "gpt-4o")
server = get_judge_server(API_TYPE, MODEL_VERSION)


def timestamp_to_seconds(timestamp):
    # Split the timestamp into hours, minutes, and seconds
    h, m, s = timestamp.split(":")
    # Convert hours, minutes, and total seconds (including fractions) to float and compute total seconds
    total_seconds = int(h) * 3600 + int(m) * 60 + float(s)
    return total_seconds


def load_video(video_file, duration, max_num_frames=16):
    from decord import VideoReader

    vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
    fps = vr.get_avg_fps()
    total_valid_frames = int(duration * fps)
    num_frames = min(max_num_frames, int(duration))

    frame_indices = [int(total_valid_frames / num_frames) * i for i in range(num_frames)]

    frames = vr.get_batch(frame_indices)
    if isinstance(frames, torch.Tensor):
        frames = frames.numpy()
    else:
        frames = frames.asnumpy()
    frame_timestamps = [frame_index / fps for frame_index in frame_indices]

    return [Image.fromarray(fr).convert("RGB") for fr in frames]


def compute_frame_timestamps(duration, max_num_frames=16):
    if duration > max_num_frames:
        return [duration / max_num_frames * i for i in range(max_num_frames)]
    else:
        return [i for i in range(int(duration))]


def insert_subtitles_into_frames(frame_timestamps, subtitles, starting_timestamp_for_subtitles, duration):
    interleaved_list = []
    cur_i = 0

    for subtitle in subtitles:
        if "timestamp" in subtitle:
            start, end = subtitle["timestamp"]

            if not isinstance(end, float):
                end = duration

            start -= starting_timestamp_for_subtitles
            end -= starting_timestamp_for_subtitles

            subtitle_timestamp = (start + end) / 2
            subtitle_text = subtitle["text"]
        else:
            start, end = subtitle["start"], subtitle["end"]
            start = timestamp_to_seconds(start)
            end = timestamp_to_seconds(end)
            start -= starting_timestamp_for_subtitles
            end -= starting_timestamp_for_subtitles

            subtitle_timestamp = (start + end) / 2
            subtitle_text = subtitle["line"]

        for i, frame_timestamp in enumerate(frame_timestamps[cur_i:]):
            if frame_timestamp <= subtitle_timestamp:
                # print("frame:", frame_timestamp)
                interleaved_list.append("<image>")
                cur_i += 1
            else:
                break

        if end - start < 1:
            end = subtitle_timestamp + 0.5
            start = subtitle_timestamp - 0.5

        covering_frames = False
        for frame_timestamp in frame_timestamps:
            if frame_timestamp < end and frame_timestamp > start:
                covering_frames = True
                break

        if covering_frames:
            # print("subtitle:", subtitle_timestamp, start, end)
            interleaved_list.append(subtitle_text)
        else:
            pass
            # print("leaving out subtitle:", start, end)

    for i, frame_timestamp in enumerate(frame_timestamps[cur_i:]):
        # print(frame_timestamp)
        interleaved_list.append("<image>")

    return "\n".join(interleaved_list)

def longvideobench_doc_to_text(doc, lmms_eval_specific_kwargs):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    candidates = []

    for i in range(5):
        candidate = doc.get(f"option{i}")
        if candidate != "N/A":
            candidates.append(candidate)

    question = doc["question"] + "\n" + "\n".join([". ".join([chr(ord("A") + i), candidate]) for i, candidate in enumerate(candidates)])
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    if lmms_eval_specific_kwargs.get("insert_interleave_subtitles", False):
        with open(Path(__file__).parent / "longvideobench_val_i.yaml", "r") as f:
            raw_data = f.readlines()
            safe_data = []
            for i, line in enumerate(raw_data):
                # remove function definition since yaml load cannot handle it
                if "!function" not in line:
                    safe_data.append(line)
        cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]
        subtitle_subdir_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"].get("subtitle_subdir", "subtitles")
        cache_dir = os.path.join(base_cache_dir, cache_name, subtitle_subdir_name)
        with open(os.path.join(cache_dir, doc["subtitle_path"])) as f:
            subtitles = json.load(f)

        max_num_frames = yaml.safe_load("".join(safe_data))["dataset_kwargs"].get("max_num_frames", 16)

        frame_timestamps = compute_frame_timestamps(doc["duration"], max_num_frames)
        interleaved_prefix = insert_subtitles_into_frames(frame_timestamps, subtitles, doc["starting_timestamp_for_subtitles"], doc["duration"])
        full_prompt = "\n".join(part for part in [interleaved_prefix, question, post_prompt] if part)
    else:
        full_prompt = "\n".join(part for part in [question, post_prompt] if part)
    print("prompt:", full_prompt)
    return full_prompt


hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)


def longvideobench_doc_to_visual_v(doc):
    with open(Path(__file__).parent / "longvideobench_val_v.yaml", "r") as f:
        raw_data = f.readlines()
        safe_data = []
        for i, line in enumerate(raw_data):
            # remove function definition since yaml load cannot handle it
            if "!function" not in line:
                safe_data.append(line)
    cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]
    vid_subdir_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"].get("video_subdir", "videos/")
    cache_dir = os.path.join(base_cache_dir, cache_name, vid_subdir_name)
    video_path = doc["video_path"]
    video_path = os.path.join(cache_dir, video_path)
    return [video_path]


def longvideobench_doc_to_visual_i(doc):
    with open(Path(__file__).parent / "longvideobench_val_i.yaml", "r") as f:
        raw_data = f.readlines()
        safe_data = []
        for i, line in enumerate(raw_data):
            # remove function definition since yaml load cannot handle it
            if "!function" not in line:
                safe_data.append(line)
    cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]
    vid_subdir_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"].get("video_subdir", "videos/")
    cache_dir = os.path.join(base_cache_dir, cache_name, vid_subdir_name)
    video_path = doc["video_path"]
    video_path = os.path.join(cache_dir, video_path)
    max_num_frames = yaml.safe_load("".join(safe_data))["dataset_kwargs"].get("max_num_frames", 16)
    return load_video(video_path, doc["duration"], max_num_frames)


def get_multi_choice_info(options):
    """
    Given the list of options for multiple choice question
    Return the index2ans and all_choices
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/data_utils.py#L54
    """

    start_chr = "A"
    all_choices = []
    index2ans = {}
    for i, option in enumerate(options):
        index2ans[chr(ord(start_chr) + i)] = option
        all_choices.append(chr(ord(start_chr) + i))

    return index2ans, all_choices


def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Changed from MMMU-style complex parsing into simple parsing.
    Fixed to avoid 'D. A book' be parsed as A.
    Same as original LongVideoBench paper (from author Haoning Wu), if parsing failed, it will assign a random choice to model.
    """
    s = response.strip()
    answer_prefixes = [
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:",
        "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCDE]", s):
        return random.choice(all_choices)

    matches = re.search(r"[ABCDE]", s)
    if matches is None:
        return random.choice(all_choices)
    return matches[0]


def evaluate_longvideobench(samples):
    pred_correct = 0
    judge_dict = dict()
    for sample in samples:
        question = sample.get("question", "")
        options = sample.get("options", [])
        options_text = "\n".join(
            [f"{chr(ord('A') + i)}. {opt}" for i, opt in enumerate(options)]
        )
        full_question = f"{question}\n{options_text}" if options_text else question
        answer = sample.get("answer", "")
        prediction = sample.get("raw_output", sample.get("parsed_pred", ""))
        judge_prediction = extract_answer_content(prediction)

        custom_prompt = """You are a strict evaluator assessing answer correctness. You must output 1 for fully correct answers and 0 for any other case.
Question:
{question}
Ground Truth Answer:
{answer}
Model Prediction:
{prediction}
# Evaluation Rules for Multiple Choice Questions
- The model prediction may contain reasoning, but focus on the final answer.
- Score 1 if the predicted answer matches the ground truth answer.
- The answer can be given as just the letter (A, B, C, ...) or include the full option text.
- Ignore minor differences in formatting, capitalization, or spacing.
- Score 0 for any incorrect answer, even if the reasoning process seems correct.

Return only "1" or "0" with no additional text or formatting."""
        if use_vlm_judge() and server is not None:
            try:
                result = server.evaluate_binary(
                    question=full_question,
                    answer=answer,
                    prediction=judge_prediction,
                    output_format="0/1",
                    custom_prompt=custom_prompt,
                )
                print("Judge response:", result)

                if result["success"]:
                    judge_score = str(result["result"]).strip()
                    correct = judge_score == "1"
                    print(f"Judge evaluation score: {judge_score}, correct: {correct}")
                else:
                    eval_logger.error(
                        f"Judge evaluation failed: {result.get('raw_response', 'Unknown error')}"
                    )
                    correct = False
            except Exception as e:
                eval_logger.error(f"Error getting judge response: {e}")
                correct = False
        else:
            correct = eval_multi_choice(answer, sample.get("parsed_pred", ""))

        if correct:
            judge_dict[sample["id"]] = "Correct"
            pred_correct += 1
        else:
            judge_dict[sample["id"]] = "Wrong"

    if len(samples) == 0:
        return judge_dict, {"acc": 0}
    return judge_dict, {"acc": pred_correct / len(samples)}


def eval_multi_choice(gold_i, pred_i):
    correct = False
    # only they are exactly the same, we consider it as correct
    if isinstance(gold_i, list):
        for answer in gold_i:
            if answer == pred_i:
                correct = True
                break
    else:  # gold_i is a string
        if gold_i == pred_i:
            correct = True
    return correct


def calculate_ins_level_acc(results):
    """Calculate the instruction level accuracy for given Subject results
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L246
    """
    acc = 0
    ins_num = 0
    for cat_results in results.values():
        acc += cat_results["acc"] * cat_results["num_example"]
        ins_num += cat_results["num_example"]
    if ins_num == 0:
        return 0
    return acc / ins_num


def longvideobench_process_results(doc, results):
    pred = results[0]
    all_choices = []
    index2ans = {}
    options = []
    for i in range(5):
        option = doc.get(f"option{i}")
        if option == "N/A":
            break
        index2ans[chr(ord("A") + i)] = option
        all_choices.append(chr(ord("A") + i))
        options.append(option)

    parsed_pred = parse_multi_choice_response(pred, all_choices, index2ans)
    id = doc["id"]
    lvb_acc = {
        "id": id,
        "duration_group": doc["duration_group"],
        "question_category": doc["question_category"],
        "question": doc.get("question", ""),
        "options": options,
        "answer": chr(ord("A") + doc["correct_choice"]),
        "parsed_pred": parsed_pred,
        "raw_output": pred,
    }
    return {
        "lvb_acc": lvb_acc,
        "submission": {
            id: pred,
        },
    }


def longvideobench_aggregate_results(results):
    evaluation_result = {}
    subset_to_eval_samples = defaultdict(list)
    for result in results:
        subset_to_eval_samples[result["duration_group"]].append(result)
        subset_to_eval_samples[result["question_category"]].append(result)
    for subset, sub_eval_samples in subset_to_eval_samples.items():
        judge_dict, metric_dict = evaluate_longvideobench(sub_eval_samples)
        metric_dict.update({"num_example": len(sub_eval_samples)})
        evaluation_result[subset] = metric_dict
    printable_results = {}

    for cat_name, cat_results in evaluation_result.items():
        printable_results[cat_name] = {
            "num": int(cat_results["num_example"]),
            "acc": round(cat_results["acc"], 5),
        }
    all_ins_acc = calculate_ins_level_acc(evaluation_result)
    printable_results["Overall"] = {
        "num": sum([cat_results["num_example"] for cat_results in evaluation_result.values()]),
        "acc": round(all_ins_acc, 5),
    }
    printable_results["Evaluation mode"] = evaluation_mode()
    eval_logger.info(printable_results)
    return printable_results["Overall"]["acc"]


def longvideobench_aggregate_results_for_submission(results, args):
    path = generate_submission_file("longvideobench_test_for_submission.json", args)
    results_dict = {list(item.keys())[0]: list(item.values())[0] for item in results}
    with open(path, "w") as f:
        json.dump(results_dict, f)
    eval_logger.info(f"Results saved to {path}.")
