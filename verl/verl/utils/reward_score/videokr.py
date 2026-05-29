# Copyright 2026 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import re
from typing import Any

from rouge_score import rouge_scorer

_ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
_FORMAT_PATTERN = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
_ROUGE_TYPES = ["rouge1", "rouge2", "rougeL"]
_ACCURACY_WEIGHT = 0.9
_FORMAT_WEIGHT = 0.1

SUPPORTED_PROBLEM_TYPES = {"multiple choice", "open-ended"}


def extract_answer(text: str) -> str:
    text = str(text or "")
    match = _ANSWER_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return ""


def compute_rouge_score(reference: str, hypothesis: str) -> float:
    scorer = rouge_scorer.RougeScorer(_ROUGE_TYPES, use_stemmer=True)
    scores = scorer.score(str(reference), str(hypothesis))
    return (scores["rouge1"].fmeasure + scores["rouge2"].fmeasure + scores["rougeL"].fmeasure) / 3.0


def format_reward(solution_str: str) -> float:
    return 1.0 if _FORMAT_PATTERN.fullmatch(str(solution_str or "")) else 0.0


def accuracy_reward(solution_str: str, ground_truth: str, problem_type: str) -> float:
    output_ans = extract_answer(solution_str)
    gt_ans = extract_answer(ground_truth)

    if problem_type == "multiple choice":
        return 1.0 if output_ans.strip() == gt_ans.strip() else 0.0

    if problem_type == "open-ended":
        score = compute_rouge_score(gt_ans, output_ans)
        return max(0.0, min(1.0, score))

    return 0.0


def compute_score(
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    format_weight: float = 1.0,
    use_format_reward: bool = True,
    return_dict: bool = False,
    data_source: str | None = None,
    **kwargs,
) -> float | dict[str, float]:
    """Compute the VideoKR rule reward.

    The scoring design is intentionally unchanged:
    - multiple-choice questions use exact answer matching;
    - open-ended questions use the average F1 of ROUGE-1/2/L;
    - final score is 0.9 * accuracy_score + 0.1 * format_score.
    """
    _ = (format_weight, data_source, kwargs)

    problem_type = ""
    if extra_info and hasattr(extra_info, "get"):
        problem_type = extra_info.get("problem_type") or ""
    problem_type = str(problem_type)

    try:
        accurate_score = accuracy_reward(solution_str=solution_str, ground_truth=ground_truth, problem_type=problem_type)
    except Exception:
        accurate_score = 0.0

    format_score = format_reward(solution_str) if use_format_reward else 0.0
    score = _ACCURACY_WEIGHT * accurate_score + _FORMAT_WEIGHT * format_score

    if return_dict:
        return {
            "score": score,
            "format_score": format_score,
            "accurate_score": accurate_score,
        }
    return score
