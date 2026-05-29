#!/usr/bin/env python3
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

"""Convert VideoKR RL data into verl parquet files.

The default input is the RL jsonl file in the Hugging Face dataset
`minuzero/VideoKR-Train`. A local JSON/JSONL path can be supplied with
`--input_json` when the dataset has already been downloaded.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

import datasets

ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", flags=re.IGNORECASE | re.DOTALL)
VIDEO_PREFIX = "<video>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare VideoKR RL parquet data for verl")
    parser.add_argument("--dataset_name", default="minuzero/VideoKR-Train")
    parser.add_argument("--data_file", default="VideoKR-RL-114K.jsonl")
    parser.add_argument("--input_json", type=Path, default=None, help="Local JSON/JSONL file. Overrides HF loading.")
    parser.add_argument("--output_dir", type=Path, default=Path("data/videokr_rl"))
    parser.add_argument(
        "--video_base_dir",
        type=Path,
        default=None,
        help="Optional local dataset root or Videos directory used to resolve relative video paths.",
    )
    parser.add_argument("--train_size", type=int, default=-1, help="-1 keeps all non-validation samples.")
    parser.add_argument("--val_size", type=int, default=500)
    parser.add_argument("--max_samples", type=int, default=-1, help="Optional cap for smoke tests.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max_frames", type=int, default=128)
    parser.add_argument("--min_frames", type=int, default=4)
    parser.add_argument("--max_pixels", type=int, default=128 * 28 * 28)
    parser.add_argument("--min_pixels", type=int, default=128 * 28 * 28)
    parser.add_argument("--use_file_uri", action="store_true")
    parser.add_argument("--save_json", action="store_true", help="Also save the transformed full JSON.")
    return parser.parse_args()


def load_local_items(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        head = f.read(4096)
        if not head:
            return []
        if head.lstrip().startswith("["):
            return json.loads(head + f.read())

    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
    return items


def load_items(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.input_json is not None:
        return load_local_items(args.input_json)

    dataset = datasets.load_dataset(args.dataset_name, data_files=args.data_file, split="train")
    return [dict(item) for item in dataset]


def extract_answer(text: Any) -> str:
    raw = str(text or "").strip()
    match = ANSWER_PATTERN.search(raw)
    if match:
        return match.group(1).strip()
    return raw


def ensure_answer_tags(text: Any) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    if ANSWER_PATTERN.search(raw):
        return raw
    return f"<answer>{raw}</answer>"


def strip_video_prefix(text: Any) -> str:
    prompt = str(text or "").strip()
    if prompt.lower().startswith(VIDEO_PREFIX):
        return prompt[len(VIDEO_PREFIX) :].lstrip()
    return prompt


def ensure_video_prefix(text: str) -> str:
    text = text.strip()
    if text.lower().startswith(VIDEO_PREFIX):
        return text
    return f"{VIDEO_PREFIX}\n{text}"


def normalize_problem_type(item: dict[str, Any]) -> str:
    problem_type = str(item.get("problem_type") or item.get("question_type") or "").strip().lower()
    if "multiple" in problem_type and "choice" in problem_type:
        return "multiple choice"
    if problem_type in {"open-ended", "open_ended", "free-form", "free form", "numerical"}:
        return "open-ended"
    if item.get("options"):
        return "multiple choice"
    return "open-ended"


def normalize_options(options: Any) -> list[str]:
    if options is None:
        return []
    if isinstance(options, str):
        return [line.strip() for line in options.splitlines() if line.strip()]
    if isinstance(options, dict):
        return [f"{key}. {value}" for key, value in sorted(options.items())]
    if isinstance(options, list):
        return [str(option).strip() for option in options if str(option).strip()]
    return []


def build_question(question: Any, options: list[str], problem_type: str) -> str:
    question_text = strip_video_prefix(question)
    if problem_type != "multiple choice" or not options:
        return question_text

    if "Options:" in question_text or "\nA." in question_text or "\nA:" in question_text:
        return question_text

    return question_text.rstrip() + "\nOptions:\n" + "\n".join(options)


def build_prompt(question: Any, options: list[str], problem_type: str) -> str:
    question_text = build_question(question, options, problem_type)
    instruction = (
        "\nPlease answer this question based on the visual content.\n"
        "Provide your thinking process between the <think> and </think> tags, "
        "and then give your final answer between the <answer> and </answer> tags.\n"
    )
    if problem_type == "multiple choice":
        instruction += (
            "Please provide only the single option letter (e.g., A, B, C, D, etc.) "
            "within the <answer>...</answer> tags."
        )
    else:
        instruction += "Please provide only your final answer within the <answer>...</answer> tags."
    return ensure_video_prefix(question_text + instruction)


def collect_video_paths(item: dict[str, Any]) -> list[str]:
    raw_videos = item.get("videos")
    if isinstance(raw_videos, list):
        paths = []
        for entry in raw_videos:
            if isinstance(entry, dict):
                path = entry.get("video") or entry.get("path")
            else:
                path = entry
            if path:
                paths.append(str(path))
        if paths:
            return paths

    for key in ("file_name", "video", "video_path", "video_file", "path"):
        value = item.get(key)
        if value:
            return [str(value)]
    return []


def resolve_video_path(raw_path: str, video_base_dir: Path | None, use_file_uri: bool) -> str:
    raw_path = raw_path.strip()
    if not raw_path or "://" in raw_path:
        return raw_path

    path = Path(raw_path)
    if path.is_absolute():
        resolved = path.resolve()
    elif video_base_dir is None:
        return path.resolve().as_uri() if use_file_uri else raw_path
    else:
        stripped = raw_path.lstrip("./")
        candidates = [video_base_dir / stripped]
        if stripped.startswith("Videos/"):
            candidates.append(video_base_dir / stripped[len("Videos/") :])
        else:
            candidates.append(video_base_dir / "Videos" / stripped)
        resolved = next((candidate for candidate in candidates if candidate.exists()), candidates[0])

    resolved = resolved.resolve()
    return resolved.as_uri() if use_file_uri else str(resolved)


def get_ground_truth(item: dict[str, Any]) -> str:
    for key in ("solution", "answer", "correct_answer", "ground_truth"):
        value = item.get(key)
        if value is not None and str(value).strip():
            return ensure_answer_tags(value)
    return ""


def make_record(item: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    problem_type = normalize_problem_type(item)
    options = normalize_options(item.get("options"))
    prompt_text = build_prompt(item.get("question") or item.get("prompt") or "", options, problem_type)
    raw_video_paths = collect_video_paths(item)
    resolved_video_paths = [
        resolve_video_path(path, args.video_base_dir, args.use_file_uri) for path in raw_video_paths
    ]
    ground_truth = get_ground_truth(item)

    return {
        "data_source": "videokr",
        "prompt": [{"role": "user", "content": prompt_text}],
        "videos": [
            {
                "video": video_path,
                "fps": args.fps,
                "max_frames": args.max_frames,
                "min_frames": args.min_frames,
                "max_pixels": args.max_pixels,
                "min_pixels": args.min_pixels,
            }
            for video_path in resolved_video_paths
        ],
        "ability": "video_qa",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {
            "problem_id": item.get("problem_id") or item.get("id") or item.get("file_name"),
            "problem_type": problem_type,
            "answer": extract_answer(ground_truth),
            "solution": item.get("solution", ""),
            "source_path": raw_video_paths[0] if len(raw_video_paths) == 1 else raw_video_paths,
        },
    }


def split_records(records: list[dict[str, Any]], train_size: int, val_size: int, seed: int) -> tuple[list, list]:
    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)

    val_size = min(max(val_size, 0), len(indices))
    val_indices = set(indices[:val_size])
    train_candidates = indices[val_size:]
    if train_size >= 0:
        train_candidates = train_candidates[: min(train_size, len(train_candidates))]
    train_indices = set(train_candidates)

    train_records = []
    val_records = []
    train_index = 0
    val_index = 0
    for index, record in enumerate(records):
        if index in val_indices:
            record["extra_info"]["split"] = "test"
            record["extra_info"]["index"] = val_index
            val_records.append(record)
            val_index += 1
        elif index in train_indices:
            record["extra_info"]["split"] = "train"
            record["extra_info"]["index"] = train_index
            train_records.append(record)
            train_index += 1

    return train_records, val_records


def main() -> None:
    args = parse_args()
    raw_items = load_items(args)
    if args.max_samples >= 0:
        raw_items = raw_items[: args.max_samples]

    records = [make_record(item, args) for item in raw_items]
    records = [record for record in records if record["videos"] and record["reward_model"]["ground_truth"]]
    if not records:
        raise ValueError("No valid VideoKR records found after filtering missing videos or answers.")

    train_records, val_records = split_records(records, args.train_size, args.val_size, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_path = args.output_dir / "train.parquet"
    val_path = args.output_dir / "test.parquet"
    datasets.Dataset.from_list(train_records).to_parquet(str(train_path))
    datasets.Dataset.from_list(val_records).to_parquet(str(val_path))

    if args.save_json:
        (args.output_dir / "videokr_rlhf_full.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"input_total={len(raw_items)}")
    print(f"valid_records={len(records)}")
    print(f"train_size={len(train_records)}")
    print(f"test_size={len(val_records)}")
    print(f"train_output={train_path}")
    print(f"test_output={val_path}")


if __name__ == "__main__":
    main()
