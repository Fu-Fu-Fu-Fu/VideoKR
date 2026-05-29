#!/usr/bin/env python3
"""Convert VideoKR-Train COT jsonl into LLaMA-Factory ShareGPT SFT format."""

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", default="data/raw/VideoKR-COT-201K.jsonl", help="Path to VideoKR-COT jsonl.")
    parser.add_argument("--output", default="data/videokr_train.json", help="Output SFT json file.")
    parser.add_argument(
        "--video-prefix",
        default="",
        help="Optional prefix added to every video path, for example `videos` if files live under data/videos/.",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Convert only the first N valid samples.")
    parser.add_argument(
        "--skip-missing-video",
        action="store_true",
        help="Skip samples whose video file is missing under --media-dir.",
    )
    parser.add_argument(
        "--media-dir",
        default=None,
        help="Optional root used with --skip-missing-video. Defaults to the output file directory.",
    )
    return parser.parse_args()


def normalize_video_path(path: str, prefix: str = "") -> str:
    path = path.strip()
    while path.startswith("./"):
        path = path[2:]
    path = path.lstrip("/")

    if prefix:
        prefix = prefix.strip().strip("/")
        if prefix and not path.startswith(prefix + "/"):
            path = f"{prefix}/{path}"

    return path


def is_multiple_choice(example: dict[str, Any]) -> bool:
    problem_type = str(example.get("problem_type", "")).strip().lower()
    if problem_type in {"multiple choice", "multiple-choice", "mcq"}:
        return True
    return bool(example.get("options"))


def format_options(options: Any) -> str:
    if not options:
        return ""

    if isinstance(options, dict):
        lines = [f"{key}. {value}" for key, value in options.items()]
    elif isinstance(options, list):
        lines = [str(option).strip() for option in options]
    else:
        lines = [str(options).strip()]

    lines = [line for line in lines if line.strip()]
    if not lines:
        return ""

    return "\n\nOptions:\n" + "\n".join(lines)


def build_user_instruction(example: dict[str, Any]) -> str:
    question = str(example.get("question", "")).strip()
    user_instruction = f"{question}{format_options(example.get('options'))}"

    if is_multiple_choice(example):
        user_instruction += (
            "\nPlease answer this question based on the visual content.\n"
            "Provide your thinking process between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags.\n"
            "Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer>...</answer> tags."
        )
    else:
        user_instruction += (
            "\nPlease answer this question based on the visual content.\n"
            "Provide your thinking process between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags.\n"
            "Please provide only your final answer within the <answer>...</answer> tags.\n"
        )

    return user_instruction


def strip_tag(text: str, tag: str) -> str:
    text = text.strip()
    open_tag = f"<{tag}>"
    close_tag = f"</{tag}>"
    if text.startswith(open_tag) and close_tag in text:
        text = text.replace(open_tag, "", 1)
        text = text.rsplit(close_tag, 1)[0]
    return text.strip()


def build_response(example: dict[str, Any]) -> str:
    solution = str(example.get("solution", "")).strip()
    think = strip_tag(str(example.get("think", "")).strip(), "think")
    answer = strip_tag(solution, "answer")

    if not think or not answer:
        return ""

    return f"<think>{think}</think>\n<answer>{answer}</answer>"


def convert_example(example: dict[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
    user_instruction = build_user_instruction(example)
    answer = build_response(example)
    raw_video_path = example.get("video_path") or example.get("path") or example.get("file_name") or ""
    video_path = normalize_video_path(str(raw_video_path), args.video_prefix)

    if not user_instruction or not answer or not video_path:
        return None

    if args.skip_missing_video:
        media_dir = Path(args.media_dir) if args.media_dir else Path(args.output).resolve().parent
        if not (media_dir / video_path).is_file():
            return None

    return {
        "messages": [
            {"role": "user", "content": f"<video>{user_instruction}"},
            {"role": "assistant", "content": answer},
        ],
        "videos": [video_path],
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    converted = []
    skipped = 0
    with input_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                example = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON on line {line_no}: {exc}") from exc

            item = convert_example(example, args)
            if item is None:
                skipped += 1
                continue

            converted.append(item)
            if args.max_samples is not None and len(converted) >= args.max_samples:
                break

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Wrote {len(converted)} samples to {output_path}")
    if skipped:
        print(f"Skipped {skipped} samples")


if __name__ == "__main__":
    main()
