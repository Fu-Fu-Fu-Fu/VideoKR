import os
import re
from pathlib import Path
from collections import defaultdict

from loguru import logger as eval_logger
from lmms_eval.tasks._task_utils.judge_utils import (
    evaluation_mode,
    extract_answer_content,
    get_judge_server,
    normalize_answer_letter,
    rule_correct_multiple_choice,
    use_vlm_judge,
)
  
# Initialize the optional judge server when API credentials are available.
API_TYPE = os.getenv("API_TYPE", "azure")
MODEL_VERSION = os.getenv("MODEL_VERSION", "gpt-4o")  
server = get_judge_server(API_TYPE, MODEL_VERSION)

_VIDEOKR_ROOT_CACHE = None
_VIDEO_SUFFIXES = ("", ".mp4", ".MP4", ".mkv", ".avi", ".mov")


def _get_videokr_root():
    """Return a local VideoKR dataset root without hardcoding machine paths."""
    global _VIDEOKR_ROOT_CACHE
    if _VIDEOKR_ROOT_CACHE is not None:
        return _VIDEOKR_ROOT_CACHE

    local_root = os.getenv("VIDEOKR_EVAL_ROOT")
    if local_root:
        _VIDEOKR_ROOT_CACHE = Path(local_root).expanduser()
        return _VIDEOKR_ROOT_CACHE

    repo_id = os.getenv("VIDEOKR_EVAL_REPO", "minuzero/VideoKR-Eval")
    try:
        from huggingface_hub import snapshot_download

        _VIDEOKR_ROOT_CACHE = Path(snapshot_download(repo_id=repo_id, repo_type="dataset"))
        return _VIDEOKR_ROOT_CACHE
    except Exception as exc:
        eval_logger.warning(
            "Unable to locate VideoKR videos automatically. Set VIDEOKR_EVAL_ROOT "
            f"to a local dataset clone or snapshot. Original error: {exc}"
        )
        return None


def _video_path_variants(path: Path):
    seen = set()
    for suffix in _VIDEO_SUFFIXES:
        candidate = path if suffix == "" else path.with_suffix(suffix)
        if candidate not in seen:
            seen.add(candidate)
            yield candidate


def videokr_eval_doc_to_visual(doc):
    raw_video_path = str(doc.get("video_path", "")).strip()
    if not raw_video_path:
        raise FileNotFoundError("Missing video_path in document.")

    raw_path = Path(raw_video_path).expanduser()
    candidates = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        dataset_root = _get_videokr_root()
        if dataset_root is not None:
            candidates.append(dataset_root / raw_video_path.lstrip("./"))
        candidates.append(Path.cwd() / raw_video_path.lstrip("./"))

    for candidate in candidates:
        for variant in _video_path_variants(candidate):
            if variant.exists():
                return [str(variant)]

    raise FileNotFoundError(
        f"Video not found for '{raw_video_path}'. Set VIDEOKR_EVAL_ROOT to the "
        "directory containing the VideoKR video files."
    )

def format_options(opts):
    if not opts:
        return ""
    if isinstance(opts, dict):
        keys = sorted(opts.keys())
        return "\n".join(f"{k}. {opts[k]}" for k in keys)
    if isinstance(opts, list):
        return "\n".join(str(opt).strip() for opt in opts)
    raise TypeError(f"Unsupported options type: {type(opts)}")

def videokr_eval_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    post_prompt_mc = lmms_eval_specific_kwargs.get(
        "post_prompt_mc",
        "",
    )
    post_prompt_oe = lmms_eval_specific_kwargs.get(
        "post_prompt_oe",
        "",
    )
    question = doc.get("question", "")
    options_text = format_options(doc.get("option", doc.get("options", [])))
    problem_type = str(doc.get("problem_type", "")).strip().lower().replace(" ", "-")
    if not problem_type:
        problem_type = "multiple-choice" if options_text else "open-ended"
    if problem_type == "multiple-choice":
        post_prompt = post_prompt_mc if post_prompt == "" else post_prompt
    else:
        post_prompt = post_prompt_oe if post_prompt == "" else post_prompt
    if options_text:
        input_text = f"{pre_prompt}{question}\n{options_text}\n{post_prompt}"
    else:
        input_text = f"{pre_prompt}{question}\n{post_prompt}"
    return input_text


def videokr_eval_doc_to_choice(doc):
    return ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

def videokr_eval_doc_to_target(doc):
    return str(doc.get("solution", doc.get("answer", doc.get("ground_truth", "")))).strip()

def extract_answer_letter(s):
    s = s.strip()
    match = re.search(r"<answer>\\s*([A-Z])\\s*</answer>", s, flags=re.I)
    if match:
        return match.group(1).upper()

    blocks = re.findall(r"<answer[^>]*>(.*?)</answer>", s, flags=re.I | re.S)
    if blocks:
        inner = " ".join(blocks)
        prefixes = [
            "The answer is",
            "The correct answer is",
            "Answer:",
            "Option:",
            "the final answer is",
        ]
        for prefix in prefixes:
            inner = inner.replace(prefix, "")
        m2 = re.search(r"\\b([A-Z])\\b", inner, flags=re.I)
        if m2:
            return m2.group(1).upper()

    m3 = re.search(r"\\b([A-Z])\\b", s, flags=re.I)
    if m3:
        return m3.group(1).upper()
    return ""


def videokr_eval_process_results(doc, results):
    prediction = results[0].strip() if isinstance(results, list) else str(results).strip()
    judge_prediction = extract_answer_content(prediction)

    question = doc.get("question", "")
    options_text = format_options(doc.get("option", doc.get("options", [])))
    full_question = f"{question}\n{options_text}" if options_text else question
    answer = str(doc.get("solution", doc.get("answer", ""))).strip()
    problem_type = str(doc.get("problem_type", "")).strip().lower().replace(" ", "-")
    if not problem_type:
        problem_type = "multiple-choice" if options_text else "open-ended"

    if problem_type == "multiple-choice":
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
    else:
        custom_prompt = """You are a fair evaluator assessing answer correctness.
Question:
{question}
Ground Truth Answer:
{answer}
Model Prediction:
{prediction}
# Scoring Rules
- Score 1: The prediction is correct - it captures the key information from the ground truth answer. Minor wording differences, extra non-contradictory details, or more verbose phrasing are acceptable.
- Score 0: The prediction is clearly wrong - it contradicts the ground truth, is completely irrelevant, or misses all key information.

# Additional Guidelines
- Focus on the final answer in the prediction, ignoring any reasoning process.
- Ignore differences in formatting, capitalization, or spacing.
- Treat numerical answers as correct if they match within reasonable precision.
- If the ground truth is very short (e.g., a single word or symbol), score 1 if the prediction contains that information in a reasonable context.
- A more detailed answer that includes the ground truth information plus reasonable elaboration should score 1, not be penalized.
- Synonyms, abbreviations, and equivalent expressions (e.g., "DHT" for "Dihydrotestosterone", "O(n log n)" for "n log n", "High blood pressure" for "Hypertension") should score 1.

Return only "1" or "0" with no additional text or formatting."""

    if problem_type == "multiple-choice":
        if use_vlm_judge() and server is not None:
            try:
                result = server.evaluate_binary(
                    question=full_question,
                    answer=answer,
                    prediction=judge_prediction,
                    output_format="0/1",
                    custom_prompt=custom_prompt,
                )
                if result["success"]:
                    judge_score = str(result["result"]).strip()
                    correct = judge_score == "1"
                else:
                    eval_logger.error(f"Judge evaluation failed: {result.get('raw_response', 'Unknown error')}")
                    correct = False
            except Exception as e:
                eval_logger.error(f"Error getting judge response: {e}")
                correct = False
        else:
            correct = rule_correct_multiple_choice(judge_prediction, answer, choices="ABCDEFGHIJ")
        extracted_answer = normalize_answer_letter(judge_prediction, choices="ABCDEFGHIJ") or extract_answer_letter(judge_prediction) or judge_prediction
        skipped = False
    else:
        skipped = not (use_vlm_judge() and server is not None)
        if skipped:
            correct = False
        else:
            try:
                result = server.evaluate_binary(
                    question=full_question,
                    answer=answer,
                    prediction=judge_prediction,
                    output_format="0/1",
                    custom_prompt=custom_prompt,
                )
                if result["success"]:
                    judge_score = str(result["result"]).strip()
                    correct = judge_score == "1"
                else:
                    eval_logger.error(f"Judge evaluation failed: {result.get('raw_response', 'Unknown error')}")
                    correct = False
            except Exception as e:
                eval_logger.error(f"Error getting judge response: {e}")
                correct = False
        extracted_answer = judge_prediction[:100] + "..." if len(judge_prediction) > 100 else judge_prediction

    data_dict = {
        "id": doc.get("id", doc.get("video_path", "")),
        "dataset": doc.get("dataset", "UNKNOWN"),
        "question_type": problem_type,
        "subject": doc.get("subject", doc.get("source", "UNKNOWN")),
        "pred_answer": extracted_answer,
        "answer": answer,
        "correct": correct,
        "skipped": skipped,
        "evaluation_mode": evaluation_mode(),
        "raw_output": results[0] if isinstance(results, list) else results,
    }

    return {
        "videokr_eval_acc": data_dict
    }


def videokr_eval_aggregate_results(results):
    by_qtype = defaultdict(lambda: {"correct": 0, "total": 0})
    by_subject = defaultdict(lambda: {"correct": 0, "total": 0})
    by_dataset = defaultdict(lambda: {"correct": 0, "total": 0})

    total_correct = 0
    total_examples = 0
    skipped_examples = 0

    for r in results:
        if r.get("skipped", False):
            skipped_examples += 1
            continue
        qtype = r.get("question_type", "UNKNOWN")
        subject = r.get("subject", "UNKNOWN")
        dataset = r.get("dataset", "UNKNOWN")
        is_correct = 1 if r.get("correct", False) else 0

        by_qtype[qtype]["correct"] += is_correct
        by_qtype[qtype]["total"] += 1
        by_subject[subject]["correct"] += is_correct
        by_subject[subject]["total"] += 1
        by_dataset[dataset]["correct"] += is_correct
        by_dataset[dataset]["total"] += 1

        total_correct += is_correct
        total_examples += 1

    def summarize(bucket):
        out = {}
        for k, v in bucket.items():
            acc = (v["correct"] / v["total"]) if v["total"] > 0 else 0.0
            out[k] = {"num": v["total"], "acc": round(acc * 100, 2)}
        return out

    printable_qtype = summarize(by_qtype)
    printable_subject = summarize(by_subject)
    printable_dataset = summarize(by_dataset)
    overall_acc = round((total_correct / total_examples) * 100, 2) if total_examples else 0.0

    print("\nVideoKR Evaluation Results:")
    print(f"Evaluation mode: {evaluation_mode()}")
    if skipped_examples:
        print(f"Skipped open-ended examples without VLM judge: {skipped_examples}")
    print(f"Overall Accuracy: {overall_acc}%")

    print("\nStatistics by Question Type:")
    for k, v in printable_qtype.items():
        print(f"{k}: {v['acc']}% (samples: {v['num']})")

    print("\nStatistics by Subject:")
    for k, v in printable_subject.items():
        print(f"{k}: {v['acc']}% (samples: {v['num']})")

    print("\nStatistics by Dataset:")
    for k, v in printable_dataset.items():
        print(f"{k}: {v['acc']}% (samples: {v['num']})")

    return {
        "overall_acc": overall_acc,
        "evaluated_examples": total_examples,
        "skipped_examples": skipped_examples,
        "evaluation_mode": evaluation_mode(),
        "by_question_type": printable_qtype,
        "by_subject": printable_subject,
        "by_dataset": printable_dataset,
    }
