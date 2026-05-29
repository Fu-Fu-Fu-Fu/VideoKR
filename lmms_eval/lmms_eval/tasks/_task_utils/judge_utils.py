import os
import re


def _env_truthy(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def use_vlm_judge():
    override = os.getenv("VIDEOKR_USE_VLM_JUDGE")
    if override is not None:
        return _env_truthy(override)

    api_type = os.getenv("API_TYPE", "azure").strip().lower()
    if api_type == "azure":
        return bool(os.getenv("AZURE_API_KEY") and os.getenv("AZURE_ENDPOINT"))
    if api_type == "openai":
        return bool(os.getenv("OPENAI_API_KEY"))
    return False


def get_judge_server(api_type=None, model_version=None):
    if not use_vlm_judge():
        return None

    from lmms_eval.llm_judge import ServerConfig, get_server

    server_config = ServerConfig(
        model_name=model_version or os.getenv("MODEL_VERSION", "gpt-4o"),
        temperature=1,
        max_tokens=256,
    )
    return get_server(server_name=api_type or os.getenv("API_TYPE", "azure"), config=server_config)


def evaluation_mode():
    return "vlm_judge" if use_vlm_judge() else "rule"


def extract_answer_content(prediction):
    prediction = str(prediction)
    match = re.search(r"<answer>(.*?)</answer>", prediction, flags=re.IGNORECASE | re.DOTALL)
    if match:
        answer_text = match.group(1).strip()
        if answer_text:
            return answer_text
    return prediction


def normalize_answer_letter(text, choices="ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    text = extract_answer_content(text)
    s = str(text).strip()
    if not s:
        return ""

    choices = "".join(dict.fromkeys(choices.upper()))
    choice_class = re.escape(choices)

    patterns = [
        rf"^\s*[\(\[]?([{choice_class}])[\)\]\.:\s]",
        rf"\b(?:answer|option|choice)\s*(?:is|:)?\s*[\(\[]?([{choice_class}])[\)\]]?\b",
        rf"\b([{choice_class}])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, s, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return ""


def rule_correct_multiple_choice(prediction, answer, choices="ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    pred_letter = normalize_answer_letter(prediction, choices)
    answer_letter = normalize_answer_letter(answer, choices)
    if not answer_letter and len(str(answer).strip()) == 1:
        candidate = str(answer).strip().upper()
        if candidate in choices.upper():
            answer_letter = candidate
    return bool(pred_letter and answer_letter and pred_letter == answer_letter)
