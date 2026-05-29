# VideoKR Evaluation

This folder contains the `lmms_eval` code used to evaluate VideoKR models on video understanding benchmarks.

## Environment

Create a conda environment:

```bash
conda create -n videokr-eval python=3.11 -y
conda activate videokr-eval

pip install --upgrade pip setuptools wheel
pip install -e ".[vllm]"
```

If your machine needs a specific CUDA/PyTorch/vLLM version, install the matching `torch` and `vllm` first, then run:

```bash
pip install -e .
```

## Data

`videokr_eval` uses the Hugging Face dataset [minuzero/VideoKR-Eval](https://huggingface.co/datasets/minuzero/VideoKR-Eval). The first run will download it automatically.

If you already have the dataset locally, set:

```bash
export VIDEOKR_EVAL_ROOT=/path/to/VideoKR-Eval
```

## Evaluate on VideoKR-Eval

Set the model path and run the evaluation script:

```bash
conda activate videokr-eval
cd /path/to/VideoKR/lmms_eval

export CUDA_VISIBLE_DEVICES=0
export VIDEOKR_MODEL=/path/to/your/VideoKR-model
export TASKS=videokr_eval
export BATCH_SIZE=1
export RUN_NAME=videokr_eval_test

bash examples/models/videokr_vllm.sh
```

For a quick sanity check, evaluate only 10 samples:

```bash
export LIMIT=10
bash examples/models/videokr_vllm.sh
```

For full evaluation, leave `LIMIT` empty:

```bash
export LIMIT=
bash examples/models/videokr_vllm.sh
```

Results are saved under:

```text
outputs/videokr/${RUN_NAME}_vllm/
```

## API Judge

If no API key is provided, the evaluator runs in rule mode:

- multiple-choice questions are evaluated by answer-letter matching
- open-ended questions are skipped
- the final result reports `evaluation_mode` and `skipped_examples`

To use a VLM judge for open-ended questions, set either Azure OpenAI:

```bash
export API_TYPE=azure
export AZURE_API_KEY=...
export AZURE_ENDPOINT=...
export API_VERSION=2024-02-15-preview
export MODEL_VERSION=gpt-4o
```

or OpenAI-compatible API:

```bash
export API_TYPE=openai
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://api.openai.com/v1
export MODEL_VERSION=gpt-4o
```

Then run:

```bash
bash examples/models/videokr_vllm.sh
```
