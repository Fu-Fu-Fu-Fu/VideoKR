# VideoKR RL Training

This folder contains the VideoKR reinforcement-learning training code based on
`verl`. It provides:

- `local_script/prepare_videokr_rl_data.py`: convert VideoKR RL data to verl parquet files
- `local_script/train_videokr_grpo.sh`: train Qwen2.5-VL or Qwen3-VL with GRPO
- `local_script/merge_videokr_checkpoint.sh`: merge verl FSDP checkpoints to Hugging Face format

The RL data is released at
[minuzero/VideoKR-Train](https://huggingface.co/datasets/minuzero/VideoKR-Train).

## Environment

Create a conda environment:

```bash
conda create -n videokr_verl python=3.11 -y
conda activate videokr_verl

pip install --upgrade pip setuptools wheel
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 xformers==0.0.32.post1
pip install "vllm==0.11.0" "transformers==4.57.6" "ray[default]==2.55.1" "numpy==1.26.4"
pip install -e ".[vllm,gpu]" --no-build-isolation
pip install qwen-vl-utils rouge-score
pip check
```

The training script keeps the original VideoKR GRPO hyperparameters by default,
including `train_batch_size=32`, `lr=5e-6`, `kl_loss_coef=0.01`,
`use_remove_padding=True`, and `use_fused_kernels=True`.

## Prepare Data

Download the VideoKR RL annotation file and videos from
[minuzero/VideoKR-Train](https://huggingface.co/datasets/minuzero/VideoKR-Train).
Then convert `VideoKR-RL-114K.jsonl` to verl parquet files:

```bash
python local_script/prepare_videokr_rl_data.py \
  --dataset_name minuzero/VideoKR-Train \
  --data_file VideoKR-RL-114K.jsonl \
  --video_base_dir /path/to/VideoKR-Train \
  --output_dir data/videokr_rl
```

`--video_base_dir` should point to the extracted dataset root, or to the
directory that contains the video files. The script writes:

```text
data/videokr_rl/train.parquet
data/videokr_rl/test.parquet
```

## Train

Run GRPO training with a Qwen-VL base model:

```bash
MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct \
N_GPUS=8 \
bash local_script/train_videokr_grpo.sh
```

For Qwen2.5-VL, only change `MODEL_PATH`:

```bash
MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct \
N_GPUS=8 \
bash local_script/train_videokr_grpo.sh
```

Useful overrides:

```bash
DATA_DIR=data/videokr_rl
OUTPUT_DIR=outputs/videokr_grpo/videokr_qwen_vl
TRAIN_BATCH_SIZE=32
ROLLOUT_N=8
SAVE_FREQ=20
TEST_FREQ=20
```

Checkpoints are saved under:

```text
outputs/videokr_grpo/videokr_qwen_vl/checkpoints/global_step_x/actor
```

## Merge Checkpoint

After training, merge the verl FSDP checkpoint to Hugging Face format:

```bash
bash local_script/merge_videokr_checkpoint.sh \
  outputs/videokr_grpo/videokr_qwen_vl/checkpoints/global_step_20/actor \
  outputs/videokr_grpo/videokr_qwen_vl/merged_hf
```

The merged model can be loaded with Hugging Face `AutoProcessor` and
`AutoModelForImageTextToText`.

## Reward

The VideoKR reward is registered as `data_source="videokr"`.
Multiple-choice questions use exact matching. Open-ended questions use the
average F1 of ROUGE-1/ROUGE-2/ROUGE-L. The final reward is:

```text
0.9 * answer_score + 0.1 * format_score
```

where the expected output format is:

```text
<think>...</think><answer>...</answer>
```

## Acknowledgement

This codebase is built on top of
[verl](https://github.com/verl-project/verl), released under the Apache-2.0
license.
