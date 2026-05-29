# VideoKR Training

This folder contains the LLaMA-Factory training code used for VideoKR supervised fine-tuning on Qwen2.5-VL and Qwen3-VL.

## Environment

```bash
conda create -n videokr_train python=3.11 pip -y
conda activate videokr_train

pip install --upgrade pip setuptools wheel
pip install -e . -r requirements/deepspeed.txt
```

The install step downloads PyTorch, CUDA runtime wheels, Transformers, DeepSpeed, and the video processing dependencies.

## Data Preparation

The training data and videos are hosted at [minuzero/VideoKR-Train](https://huggingface.co/datasets/minuzero/VideoKR-Train). This repository does not include raw data, videos, or checkpoints.

Download the SFT annotation file `VideoKR-COT-201K.jsonl` and convert it to the LLaMA-Factory ShareGPT format:

```bash
mkdir -p data/raw

huggingface-cli download minuzero/VideoKR-Train \
  --repo-type dataset \
  --local-dir data/raw \
  --include "VideoKR-COT-201K.jsonl"

python local_script/prepare_videokr_sft_data.py \
  --input data/raw/VideoKR-COT-201K.jsonl \
  --output data/videokr_train.json
```

The converter creates `data/videokr_train.json`. `data/dataset_info.json` is already configured to use this file. It uses the VideoKR training prompt and writes assistant responses as `<think>...</think>` followed by `<answer>...</answer>`.

Then download and extract the videos. The converted examples use paths such as `videos_0007/xxx.mp4`, so the default layout should be `data/videos_0007/xxx.mp4`.

```bash
huggingface-cli download minuzero/VideoKR-Train \
  --repo-type dataset \
  --local-dir data/raw \
  --include "Videos/*"

shopt -s nullglob
for f in data/raw/Videos/*.tar; do
  tar -xf "$f" -C data
done

for f in data/raw/Videos/*.tar.part.aa; do
  base="${f%.part.aa}"
  cat "${base}".part.* > "${base}"
  tar -xf "${base}" -C data
done
```

If your videos are stored elsewhere, pass `media_dir=/path/to/video_root` when launching training.

## Train

Train the Qwen2.5-VL model:

```bash
bash local_script/train_videokr.sh qwen2_5vl
```

Train the Qwen3-VL model:

```bash
bash local_script/train_videokr.sh qwen3vl
```

By default, the script uses DeepSpeed ZeRO-3 and writes checkpoints to `saves/videokr/`.

To use local model weights or a custom output directory:

```bash
MODEL_NAME_OR_PATH=/path/to/Qwen3-VL-8B-Instruct \
OUTPUT_DIR=saves/videokr/qwen3vl_sft \
bash local_script/train_videokr.sh qwen3vl
```

For quick debugging on a small run, extra LLaMA-Factory YAML overrides can be appended:

```bash
FORCE_TORCHRUN=0 \
MODEL_NAME_OR_PATH=/path/to/model \
bash local_script/train_videokr.sh qwen3vl \
  max_steps=1 \
  max_samples=8 \
  deepspeed=null \
  flash_attn=disabled
```

## Evaluation

After training, evaluate the saved checkpoint with the `lmms_eval` folder in this repository.
