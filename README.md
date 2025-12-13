<div align="center" style="font-family: charter;">
<h1 align="center">SRPO + MIA: Mitigating Long-Tail Bias in Text-to-Image Generation</h1>
<div align="center">
  <a href='https://arxiv.org/abs/2509.06942'><img src='https://img.shields.io/badge/ArXiv-SRPO-red?logo=arxiv'></a>  &nbsp;
  <a href='https://huggingface.co/tencent/SRPO/'><img src='https://img.shields.io/badge/Model-blue?logo=huggingface'></a> &nbsp; 
</div>
</div>

![head](assets/head.jpg)

## Overview

State-of-the-art text-to-image (T2I) systems underrepresent cultural concepts in the long tail of the data distribution. This project studies whether **Semantic Relative Preference Optimization (SRPO)**, an online RL framework for aligning diffusion models with fine-grained text-conditioned rewards, when combined with **Marginal Information Attribution (MIA)** from the **CuRe** benchmark, can directly mitigate long-tail bias without sacrificing overall fidelity.

We adapt SRPO to construct positive/negative signals from MIA-based attribute ablations, compare on- vs. off-policy training, and evaluate on CuRe using multimodal large language models (MLLMs), human-aligned reward models (e.g., HPSv2), and alignment proxies.

### Key Contributions

1. **MIA-Enhanced SRPO**: Integration of Marginal Information Attribution from CuRe benchmark to identify and address underrepresented cultural concepts
2. **Online & Offline RL**: Support for both online and offline reinforcement learning paradigms for flexible training strategies
3. **CuRe Dataset Integration**: Training prompts sourced from the CuRe dataset, focusing on long-tail cultural representations
4. **Bias Mitigation**: Direct approach to reducing long-tail bias in T2I generation without compromising overall image quality

<!-- ## Key Features

1. **MIA-Based Signal Construction**: Automatically constructs positive/negative training signals from MIA attribute ablations (category, region) to guide model learning toward better cultural representation
2. **Dual RL Paradigms**: Supports both online RL (rollout-based) and offline RL (dataset-based) training modes for different computational and data availability scenarios
3. **CuRe Dataset Support**: Native integration with CuRe dataset prompts and MIA attributes for targeted bias mitigation
4. **Flexible Reward Models**: Compatible with multiple reward models including HPSv2, PickScore, and MLLM-based evaluators
5. **Direct Alignment**: Inherits SRPO's direct trajectory alignment approach for stable and efficient training -->

## Dependencies and Installation

```bash
conda create -n SRPO python=3.10.16 -y
conda activate SRPO
bash ./env_setup.sh 
```

## Download Models

```bash
# Base SRPO checkpoint
mkdir ./srpo
huggingface-cli login
huggingface-cli download --resume-download Tencent/SRPO diffusion_pytorch_model.safetensors --local-dir ./srpo/

# FLUX base model
mkdir -p ./data/flux
huggingface-cli download --resume-download black-forest-labs/FLUX.1-dev --local-dir ./data/flux

# HPSv2 reward model
mkdir -p ./data/hps_ckpt
huggingface-cli download --resume-download xswu/HPSv2 HPS_v2.1_compressed.pt --local-dir ./data/hps_ckpt
huggingface-cli download --resume-download laion/CLIP-ViT-H-14-laion2B-s32B-b79K open_clip_pytorch_model.bin --local-dir ./data/hps_ckpt
```

## Inference

```python
from diffusers import FluxPipeline
from safetensors.torch import load_file
import torch

prompt = 'A traditional ceremony in a specific cultural context'
pipe = FluxPipeline.from_pretrained(
    './data/flux',
    torch_dtype=torch.bfloat16,
    use_safetensors=True
).to("cuda")
state_dict = load_file("./srpo/diffusion_pytorch_model.safetensors")
pipe.transformer.load_state_dict(state_dict)

image = pipe(
    prompt,
    guidance_scale=3.5,
    height=1024,
    width=1024,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator(device="cuda")
).images[0]
```


## Training

### Data Preparation

This fork uses prompts from the **CuRe dataset** with MIA attributes. The dataset should include:

```json
{
  "caption": "Base caption text",
  "mia_category": "Cultural category",
  "mia_region": "Geographic region",
  "pos_prompt": "Optional: Explicit positive prompt",
  "neg_prompt": "Optional: Explicit negative prompt",
  "prompt_embed_path": "path/to/embedding",
  "pooled_prompt_embeds_path": "path/to/pooled_embeds",
  "text_ids": "path/to/text_ids"
}

```
### MIA-Based Signal Construction

The training automatically constructs positive/negative signals from MIA attributes:

- **Positive prompt**: `{base_caption}. {mia_category}, {mia_region}` (enriched with cultural attributes)
- **Negative prompt**: `{base_caption}` (under-specified, missing cultural context)

This guides the model to learn better representations of long-tail cultural concepts.
```bash
# Prepare CuRe dataset prompts in ./prompts.txt
# Format: JSON with fields: caption, mia_category, mia_region, (optional) pos_prompt, neg_prompt

# Pre-extract text embeddings for efficiency
bash scripts/preprocess/preprocess_flux_rl_embeddings.sh
cp videos2caption2.json ./data/rl_embeddings
```

### Online RL Training

```bash
# HPSv2 as reward model
bash scripts/finetune/SRPO_training_hpsv2.sh
```

### Offline RL Training
Offline RL uses a fixed dataset of images, enabling training with fewer computational resources.

The codebase supports offline RL by providing image-text pairs in the dataset. Configure your training script to use the offline dataset mode.

### Distributed Training

```bash
#!/bin/bash
echo "$NODE_IP_LIST" | tr ',' '\n' | sed 's/:8$//' | grep -v '1.1.1.1' > /tmp/pssh.hosts
node_ip=$(paste -sd, /tmp/pssh.hosts)
pdsh -w $node_ip "conda activate SRPO;cd <project path>; bash scripts/finetune/SRPO_training_hpsv2.sh"
```

### Hyperparameter Recommendations

1. **Batch_size**: Larger sizes generally improve quality. For FLUX.1.dev, 32 works well.
2. **Learning_rate**: 1e-5 to 1e-6 works for most models.
3. **Train_timestep**: Focus on early-to-middle diffusion stages. Too early (e.g., sigmas>0.99) causes structural distortions; too late encourages color-based reward hacking.
4. **Discount_inv** & **Discount_denoise**: Let discount_inv = [a, b], discount_denoise = [c, d]. Preserve structure by setting c slightly > b (avoids early layout corruption). Fix color oversaturation by setting a slightly > d (tempers aggressive tones).

## Evaluation

Evaluation is performed on the CuRe benchmark using:

- **Multimodal Large Language Models (MLLMs)**: For semantic and cultural alignment assessment
Check ``Cube_Clean.ipynb`` for the detailed evaluation setup. 


## Customization

### Supporting Custom Models

1. Modify `preprocess_flux_embedding.py` and `latent_flux_rl_datasets.py` to pre-extract text embeddings from your custom training dataset.
2. Adjust `args.vis_sampling_step` to modify sigma_schedule. Typically, this value matches the model's regular inference steps.
3. Direct-propagation needs significant GPU memory. Enabling VAE gradient checkpointing before reward calculation reduces this greatly.
4. If implementing outside FastVideo, first disable the inversion branch to check for reward hacking—its presence likely indicates correct implementation.

### Dataset Format

The dataset JSON should include MIA-related fields:

```json
{
  "caption": "Base caption text",
  "mia_category": "Cultural category",
  "mia_region": "Geographic region",
  "pos_prompt": "Optional: Explicit positive prompt",
  "neg_prompt": "Optional: Explicit negative prompt",
  "prompt_embed_path": "path/to/embedding",
  "pooled_prompt_embeds_path": "path/to/pooled_embeds",
  "text_ids": "path/to/text_ids"
}
```

---

## FLUX LoRA Training on Modal

For quick experimentation, we provide Modal scripts for cloud-based LoRA fine-tuning.

### Modal Setup

```bash
# Install Modal CLI
pip install modal
modal setup

# Create HuggingFace secret
modal secret create huggingface HF_TOKEN=your_token_here
modal volume create hf-cache srpo-data srpo-outputs
```

### Dataset

We use [mahesh111000/cure-annotated](https://huggingface.co/datasets/mahesh111000/cure-annotated) which contains:
- **Images**: Cultural foods, artifacts, clothing from around the world
- **Item**: Short name (e.g., `"Panta_Bhat"`, `"Jollof_Rice"`)
- **generated_caption**: Detailed description of the item

### Training

```bash
cd scripts/modal

# Train with detailed captions (recommended)
python flux_lora_train.py --caption_column generated_caption --steps 1000

# Train with item names (for "an image of X" prompts)
python flux_lora_train.py --caption_column Item --steps 1000
```

**Training Configuration:**
- GPUs: 4x A100-80GB
- Mixed Precision: FP16
- Optimizer: Prodigy (adaptive lr)
- Effective Batch: 16 (1 × 4 grad_accum × 4 GPUs)
- Checkpoints: Every 100 steps

### Inference

```bash
cd scripts/modal

# Generate 8 images per item (parallel on up to 8 GPUs)
python flux_lora_inference.py --json_file items.json --num_images 8

# Generate for specific items
python flux_lora_inference.py --items "Panta_Bhat" "Jollof_Rice" --num_images 8

# Sequential mode (single GPU)
python flux_lora_inference.py --json_file items.json --sequential
```

### Download Results

```bash
# List checkpoints
modal volume ls srpo-outputs flux-cure-lora/

# Download model
modal volume get srpo-outputs flux-cure-lora/ ./flux-cure-lora/

# Download generated images
modal volume get srpo-outputs flux-cure-lora/generated/ ./generated/
```

---

## Acknowledgement

We build upon and acknowledge the following works:

- **[SRPO](https://github.com/Tencent-Hunyuan/SRPO)**: Original Semantic Relative Preference Optimization framework
- **[CuRe Benchmark](https://github.com/aniketrege/cure)**: Cultural Representation benchmark with MIA
- **[FastVideo](https://github.com/hao-ai-lab/FastVideo)**: Base training infrastructure
- **[DanceGRPO](https://github.com/XueZeyue/DanceGRPO)**: Related RLHF work for diffusion models

## License

This project is forked from [SRPO](https://github.com/Tencent-Hunyuan/SRPO). Please refer to the original license and any additional terms for the CuRe dataset integration.
