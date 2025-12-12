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

The environment dependency is basically the same as DanceGRPO

## Download Models

1. **Base SRPO Model**: Download the original SRPO checkpoint
```bash
mkdir ./srpo
huggingface-cli login
huggingface-cli download --resume-download Tencent/SRPO diffusion_pytorch_model.safetensors --local-dir ./srpo/
```

2. **FLUX Base Model**: Download FLUX.1-dev for training
```bash
mkdir ./data/flux
huggingface-cli login
huggingface-cli download --resume-download black-forest-labs/FLUX.1-dev --local-dir ./data/flux
```

3. **Reward Models**: Download HPSv2 and/or PickScore
```bash
# HPSv2
mkdir ./data/hps_ckpt
huggingface-cli download --resume-download xswu/HPSv2 HPS_v2.1_compressed.pt --local-dir ./data/hps_ckpt
huggingface-cli download --resume-download laion/CLIP-ViT-H-14-laion2B-s32B-b79K open_clip_pytorch_model.bin --local-dir ./data/hps_ckpt

# PickScore (optional)
mkdir ./data/ps
python ./scripts/huggingface/download_hf.py --repo_id yuvalkirstain/PickScore_v1 --local_dir ./data/ps
python ./scripts/huggingface/download_hf.py --repo_id laion/CLIP-ViT-H-14-laion2B-s32B-b79K --local_dir ./data/clip
```

## Inference

### Quick Start
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

### Using ComfyUI

You can use it in [ComfyUI](https://github.com/comfyanonymous/ComfyUI). Load the workflow from [SRPO-workflow](comfyui/SRPO-workflow.json):

![Example](comfyui/SRPO-workflow.png)

## Training

### Data

This fork uses prompts from the **CuRe dataset** with MIA attributes. The dataset should include:

- Base captions
- MIA attributes: `mia_category` and `mia_region` 
- Optional: Pre-computed `pos_prompt` and `neg_prompt` for explicit positive/negative signal construction

```bash
# Prepare CuRe dataset prompts in ./prompts.txt
# Format: JSON with fields: caption, mia_category, mia_region, (optional) pos_prompt, neg_prompt

# Pre-extract text embeddings for efficiency
bash scripts/preprocess/preprocess_flux_rl_embeddings.sh
cp videos2caption2.json ./data/rl_embeddings
```

### Training Modes

#### Online RL Training
Online RL generates rollouts during training and optimizes directly on the reward signal.

```bash
# HPSv2 as reward model
bash scripts/finetune/SRPO_training_hpsv2.sh
```

#### Offline RL Training
Offline RL uses a fixed dataset of images, enabling training with fewer computational resources.

The codebase supports offline RL by providing image-text pairs in the dataset. Configure your training script to use the offline dataset mode.

### MIA-Based Signal Construction

The training automatically constructs positive/negative signals from MIA attributes:

- **Positive prompt**: `{base_caption}. {mia_category}, {mia_region}` (enriched with cultural attributes)
- **Negative prompt**: `{base_caption}` (under-specified, missing cultural context)

This guides the model to learn better representations of long-tail cultural concepts.

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
Check ``Cube_Clean.ipynb`` for deatiled evaluation setup. 


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

## Acknowledgement

We build upon and acknowledge the following works:

- **[SRPO](https://github.com/Tencent-Hunyuan/SRPO)**: Original Semantic Relative Preference Optimization framework
- **[CuRe Benchmark](https://github.com/your-cure-repo)**: Cultural Representation benchmark with MIA
- **[FastVideo](https://github.com/hao-ai-lab/FastVideo)**: Base training infrastructure
- **[DanceGRPO](https://github.com/XueZeyue/DanceGRPO)**: Related RLHF work for diffusion models

<!-- ## Citation

If you use this work, please cite both SRPO and CuRe:

```bibtex
@misc{shen2025directlyaligningdiffusiontrajectory,
      title={Directly Aligning the Full Diffusion Trajectory with Fine-Grained Human Preference}, 
      author={Xiangwei Shen and Zhimin Li and Zhantao Yang and Shiyi Zhang and Yingfang Zhang and Donghao Li and Chunyu Wang and Qinglin Lu and Yansong Tang},
      year={2025},
      eprint={2509.06942},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2509.06942}, 
}

@misc{cure2024benchmark,
      title={CuRe: A Benchmark for Cultural Representation in Text-to-Image Generation},
      author={CuRe Authors},
      year={2024},
      url={https://github.com/your-cure-repo}
}
``` -->

## License

This project is forked from [SRPO](https://github.com/Tencent-Hunyuan/SRPO). Please refer to the original license and any additional terms for the CuRe dataset integration.
