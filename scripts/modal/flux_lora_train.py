#!/usr/bin/env python3
"""
FLUX LoRA Training on Modal

This script fine-tunes FLUX.1-dev using LoRA (Low-Rank Adaptation) on the CuRe dataset.
It runs on Modal's cloud infrastructure with 4x A100-80GB GPUs.

Dataset: Kaousheik/cure-annotated
- Contains cultural food/artifact images with detailed captions
- Columns: image, Item (name), generated_caption (detailed description)

Training Configuration:
- LoRA rank: 16
- Optimizer: Prodigy (adaptive learning rate)
- Mixed precision: FP16
- Resolution: 1024x1024
- Effective batch size: 16 (batch=1 × grad_accum=4 × GPUs=4)

Usage:
    # Train with detailed captions (recommended)
    python flux_lora_train.py --mode train --caption_column generated_caption
    
    # Train with item names (e.g., "an image of Panta_Bhat")
    python flux_lora_train.py --mode train --caption_column Item
    
    # Custom training steps
    python flux_lora_train.py --mode train --steps 2000

Output:
    Checkpoints saved to Modal volume: srpo-outputs/flux-cure-lora/
    Download with: modal volume get srpo-outputs flux-cure-lora/ ./flux-cure-lora/
"""

import modal
import os
from pathlib import Path

# =============================================================================
# Modal App Configuration
# =============================================================================

app = modal.App("flux-lora-train")

# Persistent volumes for caching and outputs
hf_cache_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
srpo_data_vol = modal.Volume.from_name("srpo-data", create_if_missing=True)
srpo_outputs_vol = modal.Volume.from_name("srpo-outputs", create_if_missing=True)

# HuggingFace secret for authenticated downloads
hf_secret = modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])

# Docker image with all dependencies
flux_lora_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    # PyTorch with CUDA 12.4
    .pip_install(
        "torch==2.6.0+cu124",
        "torchvision==0.21.0+cu124",
        "torchaudio==2.6.0+cu124",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    # Cleanup unnecessary packages
    .run_commands(
        "pip uninstall -y bitsandbytes triton || true",
        "pip install --no-cache-dir --no-deps protobuf==3.20.0",
    )
    # Training dependencies
    .pip_install(
        "accelerate",
        "transformers",
        "safetensors",
        "peft",
        "sentencepiece==0.1.99",
        "datasets",
        "prodigyopt",
        "pyyaml",
    )
    # Install diffusers from source (required for FLUX training scripts)
    .run_commands(
        "git clone https://github.com/huggingface/diffusers.git",
        "cd diffusers && pip install -e .",
        "cd diffusers/examples/dreambooth && pip install -r requirements_flux.txt",
    )
)


# =============================================================================
# Helper Functions
# =============================================================================

def _write_accelerate_config(num_gpus: int = 4):
    """Write accelerate config for multi-GPU DDP training."""
    import yaml
    
    config = {
        "compute_environment": "LOCAL_MACHINE",
        "distributed_type": "MULTI_GPU",
        "main_training_function": "main",
        "mixed_precision": "fp16",
        "num_machines": 1,
        "num_processes": num_gpus,
        "rdzv_backend": "static",
        "same_network": True,
        "use_cpu": False,
    }
    
    config_dir = Path("/root/.cache/huggingface/accelerate")
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "default_config.yaml"
    
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✓ Accelerate config: {num_gpus} GPUs, FP16, DDP", flush=True)


# =============================================================================
# Training Function
# =============================================================================

@app.function(
    image=flux_lora_image,
    gpu="A100-80GB:4",
    timeout=60 * 60 * 12,  # 12 hours max
    secrets=[hf_secret],
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/srpo_data": srpo_data_vol,
        "/root/srpo_outputs": srpo_outputs_vol,
    },
)
def train_flux_lora(
    dataset_name: str = "Kaousheik/cure-annotated",
    caption_column: str = "generated_caption",
    output_dir: str = "/root/srpo_outputs/flux-cure-lora",
    steps: int = 1000,
    resolution: int = 1024,
    train_batch_size: int = 1,
    grad_accum: int = 4,
    rank: int = 16,
    checkpointing_steps: int = 100,
):
    """
    Fine-tune FLUX with LoRA on the CuRe dataset.
    
    Args:
        dataset_name: HuggingFace dataset identifier
        caption_column: Column to use for captions ("generated_caption" or "Item")
        output_dir: Where to save checkpoints (on Modal volume)
        steps: Total training steps
        resolution: Image resolution (1024 recommended)
        train_batch_size: Per-GPU batch size
        grad_accum: Gradient accumulation steps
        rank: LoRA rank (higher = more parameters)
        checkpointing_steps: Save checkpoint every N steps
    """
    import subprocess
    
    _write_accelerate_config(num_gpus=4)
    
    # Set HuggingFace token
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        print("✓ HuggingFace token configured", flush=True)
    
    # Disable unnecessary quantization imports
    os.environ["BNB_DISABLE_AUTODETECT"] = "1"
    os.environ["DISABLE_BNB"] = "1"
    os.environ["DIFFUSERS_NO_BNB_IMPORT"] = "1"
    
    base_model = "/root/srpo_data/flux"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Determine instance prompt based on caption column
    if caption_column == "Item":
        instance_prompt = "an image of"
        print(f"✓ Training with Item names: 'an image of {{Item}}'", flush=True)
    else:
        instance_prompt = ""
        print(f"✓ Training with full captions from '{caption_column}'", flush=True)
    
    print(f"✓ Dataset: {dataset_name}", flush=True)
    print(f"✓ Output: {output_dir}", flush=True)
    print(f"✓ Steps: {steps}, Resolution: {resolution}x{resolution}", flush=True)
    print(f"✓ LoRA rank: {rank}, Batch: {train_batch_size}x{grad_accum}x4 = {train_batch_size * grad_accum * 4}", flush=True)
    
    cmd = [
        "accelerate", "launch",
        "examples/dreambooth/train_dreambooth_lora_flux.py",
        "--pretrained_model_name_or_path", base_model,
        "--dataset_name", dataset_name,
        "--image_column", "image",
        "--caption_column", caption_column,
        "--instance_prompt", instance_prompt,
        "--resolution", str(resolution),
        "--train_batch_size", str(train_batch_size),
        "--gradient_accumulation_steps", str(grad_accum),
        "--rank", str(rank),
        "--mixed_precision", "fp16",
        "--optimizer", "prodigy",
        "--learning_rate", "1.0",
        "--guidance_scale", "1",
        "--lr_scheduler", "constant",
        "--lr_warmup_steps", "0",
        "--checkpointing_steps", str(checkpointing_steps),
        "--max_train_steps", str(steps),
        "--output_dir", output_dir,
        "--seed", "42",
    ]
    
    print("\n" + "="*60, flush=True)
    print("Starting training...", flush=True)
    print("="*60 + "\n", flush=True)
    
    subprocess.run(cmd, check=True, cwd="/diffusers")
    
    print("\n" + "="*60, flush=True)
    print(f"✓ Training complete! Checkpoints saved to: {output_dir}", flush=True)
    print("="*60, flush=True)
    
    return {"output_dir": output_dir, "steps": steps}


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="FLUX LoRA Training on Modal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with detailed captions (recommended)
  python flux_lora_train.py --caption_column generated_caption
  
  # Train with item names
  python flux_lora_train.py --caption_column Item
  
  # Custom settings
  python flux_lora_train.py --steps 2000 --rank 32
        """
    )
    parser.add_argument("--dataset", default="Kaousheik/cure-annotated",
                        help="HuggingFace dataset name")
    parser.add_argument("--caption_column", default="generated_caption",
                        choices=["generated_caption", "Item"],
                        help="Caption column: 'generated_caption' (detailed) or 'Item' (names)")
    parser.add_argument("--steps", type=int, default=1000,
                        help="Training steps (default: 1000)")
    parser.add_argument("--resolution", type=int, default=1024,
                        help="Image resolution (default: 1024)")
    parser.add_argument("--rank", type=int, default=16,
                        help="LoRA rank (default: 16)")
    parser.add_argument("--output_dir", default="/root/srpo_outputs/flux-cure-lora",
                        help="Output directory on Modal volume")
    args = parser.parse_args()
    
    print("="*60)
    print("FLUX LoRA Training on Modal")
    print("="*60)
    print(f"Dataset:        {args.dataset}")
    print(f"Caption column: {args.caption_column}")
    print(f"Steps:          {args.steps}")
    print(f"Resolution:     {args.resolution}x{args.resolution}")
    print(f"LoRA rank:      {args.rank}")
    print(f"Output:         {args.output_dir}")
    print(f"GPUs:           4x A100-80GB")
    print("="*60 + "\n")
    
    with modal.enable_output():
        with app.run():
            result = train_flux_lora.remote(
                dataset_name=args.dataset,
                caption_column=args.caption_column,
                output_dir=args.output_dir,
                steps=args.steps,
                resolution=args.resolution,
                rank=args.rank,
            )
            print(f"\n✓ Done: {result}")
