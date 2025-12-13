#!/usr/bin/env python3
"""
FLUX LoRA Inference on Modal

Generate images using a trained FLUX LoRA model. Supports parallel generation
across multiple GPUs for fast batch processing.

Usage:
    # Generate 8 images per item from a JSON file
    python flux_lora_inference.py --json_file items.json --num_images 8
    
    # Generate from specific items
    python flux_lora_inference.py --items "Panta_Bhat" "Jollof_Rice" "Biryani"
    
    # Use a specific LoRA checkpoint
    python flux_lora_inference.py --lora_dir /root/srpo_outputs/flux-cure-lora-item
    
    # Test with a few items first
    python flux_lora_inference.py --json_file items.json --limit 5

Output:
    Images saved to Modal volume: srpo-outputs/flux-cure-lora/generated/
    Download with: modal volume get srpo-outputs flux-cure-lora/generated/ ./generated/
"""

import modal
import json
from pathlib import Path

# =============================================================================
# Modal App Configuration
# =============================================================================

app = modal.App("flux-lora-inference")

# Persistent volumes
hf_cache_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
srpo_data_vol = modal.Volume.from_name("srpo-data", create_if_missing=True)
srpo_outputs_vol = modal.Volume.from_name("srpo-outputs", create_if_missing=True)

# Docker image with inference dependencies
flux_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.6.0+cu124",
        "torchvision==0.21.0+cu124",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "diffusers",
        "transformers",
        "accelerate",
        "safetensors",
        "peft",
        "sentencepiece",
    )
)


# =============================================================================
# Single Item Generation (for parallel .map())
# =============================================================================

@app.function(
    image=flux_image,
    gpu="A100-80GB",
    timeout=60 * 30,  # 30 min per item
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/srpo_data": srpo_data_vol,
        "/root/srpo_outputs": srpo_outputs_vol,
    },
    concurrency_limit=8,  # Up to 8 parallel GPUs
)
def generate_for_item(
    item: str,
    num_images: int = 8,
    lora_dir: str = "/root/srpo_outputs/flux-cure-lora",
    output_dir: str = "/root/srpo_outputs/flux-cure-lora/generated",
    prompt_template: str = "an image of {item}",
    steps: int = 28,
    guidance: float = 3.5,
):
    """
    Generate multiple images for a single item.
    
    Args:
        item: Item name (e.g., "Panta_Bhat")
        num_images: Number of images to generate per item
        lora_dir: Path to LoRA weights
        output_dir: Where to save generated images
        prompt_template: Template for prompts (use {item} as placeholder)
        steps: Inference steps (28 is good for FLUX)
        guidance: Guidance scale (3.5 recommended)
    """
    import torch
    from diffusers import FluxPipeline
    
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean item name for prompt (replace underscores with spaces)
    clean_name = item.replace("_", " ")
    prompt = prompt_template.format(item=clean_name)
    
    # Safe filename (keep underscores for filesystem)
    safe_name = item.replace("/", "_")[:50]
    
    print(f"\n{'='*50}", flush=True)
    print(f"Item: {item}", flush=True)
    print(f"Prompt: {prompt}", flush=True)
    print(f"{'='*50}", flush=True)
    
    # Load model
    print("Loading FLUX model...", flush=True)
    pipe = FluxPipeline.from_pretrained(
        "/root/srpo_data/flux",
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    
    print(f"Loading LoRA from {lora_dir}...", flush=True)
    pipe.load_lora_weights(lora_dir)
    
    # Generate images
    results = []
    for i in range(num_images):
        img = pipe(
            prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            height=1024,
            width=1024,
        ).images[0]
        
        # Save with index
        out_path = out_dir / f"{safe_name}_{i+1}.png"
        img.save(out_path)
        results.append(str(out_path))
        print(f"  [{i+1}/{num_images}] Saved: {out_path.name}", flush=True)
    
    # Commit volume changes
    srpo_outputs_vol.commit()
    
    print(f"\n✓ Done: {item} ({num_images} images)", flush=True)
    
    return {
        "item": item,
        "prompt": prompt,
        "images": results,
        "count": len(results),
    }


# =============================================================================
# Batch Generation (sequential, single GPU)
# =============================================================================

@app.function(
    image=flux_image,
    gpu="A100-80GB",
    timeout=60 * 60 * 6,  # 6 hours for large batches
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/srpo_data": srpo_data_vol,
        "/root/srpo_outputs": srpo_outputs_vol,
    },
)
def generate_batch(
    items: list,
    num_images: int = 8,
    lora_dir: str = "/root/srpo_outputs/flux-cure-lora",
    output_dir: str = "/root/srpo_outputs/flux-cure-lora/generated",
    prompt_template: str = "an image of {item}",
    steps: int = 28,
    guidance: float = 3.5,
):
    """
    Generate images for multiple items sequentially on a single GPU.
    Use this for smaller batches or when parallel processing isn't needed.
    """
    import torch
    from diffusers import FluxPipeline
    
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading FLUX model...", flush=True)
    pipe = FluxPipeline.from_pretrained(
        "/root/srpo_data/flux",
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    
    print(f"Loading LoRA from {lora_dir}...", flush=True)
    pipe.load_lora_weights(lora_dir)
    
    print(f"\nGenerating {len(items)} items x {num_images} images = {len(items) * num_images} total\n", flush=True)
    
    all_results = []
    for idx, item in enumerate(items):
        clean_name = item.replace("_", " ")
        prompt = prompt_template.format(item=clean_name)
        safe_name = item.replace("/", "_")[:50]
        
        print(f"[{idx+1}/{len(items)}] {item}", flush=True)
        print(f"  Prompt: {prompt}", flush=True)
        
        item_results = []
        for i in range(num_images):
            img = pipe(
                prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                height=1024,
                width=1024,
            ).images[0]
            
            out_path = out_dir / f"{safe_name}_{i+1}.png"
            img.save(out_path)
            item_results.append(str(out_path))
            print(f"  [{i+1}/{num_images}] {out_path.name}", flush=True)
        
        all_results.append({
            "item": item,
            "prompt": prompt,
            "images": item_results,
        })
    
    print(f"\n✓ Done! {len(items)} items, {len(items) * num_images} images", flush=True)
    print(f"  Saved to: {output_dir}", flush=True)
    
    return {"output_dir": str(output_dir), "results": all_results}


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="FLUX LoRA Inference on Modal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 8 images per item from JSON file
  python flux_lora_inference.py --json_file items.json
  
  # Generate for specific items
  python flux_lora_inference.py --items "Panta_Bhat" "Jollof_Rice"
  
  # Use different LoRA checkpoint
  python flux_lora_inference.py --lora_dir /root/srpo_outputs/flux-cure-lora-item
  
  # Test with limit
  python flux_lora_inference.py --json_file items.json --limit 5
  
  # Sequential mode (single GPU)
  python flux_lora_inference.py --json_file items.json --sequential
        """
    )
    parser.add_argument("--json_file", type=str, default=None,
                        help="JSON file with items (expects 'Item' field)")
    parser.add_argument("--items", nargs="+", default=None,
                        help="List of item names to generate")
    parser.add_argument("--num_images", type=int, default=8,
                        help="Images per item (default: 8)")
    parser.add_argument("--lora_dir", default="/root/srpo_outputs/flux-cure-lora",
                        help="LoRA weights directory")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: {lora_dir}/generated)")
    parser.add_argument("--prompt_template", default="an image of {item}",
                        help="Prompt template (use {item} placeholder)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of items to process")
    parser.add_argument("--sequential", action="store_true",
                        help="Use sequential mode (single GPU) instead of parallel")
    parser.add_argument("--steps", type=int, default=28,
                        help="Inference steps (default: 28)")
    parser.add_argument("--guidance", type=float, default=3.5,
                        help="Guidance scale (default: 3.5)")
    args = parser.parse_args()
    
    # Determine items list
    if args.items:
        items = args.items
    elif args.json_file:
        with open(args.json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        items = list(set(entry["Item"] for entry in data if "Item" in entry))
        items.sort()
    else:
        # Default test items
        items = ["Panta_Bhat", "Jollof_Rice", "Biryani", "Tacos"]
    
    if args.limit:
        items = items[:args.limit]
    
    # Default output directory
    output_dir = args.output_dir or f"{args.lora_dir}/generated"
    
    # Print config
    print("="*60)
    print("FLUX LoRA Inference on Modal")
    print("="*60)
    print(f"Items:          {len(items)}")
    print(f"Images/item:    {args.num_images}")
    print(f"Total images:   {len(items) * args.num_images}")
    print(f"LoRA:           {args.lora_dir}")
    print(f"Output:         {output_dir}")
    print(f"Mode:           {'Sequential (1 GPU)' if args.sequential else 'Parallel (up to 8 GPUs)'}")
    print(f"Prompt:         {args.prompt_template}")
    print("="*60)
    print("\nFirst 10 items:")
    for item in items[:10]:
        clean = item.replace("_", " ")
        print(f"  {item} → {args.prompt_template.format(item=clean)}")
    if len(items) > 10:
        print(f"  ... and {len(items) - 10} more")
    print()
    
    with modal.enable_output():
        with app.run():
            if args.sequential:
                # Sequential mode - single GPU
                result = generate_batch.remote(
                    items=items,
                    num_images=args.num_images,
                    lora_dir=args.lora_dir,
                    output_dir=output_dir,
                    prompt_template=args.prompt_template,
                    steps=args.steps,
                    guidance=args.guidance,
                )
                print(f"\n✓ Done: {result['output_dir']}")
            else:
                # Parallel mode - multiple GPUs
                print(f"Starting parallel generation on up to 8 GPUs...\n")
                results = list(generate_for_item.map(
                    items,
                    kwargs={
                        "num_images": args.num_images,
                        "lora_dir": args.lora_dir,
                        "output_dir": output_dir,
                        "prompt_template": args.prompt_template,
                        "steps": args.steps,
                        "guidance": args.guidance,
                    },
                ))
                
                total_images = sum(r["count"] for r in results)
                print(f"\n{'='*60}")
                print(f"✓ Done! {len(results)} items, {total_images} images")
                print(f"  Output: {output_dir}")
                print("="*60)
