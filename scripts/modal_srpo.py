import os
import subprocess
from pathlib import Path
import modal


app = modal.App("srpo-train")

# Resolve repo root robustly regardless of CLI CWD
REPO_ROOT = Path(__file__).resolve().parent.parent


def _add_code_dirs(img: modal.Image) -> modal.Image:
    return img.add_local_dir(
        str(REPO_ROOT / "fastvideo"), remote_path="/workspace/SRPO/fastvideo"
    ).add_local_dir(str(REPO_ROOT / "scripts"), remote_path="/workspace/SRPO/scripts")


# GPU base image + deps. Torch installed with cu124 wheels.
# CPU-only slim image for quick mount checks (no GPU needed)
cpu_image = _add_code_dirs(
    modal.Image.from_registry("python:3.10-slim")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .run_commands("python -V")
)

# Hugging Face token should be provided via environment variable or function parameter
# Set HUGGINGFACE_TOKEN or HF_TOKEN environment variable, or pass token parameter
HF_TOKEN_DEFAULT = ""  # Removed hardcoded token for security

# Full GPU image for training when ready
image = _add_code_dirs(
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0", "build-essential", "clang")
    .run_commands(
        "pip install -U pip",
        # Torch/cu124 (adjust if you prefer a different CUDA/PyTorch pairing)
        "pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124",
        # Core libs (aligned to SRPO/pyproject.toml)
        "pip install transformers==4.46.1 accelerate==1.0.1 tokenizers==0.20.1",
        "pip install diffusers==0.32.0 einops==0.8.0",
        "pip install pytorch-lightning==2.4.0 pytorchvideo==0.1.5",
        "pip install albumentations==1.4.20 av==13.1.0",
        "pip install fastapi==0.115.3 h5py==3.12.1 imageio==2.36.0 matplotlib==3.9.2 numpy==1.26.3",
        "pip install omegaconf==2.3.0 opencv-python-headless==4.10.0.84 pandas==2.2.3 pydub==0.25.1",
        "pip install PyYAML==6.0.1 regex==2024.9.11 requests==2.31.0 scikit-learn==1.5.2 scipy==1.13.0 six==1.16.0",
        "pip install timm==1.0.11 torchdiffeq==0.2.4 torchmetrics==1.5.1 tqdm==4.66.5 urllib3==2.2.0",
        "pip install sentencepiece==0.2.0 beautifulsoup4==4.12.3 ftfy==6.3.0 moviepy==1.0.3 wandb==0.18.5",
        "pip install pydantic==2.9.2 huggingface_hub==0.26.1 protobuf==5.28.3 gpustat peft==0.13.2 liger_kernel==0.4.1 loguru bitsandbytes==0.44.1",
        # Required by torch.utils.tensorboard SummaryWriter
        "pip install tensorboard",
        # Build tools for FlashAttention and other native deps
        "pip install packaging ninja wheel setuptools cmake",
        # (FlashAttention removed per request to avoid long compiles)
        # Ensure decord and hf_transfer are preinstalled (no runtime installs)
        "pip install decord==0.6.0 hf_transfer==0.1.8",
        # HPSv2 (provides hpsv2.src.open_clip)
        "pip install git+https://github.com/tgxs002/HPSv2.git",
        # Enforce a Triton/bitsandbytes pair compatible with diffusers' bnb quantizers
        # to satisfy `import triton.ops` used by bnb 0.43.x paths.
        "pip uninstall -y triton bitsandbytes || true",
        "pip install --no-deps --force-reinstall triton==2.3.0",
        "pip install --no-deps --force-reinstall bitsandbytes==0.43.3",
        # Let PyTorch manage Triton (prebuilt wheel per Torch version)
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


# Persist models/datasets/outputs across runs
data_vol = modal.Volume.from_name("srpo-data", create_if_missing=True)


# Step 0: Build a minimal image and verify the repo mount without using GPUs.
@app.function(
    image=cpu_image,
    timeout=10 * 60,
)
def check_mount(path: str = "/workspace/SRPO"):
    """Print a small listing from the mounted repo to verify mount works."""
    import json

    contents = []
    try:
        for name in sorted(os.listdir(path))[:50]:
            p = os.path.join(path, name)
            try:
                st = os.stat(p)
                contents.append(
                    {
                        "name": name,
                        "is_dir": os.path.isdir(p),
                        "size": st.st_size,
                    }
                )
            except Exception:
                contents.append({"name": name, "error": "stat-failed"})
    except FileNotFoundError:
        print(json.dumps({"ok": False, "error": f"Path not found: {path}"}, indent=2))
        return
    print(json.dumps({"ok": True, "path": path, "items": contents}, indent=2))


@app.function(
    image=image,
    gpu="A100:1",  # Change to "H100:1" or "H100:4" as needed
    volumes={"/data": data_vol},
    timeout=24 * 60 * 60,
)
def prep_models(token: str = ""):
    """Download FLUX.1-dev and HPSv2 weights to the persisted volume."""
    from huggingface_hub import snapshot_download

    os.makedirs("/data/flux", exist_ok=True)
    os.makedirs("/data/hps_ckpt", exist_ok=True)
    os.makedirs("/data/clip", exist_ok=True)
    os.makedirs("/data/.cache", exist_ok=True)

    # Prefer explicit flag; fallback to environment; finally to hardcoded default
    token = (
        token
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HF_TOKEN")
        or HF_TOKEN_DEFAULT
    )

    # Base model (FLUX.1-dev)
    snapshot_download(
        "black-forest-labs/FLUX.1-dev",
        local_dir="/data/flux",
        local_dir_use_symlinks=False,
        token=token,
    )

    # HPS reward checkpoint
    snapshot_download(
        "xswu/HPSv2",
        allow_patterns=["HPS_v2.1_compressed.pt"],
        local_dir="/data/hps_ckpt",
        local_dir_use_symlinks=False,
        token=token,
    )

    # CLIP H-14 weights (for CLIP-based reward / optional use)
    snapshot_download(
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        local_dir="/data/clip",
        local_dir_use_symlinks=False,
        token=token,
    )

    return "Models ready in volume."


@app.function(
    image=image,
    gpu="A100-80GB:4",  # Using 4x A100 per your request
    volumes={"/data": data_vol},
    timeout=24 * 60 * 60,
)
def train(
    data_json_path="",
    output_dir="",
    steps=1000,
    train_guidance=1,
    bs=1,
    acc=2,
    token: str = "",
    pos_prompt: str = "",
    neg_prompt: str = "",
    pos_prefix: str = "",
    neg_prefix: str = "",
    # Optional anti-hack / alignment knobs (pass-through)
    vis_guidence: int | None = None,
    vis_sampling_step: int | None = None,
    vis_size: int | None = None,
    train_timestep_low: int | None = None,
    train_timestep_high: int | None = None,
    discount_inv_a: float | None = None,
    discount_inv_b: float | None = None,
    discount_pos_a: float | None = None,
    discount_pos_b: float | None = None,
    eta: float | None = None,
):
    """Run SRPO training using the mounted code and persisted data volume."""
    import torch

    # Ensure local imports resolve (mounted code)
    os.environ["PYTHONPATH"] = "/workspace/SRPO:" + os.environ.get("PYTHONPATH", "")

    # Auto-detect model/data base: prefer Modal volume (/data) if populated, else repo-local (/workspace/SRPO/data)
    base_candidates = ["/data", "/workspace/SRPO/data"]
    base = None
    for b in base_candidates:
        if os.path.isdir(os.path.join(b, "flux")) and os.path.isdir(
            os.path.join(b, "hps_ckpt")
        ):
            base = b
            break
    if base is None:
        raise RuntimeError(
            "Could not find models. Expected 'flux' and 'hps_ckpt' under /data or /workspace/SRPO/data"
        )

    # SRPO.py expects './hps_ckpt/...', so set CWD to the base directory that contains it
    os.chdir(base)

    # Default paths derived from detected base, if not provided
    if not data_json_path:
        cands = [
            os.path.join(base, "rl_embeddings", "videos2caption2.json"),
            os.path.join(base, "rl_embeddings", "videos2caption.json"),
            os.path.join(base, "rl_embeddings", "data.json"),
        ]
        for c in cands:
            if os.path.isfile(c):
                data_json_path = c
                break
        if not data_json_path:
            raise RuntimeError(
                "Could not auto-detect embeddings JSON under rl_embeddings/; pass --data_json_path explicitly."
            )
    if not output_dir:
        output_dir = os.path.join(base, "output", "hps")
    os.makedirs(output_dir, exist_ok=True)

    # Expose token to huggingface_hub via env
    token = (
        token
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HF_TOKEN")
        or HF_TOKEN_DEFAULT
    )
    os.environ["HUGGINGFACE_TOKEN"] = token

    gpus = torch.cuda.device_count()
    env = os.environ.copy()
    env.update(
        {
            "HOST_NUM": "1",
            "HOST_GPU_NUM": str(gpus),
            "CHIEF_IP": "127.0.0.1",
            "INDEX": "0",
            "WANDB_DISABLED": "true",
        }
    )

    pretrained_dir = os.path.join(base, "flux")

    eta_val = str(eta) if eta is not None else "0.3"
    di_a = str(discount_inv_a) if discount_inv_a is not None else "0.3"
    di_b = str(discount_inv_b) if discount_inv_b is not None else "0.01"
    dp_a = str(discount_pos_a) if discount_pos_a is not None else "0.1"
    dp_b = str(discount_pos_b) if discount_pos_b is not None else "0.25"

    cmd = [
        "torchrun",
        "--nnodes",
        "1",
        "--nproc_per_node",
        str(gpus),
        "--node_rank",
        "0",
        "--rdzv_endpoint",
        "127.0.0.1:29601",
        "--module",
        "fastvideo.SRPO",
        "--seed",
        "42",
        "--pretrained_model_name_or_path",
        pretrained_dir,
        "--vae_model_path",
        pretrained_dir,
        "--cache_dir",
        os.path.join(base, ".cache"),
        "--data_json_path",
        data_json_path,
        "--gradient_checkpointing",
        "--train_batch_size",
        str(bs),
        "--num_latent_t",
        "1",
        "--sp_size",
        "1",
        "--train_sp_batch_size",
        "1",
        "--dataloader_num_workers",
        "4",
        "--gradient_accumulation_steps",
        str(acc),
        "--max_train_steps",
        str(steps),
        "--learning_rate",
        "5e-6",
        "--mixed_precision",
        "bf16",
        "--checkpointing_steps",
        "20",
        "--allow_tf32",
        "--train_guidence",
        str(train_guidance),
        "--output_dir",
        output_dir,
        "--h",
        "720",
        "--w",
        "720",
        "--t",
        "1",
        "--sampling_steps",
        "25",
        "--image_p",
        "srpohps",
        "--eta",
        eta_val,
        "--lr_warmup_steps",
        "0",
        "--sampler_seed",
        "1223627",
        "--max_grad_norm",
        "0.1",
        "--weight_decay",
        "0.0001",
        "--shift",
        "3",
        "--ignore_last",
        "--discount_inv",
        di_a,
        di_b,
        "--discount_pos",
        dp_a,
        dp_b,
        "--timestep_length",
        "1000",
        "--groundtruth_ratio",
        "0.9",
        "--disable_control_words",
    ]
    if pos_prompt:
        cmd += ["--pos_prompt", pos_prompt]
    if neg_prompt:
        cmd += ["--neg_prompt", neg_prompt]
    if pos_prefix:
        cmd += ["--pos_prefix", pos_prefix]
    if neg_prefix:
        cmd += ["--neg_prefix", neg_prefix]
    if vis_guidence is not None:
        cmd += ["--vis_guidence", str(vis_guidence)]
    if vis_sampling_step is not None:
        cmd += ["--vis_sampling_step", str(vis_sampling_step)]
    if vis_size is not None:
        cmd += ["--vis_size", str(vis_size)]
    if train_timestep_low is not None and train_timestep_high is not None:
        cmd += ["--train_timestep", str(train_timestep_low), str(train_timestep_high)]

    subprocess.run(cmd, env=env, check=True)


@app.local_entrypoint()
def main():
    print(
        "First, verify mount without GPU: modal run SRPO/scripts/modal_srpo.py::check_mount"
    )
    print(
        "Then (optional) download models: modal run SRPO/scripts/modal_srpo.py::prep_models"
    )
    print("Finally (optional) train: modal run SRPO/scripts/modal_srpo.py::train")


# GPU-based embedding extraction per README
@app.function(
    image=image,
    gpu="A100-80GB:4",  # Using 4x A100 per your request
    timeout=24 * 60 * 60,
)
def preprocess_embeddings(
    prompt_file: str = "/workspace/SRPO/prompts.txt",
    output_dir: str = "/workspace/SRPO/data/rl_embeddings",
    master_port: int = 19002,
):
    import torch

    # Ensure repo imports work
    os.environ["PYTHONPATH"] = "/workspace/SRPO:" + os.environ.get("PYTHONPATH", "")

    # Work from the repo root so "./data/flux" resolves for the pipeline
    os.chdir("/workspace/SRPO")

    os.makedirs(output_dir, exist_ok=True)

    gpus = torch.cuda.device_count()
    env = os.environ.copy()

    cmd = [
        "torchrun",
        "--nproc_per_node",
        str(gpus),
        "--master_port",
        str(master_port),
        "fastvideo/data_preprocess/preprocess_flux_embedding.py",
        "--model_path",
        "black-forest-labs/FLUX.1-dev",  # script uses ./data/flux internally
        "--output_dir",
        output_dir,
        "--prompt_dir",
        prompt_file,
    ]

    subprocess.run(cmd, env=env, check=True)


# List contents of the persisted volume mapped at /data
@app.function(image=cpu_image, timeout=10 * 60, volumes={"/data": data_vol})
def check_data(path: str = "/data"):
    import json

    contents = []
    try:
        for name in sorted(os.listdir(path))[:200]:
            p = os.path.join(path, name)
            try:
                st = os.stat(p)
                contents.append(
                    {
                        "name": name,
                        "is_dir": os.path.isdir(p),
                        "size": st.st_size,
                    }
                )
            except Exception:
                contents.append({"name": name, "error": "stat-failed"})
    except FileNotFoundError:
        print(json.dumps({"ok": False, "error": f"Path not found: {path}"}, indent=2))
        return
    print(json.dumps({"ok": True, "path": path, "items": contents}, indent=2))


# Extract a tar archive inside the volume (e.g., /data/rl_embeddings.tar)
@app.function(image=cpu_image, timeout=30 * 60, volumes={"/data": data_vol})
def extract_tar(
    tar_path: str = "/data/rl_embeddings.tar", dest: str = "/data", delete: bool = True
):
    import tarfile
    import json

    if not os.path.isfile(tar_path):
        print(
            json.dumps({"ok": False, "error": f"Tar not found: {tar_path}"}, indent=2)
        )
        return
    os.makedirs(dest, exist_ok=True)
    with tarfile.open(tar_path, "r") as tf:
        tf.extractall(dest)
    if delete:
        try:
            os.remove(tar_path)
        except Exception:
            pass
    print(
        json.dumps({"ok": True, "tar_path": tar_path, "extracted_to": dest}, indent=2)
    )


# Assemble split tar parts and extract (handles >128 GiB archives)
@app.function(image=cpu_image, timeout=6 * 60 * 60, volumes={"/data": data_vol})
def assemble_and_extract(
    prefix: str = "/data/rl_embeddings.tar.part.",
    dest: str = "/data",
    delete_parts: bool = True,
):
    import glob
    import json
    import tarfile

    parts = sorted(glob.glob(prefix + "*"))
    if not parts:
        print(
            json.dumps(
                {"ok": False, "error": f"No parts found for prefix: {prefix}"}, indent=2
            )
        )
        return

    combined = "/data/rl_embeddings.tar"
    os.makedirs(dest, exist_ok=True)

    # Concatenate parts into a single tar inside the volume
    with open(combined, "wb") as out:
        for p in parts:
            with open(p, "rb") as inp:
                while True:
                    chunk = inp.read(16 * 1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)

    # Extract, then cleanup
    with tarfile.open(combined, "r") as tf:
        tf.extractall(dest)

    try:
        os.remove(combined)
    except Exception:
        pass

    if delete_parts:
        for p in parts:
            try:
                os.remove(p)
            except Exception:
                pass

    print(
        json.dumps(
            {"ok": True, "assembled_from": parts, "extracted_to": dest}, indent=2
        )
    )
