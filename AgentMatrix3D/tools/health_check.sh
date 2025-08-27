#!/usr/bin/env bash
set -o pipefail

ok(){   echo -e "\033[32m✔ $*\033[0m"; }
warn(){ echo -e "\033[33m⚠ $*\033[0m"; }
err(){  echo -e "\033[31m✖ $*\033[0m"; }

echo "=== AgentMatrix3D Health Check ==="
echo "Repo root: $PWD"
date

# --- basic repo sanity
[ -f "code/panoramic_image_to_video.py" ] || { err "not at repo root (missing code/panoramic_image_to_video.py)"; exit 1; }
ok "repo root looks correct"

# --- env summary
echo "=== Environment ==="
echo "SHELL: $SHELL"
echo "CONDA PREFIX: ${CONDA_PREFIX:-<none>}"
command -v conda >/dev/null 2>&1 && conda info --envs || warn "conda not found"
echo "Python:"
which python || true
python --version || err "python not runnable"

echo "CUDA-related env:"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-<unset>}"
echo "  TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-<unset>}"
echo

# --- GPU / driver / toolkit
echo "=== NVIDIA / CUDA ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  warn "nvidia-smi not found"
fi
echo
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version || true
else
  warn "nvcc (CUDA toolkit) not found on PATH — ok if you use PyTorch wheels"
fi
echo

# --- python package versions + torch cuda status
python - <<'PY'
import sys, platform
print("=== Python / Torch stack ===")
print("Platform:", platform.platform())
try:
    import torch
    print("torch:", torch.__version__)
    print("  cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("  device count:", torch.cuda.device_count())
        for d in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(d)
            print(f"  [{d}] {p.name}  {p.total_memory/1024**3:.1f} GB  cc {p.major}.{p.minor}")
    print("  torch.version.cuda:", torch.version.cuda)
    print("  cudnn.enabled:", getattr(torch.backends, 'cudnn', None) and torch.backends.cudnn.enabled)
except Exception as e:
    print("torch import failed:", e)

for name in ["torchvision","torchaudio","xformers","deepspeed","triton","flash_attn","numpy","opencv-python","pillow"]:
    try:
        mod = __import__(name.replace("-","_"))
        ver = getattr(mod, "__version__", "<no __version__>")
        print(f"{name}: {ver}")
    except Exception as e:
        print(f"{name}: NOT FOUND ({e})")
PY
echo

# --- file structure & checkpoints
echo "=== Files / Checkpoints ==="
paths=(
  "code/DiffSynth-Studio/diffsynth/pipelines/wan_video.py"
  "code/MoGe/scripts/infer_panorama.py"
  "output/example1/pano_img.jpg"
  "output/example1/cam_8f.json"
  "output/example1/moge/mask.png"
  "checkpoints/Wan-AI/Wan2.1-I2V-14B-480P"
)
for p in "${paths[@]}"; do
  if [ -e "$p" ]; then ok "exists: $p"; else err "missing: $p"; fi
done

if [ -d "checkpoints/Wan-AI/Wan2.1-I2V-14B-480P" ]; then
  echo
  echo "Checkpoint dir size:"
  du -sh checkpoints/Wan-AI/Wan2.1-I2V-14B-480P || true
  echo "Expected key files present?"
  for f in \
    "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
    "models_t5_umt5-xxl-enc-bf16.pth" \
    "Wan2.1_VAE.pth"
  do
    if [ -f "checkpoints/Wan-AI/Wan2.1-I2V-14B-480P/$f" ]; then ok "found: $f"; else err "missing: $f"; fi
  done
  echo "Diffusion shards count (expect 7):"
  ls -1 checkpoints/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-*.safetensors 2>/dev/null | wc -l
fi
echo

# --- image & mask dimensions
python - <<'PY'
from PIL import Image
from pathlib import Path
print("=== Pano/Mask dimensions ===")
pano = Path("output/example1/pano_img.jpg")
mask = Path("output/example1/moge/mask.png")
def info(p):
    if not p.exists():
        print(f"{p}: MISSING")
        return None
    try:
        im = Image.open(p)
        print(f"{p}: {im.size} mode={im.mode}")
        return im.size
    except Exception as e:
        print(f"{p}: ERROR {e}")
        return None
ps = info(pano)
ms = info(mask)
if ps and ms and ms != (2048,1024):
    print(f"⚠ mask is {ms}, many scripts assume (2048,1024). Consider resizing the mask to match.")
PY
echo

# --- cam json sanity
python - <<'PY'
import json
from pathlib import Path
print("=== Camera JSON sanity ===")
j = Path("output/example1/cam_8f.json")
try:
    d = json.loads(j.read_text())
    if isinstance(d, dict):
        print("keys:", list(d.keys())[:10])
    elif isinstance(d, list):
        print("list length:", len(d))
    else:
        print("json type:", type(d))
except Exception as e:
    print("JSON read error:", e)
PY
echo

# --- README/install.sh hints (pinned versions)
echo "=== Hints from README.md / install.sh ==="
if [ -f README.md ]; then
  echo "-- version-like lines from README.md --"
  grep -nEi 'torch|torchvision|xformers|deepspeed|flash(-|_)attn|cuda|python' README.md || true
else
  warn "README.md missing"
fi
if [ -f code/install.sh ]; then
  echo "-- dependency lines from code/install.sh --"
  grep -nEi 'pip|conda|torch|torchvision|xformers|deepspeed|flash(-|_)attn|cuda|python' code/install.sh || true
else
  warn "code/install.sh missing"
fi
echo

# --- local edits (esp. wan_video patch)
echo "=== git status / local diffs ==="
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git status -s || true
  echo "-- diff for wan_video.py (first 200 lines of patch) --"
  git --no-pager diff -- code/DiffSynth-Studio/diffsynth/pipelines/wan_video.py | sed -n '1,200p'
else
  warn "not a git repo (cannot show diffs)"
fi

echo
ok "health check complete"
