#!/usr/bin/env bash
set -euo pipefail
keep="Matrix-3D_local"
up="third_party/Matrix-3D"
echo "== Presence checks in $keep =="
need=( "install.sh" "generate.sh" "README.md"
       "code/download_checkpoints.py"
       "code/panoramic_image_generation.py"
       "code/panoramic_image_to_video.py"
       "code/panoramic_video_to_3DScene.py" )
score=0; for f in "${need[@]}"; do
  [ -e "$keep/$f" ] && echo "  ✓ $f" && ((score++)) || echo "  ✗ $f"
done; echo "Score: $score/${#need[@]}"
echo; echo "== File diffs (keeper vs pristine) ==";
diff -ruN "$up" "$keep" > .reports/diff_vs_upstream.patch || true
echo "  Wrote .reports/diff_vs_upstream.patch"
echo; echo "== Env sanity =="
python3 -V || true
python3 - <<'PY'
import torch, torchvision, sys
print("python              :", sys.version.split()[0])
print("torch               :", torch.__version__)
print("torch.cuda.is_available:", torch.cuda.is_available())
print("cuda runtime (torch):", getattr(torch.version, "cuda", "unknown"))
try:
    import flash_attn as fa; print("flash_attn          :", getattr(fa, "__version__", "installed"))
except Exception as e:
    print("flash_attn          : not installed")
PY
command -v nvidia-smi >/dev/null && { echo; echo "== nvidia-smi =="; nvidia-smi | head -n 5; } || true
