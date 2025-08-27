#!/usr/bin/env python3
import importlib, importlib.util, sys, os, shutil, inspect

print("=== Matrix-3D Full Pipeline Verification (Stages 1/2/3) ===\n")

ok = True
repo_root = os.path.abspath(os.getcwd())

def have(path): return os.path.exists(path)
def exists_exe(name): return shutil.which(name) is not None

def import_version(modname, import_as=None):
    try:
        m = importlib.import_module(import_as or modname)
        ver = getattr(m, "__version__", "unknown")
        return True, ver, m
    except Exception as e:
        return False, f"IMPORT-ERROR: {e}", None

def check_pkg(stage, pip_name, target_substr, import_as=None, friendly=None, required=True):
    global ok
    friendly = friendly or (import_as or pip_name)
    present, ver, _ = import_version(pip_name, import_as=import_as)
    if not present:
        mark = "❌ NOT INSTALLED" if required else "⚠️  NOT INSTALLED (optional)"
        print(f"[Stage {stage}] {friendly:28} {mark} (expected {target_substr or 'any'})")
        if required: ok = False
        return
    good = (target_substr in (ver or "")) if target_substr else True
    mark = "✅ OK" if good else f"❌ Expected {target_substr}"
    print(f"[Stage {stage}] {friendly:28} {str(ver):18} {mark}")
    if not good: ok = False

def check_editable(stage, import_name, expected_subdir, friendly=None):
    global ok
    friendly = friendly or import_name
    try:
        m = importlib.import_module(import_name)
        path = os.path.abspath(inspect.getfile(m))
        exp = os.path.join(repo_root, "code", expected_subdir)
        if path.startswith(exp):
            print(f"[Stage {stage}] {friendly:28} ✅ Editable import at {expected_subdir}/")
        else:
            print(f"[Stage {stage}] {friendly:28} ❌ Not imported from code/{expected_subdir}/ (got {path})")
            ok = False
    except Exception as e:
        print(f"[Stage {stage}] {friendly:28} ❌ Import failed ({e})")
        ok = False

def check_script(stage, relpath, must_exec=False):
    global ok
    p = os.path.join(repo_root, relpath)
    if os.path.isfile(p):
        if must_exec and not os.access(p, os.X_OK):
            print(f"[Stage {stage}] {relpath:28} ❌ Found but not executable (chmod +x)")
            ok = False
        else:
            print(f"[Stage {stage}] {relpath:28} ✅ Present")
    else:
        print(f"[Stage {stage}] {relpath:28} ❌ Missing")
        ok = False

# ---------- Global pre-flight ----------
print(f"Working dir: {repo_root}")
check_script("-", "install.sh")
check_script("-", "README.md")
check_script("-", "generate.sh", must_exec=True)

print(f"\nPython version: {sys.version.split()[0]}")
if not sys.version.startswith("3.10"):
    print("❌ Python should be 3.10 per README (conda create -n ... python=3.10)")
    ok = False

# ---------- Stage 1: Panorama (text/image → panorama) ----------
print("\n--- Stage 1: Panorama generation ---")
# Core pins from README/install.sh
check_pkg(1, "torch",                   "2.7.0",  friendly="torch")
check_pkg(1, "torchvision",             "0.22.0", friendly="torchvision")
check_pkg(1, "flash_attn",              "2.7.4.post1")
check_pkg(1, "xformers",                "0.0.31")
check_pkg(1, "open_clip",               "2.7.0",  import_as="open_clip", friendly="open-clip-torch")

# Exact pins in install.sh
check_pkg(1, "diffusers",               "0.34.0")
check_pkg(1, "transformers",            "4.51.0")
check_pkg(1, "matplotlib",              "3.8.4")
check_pkg(1, "omegaconf",               "2.1.1")
check_pkg(1, "modelscope",              "1.28.2")
check_pkg(1, "jaxtyping",               "0.3.2")

# Extras from install.sh (presence check)
for name in [
    ("peft", None),
    ("easydict", None),
    ("torchsde", None),
    ("fairscale", None),
    ("natsort", None),
    ("realesrgan", None),
    ("xfuser", None),
    ("open3d", None),
    ("py360convert", None),
    ("imageio_ffmpeg", "imageio-ffmpeg"),
    ("webdataset", None),
    ("kornia", "0.6"),
    ("streamlit", "1.12.1"),
    ("einops", "0.8.0"),
    ("SwissArmyTransformer", "0.4.12"),
    ("wandb", "0.21.1"),
    ("taming", None),   # CompVis/taming-transformers
    ("clip", None),     # openai-clip
]:
    mod, pin = name
    check_pkg(1, mod, pin or "", friendly=mod, required=False)

# Submodules / editable installs frequently used in Stage 1
check_editable(1, "diffsynth", "DiffSynth-Studio", friendly="DiffSynth-Studio (diffsynth)")

# Must-have scripts
check_script(1, "code/panoramic_image_generation.py")
check_script(1, "code/pano_init/i2p_model.py")

# ---------- Stage 2: Panorama → Video ----------
print("\n--- Stage 2: Panorama → Video ---")
check_pkg(2, "pytorch_lightning",       "1.4.2")
check_pkg(2, "torchmetrics",            "0.7.0")
check_pkg(2, "pytorch3d",               "0.7.7")

check_script(2, "code/panoramic_image_to_video.py")
check_script(2, "code/MoGe/scripts/infer_panorama.py")

# ---------- Stage 3: Video → 3D Scene ----------
print("\n--- Stage 3: Video → 3D Scene ---")
check_pkg(3, "pytorch3d",               "0.7.7")  # again

# Rasterizers & graphics backends from install.sh
check_pkg(3, "nvdiffrast",              "", friendly="nvdiffrast")
check_pkg(3, "nvdiffrast.torch",        "", friendly="nvdiffrast.torch")
check_pkg(3, "simple_knn",              "", friendly="simple-knn")
check_pkg(3, "diff_gaussian_rasterization", "", friendly="diff-gaussian-rasterization")
check_pkg(3, "odgs_gaussian_rasterization","", friendly="odgs-gaussian-rasterization")

check_script(3, "code/panoramic_video_to_3DScene.py")
check_script(3, "code/Pano_GS_Opt/train.py")
check_script(3, "code/panoramic_video_480p_to_3DScene_lrm.py")

# ---------- Global runtime checks ----------
print("\n--- Global runtime checks ---")
print("torchrun                      " + ("✅ Found" if exists_exe("torchrun") else "❌ Not found in PATH"))

# CUDA / device
present, ver, m = import_version("torch")
if present and hasattr(m, "cuda"):
    try:
        gpu_ok = m.cuda.is_available()
        dev = m.cuda.get_device_name(0) if gpu_ok else "N/A"
        print(f"CUDA                          {'✅' if gpu_ok else '❌'} {dev} (CUDA {m.version.cuda})")
    except Exception as e:
        print(f"CUDA                          ❌ torch.cuda exception: {e}")
        ok = False
else:
    print("CUDA                          ❌ torch not importable")
    ok = False

# NCCL backend availability (no init to avoid false negatives on WSL)
try:
    import torch.distributed as dist
    nccl_avail = hasattr(dist, "is_nccl_available") and dist.is_nccl_available()
    print(f"NCCL backend availability     {'✅' if nccl_avail else '⚠️  not reported available'}")
except Exception as e:
    print(f"NCCL backend availability     ⚠️  check failed: {e}")

# checkpoints (at least Wan loras per README table)
wan_dir = os.path.join(repo_root, "checkpoints", "Wan-AI", "wan_lora", "checkpoints")
wan_ok = have(os.path.join(wan_dir, "pano_video_gen_480p.ckpt")) and have(os.path.join(wan_dir, "pano_video_gen_720p.bin"))
if os.path.isdir(os.path.join(repo_root, "checkpoints")) and any(True for _ in os.scandir(os.path.join(repo_root, "checkpoints"))):
    print(f"checkpoints/                   ✅ Present ({'Wan Loras OK' if wan_ok else 'partial'})")
else:
    print("checkpoints/                   ❌ Missing or empty (run: python code/download_checkpoints.py)")
    ok = False

# required stage scripts existence (summary)
print("\n--- Required command entrypoints (README usage) ---")
check_script(1, "code/panoramic_image_generation.py")
check_script(2, "code/panoramic_image_to_video.py")
check_script(3, "code/panoramic_video_to_3DScene.py")
check_script(3, "code/panoramic_video_480p_to_3DScene_lrm.py")

# Final verdict
print("\n=========================================")
if ok:
    print("STAMP_OF_APPROVAL ✅  Environment & repo match developers' intended setup.\n")
else:
    print("❌ Environment does NOT match dev repo. Fix issues above, then re-run.\n")
