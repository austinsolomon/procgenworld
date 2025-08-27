import os, sys, time, builtins, runpy
from pathlib import Path

# --- Config ---
DEFAULT_IMAGE = os.environ.get(
    "MG2_IMAGE",
    "/home/pg1221/Matrix-Game/Matrix-Game-2/demo_images/boxerqual.jpg"
)

# Find the underlying original script
CANDIDATES = [Path("inference_streaming_original.py"), Path("inference_streaming.py.bak")]
UNDERLYING = next((p for p in CANDIDATES if p.exists()), None)
if UNDERLYING is None:
    print("[error] Could not find the original script to delegate to.")
    sys.exit(1)

# --- Choreography plan (210 steps total) ---
# Mouse: I(up), K(down), J(left), L(right), U(no move)
# Keys : W(forward), S(back), A(left), D(right), Q(no move)
MOUSE_SEQ = (["K"]*15) + (["I"]*15) + (["L"]*120) + (["U"]*60)
KEY_SEQ   = (["Q"]*(15+15+120)) + (["W"]*60)
TOTAL = len(MOUSE_SEQ)
assert TOTAL == len(KEY_SEQ)

t0 = time.time()
mouse_i = 0
key_i = 0
real_input = builtins.input

def _fmt(sec: float) -> str:
    sec = int(max(0, sec))
    m, s = divmod(sec, 60)
    return f"{m:02d}:{s:02d}"

def patched_input(prompt: str = "") -> str:
    global mouse_i, key_i
    p = (prompt or "").lower()

    # Auto-supply image path once
    if "image path" in p:
        path = DEFAULT_IMAGE
        print(path, flush=True)
        return path

    # Ignore "press n to stop" prompts
    if "press `n` to stop" in p or "press 'n' to stop" in p:
        print("", flush=True)
        return ""

    # Feed mouse actions
    if "mouse action" in p:
        ans = MOUSE_SEQ[min(mouse_i, TOTAL-1)]
        mouse_i += 1
        print(ans, flush=True)
        return ans

    # Feed keyboard actions + show progress
    if "keyboard action" in p:
        ans = KEY_SEQ[min(key_i, TOTAL-1)]
        key_i += 1
        step = min(mouse_i, key_i)
        elapsed = time.time() - t0
        avg = (elapsed / step) if step > 0 else 0.0
        eta = avg * max(TOTAL - step, 0)
        pct = (step / TOTAL) * 100 if TOTAL else 100.0
        print(f"\n[progress] step {step}/{TOTAL} ({pct:5.1f}%) | elapsed {_fmt(elapsed)} | eta {_fmt(eta)} | mouse={MOUSE_SEQ[min(mouse_i-1, TOTAL-1)]} key={ans}", flush=True)
        return ans

    # Any other prompt: return empty so we never block
    print("", flush=True)
    return ""

# Validate image exists before starting
if not Path(DEFAULT_IMAGE).exists():
    print(f"[error] Image not found: {DEFAULT_IMAGE}")
    sys.exit(1)

# Patch input and delegate to the original script with the same argv
try:
    builtins.input = patched_input
    runpy.run_path(str(UNDERLYING), run_name="__main__")
finally:
    builtins.input = real_input
