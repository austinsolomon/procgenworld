cd ~/Matrix-Game/Matrix-Game-2 && \
cp -n inference_streaming.py inference_streaming.py.bak 2>/dev/null || true && \
cat > inference_streaming.py <<'PY'
# auto-driver wrapper: feeds actions + shows progress, then hands off to your original script
import runpy, builtins, time, sys
from pathlib import Path

# pick the underlying original script to run
u = next((Path(p) for p in ("inference_streaming_original.py","inference_streaming.py.bak") if Path(p).exists()), None)
if not u:
    print("[error] Neither inference_streaming_original.py nor inference_streaming.py.bak found.")
    sys.exit(1)

# choreography: mouse then keyboard per step
# Mouse: I(up) K(down) J(left) L(right) U(no move)
# Keys : W(forward) S(back) A(left) D(right) Q(no move)
m = ["K"]*15 + ["I"]*15 + ["L"]*120 + ["U"]*60
k = ["Q"]*(15+15+120) + ["W"]*60

N=len(m); i=0; t0=None
real = builtins.input

def eta(s):
    s=max(0,int(s)); return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"

def inp(p=""):
    global i,t0
    pl=p.lower()
    if "image path" in pl:
        print("[hint] paste your image path (e.g. /home/pg1221/Matrix-Game/Matrix-Game-2/demo_images/temeraire.jpg) and press Enter")
        return real(p)
    if t0 is None:
        t0=time.time()
    if "mouse action" in pl:
        done=min(i,N); el=time.time()-t0
        avg= el/done if done else 0.0
        rem= (N-done)*avg if avg else 0.0
        pct= int(100*done/N) if N else 100
        print(f"[progress] {done:>3}/{N} ({pct:>3}%) | elapsed {int(el//60):02d}:{int(el%60):02d} | eta {eta(rem)}")
        if i<N:
            val=m[i]; i+=1; return val
        return "U"
    if "keyboard action" in pl:
        idx=min(max(i-1,0),N-1)
        return k[idx]
    return real(p)

builtins.input=inp
try:
    runpy.run_path(str(u), run_name="__main__")
finally:
    builtins.input=real
PY

