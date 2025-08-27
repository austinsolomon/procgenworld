import argparse
import torch
from pano_init.utils.pipeline_flux import FluxPipeline

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["i2p", "t2p"], required=True)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--input_image_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--fov", type=int, default=90)
    parser.add_argument("--output_path", type=str, required=True)
    return parser.parse_args()

def main(args):
    torch.manual_seed(args.seed)

    # --- load pipeline with memory-aware placement ---
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # VRAM optimizations (no quality loss)
    pipe.enable_attention_slicing()
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()

    if args.mode == "t2p":
        if not args.prompt:
            raise ValueError("Text-to-panorama requires --prompt")
        image = pipe(prompt=args.prompt).images[0]
    elif args.mode == "i2p":
        if not args.input_image_path:
            raise ValueError("Image-to-panorama requires --input_image_path")
        image = pipe(image=args.input_image_path).images[0]
    else:
        raise ValueError(f"Unknown mode {args.mode}")

    image.save(f"{args.output_path}/pano_img.jpg")
    with open(f"{args.output_path}/prompt.txt", "w", encoding="utf-8") as f:
        f.write(args.prompt or "")

if __name__ == "__main__":
    args = parse_args()
    main(args)

