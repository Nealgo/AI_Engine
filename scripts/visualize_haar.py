import argparse
import os

import numpy as np
import torch
from PIL import Image
from pytorch_wavelets import DWTForward
from torchvision import transforms


def to_gray_uint8(tensor, signed=False):
    """Convert a CHW or HW tensor to uint8 grayscale image data."""
    t = tensor.detach().cpu().float()

    if t.dim() == 3:
        # Aggregate RGB channels for visualization.
        t = t.mean(dim=0)
    elif t.dim() != 2:
        raise ValueError(f"Expected tensor with dim 2 or 3, got {t.dim()}")

    if signed:
        max_abs = float(t.abs().max().item())
        if max_abs < 1e-8:
            norm = torch.zeros_like(t)
        else:
            # Map [-max_abs, max_abs] to [0, 255], where 128 is zero.
            norm = (t / max_abs) * 127.5 + 127.5
    else:
        t_min = float(t.min().item())
        t_max = float(t.max().item())
        if abs(t_max - t_min) < 1e-8:
            norm = torch.zeros_like(t)
        else:
            norm = (t - t_min) / (t_max - t_min)
            norm = norm * 255.0

    return norm.clamp(0, 255).byte().numpy()


def make_grid(ll_img, lh_img, hl_img, hh_img):
    h, w = ll_img.shape
    canvas = np.zeros((h * 2, w * 2), dtype=np.uint8)
    canvas[0:h, 0:w] = ll_img
    canvas[0:h, w:2 * w] = lh_img
    canvas[h:2 * h, 0:w] = hl_img
    canvas[h:2 * h, w:2 * w] = hh_img
    return canvas


def main():
    parser = argparse.ArgumentParser(description="Visualize Haar wavelet LL/LH/HL/HH components")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output_dir", type=str, default="results/haar_vis", help="Directory to save results")
    parser.add_argument(
        "--resize",
        type=str,
        default=None,
        help="Optional resize as 'height,width', e.g. 480,720",
    )
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Compute device")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Input image not found: {args.image}")

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    image = Image.open(args.image).convert("RGB")
    if args.resize is not None:
        height, width = map(int, args.resize.split(","))
        image = image.resize((width, height), Image.BILINEAR)

    to_tensor = transforms.ToTensor()
    x = to_tensor(image).unsqueeze(0).to(device)

    dwt = DWTForward(J=1, mode="zero", wave="haar").to(device)
    with torch.no_grad():
        yL, yH = dwt(x)

    # yL: [B, C, H/2, W/2]
    # yH[0]: [B, C, 3, H/2, W/2], orientation order: LH, HL, HH
    ll = yL[0]
    lh = yH[0][0, :, 0, :, :]
    hl = yH[0][0, :, 1, :, :]
    hh = yH[0][0, :, 2, :, :]

    ll_img = to_gray_uint8(ll, signed=False)
    lh_img = to_gray_uint8(lh, signed=True)
    hl_img = to_gray_uint8(hl, signed=True)
    hh_img = to_gray_uint8(hh, signed=True)
    grid_img = make_grid(ll_img, lh_img, hl_img, hh_img)

    os.makedirs(args.output_dir, exist_ok=True)
    Image.fromarray(ll_img, mode="L").save(os.path.join(args.output_dir, "LL_low_freq.png"))
    Image.fromarray(lh_img, mode="L").save(os.path.join(args.output_dir, "LH_horizontal_detail.png"))
    Image.fromarray(hl_img, mode="L").save(os.path.join(args.output_dir, "HL_vertical_detail.png"))
    Image.fromarray(hh_img, mode="L").save(os.path.join(args.output_dir, "HH_diagonal_detail.png"))
    Image.fromarray(grid_img, mode="L").save(os.path.join(args.output_dir, "haar_components_grid.png"))

    print("Saved wavelet component visualizations:")
    print(os.path.join(args.output_dir, "LL_low_freq.png"))
    print(os.path.join(args.output_dir, "LH_horizontal_detail.png"))
    print(os.path.join(args.output_dir, "HL_vertical_detail.png"))
    print(os.path.join(args.output_dir, "HH_diagonal_detail.png"))
    print(os.path.join(args.output_dir, "haar_components_grid.png"))


if __name__ == "__main__":
    main()
