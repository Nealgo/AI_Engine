import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps
from torchvision import transforms

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from model.wdnet import DeformConvTranspose2d


def to_gray_uint8(tensor, signed=False):
    t = tensor.detach().cpu().float()
    if t.dim() == 3:
        t = t.mean(dim=0)
    elif t.dim() != 2:
        raise ValueError(f"Expected dim 2 or 3 tensor, got dim={t.dim()}")

    if signed:
        max_abs = float(t.abs().max().item())
        if max_abs < 1e-8:
            norm = torch.zeros_like(t)
        else:
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


def tensor_to_rgb_uint8(tensor):
    t = tensor.detach().cpu().float()

    if t.dim() == 4:
        t = t[0]
    if t.dim() == 2:
        t = t.unsqueeze(0)
    if t.dim() != 3:
        raise ValueError(f"Expected CHW or BCHW tensor, got dim={t.dim()}")

    c, h, w = t.shape
    if c == 1:
        t = t.repeat(3, 1, 1)
    elif c >= 3:
        t = t[:3]
    else:
        t = torch.cat([t, t[:1]], dim=0)

    t_min = float(t.min().item())
    t_max = float(t.max().item())
    if abs(t_max - t_min) < 1e-8:
        t = torch.zeros_like(t)
    else:
        t = (t - t_min) / (t_max - t_min)

    rgb = (t * 255.0).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    return rgb


def colorize_signed(gray_img, neg=(52, 96, 242), zero=(34, 34, 34), pos=(242, 88, 52)):
    pil_img = Image.fromarray(gray_img, mode="L")
    return np.array(ImageOps.colorize(pil_img, black=neg, mid=zero, white=pos))


def colorize_unsigned(gray_img, low=(14, 31, 74), mid=(42, 163, 177), high=(255, 230, 92)):
    pil_img = Image.fromarray(gray_img, mode="L")
    return np.array(ImageOps.colorize(pil_img, black=low, mid=mid, white=high))


def make_overlay(base_rgb, gray_map, alpha=0.65):
    heat = colorize_unsigned(gray_map, low=(20, 52, 128), mid=(242, 171, 60), high=(224, 58, 46)).astype(np.float32)
    base = base_rgb.astype(np.float32)
    a = (gray_map.astype(np.float32) / 255.0) * float(alpha)
    a = a[..., None]
    out = base * (1.0 - a) + heat * a
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_grid_image(grid_xy, step=16, bg=(16, 16, 16), line=(128, 220, 255)):
    # grid_xy: [H, W, 2] in pixel coordinates
    h, w, _ = grid_xy.shape
    img = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(img)

    for y in range(0, h, step):
        points = []
        for x in range(0, w, step):
            px, py = grid_xy[y, x]
            points.append((float(px), float(py)))
        if len(points) > 1:
            draw.line(points, fill=line, width=1)

    for x in range(0, w, step):
        points = []
        for y in range(0, h, step):
            px, py = grid_xy[y, x]
            points.append((float(px), float(py)))
        if len(points) > 1:
            draw.line(points, fill=line, width=1)

    return np.array(img)


def main():
    parser = argparse.ArgumentParser(description="Visualize DeformConvTranspose2d and offset sampling internals")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output_dir", type=str, default="results/deform_vis", help="Output directory")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Compute device")
    parser.add_argument("--resize", type=str, default=None, help="Optional resize as 'height,width', e.g. 480,720")
    parser.add_argument("--in_channels", type=int, default=3, help="DeformConvTranspose2d input channels")
    parser.add_argument("--out_channels", type=int, default=3, help="DeformConvTranspose2d output channels")
    parser.add_argument("--kernel_size", type=int, default=3, help="Transpose conv kernel size")
    parser.add_argument("--stride", type=int, default=2, help="Transpose conv stride")
    parser.add_argument("--padding", type=int, default=1, help="Transpose conv padding")
    parser.add_argument("--output_padding", type=int, default=1, help="Transpose conv output padding")
    parser.add_argument("--grid_step", type=int, default=24, help="Grid line step for visualization")
    parser.add_argument("--overlay_alpha", type=float, default=0.65, help="Overlay alpha in [0, 1]")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Input image not found: {args.image}")

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    image = Image.open(args.image).convert("RGB")
    if args.resize is not None:
        height, width = map(int, args.resize.split(","))
        image = image.resize((width, height), Image.BILINEAR)

    x = transforms.ToTensor()(image).unsqueeze(0)
    if x.shape[1] != args.in_channels:
        raise ValueError(f"Input image channels={x.shape[1]} but --in_channels={args.in_channels}")
    x = x.to(device)

    module = DeformConvTranspose2d(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        kernel_size=args.kernel_size,
        stride=args.stride,
        padding=args.padding,
        output_padding=args.output_padding,
    ).to(device)
    module.eval()

    with torch.no_grad():
        offset1 = module.offset_conv1(x)
        offset2 = module.offset_conv2(x)
        offset_cat = torch.cat([offset1, offset2], dim=1)
        offset_fused = torch.tanh(module.offset_fuse(offset_cat))

        deconv_out = module.conv_transpose(x)
        n, _, h, w = deconv_out.shape

        offset_up = F.interpolate(offset_fused, size=(h, w), mode="bilinear", align_corners=True)

        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=x.device),
            torch.arange(w, device=x.device),
            indexing="ij",
        )
        base_grid = torch.stack((grid_x, grid_y), dim=-1).float()
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        offset_nhw2 = offset_up.permute(0, 2, 3, 1)
        warped_grid = base_grid + offset_nhw2

        x_denom = float(max(w - 1, 1))
        y_denom = float(max(h - 1, 1))
        norm = torch.tensor([x_denom, y_denom], device=x.device, dtype=warped_grid.dtype)
        grid_norm = (warped_grid / norm - 0.5) * 2.0

        sampled_out = F.grid_sample(deconv_out, grid_norm, align_corners=True)
        final_out = sampled_out + deconv_out

    offset_x = offset_up[0, 0]
    offset_y = offset_up[0, 1]
    offset_mag = torch.sqrt(offset_x ** 2 + offset_y ** 2)

    input_rgb = np.array(image)
    deconv_rgb = tensor_to_rgb_uint8(deconv_out)
    sampled_rgb = tensor_to_rgb_uint8(sampled_out)
    final_rgb = tensor_to_rgb_uint8(final_out)

    offset_x_img = to_gray_uint8(offset_x, signed=True)
    offset_y_img = to_gray_uint8(offset_y, signed=True)
    offset_mag_img = to_gray_uint8(offset_mag, signed=False)
    offset_x_color = colorize_signed(offset_x_img)
    offset_y_color = colorize_signed(offset_y_img)
    offset_mag_color = colorize_unsigned(offset_mag_img)

    diff_sample_vs_deconv = (sampled_out - deconv_out).abs().mean(dim=1).squeeze(0)
    diff_final_vs_deconv = (final_out - deconv_out).abs().mean(dim=1).squeeze(0)
    diff_sample_img = to_gray_uint8(diff_sample_vs_deconv, signed=False)
    diff_final_img = to_gray_uint8(diff_final_vs_deconv, signed=False)

    alpha = float(np.clip(args.overlay_alpha, 0.0, 1.0))
    diff_sample_overlay = make_overlay(deconv_rgb, diff_sample_img, alpha=alpha)
    diff_final_overlay = make_overlay(deconv_rgb, diff_final_img, alpha=alpha)

    base_grid_img = draw_grid_image(base_grid[0].detach().cpu().numpy(), step=args.grid_step, line=(120, 220, 255))
    warped_grid_img = draw_grid_image(warped_grid[0].detach().cpu().numpy(), step=args.grid_step, line=(255, 132, 102))

    os.makedirs(args.output_dir, exist_ok=True)

    Image.fromarray(input_rgb, mode="RGB").save(os.path.join(args.output_dir, "input.png"))
    Image.fromarray(deconv_rgb, mode="RGB").save(os.path.join(args.output_dir, "deconv_output.png"))
    Image.fromarray(sampled_rgb, mode="RGB").save(os.path.join(args.output_dir, "offset_sampled_output.png"))
    Image.fromarray(final_rgb, mode="RGB").save(os.path.join(args.output_dir, "final_residual_output.png"))

    Image.fromarray(offset_x_img, mode="L").save(os.path.join(args.output_dir, "offset_x_gray.png"))
    Image.fromarray(offset_y_img, mode="L").save(os.path.join(args.output_dir, "offset_y_gray.png"))
    Image.fromarray(offset_mag_img, mode="L").save(os.path.join(args.output_dir, "offset_magnitude_gray.png"))
    Image.fromarray(offset_x_color, mode="RGB").save(os.path.join(args.output_dir, "offset_x_color.png"))
    Image.fromarray(offset_y_color, mode="RGB").save(os.path.join(args.output_dir, "offset_y_color.png"))
    Image.fromarray(offset_mag_color, mode="RGB").save(os.path.join(args.output_dir, "offset_magnitude_color.png"))

    Image.fromarray(diff_sample_img, mode="L").save(os.path.join(args.output_dir, "diff_sample_vs_deconv_gray.png"))
    Image.fromarray(diff_final_img, mode="L").save(os.path.join(args.output_dir, "diff_final_vs_deconv_gray.png"))
    Image.fromarray(diff_sample_overlay, mode="RGB").save(os.path.join(args.output_dir, "diff_sample_vs_deconv_overlay.png"))
    Image.fromarray(diff_final_overlay, mode="RGB").save(os.path.join(args.output_dir, "diff_final_vs_deconv_overlay.png"))

    Image.fromarray(base_grid_img, mode="RGB").save(os.path.join(args.output_dir, "base_sampling_grid.png"))
    Image.fromarray(warped_grid_img, mode="RGB").save(os.path.join(args.output_dir, "warped_sampling_grid.png"))

    print("Saved DeformConvTranspose2d visualizations:")
    print(os.path.join(args.output_dir, "input.png"))
    print(os.path.join(args.output_dir, "deconv_output.png"))
    print(os.path.join(args.output_dir, "offset_sampled_output.png"))
    print(os.path.join(args.output_dir, "final_residual_output.png"))
    print(os.path.join(args.output_dir, "offset_x_color.png"))
    print(os.path.join(args.output_dir, "offset_y_color.png"))
    print(os.path.join(args.output_dir, "offset_magnitude_color.png"))
    print(os.path.join(args.output_dir, "diff_sample_vs_deconv_overlay.png"))
    print(os.path.join(args.output_dir, "diff_final_vs_deconv_overlay.png"))
    print(os.path.join(args.output_dir, "base_sampling_grid.png"))
    print(os.path.join(args.output_dir, "warped_sampling_grid.png"))


if __name__ == "__main__":
    main()