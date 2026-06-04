import argparse
import json
import os
from glob import glob

import cv2 as cv
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--target-size", type=int, default=2000)
    parser.add_argument("--pad-frac", type=float, default=0.15)
    return parser.parse_args()


def mask_bbox(mask):
    ys, xs = np.where(mask > 127)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def resize_to_square(image, mask, target_size):
    h, w = image.shape[:2]
    scale = target_size / float(max(w, h))
    out_w = max(1, int(round(w * scale)))
    out_h = max(1, int(round(h * scale)))

    image_interp = cv.INTER_AREA if scale < 1.0 else cv.INTER_CUBIC
    image_rs = cv.resize(image, (out_w, out_h), interpolation=image_interp)
    mask_rs = cv.resize(mask, (out_w, out_h), interpolation=cv.INTER_NEAREST)
    mask_rs = np.where(mask_rs > 127, 255, 0).astype(np.uint8)

    if image.ndim == 2:
        canvas = np.zeros((target_size, target_size), dtype=image.dtype)
    else:
        canvas = np.zeros((target_size, target_size, image.shape[2]), dtype=image.dtype)
    mask_canvas = np.zeros((target_size, target_size), dtype=np.uint8)

    x0 = (target_size - out_w) // 2
    y0 = (target_size - out_h) // 2
    canvas[y0:y0 + out_h, x0:x0 + out_w] = image_rs
    mask_canvas[y0:y0 + out_h, x0:x0 + out_w] = mask_rs
    return canvas, mask_canvas, scale, x0, y0, out_w, out_h


def main():
    args = parse_args()
    out_images = os.path.join(args.out, "images")
    out_masks = os.path.join(args.out, "masques_sleeve")
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_masks, exist_ok=True)

    image_paths = sorted(glob(os.path.join(args.images, "*")))
    if not image_paths:
        raise RuntimeError(f"No images found in {args.images}")

    per_image = []
    written = 0
    for image_path in image_paths:
        name = os.path.basename(image_path)
        mask_path = os.path.join(args.masks, name)
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Missing matching mask for {name}: {mask_path}")

        image = cv.imread(image_path, cv.IMREAD_UNCHANGED)
        mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        if mask is None:
            raise RuntimeError(f"Could not read mask: {mask_path}")
        if mask.shape[:2] != image.shape[:2]:
            raise RuntimeError(f"Image/mask size mismatch for {name}: {image.shape[:2]} vs {mask.shape[:2]}")

        bbox = mask_bbox(mask)
        if bbox is None:
            raise RuntimeError(f"Mask is empty for {name}")

        x1, y1, x2, y2 = bbox
        box_w = x2 - x1
        box_h = y2 - y1
        pad = int(np.ceil(max(box_w, box_h) * args.pad_frac))
        h, w = image.shape[:2]
        crop_x1 = max(0, x1 - pad)
        crop_y1 = max(0, y1 - pad)
        crop_x2 = min(w, x2 + pad)
        crop_y2 = min(h, y2 + pad)

        image_crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
        mask_crop = mask[crop_y1:crop_y2, crop_x1:crop_x2]
        image_out, mask_out, scale, paste_x, paste_y, resized_w, resized_h = resize_to_square(
            image_crop, mask_crop, args.target_size
        )

        cv.imwrite(os.path.join(out_images, name), image_out)
        cv.imwrite(os.path.join(out_masks, name), mask_out)

        per_image.append({
            "file": name,
            "original_size": [int(w), int(h)],
            "mask_bbox_xyxy": [x1, y1, x2, y2],
            "crop_xyxy": [crop_x1, crop_y1, crop_x2, crop_y2],
            "crop_size": [int(crop_x2 - crop_x1), int(crop_y2 - crop_y1)],
            "resize_scale": float(scale),
            "resized_size": [int(resized_w), int(resized_h)],
            "paste_xy": [int(paste_x), int(paste_y)],
        })
        written += 1

    crop_widths = np.array([item["crop_size"][0] for item in per_image], dtype=np.int32)
    crop_heights = np.array([item["crop_size"][1] for item in per_image], dtype=np.int32)
    scales = np.array([item["resize_scale"] for item in per_image], dtype=np.float32)
    meta = {
        "mode": "per_image_square",
        "source_images": args.images,
        "source_masks": args.masks,
        "target_size": [args.target_size, args.target_size],
        "pad_frac": args.pad_frac,
        "written": written,
        "crop_width_min_median_max": [int(crop_widths.min()), int(np.median(crop_widths)), int(crop_widths.max())],
        "crop_height_min_median_max": [int(crop_heights.min()), int(np.median(crop_heights)), int(crop_heights.max())],
        "resize_scale_min_median_max": [float(scales.min()), float(np.median(scales)), float(scales.max())],
        "frames": per_image,
    }
    with open(os.path.join(args.out, "crop_meta.json"), "w") as fp:
        json.dump(meta, fp, indent=2)

    summary = {key: value for key, value in meta.items() if key != "frames"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
