import argparse
import json
import os
from copy import deepcopy
from glob import glob

import cv2 as cv
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dataset", required=True)
    parser.add_argument("--source-images", required=True)
    parser.add_argument("--source-masks", required=True)
    parser.add_argument("--out-dataset", required=True)
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
        image_out = np.zeros((target_size, target_size), dtype=image.dtype)
    else:
        image_out = np.zeros((target_size, target_size, image.shape[2]), dtype=image.dtype)
    mask_out = np.zeros((target_size, target_size), dtype=np.uint8)

    paste_x = (target_size - out_w) // 2
    paste_y = (target_size - out_h) // 2
    image_out[paste_y:paste_y + out_h, paste_x:paste_x + out_w] = image_rs
    mask_out[paste_y:paste_y + out_h, paste_x:paste_x + out_w] = mask_rs
    return image_out, mask_out, scale, paste_x, paste_y, out_w, out_h


def frame_intrinsics(frame, data):
    fl_x = float(frame.get("fl_x", data.get("fl_x", 0.0)))
    fl_y = float(frame.get("fl_y", frame.get("fl_x", data.get("fl_y", data.get("fl_x", 0.0)))))
    cx = float(frame.get("cx", data.get("cx", 0.0)))
    cy = float(frame.get("cy", data.get("cy", 0.0)))
    w = float(frame.get("w", data.get("w", 0.0)))
    h = float(frame.get("h", data.get("h", 0.0)))
    if fl_x <= 0 or fl_y <= 0 or w <= 0 or h <= 0:
        raise RuntimeError("Missing source intrinsics in transforms file")
    if cx <= 0:
        cx = w * 0.5
    if cy <= 0:
        cy = h * 0.5
    return fl_x, fl_y, cx, cy, w, h


def process_frame(frame, data, args, out_images, out_masks, meta_frames):
    name = os.path.basename(frame["file_path"])
    image_path = os.path.join(args.source_images, name)
    mask_path = os.path.join(args.source_masks, name)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Missing source image: {image_path}")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Missing source mask: {mask_path}")

    image = cv.imread(image_path, cv.IMREAD_UNCHANGED)
    mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    if mask is None:
        raise RuntimeError(f"Could not read mask: {mask_path}")
    if image.shape[:2] != mask.shape[:2]:
        raise RuntimeError(f"Image/mask size mismatch for {name}: {image.shape[:2]} vs {mask.shape[:2]}")

    bbox = mask_bbox(mask)
    if bbox is None:
        raise RuntimeError(f"Mask is empty for {name}")
    x1, y1, x2, y2 = bbox
    box_w = x2 - x1
    box_h = y2 - y1
    pad = int(np.ceil(max(box_w, box_h) * args.pad_frac))
    img_h, img_w = image.shape[:2]
    crop_x1 = max(0, x1 - pad)
    crop_y1 = max(0, y1 - pad)
    crop_x2 = min(img_w, x2 + pad)
    crop_y2 = min(img_h, y2 + pad)

    image_crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
    mask_crop = mask[crop_y1:crop_y2, crop_x1:crop_x2]
    image_out, mask_out, scale, paste_x, paste_y, resized_w, resized_h = resize_to_square(
        image_crop, mask_crop, args.target_size
    )

    cv.imwrite(os.path.join(out_images, name + ".png"), image_out)
    cv.imwrite(os.path.join(out_masks, name + ".png"), mask_out)

    fl_x, fl_y, cx, cy, src_w, src_h = frame_intrinsics(frame, data)
    sx = img_w / src_w
    sy = img_h / src_h
    fl_x *= sx
    fl_y *= sy
    cx *= sx
    cy *= sy

    updated = deepcopy(frame)
    updated["file_path"] = f"./images/{name}"
    updated["fl_x"] = fl_x * scale
    updated["fl_y"] = fl_y * scale
    updated["cx"] = (cx - crop_x1) * scale + paste_x
    updated["cy"] = (cy - crop_y1) * scale + paste_y
    updated["w"] = float(args.target_size)
    updated["h"] = float(args.target_size)
    updated["camera_angle_x"] = float(2.0 * np.arctan(args.target_size / (2.0 * updated["fl_x"])))
    updated["camera_angle_y"] = float(2.0 * np.arctan(args.target_size / (2.0 * updated["fl_y"])))

    meta_frames[name] = {
        "source_size": [int(img_w), int(img_h)],
        "mask_bbox_xyxy": [x1, y1, x2, y2],
        "crop_xyxy": [crop_x1, crop_y1, crop_x2, crop_y2],
        "crop_size": [int(crop_x2 - crop_x1), int(crop_y2 - crop_y1)],
        "resize_scale": float(scale),
        "resized_size": [int(resized_w), int(resized_h)],
        "paste_xy": [int(paste_x), int(paste_y)],
        "fl_x": float(updated["fl_x"]),
        "fl_y": float(updated["fl_y"]),
        "cx": float(updated["cx"]),
        "cy": float(updated["cy"]),
    }
    return updated


def write_transforms(src_path, dst_path, args, out_images, out_masks, meta_frames):
    with open(src_path, "r") as fp:
        data = json.load(fp)

    out = deepcopy(data)
    out["w"] = float(args.target_size)
    out["h"] = float(args.target_size)
    out["frames"] = [
        process_frame(frame, data, args, out_images, out_masks, meta_frames)
        for frame in data["frames"]
    ]

    if out["frames"]:
        first = out["frames"][0]
        for key in ("fl_x", "fl_y", "cx", "cy", "camera_angle_x", "camera_angle_y"):
            out[key] = first[key]

    with open(dst_path, "w") as fp:
        json.dump(out, fp, indent=2)


def main():
    args = parse_args()
    out_images = os.path.join(args.out_dataset, "images")
    out_masks = os.path.join(args.out_dataset, "masks")
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_masks, exist_ok=True)

    meta_frames = {}
    for name in ("transforms.json", "transforms_train.json", "transforms_test.json"):
        src = os.path.join(args.base_dataset, name)
        if not os.path.exists(src):
            continue
        write_transforms(src, os.path.join(args.out_dataset, name), args, out_images, out_masks, meta_frames)

    crop_widths = np.array([item["crop_size"][0] for item in meta_frames.values()], dtype=np.int32)
    crop_heights = np.array([item["crop_size"][1] for item in meta_frames.values()], dtype=np.int32)
    scales = np.array([item["resize_scale"] for item in meta_frames.values()], dtype=np.float32)
    meta = {
        "mode": "pose_fixed_per_image_square",
        "base_dataset": args.base_dataset,
        "source_images": args.source_images,
        "source_masks": args.source_masks,
        "target_size": [args.target_size, args.target_size],
        "pad_frac": args.pad_frac,
        "written_unique_frames": len(meta_frames),
        "crop_width_min_median_max": [int(crop_widths.min()), int(np.median(crop_widths)), int(crop_widths.max())],
        "crop_height_min_median_max": [int(crop_heights.min()), int(np.median(crop_heights)), int(crop_heights.max())],
        "resize_scale_min_median_max": [float(scales.min()), float(np.median(scales)), float(scales.max())],
        "frames": meta_frames,
    }
    with open(os.path.join(args.out_dataset, "crop_meta.json"), "w") as fp:
        json.dump(meta, fp, indent=2)

    summary = {key: value for key, value in meta.items() if key != "frames"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
