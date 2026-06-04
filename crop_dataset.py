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
    parser.add_argument("--target-width", type=int, default=2000)
    parser.add_argument("--pad-frac", type=float, default=0.1)
    return parser.parse_args()


def mask_bbox(mask):
    ys, xs = np.where(mask > 127)
    if len(xs) == 0:
        return None
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def main():
    args = parse_args()
    out_images = os.path.join(args.out, "images")
    out_masks = os.path.join(args.out, "masques_sleeve")
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_masks, exist_ok=True)

    mask_paths = sorted(glob(os.path.join(args.masks, "*")))
    boxes = []
    image_shape = None
    for mask_path in mask_paths:
        mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        if image_shape is None:
            image_shape = mask.shape[:2]
        bbox = mask_bbox(mask)
        if bbox is not None:
            boxes.append(bbox)

    if not boxes:
        raise RuntimeError(f"No non-empty masks found in {args.masks}")

    boxes_np = np.asarray(boxes, dtype=np.float32)
    x1, y1 = boxes_np[:, 0].min(), boxes_np[:, 1].min()
    x2, y2 = boxes_np[:, 2].max(), boxes_np[:, 3].max()
    box_w = x2 - x1
    box_h = y2 - y1
    pad = max(box_w, box_h) * args.pad_frac

    height, width = image_shape
    crop_x1 = max(0, int(np.floor(x1 - pad)))
    crop_y1 = max(0, int(np.floor(y1 - pad)))
    crop_x2 = min(width, int(np.ceil(x2 + pad)))
    crop_y2 = min(height, int(np.ceil(y2 + pad)))

    crop_w = crop_x2 - crop_x1
    crop_h = crop_y2 - crop_y1
    target_h = int(round(crop_h * args.target_width / crop_w))
    target_size = (args.target_width, target_h)

    image_paths = sorted(glob(os.path.join(args.images, "*")))
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

        image_crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
        mask_crop = mask[crop_y1:crop_y2, crop_x1:crop_x2]

        image_out = cv.resize(image_crop, target_size, interpolation=cv.INTER_AREA)
        mask_out = cv.resize(mask_crop, target_size, interpolation=cv.INTER_NEAREST)
        mask_out = np.where(mask_out > 127, 255, 0).astype(np.uint8)

        cv.imwrite(os.path.join(out_images, name), image_out)
        cv.imwrite(os.path.join(out_masks, name), mask_out)
        written += 1

    meta = {
        "source_images": args.images,
        "source_masks": args.masks,
        "original_size": [int(width), int(height)],
        "crop_xyxy": [crop_x1, crop_y1, crop_x2, crop_y2],
        "crop_size": [crop_w, crop_h],
        "target_size": [target_size[0], target_size[1]],
        "target_width": args.target_width,
        "pad_frac": args.pad_frac,
        "written": written,
    }
    with open(os.path.join(args.out, "crop_meta.json"), "w") as fp:
        json.dump(meta, fp, indent=2)

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
