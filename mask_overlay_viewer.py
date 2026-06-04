#!/usr/bin/env python3
import argparse
from pathlib import Path
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageTk


DEFAULT_DATASET = Path(
    "/ref_neus/work/sleeve5-2_posefixed_perimage_crop2000_nerfply_20260528_100340/dataset"
)
DEFAULT_OUTPUT = Path("/data/exp/sleeve5-2/mask_overlays")


def list_pairs(dataset_dir):
    image_dir = dataset_dir / "images"
    mask_dir = dataset_dir / "masks"
    pairs = []
    for image_path in sorted(image_dir.glob("*")):
        mask_path = mask_dir / image_path.name
        if image_path.is_file() and mask_path.is_file():
            pairs.append((image_path, mask_path))
    return pairs


def make_overlay(image_path, mask_path, alpha=0.38, max_size=None):
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.Resampling.NEAREST)

    binary = mask.point(lambda p: 255 if p > 127 else 0)
    foreground = Image.new("RGB", image.size, (0, 210, 135))
    background = Image.new("RGB", image.size, (230, 40, 90))
    overlay = Image.composite(foreground, background, binary)
    blended = Image.blend(image, overlay, alpha)

    # Yellow boundary makes small mask misses at thin geometry much easier to see.
    edge = ImageChops.difference(binary.filter(ImageFilter.MaxFilter(5)), binary.filter(ImageFilter.MinFilter(5)))
    draw = ImageDraw.Draw(blended)
    draw.bitmap((0, 0), edge, fill=(255, 230, 0))

    if max_size is not None:
        blended.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return blended


def export_overlays(dataset_dir, output_dir, alpha):
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = list_pairs(dataset_dir)
    for index, (image_path, mask_path) in enumerate(pairs, start=1):
        overlay = make_overlay(image_path, mask_path, alpha=alpha)
        out_path = output_dir / f"{index:04d}_{image_path.name}"
        overlay.save(out_path)
    return len(pairs)


class Viewer:
    def __init__(self, root, dataset_dir, alpha):
        self.root = root
        self.dataset_dir = dataset_dir
        self.pairs = list_pairs(dataset_dir)
        self.index = 0
        self.alpha = tk.DoubleVar(value=alpha)
        self.photo = None

        root.title("Mask Overlay Viewer")
        root.geometry("1200x950")

        toolbar = ttk.Frame(root)
        toolbar.pack(fill="x", padx=8, pady=8)

        ttk.Button(toolbar, text="Prev", command=self.prev_image).pack(side="left")
        ttk.Button(toolbar, text="Next", command=self.next_image).pack(side="left", padx=(6, 12))
        ttk.Label(toolbar, text="Alpha").pack(side="left")
        slider = ttk.Scale(toolbar, from_=0.0, to=0.85, variable=self.alpha, command=lambda _: self.show())
        slider.pack(side="left", fill="x", expand=True, padx=8)

        self.status = ttk.Label(toolbar, text="")
        self.status.pack(side="right")

        self.canvas = tk.Canvas(root, background="#202020", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _: self.show())
        root.bind("<Left>", lambda _: self.prev_image())
        root.bind("<Right>", lambda _: self.next_image())
        root.bind("<space>", lambda _: self.next_image())

        self.show()

    def prev_image(self):
        if self.pairs:
            self.index = (self.index - 1) % len(self.pairs)
            self.show()

    def next_image(self):
        if self.pairs:
            self.index = (self.index + 1) % len(self.pairs)
            self.show()

    def show(self):
        self.canvas.delete("all")
        if not self.pairs:
            self.status.configure(text="No image/mask pairs found")
            return

        image_path, mask_path = self.pairs[self.index]
        width = max(self.canvas.winfo_width() - 20, 200)
        height = max(self.canvas.winfo_height() - 20, 200)
        max_size = min(width, height)
        overlay = make_overlay(image_path, mask_path, alpha=self.alpha.get(), max_size=max_size)
        self.photo = ImageTk.PhotoImage(overlay)
        self.canvas.create_image(width // 2 + 10, height // 2 + 10, image=self.photo, anchor="center")
        self.status.configure(text=f"{self.index + 1}/{len(self.pairs)}  {image_path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--alpha", type=float, default=0.38)
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()

    if args.export:
        count = export_overlays(args.dataset, args.output, args.alpha)
        print(f"Wrote {count} overlays to {args.output}")
        return

    root = tk.Tk()
    Viewer(root, args.dataset, args.alpha)
    root.mainloop()


if __name__ == "__main__":
    main()
