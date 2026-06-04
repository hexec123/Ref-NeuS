from PIL import Image
import glob
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("data_dir", type=str)
parser.add_argument("--output-dir", type=str, default=None)

args = parser.parse_args()

if args.output_dir is None:
    args.output_dir = args.data_dir

if os.path.exists(args.output_dir) and not os.path.isdir(args.output_dir):
    raise ValueError(f"{args.output_dir} is not a directory.")
os.makedirs(args.output_dir, exist_ok=True)

for f in glob.glob(f"{args.data_dir}/*.jpg"):
    Image.open(f).save(f"{args.output_dir}/{os.path.basename(f)}.png")

