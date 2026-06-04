#!/usr/bin/env bash
set -euo pipefail

SOURCE_DATASET="${SOURCE_DATASET:-/data/sleeve5-2}"
SOURCE_IMAGES="${SOURCE_IMAGES:-$SOURCE_DATASET/images}"
SOURCE_MASKS="${SOURCE_MASKS:-$SOURCE_DATASET/masques_sleeve}"
OUT_ROOT="${OUT_ROOT:-/ref_neus/work/sleeve5-2_perimage_crop2000_sleeve_pass_$(date +%Y%m%d_%H%M%S)}"
TARGET_SIZE="${TARGET_SIZE:-2000}"
CROP_PAD_FRAC="${CROP_PAD_FRAC:-0.15}"
GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

CROPPED_SOURCE="$OUT_ROOT/cropped_source"
IMAGES_DIR="$CROPPED_SOURCE/images"
MASKS_DIR="$CROPPED_SOURCE/masques_sleeve"
COLMAP_WORK="$OUT_ROOT/colmap"
COLMAP_MASKS="$COLMAP_WORK/masques_sleeve_png"
COLMAP_SPARSE="$COLMAP_WORK/sparse"
DB_PATH="$COLMAP_WORK/database.db"

for required_dir in "$SOURCE_IMAGES" "$SOURCE_MASKS"; do
    if [ ! -d "$required_dir" ]; then
        echo "Missing required directory: $required_dir" >&2
        exit 1
    fi
done

mkdir -p "$OUT_ROOT" "$COLMAP_WORK" "$COLMAP_MASKS" "$COLMAP_SPARSE"

"$PYTHON_BIN" crop_dataset_per_image.py \
    --images "$SOURCE_IMAGES" \
    --masks "$SOURCE_MASKS" \
    --out "$CROPPED_SOURCE" \
    --target-size "$TARGET_SIZE" \
    --pad-frac "$CROP_PAD_FRAC"

"$PYTHON_BIN" masks2png.py "$MASKS_DIR" --output-dir "$COLMAP_MASKS"

colmap feature_extractor \
    --database_path "$DB_PATH" \
    --image_path "$IMAGES_DIR" \
    --ImageReader.mask_path "$COLMAP_MASKS" \
    --ImageReader.single_camera 0 \
    --FeatureExtraction.gpu_index "$GPU"

colmap exhaustive_matcher \
    --database_path "$DB_PATH" \
    --FeatureMatching.gpu_index "$GPU"

colmap mapper \
    --database_path "$DB_PATH" \
    --image_path "$IMAGES_DIR" \
    --output_path "$COLMAP_SPARSE" \
    --Mapper.multiple_models 0

colmap bundle_adjuster \
    --input_path "$COLMAP_SPARSE/0" \
    --output_path "$COLMAP_SPARSE/0" \
    --BundleAdjustment.refine_principal_point 1

cat <<EOF
Stopped before model conversion, as requested.
Cropped sources: $CROPPED_SOURCE
COLMAP sparse model: $COLMAP_SPARSE/0
COLMAP database: $DB_PATH
Crop metadata: $CROPPED_SOURCE/crop_meta.json
EOF
