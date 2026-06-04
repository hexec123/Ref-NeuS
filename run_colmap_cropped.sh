#!/usr/bin/env bash
set -euo pipefail

SOURCE_DATASET="${SOURCE_DATASET:-/data/sleeve5-2}"
SOURCE_IMAGES="${SOURCE_IMAGES:-$SOURCE_DATASET/images}"
SOURCE_MASKS="${SOURCE_MASKS:-$SOURCE_DATASET/masques_sleeve}"
OUT_ROOT="${OUT_ROOT:-/ref_neus/work/sleeve5-2_crop2000_sleeve_pass_$(date +%Y%m%d_%H%M%S)}"
TARGET_WIDTH="${TARGET_WIDTH:-2000}"
CROP_PAD_FRAC="${CROP_PAD_FRAC:-0.1}"
POINT_FILTER_RADIUS="${POINT_FILTER_RADIUS:-1.10}"
RUN_TRAIN="${RUN_TRAIN:-0}"
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
COLMAP_TEXT="$COLMAP_WORK/text"
COLMAP_DENSE="$COLMAP_WORK/dense"
NERF_DATASET="$OUT_ROOT/dataset"
DB_PATH="$COLMAP_WORK/database.db"
POINTS_PLY="$COLMAP_WORK/points3D.ply"
CONF_OUT="$OUT_ROOT/sleeve5.2.conf"

for required_dir in "$SOURCE_IMAGES" "$SOURCE_MASKS"; do
    if [ ! -d "$required_dir" ]; then
        echo "Missing required directory: $required_dir" >&2
        exit 1
    fi
done

mkdir -p "$OUT_ROOT" "$COLMAP_WORK" "$COLMAP_MASKS" "$COLMAP_SPARSE" "$COLMAP_TEXT" "$COLMAP_DENSE" "$NERF_DATASET/images" "$NERF_DATASET/masks"

"$PYTHON_BIN" crop_dataset.py \
    --images "$SOURCE_IMAGES" \
    --masks "$SOURCE_MASKS" \
    --out "$CROPPED_SOURCE" \
    --target-width "$TARGET_WIDTH" \
    --pad-frac "$CROP_PAD_FRAC"
cp "$CROPPED_SOURCE/crop_meta.json" "$NERF_DATASET/crop_meta.json"

"$PYTHON_BIN" masks2png.py "$MASKS_DIR" --output-dir "$COLMAP_MASKS"

colmap feature_extractor \
    --database_path "$DB_PATH" \
    --image_path "$IMAGES_DIR" \
    --ImageReader.mask_path "$COLMAP_MASKS" \
    --ImageReader.single_camera 1 \
    --FeatureExtraction.gpu_index "$GPU"

colmap exhaustive_matcher \
    --database_path "$DB_PATH" \
    --FeatureMatching.gpu_index "$GPU"

colmap mapper \
    --database_path "$DB_PATH" \
    --image_path "$IMAGES_DIR" \
    --output_path "$COLMAP_SPARSE"

colmap bundle_adjuster \
    --input_path "$COLMAP_SPARSE/0" \
    --output_path "$COLMAP_SPARSE/0" \
    --BundleAdjustment.refine_principal_point 1

colmap model_converter \
    --input_path "$COLMAP_SPARSE/0" \
    --output_path "$COLMAP_TEXT" \
    --output_type TXT

colmap model_converter \
    --input_path "$COLMAP_SPARSE/0" \
    --output_path "$POINTS_PLY" \
    --output_type PLY

"$PYTHON_BIN" colmap2nerf.py \
    --images "$IMAGES_DIR" \
    --text "$COLMAP_TEXT" \
    --out "$NERF_DATASET/transforms.json"

colmap image_undistorter \
    --image_path "$IMAGES_DIR" \
    --input_path "$COLMAP_SPARSE/0" \
    --output_path "$COLMAP_DENSE" \
    --output_type COLMAP

colmap patch_match_stereo \
    --workspace_path "$COLMAP_DENSE" \
    --workspace_format COLMAP \
    --PatchMatchStereo.gpu_index "$GPU" \
    --PatchMatchStereo.geom_consistency true \
    --PatchMatchStereo.window_radius 3 \
    --PatchMatchStereo.num_samples 15 \
    --PatchMatchStereo.num_iterations 7

colmap stereo_fusion \
    --workspace_path "$COLMAP_DENSE" \
    --workspace_format COLMAP \
    --output_path "$COLMAP_DENSE/fused.ply"

"$PYTHON_BIN" masks2png.py "$IMAGES_DIR" --output-dir "$NERF_DATASET/images"
"$PYTHON_BIN" masks2png.py "$MASKS_DIR" --output-dir "$NERF_DATASET/masks"
cp "$POINTS_PLY" "$NERF_DATASET/points_of_interest_colmap_raw.ply"

"$PYTHON_BIN" colmap_points_to_nerf.py \
    --colmap-images "$COLMAP_TEXT/images.txt" \
    --transforms "$NERF_DATASET/transforms.json" \
    --points "$POINTS_PLY" \
    --out "$NERF_DATASET/points_of_interest.ply" \
    --filter-radius "$POINT_FILTER_RADIUS"

"$PYTHON_BIN" split_transforms.py "$NERF_DATASET"

sed \
    -e "s#base_exp_dir = .*#base_exp_dir = $OUT_ROOT/exp/Ref_NeuS,#" \
    -e "s#data_dir = .*#data_dir = $NERF_DATASET,#" \
    -e "s#batch_size = .*#batch_size = 512,#" \
    -e "s#val_mesh_freq = .*#val_mesh_freq = 999999,#" \
    confs/sleeve5.2.conf > "$CONF_OUT"

cat <<EOF
Prepared cropped masked sleeve dataset at: $NERF_DATASET
Cropped sources at: $CROPPED_SOURCE
COLMAP outputs at: $COLMAP_WORK
Training config at: $CONF_OUT
EOF

if [ "$RUN_TRAIN" = "1" ]; then
    "$PYTHON_BIN" exp_runner.py --conf "$CONF_OUT" --mode train --gpu "$GPU"
else
    echo "Skipping training because RUN_TRAIN=$RUN_TRAIN"
fi
