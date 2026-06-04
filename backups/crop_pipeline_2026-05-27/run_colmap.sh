#!/usr/bin/env bash
set -euo pipefail

SOURCE_DATASET="${SOURCE_DATASET:-/data/sleeve5-2}"
IMAGES_DIR="${IMAGES_DIR:-$SOURCE_DATASET/images}"
MASKS_DIR="${MASKS_DIR:-$SOURCE_DATASET/masques_sleeve}"
OUT_ROOT="${OUT_ROOT:-/ref_neus/work/sleeve5-2_sleeve_pass_$(date +%Y%m%d_%H%M%S)}"
RUN_TRAIN="${RUN_TRAIN:-1}"
GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

COLMAP_WORK="$OUT_ROOT/colmap"
COLMAP_MASKS="$COLMAP_WORK/masques_sleeve_png"
COLMAP_SPARSE="$COLMAP_WORK/sparse"
COLMAP_TEXT="$COLMAP_WORK/text"
COLMAP_DENSE="$COLMAP_WORK/dense"
NERF_DATASET="$OUT_ROOT/dataset"
DB_PATH="$COLMAP_WORK/database.db"
POINTS_PLY="$COLMAP_WORK/points3D.ply"
CONF_OUT="$OUT_ROOT/sleeve5.2.conf"

for required_dir in "$IMAGES_DIR" "$MASKS_DIR"; do
    if [ ! -d "$required_dir" ]; then
        echo "Missing required directory: $required_dir" >&2
        exit 1
    fi
done

mkdir -p "$COLMAP_WORK" "$COLMAP_MASKS" "$COLMAP_SPARSE" "$COLMAP_TEXT" "$COLMAP_DENSE" "$NERF_DATASET/images" "$NERF_DATASET/masks"

"$PYTHON_BIN" masks2png.py "$MASKS_DIR" --output-dir "$COLMAP_MASKS"

colmap feature_extractor \
    --database_path "$DB_PATH" \
    --image_path "$IMAGES_DIR" \
    --ImageReader.mask_path "$COLMAP_MASKS" \
    --ImageReader.single_camera 1

colmap exhaustive_matcher \
    --database_path "$DB_PATH"

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
cp "$POINTS_PLY" "$NERF_DATASET/points_of_interest.ply"

"$PYTHON_BIN" split_transforms.py "$NERF_DATASET"

sed \
    -e "s#base_exp_dir = .*#base_exp_dir = $OUT_ROOT/exp/Ref_NeuS,#" \
    -e "s#data_dir = .*#data_dir = $NERF_DATASET,#" \
    confs/sleeve5.2.conf > "$CONF_OUT"

cat <<EOF
Prepared masked sleeve dataset at: $NERF_DATASET
COLMAP outputs at: $COLMAP_WORK
Training config at: $CONF_OUT
EOF

if [ "$RUN_TRAIN" = "1" ]; then
    "$PYTHON_BIN" exp_runner.py --conf "$CONF_OUT" --mode train --gpu "$GPU"
else
    echo "Skipping training because RUN_TRAIN=$RUN_TRAIN"
fi
