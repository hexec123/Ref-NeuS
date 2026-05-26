DATASET_PATH=data/sleeve5-2

python masks2png.py \
    $DATASET_PATH/masques_connecteur/ \
    --output-dir $DATASET_PATH/masques_connecteur_png/

python masks2png.py \
    $DATASET_PATH/masques_sleeve/ \
    --output-dir $DATASET_PATH/masques_sleeve_png/

colmap feature_extractor \
    --database_path database.db \
    --image_path $DATASET_PATH/images \
    --ImageReader.mask_path $DATASET_PATH/masques_sleeve_png \
    --ImageReader.single_camera 1

colmap exhaustive_matcher \
    --database_path database.db

mkdir $DATASET_PATH/sparse

colmap mapper \
    --database_path database.db \
    --image_path $DATASET_PATH/images \
    --output_path $DATASET_PATH/sparse

colmap model_converter \
    --input_path $DATASET_PATH/sparse/0 \
    --output_path $DATASET_PATH/sparse/0/points3D.ply \
    --output_type PLY

python colmap2nerf.py \
    --images $DATASET_PATH/images/ \
    --text $DATASET_PATH/sparse/0/ \
    --out transforms.json

#make dataset
mkdir $DATASET_PATH/dense

colmap image_undistorter \
    --image_path $DATASET_PATH/images \
    --input_path $DATASET_PATH/sparse/0 \
    --output_path $DATASET_PATH/dense \
    --output_type COLMAP

colmap patch_match_stereo \
    --workspace_path $DATASET_PATH/dense \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency true \
    --PatchMatchStereo.window_radius 3 \
    --PatchMatchStereo.num_samples 15 \
    --PatchMatchStereo.num_iterations 7

colmap stereo_fusion \
    --workspace_path $DATASET_PATH/dense \
    --workspace_format COLMAP \
    --output_path $DATASET_PATH/dense/fused.ply


mkdir $DATASET_PATH/dataset

python masks2png.py $DATASET_PATH/images --output-dir $DATASET_PATH/dataset/images_png/
python masks2png.py $DATASET_PATH/masques_connecteur/ --output-dir $DATASET_PATH/dataset/masks_connecteur_png/
python masks2png.py $DATASET_PATH/masques_sleeve/ --output-dir $DATASET_PATH/dataset/masks_sleeve_png/

cp -r $DATASET_PATH/images_png $DATASET_PATH/dataset/images
#cp -r $DATASET_PATH/masques_connecteur_png/ $DATASET_PATH/dataset/masks
cp -r $DATASET_PATH/masques_sleeve_png/ $DATASET_PATH/dataset/masks
cp transforms.json $DATASET_PATH/dataset/
cp $DATASET_PATH/sparse/0/points3D.ply $DATASET_PATH/dataset/points_of_interest.ply

python split_transforms.py $DATASET_PATH/dataset

python exp_runner.py --conf confs/wmask.conf --mode train --gpu 0
