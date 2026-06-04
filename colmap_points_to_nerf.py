import argparse
import json
import os

import numpy as np
import trimesh

from colmap2nerf import qvec2rotmat


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--colmap-images", required=True)
    parser.add_argument("--transforms", required=True)
    parser.add_argument("--points", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--filter-radius", type=float, default=1.10)
    return parser.parse_args()


def parse_colmap_camera_centers(images_txt):
    centers = {}
    with open(images_txt, "r") as fp:
        lines = [line.strip() for line in fp if line.strip() and not line.startswith("#")]

    for i in range(0, len(lines), 2):
        elems = lines[i].split()
        qvec = np.asarray(tuple(map(float, elems[1:5])))
        tvec = np.asarray(tuple(map(float, elems[5:8]))).reshape(3, 1)
        name = "_".join(elems[9:])

        rotation = qvec2rotmat(-qvec)
        world_to_camera = np.eye(4)
        world_to_camera[:3, :3] = rotation
        world_to_camera[:3, 3:4] = tvec
        camera_to_world = np.linalg.inv(world_to_camera)
        centers[name] = camera_to_world[:3, 3]

    return centers


def fit_similarity(source_points, target_points):
    source_points = np.asarray(source_points)
    target_points = np.asarray(target_points)
    source_mean = source_points.mean(axis=0)
    target_mean = target_points.mean(axis=0)
    source_centered = source_points - source_mean
    target_centered = target_points - target_mean

    covariance = (target_centered.T @ source_centered) / len(source_points)
    u, singular_values, vt = np.linalg.svd(covariance)
    handedness = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        handedness[-1, -1] = -1
    rotation = u @ handedness @ vt
    variance = (source_centered * source_centered).sum() / len(source_points)
    scale = np.trace(np.diag(singular_values) @ handedness) / variance
    translation = target_mean - scale * rotation @ source_mean
    return scale, rotation, translation


def main():
    args = parse_args()

    colmap_centers = parse_colmap_camera_centers(args.colmap_images)
    with open(args.transforms, "r") as fp:
        transforms = json.load(fp)

    source_centers = []
    target_centers = []
    for frame in transforms["frames"]:
        name = os.path.basename(frame["file_path"])
        if name in colmap_centers:
            source_centers.append(colmap_centers[name])
            target_centers.append(np.asarray(frame["transform_matrix"])[:3, 3])

    if len(source_centers) < 3:
        raise RuntimeError("Not enough matching COLMAP/transform cameras to fit a similarity transform")

    scale, rotation, translation = fit_similarity(source_centers, target_centers)
    fitted = scale * (rotation @ np.asarray(source_centers).T).T + translation
    fit_error = np.linalg.norm(fitted - np.asarray(target_centers), axis=1)

    point_cloud = trimesh.load(args.points)
    vertices = np.asarray(point_cloud.vertices)
    keep = np.ones(len(vertices), dtype=bool)
    if args.filter_radius > 0:
        median = np.median(vertices, axis=0)
        keep = np.linalg.norm(vertices - median, axis=1) <= args.filter_radius

    filtered_vertices = vertices[keep]
    transformed_vertices = scale * (rotation @ filtered_vertices.T).T + translation

    out_cloud = trimesh.points.PointCloud(transformed_vertices)
    if hasattr(point_cloud.visual, "vertex_colors") and len(point_cloud.visual.vertex_colors) == len(vertices):
        out_cloud.colors = np.asarray(point_cloud.visual.vertex_colors)[keep]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_cloud.export(args.out)

    bounds_min = transformed_vertices.min(axis=0)
    bounds_max = transformed_vertices.max(axis=0)
    center = (bounds_min + bounds_max) * 0.5
    radius = np.linalg.norm(transformed_vertices - center, axis=1).max()

    report = {
        "matched_cameras": len(source_centers),
        "fit_error_max": float(fit_error.max()),
        "fit_error_mean": float(fit_error.mean()),
        "input_vertices": int(len(vertices)),
        "output_vertices": int(len(transformed_vertices)),
        "removed_vertices": int(len(vertices) - len(transformed_vertices)),
        "filter_radius": args.filter_radius,
        "bounds_min": bounds_min.tolist(),
        "bounds_max": bounds_max.tolist(),
        "center": center.tolist(),
        "radius": float(radius),
        "out": args.out,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
