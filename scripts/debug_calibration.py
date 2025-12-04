import sys
from pathlib import Path
import toml

import numpy as np
from mokap.geometry.backend import xp

import matplotlib.pyplot as plt

from mokap.utils.visualisation import visualise_calibration_scene
from mokap.geometry import (triangulate, compose_transform_matrix, invert_transform,
                            project_to_cameras, reprojection_errors)


def load_calibration_data(folder: Path):
    """Loads camera parameters and volume from the folder."""
    # TODO: get rid of this and use the new centralised loaders when they're ready

    param_file = folder / 'parameters.toml'
    if not param_file.exists():
        print(f"Error: {param_file} does not exist.")
        sys.exit(1)

    print(f"Loading parameters from: {param_file}")
    data = toml.load(param_file)

    # cam_names = sorted(list(data.keys()))
    cam_names = ['avocado', 'coconut', 'banana', 'strawberry', 'blueberry'] # order is from the VIDEO FILES # TODO: fix this

    print(f"Found {len(cam_names)} cameras: {cam_names}")

    Ks = []
    Ds = []
    rvecs = []
    tvecs = []

    for name in cam_names:
        cam_data = data[name]
        try:
            Ks.append(cam_data['camera_matrix'])
            Ds.append(cam_data['dist_coeffs'])
            rvecs.append(cam_data['rvec'])
            tvecs.append(cam_data['tvec'])
        except KeyError as e:
            print(f"Error: Camera '{name}' is missing parameter {e}")
            sys.exit(1)

    Ks = np.array(Ks, dtype=np.float32)
    Ds = np.array(Ds, dtype=np.float32)
    T_c2w = compose_transform_matrix(xp.stack(rvecs), xp.stack(tvecs))

    # Load Volume of Trust if it exists
    volume = None
    vol_path = folder / 'volume.toml'
    if vol_path.exists():
        volume = toml.load(vol_path)
        print("Loaded Volume of Trust.")

    return cam_names, Ks, Ds, T_c2w, volume


if __name__ == "__main__":

    folder = Path.home() / "Desktop/3d_ant_data/240905-1616/calibration"
    frame_idx = None

    if not folder.exists():
        print(f"Folder not found: {folder}")
        sys.exit(1)

    # Load data
    cam_names, K, D, T_c2w, volume = load_calibration_data(folder)

    # Check for debug point data
    pts_path = folder / 'points2d_stacked.npz'
    vis_path = folder / 'visibility_masks_stacked.npz'

    has_points = pts_path.exists() and vis_path.exists()

    if not has_points:
        print("No debug points found (points2d_stacked.npz). Plotting cameras only.")

        visualise_calibration_scene(
            T_c2w=T_c2w,
            K=K,
            D=D,
            trust_volume=volume,
            camera_names=cam_names,
            orientation='upright'
        )

        plt.title(f"Calibration Setup: {folder.name}")
        plt.show()

    # Triangulation visualisation
    print("Debug points found. Calculating scene...")

    # Load arrays
    points2d_stacked = np.load(pts_path)['arr_0']
    visibility_masks_stacked = np.load(vis_path)['arr_0']

    if frame_idx is None:
        # Find the frame with the most detections across all cameras
        detections_per_frame = np.sum(visibility_masks_stacked, axis=(0, 2))
        frame_idx = np.argmax(detections_per_frame)
        count = detections_per_frame[frame_idx]
        print(f"Auto-selected frame {frame_idx} (Total {int(count)} detections across views)")

    # Prepare data for this specific frame
    points2d_frame = points2d_stacked[:, frame_idx, :, :]
    vis_mask_frame = visibility_masks_stacked[:, frame_idx, :]

    # Mask out invalid points with NaN to clean the plot
    points2d_frame[~vis_mask_frame] = np.nan

    # Triangulate
    T_w2c = invert_transform(T_c2w)

    points3d = triangulate(
        points2d_frame,
        T_w2c,
        K, D,
        weights=vis_mask_frame,
        distortion_model='standard'
    )

    reproj_points, _ = project_to_cameras(
        points3d, T_w2c, K, D, distortion_model='standard'
    )

    err_metrics = reprojection_errors(
        points2d_observed=points2d_frame,
        points2d_reprojected=reproj_points,
        visibility_mask=vis_mask_frame,
        per_point_errors=True
    )

    # (C, N) array of Euclidean distances
    per_point_dists = err_metrics['mre_per_point']

    # Per-camera stats
    camera_stats = {}
    print("\nReprojection errors (Frame {})".format(frame_idx))
    for i, name in enumerate(cam_names):
        cam_errs = per_point_dists[i, :]
        valid_errs = cam_errs[~np.isnan(cam_errs)]

        if len(valid_errs) > 0:
            mean_e = np.mean(valid_errs)
            max_e = np.max(valid_errs)
        else:
            mean_e, max_e = np.nan, np.nan

        camera_stats[name] = {'mean': mean_e, 'max': max_e}
        print(f"  {name:<15}: Mean={mean_e:.4f} px, Max={max_e:.4f} px")
    print("──────────────────────────────────────")

    # Find worst point
    max_err_per_point = np.nanmax(per_point_dists, axis=0)
    if np.all(np.isnan(max_err_per_point)):
        worst_idx = None
    else:
        worst_idx = np.nanargmax(max_err_per_point)
        worst_val = max_err_per_point[worst_idx]
        print(f"Worst 3D point (idx={worst_idx}): Max Error = {worst_val:.4f} px)")

    # Plot
    visualise_calibration_scene(
        T_c2w=T_c2w,
        K=K,
        D=D,
        points3d=points3d,
        points2d=points2d_frame,
        visibility_mask=vis_mask_frame,
        trust_volume=volume,
        point_errors=per_point_dists,
        worst_point_idx=worst_idx,
        camera_names=cam_names,
        frustum_scale=0.5,
        orientation='upright'
    )

    plt.suptitle(f"Calibration Scene (frame {frame_idx})")
    plt.show()