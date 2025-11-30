import sys
from pathlib import Path
import numpy as np
import toml
import matplotlib.pyplot as plt

from mokap.utils.visualisation import plot_cameras_3d, plot_triangulation_scene
from mokap.utils.geometry.projective import triangulate
from mokap.utils.geometry.transforms import invert_vectors


def load_calibration_data(folder: Path):
    """Loads camera parameters and volume from the folder."""
    # TODO: get rid of this and use the new centralised loaders when they're ready

    param_file = folder / 'parameters.toml'
    if not param_file.exists():
        print(f"Error: {param_file} does not exist.")
        sys.exit(1)

    print(f"Loading parameters from: {param_file}")
    data = toml.load(param_file)

    cam_names = sorted(list(data.keys()))
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
    rvecs_c2w = np.array(rvecs, dtype=np.float32)
    tvecs_c2w = np.array(tvecs, dtype=np.float32)

    # Load Volume of Trust if it exists
    volume = None
    vol_path = folder / 'volume.toml'
    if vol_path.exists():
        volume = toml.load(vol_path)
        print("Loaded Volume of Trust.")

    return cam_names, Ks, Ds, rvecs_c2w, tvecs_c2w, volume


if __name__ == "__main__":

    folder = Path.home() / "Desktop/3d_ant_data/240905-1616/calibration"
    frame_idx = None

    if not folder.exists():
        print(f"Folder not found: {folder}")
        sys.exit(1)

    # Load data
    cam_names, Ks, Ds, rvecs_c2w, tvecs_c2w, volume = load_calibration_data(folder)

    # Check for debug point data
    pts_path = folder / 'points2d_stacked.npz'
    vis_path = folder / 'visibility_masks_stacked.npz'

    has_points = pts_path.exists() and vis_path.exists()

    if not has_points:
        print("No debug points found (points2d_stacked.npz). Plotting cameras only.")

        plot_cameras_3d(
            rvecs_c2w=rvecs_c2w,
            tvecs_c2w=tvecs_c2w,
            camera_matrices=Ks,
            dist_coeffs=Ds,
            cameras_names=cam_names,
            trust_volume=volume,
            depth_ratio=0.5
        )
        plt.title(f"Calibration Setup: {folder.name}")
        plt.show()

    # Triangulation Visualisation
    print("Debug points found. Calculating scene...")

    # Load arrays
    points2d_stacked = np.load(pts_path)['arr_0']
    visibility_masks_stacked = np.load(vis_path)['arr_0']

    if frame_idx is None:
        # Find the frame with the most detections across all cameras
        # Sum visibility over Cameras (axis 0) and Points (axis 2) -> Result (Frames,)
        detections_per_frame = np.sum(visibility_masks_stacked, axis=(0, 2))
        frame_idx = np.argmax(detections_per_frame)
        count = detections_per_frame[frame_idx]
        print(f"Auto-selected frame {frame_idx} (Total {int(count)} detections across views)")

    # Prepare data for this specific frame
    points2d_frame = points2d_stacked[:, frame_idx, :, :]
    vis_mask_frame = visibility_masks_stacked[:, frame_idx, :]

    # Mask out invalid points with NaN for clean plotting
    points2d_frame[~vis_mask_frame] = np.nan

    # Triangulate
    rvecs_w2c, tvecs_w2c = invert_vectors(rvecs_c2w, tvecs_c2w)

    points3d = triangulate(
        points2d_frame,
        Ks, Ds,
        rvecs_w2c, tvecs_w2c,
        weights=vis_mask_frame
    )

    # Plot
    plot_triangulation_scene(
        points3d=points3d,
        points2d=points2d_frame,
        rvecs_c2w=rvecs_c2w,
        tvecs_c2w=tvecs_c2w,
        camera_matrices=Ks,
        dist_coeffs=Ds,
        visibility_mask=vis_mask_frame,
        cameras_names=cam_names,
        imsizes=(1440, 1080),
        frustums_depth=0.5,
        detections_depth=0.8,
        trust_volume=volume
    )

    plt.suptitle(f"Calibration Scene (frame {frame_idx})")
    plt.show()
