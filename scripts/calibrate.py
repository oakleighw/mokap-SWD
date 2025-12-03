import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import numpy as np
import toml

from mokap.calibration.monocular import MonocularCalibrationTool
from mokap.calibration.multiview import MultiviewCalibrationTool
from mokap.geometry import decompose_transform_matrix
from mokap.utils import fileio
from mokap.utils.datatypes import CharucoBoard, DetectionPayload


# ──────────────────────────────────────────────────── Config ──────────────────────────────────────────────────────────

DEFAULT_BOARD = CharucoBoard(rows=6, cols=5, square_length=1.5, markers_size=4)
DEFAULT_DIST_MODEL = 'simple'  # or even none? Basler lenses are pretty linear

# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────


def get_video_files(folder: Path) -> Tuple[List[Path], List[str], Path]:
    """Scans folder for mp4s and returns paths and camera names."""
    if folder.name != "calibration" and (folder / "calibration").is_dir():
        folder = folder / "calibration"

    video_paths = sorted(folder.glob("*.mp4"))
    if not video_paths:
        raise FileNotFoundError(f"No .mp4 files found in {folder}")

    cam_names = []
    for vp in video_paths:
        parts = vp.stem.split("_")
        if len(parts) >= 2:
            cam_names.append(parts[-2])
        else:
            cam_names.append(vp.stem)

    return video_paths, cam_names, folder


def open_cameras(video_paths: List[Path]) -> Tuple[List[cv2.VideoCapture], np.ndarray]:
    """Opens video captures and returns handles + (Height, Width) array."""
    caps = [cv2.VideoCapture(str(vp)) for vp in video_paths]
    sizes_hw = []
    for vp in video_paths:
        meta = fileio.probe_video(vp)
        sizes_hw.append((meta['height'], meta['width']))
    return caps, np.array(sizes_hw)


def print_report(errors: Dict[str, float], title: str = "Report"):
    """Pretty prints a dictionary of errors."""
    print(f"\n─── {title} ───")
    for name, err in errors.items():
        err_str = f"{err:.4f}" if np.isfinite(err) else "Inf"
        print(f"  {name:<15} : {err_str} px RMSE")
    print("────────────────────────────────")


def print_intrinsics_details(names: List[str], tools: List[MonocularCalibrationTool]):
    """Prints detailed intrinsic parameters."""
    print("\n─── Intrinsics ───")
    for name, tool in zip(names, tools):
        if not tool.has_intrinsics:
            continue

        K, D = tool.intrinsics
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # Flatten and format distortion coeffs
        d_str = ", ".join([f"{x:.4g}" for x in D.flatten()])

        print(f"[{name}]")
        print(f"  Focal:   fx={fx:.2f}, fy={fy:.2f}")
        print(f"  Center:  cx={cx:.2f}, cy={cy:.2f}")
        print(f"  Dist:    [{d_str}]")
    print("───────────────────────────")


def confirm_save(prompt: str = "Save these parameters to disk?") -> bool:
    """Interactively ask user to confirm action."""
    while True:
        choice = input(f"\n{prompt} [y/N]: ").strip().lower()
        if choice in ['y', 'yes']:
            return True
        if choice in ['n', 'no', '']:
            return False

# ─────────────────────────────────────────────────── Intrinsics ───────────────────────────────────────────────────────

def run_intrinsics(folder: Path,
                   step: int = 1,
                   min_coverage: float = 75.0,
                   min_samples: int = 30,
                   stabilize_frames: int = 500) -> bool:
    """
    Runs the monocular intrinsic calibration loop.
    Returns True if calibration finished and was saved/accepted.
    """
    print(f"\n[INTRINSICS] Starting calibration in: {folder}")
    video_paths, cam_names, work_dir = get_video_files(folder)
    caps, sizes_hw = open_cameras(video_paths)
    C = len(caps)

    tools = [
        MonocularCalibrationTool(
            calibration_board=DEFAULT_BOARD,
            imsize_hw=sizes_hw[i],
            distortion_model=DEFAULT_DIST_MODEL)
        for i in range(C)
    ]

    # State tracking
    refining = [False] * C
    best_errors = [np.inf] * C
    frames_no_imp = [0] * C
    frame_idx = 0

    try:
        while True:
            # Read Frames
            frames = []
            for cap in caps:
                ret, fr = cap.read()
                frames.append(fr if ret else None)

            if any(f is None for f in frames):
                print("\n[Video End] Reached end of stream.")
                break

            # Skip redundant frames if requested
            if frame_idx % step != 0:
                frame_idx += 1
                continue

            # Process
            system_stable = True
            any_refining = False

            for i, m in enumerate(tools):
                # Always detect and try to register novel samples
                m.detect(frames[i])
                m.register_sample(min_new_area=0.1)

                # State Machine
                if not refining[i]:
                    system_stable = False  # not stable if still collecting
                    if m.pct_coverage >= min_coverage and m.curr_nb_samples >= min_samples:
                        refining[i] = True
                        print(f"[Cam {cam_names[i]}] Target coverage reached. Starting refinement.")
                else:
                    any_refining = True
                    frames_no_imp[i] += step

                    # Check for improvement every 25 processed frames
                    if (frame_idx // step) % 25 == 0:

                        m.compute_intrinsics(keep_stacks=True, fix_aspect_ratio=True, distortion_model='simple')
                        curr_err = np.nanmean(m.intrinsics_errors)

                        if curr_err < best_errors[i]:
                            frames_no_imp[i] = 0
                            best_errors[i] = curr_err
                        else:
                            # if individual cam hasn't improved in X frames it contributes to stability
                            if frames_no_imp[i] < stabilize_frames:
                                system_stable = False

            # Check stability
            # System is stable if everyone who is refining has stopped improving,
            # and everyone is refining (or we decided to stop waiting for laggards?)
            # if all active cameras are stable, we quit.
            if system_stable and any_refining:
                # don't exit if some cameras are lagging significantly in collection unless video is ending
                print(f"\n[Stable] No improvement for {stabilize_frames} frames.")
                break

            # Status print
            if frame_idx > 0 and (frame_idx // step) % 10 == 0:
                stats = []
                for i in range(C):
                    state = "Refining..." if refining[i] else "Collecting..."
                    err_s = f"{best_errors[i]:.3f}" if best_errors[i] != np.inf else "-"
                    stats.append(f"'{cam_names[i]}': {state} (err: {err_s} px)\n")
                print(f"\rFrame {frame_idx} | " + " | ".join(stats), end="")

            frame_idx += 1

    except KeyboardInterrupt:
        print("\n[Interrupt] User stopped processing.")
    finally:
        for c in caps: c.release()

    # Final computation
    print("\n\nFinalizing Intrinsics")
    results = {}

    for i, (name, m) in enumerate(zip(cam_names, tools)):
        if m.curr_nb_samples > min_samples:
            print(f"Computing final calibration for camera '{name}' ({m.curr_nb_samples} samples)...")
            m.compute_intrinsics(keep_stacks=False, fix_aspect_ratio=False, distortion_model='standard')
            err = np.nanmean(m.intrinsics_errors)
            results[name] = err
        else:
            results[name] = np.inf
            print(f"Skipping {name}: Not enough samples.")

    print_report(results, "Final Intrinsic Errors")
    print_intrinsics_details(cam_names, tools)

    if confirm_save("Overwrite intrinsic files in calibration folder?"):
        for i, (name, m) in enumerate(zip(cam_names, tools)):
            if results[name] != np.inf:
                K, D = m.intrinsics
                fileio.write_intrinsics(work_dir, name, np.asarray(K), np.asarray(D))
        print("Intrinsics saved.")
        return True

    return False

# ─────────────────────────────────────────────────── Extrinsics ───────────────────────────────────────────────────────

def run_extrinsics(folder: Path,
                   step: int = 1,
                   origin_cam: str = "",
                   ba_frames: int = 100) -> bool:
    """
    Runs the multi-view extrinsic calibration loop.
    Requires intrinsics to exist on disk (or be recently written).
    """
    print(f"\n[EXTRINSICS] Starting calibration in: {folder}")
    video_paths, cam_names, work_dir = get_video_files(folder)

    # Identify origin cam
    if not origin_cam:
        origin_cam = cam_names[0]
        print(f"Origin not specified. Defaulting to: {origin_cam}")

    if origin_cam not in cam_names:
        print(f"Error: Origin camera '{origin_cam}' not found in {cam_names}")
        return False

    origin_idx = cam_names.index(origin_cam)

    # Load intrinsics & setup tools
    Ks, Ds = [], []
    try:
        for name in cam_names:
            p = fileio.read_parameters(work_dir, camera_name=name)
            Ks.append(p["camera_matrix"])
            Ds.append(p["dist_coeffs"])
    except FileNotFoundError:
        print("Error: Could not load intrinsics. Run 'intrinsics' mode first.")
        return False

    Ks = np.stack(Ks)
    Ds = np.stack(Ds)

    caps, sizes_hw = open_cameras(video_paths)
    C = len(caps)

    # Tools
    mono_tools = []
    for i in range(C):
        m = MonocularCalibrationTool(
            calibration_board=DEFAULT_BOARD,
            imsize_hw=sizes_hw[i],
            distortion_model=DEFAULT_DIST_MODEL)
        m.set_intrinsics(Ks[i], Ds[i])
        mono_tools.append(m)

    mv_tool = MultiviewCalibrationTool(
        nb_cameras=C,
        images_sizes_hw=sizes_hw[:, :2],
        object_points=DEFAULT_BOARD.object_points,
        K_init=Ks,
        D_init=Ds,
        origin_idx=origin_idx,
        min_detections=ba_frames,
        max_detections=ba_frames * 2,
        distortion_model=DEFAULT_DIST_MODEL
    )

    frame_idx = 0
    target_coverage = 60.0

    try:
        while True:
            # Read
            frames = []
            for cap in caps:
                ret, fr = cap.read()
                frames.append(fr if ret else None)

            if any(f is None for f in frames):
                break  # Video ended

            if frame_idx % step != 0:
                frame_idx += 1
                continue

            # Detect
            detections = [None] * C
            for i, img in enumerate(frames):
                mono_tools[i].detect(img)
                if mono_tools[i].has_detection:
                    detections[i] = mono_tools[i].detection

            # Register decision (coverage hunt + buffer fill)
            should_add = False

            # Are all cameras covered enough?
            all_covered = all(m.pct_coverage >= target_coverage for m in mono_tools)

            if not all_covered:
                # Add if it helps a camera that needs coverage
                for i, m in enumerate(mono_tools):
                    if m.pct_coverage < target_coverage and detections[i]:
                        if m.register_sample(min_new_area=0.1):
                            should_add = True
                            break
            elif mv_tool.ba_sample_count < ba_frames:
                # Coverage done, just filling the BA buffer
                should_add = True

            # Add to MultiView tool
            if should_add:
                # Must have at least 2 views for useful extrinsics
                views = sum(1 for d in detections if d is not None)
                if views >= 2:
                    for i, det in enumerate(detections):
                        if det:
                            payload = DetectionPayload(
                                frame=frame_idx,
                                points2D=det[0],
                                pointsIDs=det[1]
                            )
                            mv_tool.register(i, payload)

            # Check termination
            if all_covered and mv_tool.ba_sample_count >= ba_frames:
                print("\n[Ready] Sufficient coverage and BA samples collected.")
                break

            # Status
            if frame_idx > 0 and (frame_idx // step) % 10 == 0:
                cov_str = ", ".join([f"{m.pct_coverage:.0f}%".rjust(4) for m in mono_tools])
                print(f"\rFrame {frame_idx} | BA Samples: {mv_tool.ba_sample_count} | Coverage: [{cov_str}]", end="")

            frame_idx += 1

    except KeyboardInterrupt:
        print("\n[Interrupt] Stopping collection.")
    finally:
        for c in caps: c.release()

    if mv_tool.ba_sample_count < 10:
        print("\n[Error] Not enough common samples found for Bundle Adjustment.")
        return False

    # Bundle Adjustment
    print("\nRunning Bundle Adjustment (this may take a moment)...")
    success = mv_tool.refine_all()

    if not success:
        print("Bundle Adjustment Failed.")
        return False

    # Finish
    K_opt, D_opt = mv_tool.refined_intrinsics
    T_opt = mv_tool.refined_extrinsics

    # TODO: New io classes will save the matrix directly
    r_opt, t_opt = decompose_transform_matrix(T_opt)

    r_opt_np = np.asarray(r_opt)
    t_opt_np = np.asarray(t_opt)
    K_opt_np = np.asarray(K_opt)
    D_opt_np = np.asarray(D_opt)

    print("\nCalibration Refinement Complete.")

    # Create temporary tools to print details easily
    final_tools = []
    for i in range(C):
        m = MonocularCalibrationTool(DEFAULT_BOARD, imsize_hw=sizes_hw[i], distortion_model=DEFAULT_DIST_MODEL)
        m.set_intrinsics(K_opt_np[i], D_opt_np[i])
        final_tools.append(m)

    print_intrinsics_details(cam_names, final_tools)

    if confirm_save("Save Extrinsics (and refined Intrinsics) to disk?"):
        for i, name in enumerate(cam_names):
            fileio.write_intrinsics(work_dir, name, K_opt_np[i], D_opt_np[i])
            fileio.write_extrinsics(work_dir, name, r_opt_np[i], t_opt_np[i])

        # Save Volume of Trust
        vol = mv_tool.volume_of_trust()
        with open(work_dir / 'volume.toml', 'w') as f:
            toml.dump(vol, f)

        print(f"Saved to {work_dir}")
        return True

    return False


# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Calibration Tool")
    parser.add_argument("folder", type=str, help="Path to folder containing video files")
    parser.add_argument("--mode", choices=["intrinsics", "extrinsics", "both"], default="both",
                        help="Calibration mode.")

    # Tuning params
    parser.add_argument("--step", type=int, default=1,
                        help="Process every Nth frame (speed up processing).")
    parser.add_argument("--origin", type=str, default="",
                        help="Name of camera to be world origin (extrinsics only).")

    # Advanced thresholds
    parser.add_argument("--coverage", type=float, default=75.0, help="Min coverage % (intrinsics)")
    parser.add_argument("--samples", type=int, default=30, help="Min samples (intrinsics)")
    parser.add_argument("--ba_frames", type=int, default=100, help="Min frames for Bundle Adj (extrinsics)")

    args = parser.parse_args()

    target_folder = Path(args.folder)
    if not target_folder.exists():
        print(f"Error: Folder {target_folder} does not exist.")
        sys.exit(1)

    if args.mode in ["intrinsics", "both"]:
        success = run_intrinsics(
            target_folder,
            step=args.step,
            min_coverage=args.coverage,
            min_samples=args.samples
        )
        if not success and args.mode == "both":
            print("Intrinsics failed or were not saved. Aborting Extrinsics.")
            sys.exit(0)

    if args.mode in ["extrinsics", "both"]:
        run_extrinsics(
            target_folder,
            step=args.step,
            origin_cam=args.origin,
            ba_frames=args.ba_frames
        )