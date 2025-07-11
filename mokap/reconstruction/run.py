import pickle
import json
from pathlib import Path
from typing import Optional
from joblib import Parallel, delayed
from collections import defaultdict
from mokap.reconstruction.reconstruction import Reconstructor
from mokap.reconstruction.config import PipelineConfig, ReconstructorConfig
from mokap.reconstruction.anatomy import StatsBootstrapper, AnatomyLearner
from mokap.reconstruction.tracking import SkeletonAssembler, MultiObjectTracker
from mokap.reconstruction.linking import FragmentMerger, TrackletLinker, load_tracklets, \
    combine_chains
from mokap.reconstruction.utils import create_canonical_map
from mokap.utils import fileio


# --- Stage 1: Reconstruction ---
def reconstruct_batch(batch_of_groups, keypoints, camera_parameters, volume_bounds, config: ReconstructorConfig):
    # each worker process must have its own Reconstructor instance
    reconstructor = Reconstructor(
        camera_parameters=camera_parameters,
        volume_bounds=volume_bounds,
        config=config
    )
    soup_batch = {}
    for ftuple, df_frame in batch_of_groups:
        frame_idx = ftuple[0]
        reconstructed_points = reconstructor.reconstruct_frame(
            df_frame=df_frame,
            keypoint_names=keypoints
        )
        soup_batch[frame_idx] = reconstructed_points
    return soup_batch


def run_reconstruction(df,
                       keypoints,
                       cal_data,
                       volume_bounds,
                       config: PipelineConfig,
                       n_jobs: int,
                       batch_size: int,
                       output_file: Path):
    print("\n--- STAGE 1: Reconstructing 3D Point Soup ---")

    grouped_by_frame = df.group_by('frame', maintain_order=True)
    all_grouped_frames = list(grouped_by_frame)

    frame_batches = [
        all_grouped_frames[i: i + batch_size]
        for i in range(0, len(all_grouped_frames), batch_size)
    ]

    print(f"Cooking points soup with {len(frame_batches)} batches across {n_jobs if n_jobs != -1 else 'all'} cores...")

    results_list = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(reconstruct_batch)(
            batch, keypoints, cal_data, volume_bounds, config.reconstruction
        ) for batch in frame_batches
    )

    points_by_frame = {}
    for batch_dict in results_list:
        points_by_frame.update(batch_dict)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(points_by_frame, f)

    print(f"Reconstruction complete. Points soup with {len(points_by_frame)} frames saved to {output_file}")
    return points_by_frame


# --- Stage 2: Tracking ---
def run_tracking(points_soup,
                 bones,
                 symmetry,
                 config: PipelineConfig,
                 stats_prior_file: Optional[Path],
                 stats_output_file: Path,
                 tracklets_output_file: Path):
    print("\n--- STAGE 2: Assembling and Tracking Skeletons ---")

    # Get Anatomical stats
    bootstrapper = StatsBootstrapper(
        output_path=stats_output_file,
        bones_list=bones,
        symmetry_map=symmetry,
        prior_stats_path=stats_prior_file,
        bootstrap_data=points_soup,
        config=config.anatomy
    )
    bone_stats = bootstrapper.get_initial_stats()

    # Initialisation
    anatomy_learner = AnatomyLearner(initial_stats=bone_stats, config=config.anatomy)
    assembler = SkeletonAssembler(bones_list=bones, bone_stats=bone_stats, assembler_config=config.assembler,
                                  tracker_config=config.tracker)
    tracker = MultiObjectTracker(assembler=assembler, config=config.tracker)

    # Run
    frames_indices = sorted(points_soup.keys())
    if not frames_indices:
        print("No frames with points in the soup. Skipping tracking.")
        return {}

    min_frame, max_frame = frames_indices[0], frames_indices[-1]
    tracklets_by_id = defaultdict(list)

    print("Tracking skeletons through time...")
    for frame_idx in range(min_frame, max_frame + 1):
        # Update the assembler's anatomical model with the latest learned stats
        current_stats = anatomy_learner.get_stats()
        assembler.update_bone_stats(current_stats)

        if frame_idx in points_soup:
            active_tracklets = tracker.update(points_soup[frame_idx], frame_idx)
        else:
            active_tracklets = tracker.predict_only(frame_idx)

        for tracklet in active_tracklets:
            if tracklet.last_update_frame == frame_idx:
                anatomy_learner.add_sample(tracklet.skeleton)

            skel_dict = tracklet.skeleton.to_dict()
            skel_dict.update({
                'frame_idx': frame_idx,
                'track_idx': tracklet.track_idx,
                'track_health': tracklet.health,
                'track_anatomical_integrity': tracklet.anatomical_integrity,
                'track_uncertainty_pos': tracklet.uncertainty['position'].tolist(),
                'track_velocity': tracklet.kf.x[3:6].flatten().tolist(),
                'track_predicted_pos': tracklet.predicted_position.tolist(),
                'time_since_update': tracklet.time_since_update
            })
            tracklets_by_id[tracklet.track_idx].append(skel_dict)

    print("Tracking complete.")
    tracklets_output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(tracklets_output_file, 'wb') as f:
        pickle.dump(dict(tracklets_by_id), f)
    print(f"Saved {len(tracklets_by_id)} raw tracklets to '{tracklets_output_file}'")

    # Save the final learned stats
    with open(stats_output_file, 'w') as f:
        json.dump(anatomy_learner.get_stats(), f, indent=2)
    print(f"Saved final learned anatomy stats to '{stats_output_file}'")

    return dict(tracklets_by_id)


# --- Stage 3: Linking & Merging ---
def run_linking(all_tracked_data,
                keypoints,
                bones,
                symmetry,
                config: PipelineConfig,
                stats_file: Path,
                final_output_file: Path):
    print("\n--- STAGE 3: Merging and Linking Tracklets ---")

    if not all_tracked_data:
        print("No tracklets found to link. Exiting.")
        return

    with open(stats_file, 'r') as f:
        bone_stats = json.load(f)
    canonical_map = create_canonical_map(keypoints, symmetry)

    print("\n[3a] Merging overlapping fragments...")
    initial_tracklets = load_tracklets(all_tracked_data, config=config.linker)
    if not initial_tracklets:
        print("No valid tracklets after loading. Cannot proceed.")
        return

    merger = FragmentMerger(initial_tracklets, bone_stats, bones, config=config.merger)
    merged_tracklets = merger.merge_fragments()

    print("\n[3b] Linking temporal gaps...")
    linker = TrackletLinker(merged_tracklets, canonical_map, config=config.linker)
    chains = linker.link_tracklets()

    print("\n[3c] Combining chains into final tracks...")
    final_linked_tracks = combine_chains(chains, merged_tracklets)

    final_output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(final_output_file, 'wb') as f:
        pickle.dump(final_linked_tracks, f)

    print(f"\nPipeline complete! Saved {len(final_linked_tracks)} final linked tracks to '{final_output_file}'")


if __name__ == '__main__':

    folder = Path().home() / 'Desktop' / '3d_ant_data'
    prefix = '240905-1616'
    session = 22

    config = PipelineConfig()

    # Parallel processing settings for stage 1
    N_JOBS = -1
    BATCH_SIZE = 200

    # input and output files
    input_dir = folder / prefix / 'inputs' / 'tracking'
    output_dir = folder / prefix / 'outputs'
    output_dir.mkdir(parents=True, exist_ok=True)

    points_soup_file = output_dir / f'points_soup_session{session}.pkl'
    stats_prior_file = None
    stats_file = output_dir / f'bone_stats_session{session}.json'
    tracklets_file = output_dir / f'tracklets_session{session}.pkl'
    final_tracks_file = output_dir / f'linked_tracks_session{session}.pkl'

    # --- Load General Data ---

    df = fileio.load_session(input_dir, session=session, use_polars=True)
    cal_data = fileio.read_parameters(folder / prefix / 'calibration')
    keypoints, bones, symmetry = fileio.load_skeleton_SLEAP(input_dir, symmetry=True)
    volume_bounds = {'x': (-10.5, 13.0), 'y': (-21.0, 11.0), 'z': (180.0, 201.0)}

    # --- Run everything ---

    # Stage 1
    if not points_soup_file.exists():
        points_soup = run_reconstruction(df,
                                         keypoints,
                                         cal_data,
                                         volume_bounds,
                                         config,
                                         N_JOBS, BATCH_SIZE,
                                         points_soup_file)
    else:
        print(f"Loading existing points soup from {points_soup_file}")
        with open(points_soup_file, 'rb') as f:
            points_soup = pickle.load(f)

    # Stage 2
    if not tracklets_file.exists():
        tracked_data = run_tracking(points_soup, bones, symmetry, config, stats_prior_file, stats_file, tracklets_file)
    else:
        print(f"Loading existing tracklets from {tracklets_file}")
        with open(tracklets_file, 'rb') as f:
            tracked_data = pickle.load(f)

    # Stage 3
    run_linking(tracked_data, keypoints, bones, symmetry, config, stats_file, final_tracks_file)