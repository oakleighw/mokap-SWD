import logging
from collections import deque
from typing import Tuple, Dict, Optional, List, Any
import gc
import psutil

import numpy as np
from mokap.geometry.backend import xp, ArrayLike, set_at

from mokap.calibration import bundle_adjustment
from mokap.calibration.common import solve_pnp_robust

from mokap.utils.datatypes import DetectionPayload, DistortionModel
from mokap.geometry import (project_to_cameras, reprojection_errors, project_to_cameras_multi,
                            quaternion_average, average_qtposes, compute_bounds, flip_transform_180,
                            compose_transform_matrix, decompose_transform_matrix,
                            quaternion_from_vector, quaternion_from_matrix, vector_from_quaternion,
                            invert_transform, quaternion_distance)

logger = logging.getLogger(__name__)


class MultiviewCalibrationTool:
    def __init__(
            self,
            nb_cameras:            int,
            images_sizes_hw:       ArrayLike,
            origin_idx:            int,
            K_init:                ArrayLike,
            D_init:                ArrayLike,
            object_points:         ArrayLike,
            min_detections:        int = 100,
            max_detections:        int = 100,
            angular_thresh:        float = 10.0,  # in degrees
            translational_thresh:  float = 10.0,  # in object_points' units
            distortion_model: DistortionModel = 'standard'
        ):

        self.nb_cameras = nb_cameras
        self.origin_idx = origin_idx
        self._distortion_model = distortion_model

        images_sizes_hw = np.asarray(images_sizes_hw)
        if images_sizes_hw.ndim == 2 and images_sizes_hw.shape[0] == self.nb_cameras:
            self._images_sizes_hw = images_sizes_hw[:, :2]

        elif images_sizes_hw.ndim == 1 and 2 <= images_sizes_hw.shape[0] <= 3:
            logger.debug('Only one size passed, assuming identical image size for all cameras.')

            self._images_sizes_hw = np.asarray([images_sizes_hw[:2]] * self.nb_cameras)
        else:
            raise AttributeError("Can't understand image size.")

        self._angular_thresh_rad: float = np.deg2rad(angular_thresh)
        self._translational_thresh: float = translational_thresh

        # Known 3D board model points (N, 3)
        self._object_points = xp.asarray(object_points, dtype=xp.float32)
        self._board_pts_hom = xp.hstack([
            self._object_points, xp.ones((self._object_points.shape[0], 1), dtype=xp.float32)
        ])

        # buffers for incoming frames
        self._detection_buffer = [dict() for _ in range(nb_cameras)]
        self._last_frame = np.full(nb_cameras, -1, dtype=int)

        # State for extrinsics (camera-to-world)
        self._has_extrinsics = np.zeros(nb_cameras, dtype=bool)

        identity = xp.eye(4, dtype=xp.float32)
        self._T_c2w_all = xp.repeat(identity[None, ...], nb_cameras, axis=0)

        # State for board pose (world)
        self._latest_board_pose_w: Optional[xp.ndarray] = None
        self._board_pose_history = deque(maxlen=10)

        # intrinsics state

        init_cam_matrices_np = np.asarray(K_init)
        if init_cam_matrices_np.ndim == 2:
            logger.debug("A single camera matrix was provided. Broadcasting to all cameras.")
            self._K = xp.asarray([init_cam_matrices_np] * self.nb_cameras, dtype=xp.float32)
        else:
            self._K = xp.asarray(init_cam_matrices_np, dtype=xp.float32)

        init_dist_coeffs_np = np.asarray(D_init)
        if init_dist_coeffs_np.ndim == 1:
            logger.debug("A single set of distortion coeffs was provided. Broadcasting to all cameras.")
            self._D = xp.asarray([init_dist_coeffs_np] * self.nb_cameras, dtype=xp.float32)
        else:
            self._D = xp.asarray(init_dist_coeffs_np, dtype=xp.float32)

        if self._K.shape != (self.nb_cameras, 3, 3):
            raise ValueError(
                f"Shape mismatch for init_cam_matrices. Expected ({self.nb_cameras}, 3, 3), got {self._K.shape}")
        if self._D.shape[0] != self.nb_cameras:
            raise ValueError(
                f"Shape mismatch for init_dist_coeffs. Expected ({self.nb_cameras}, D), got {self._D.shape}")

        # triangulation & BA buffers
        self.ba_samples = deque(maxlen=max_detections)
        self._min_detections = min_detections

        # bs results
        self._refined = False
        self._refined_intrinsics = None
        self._refined_extrinsics = None  # T matrices (C, 4, 4)
        self._refined_board_poses = None  # T matrices (P, 4, 4)
        self._points2d = None
        self._visibility_mask = None
        self._volume_of_trust = None

    def _find_stale_frames(self):
        global_min = int(self._last_frame.min())

        pending = set()

        for buf in self._detection_buffer:
            pending.update(buf.keys())

        stale = [f for f in pending if f < global_min]
        return stale

    def _flush_frames(self):
        for f in self._find_stale_frames():
            cams = [c for c in range(self.nb_cameras) if f in self._detection_buffer[c]]
            if len(cams) < 2:
                for c in cams:
                    self._detection_buffer[c].pop(f, None)
                continue

            entries = [(c, *self._detection_buffer[c].pop(f)) for c in cams]
            self._process_frame(entries)

    def _gather_frame_data(self, entries):
        """ Gathers and pads frame data into JAX arrays for vectorized processing """

        C = len(entries)
        N = self._object_points.shape[0]

        cam_indices = np.array([c for c, _, _, _ in entries], dtype=np.int32)

        gt_points_padded_np = np.zeros((C, N, 2), dtype=np.float32)
        visibility_mask_np = np.zeros((C, N), dtype=bool)

        for i, (_, _, points2d, pointsids) in enumerate(entries):
            gt_points_padded_np[i, pointsids, :] = points2d
            visibility_mask_np[i, pointsids] = True

        return xp.asarray(cam_indices), xp.asarray(gt_points_padded_np), xp.asarray(visibility_mask_np)

    def _process_frame(self, entries: List[Tuple[int, Any, Any, Any]]):

        if not any(self._has_extrinsics):
            self._latest_board_pose_w = None
            return

        cam_indices, gt_points_padded, visibility_mask = self._gather_frame_data(entries)

        known_mask = xp.array([self._has_extrinsics[c] for c in cam_indices])
        if not xp.any(known_mask):
            return

        # Initial board pose estimation
        T_b2c_all = xp.stack([entry[1] for entry in entries])
        T_c2w_known = self._T_c2w_all[cam_indices[known_mask]]
        T_b2c_known = T_b2c_all[known_mask]

        # Initial vote for the board's pose based on currently known cameras
        T_b2w_votes = T_c2w_known @ T_b2c_known

        # Temporal disambiguation using a stable ref pose
        if len(self._board_pose_history) > 0:

            history_q = quaternion_from_matrix(xp.stack(list(self._board_pose_history)))

            # We average the rotation (via quaternions)
            q_ref = quaternion_average(history_q)

            # Get the alternative PnP solutions (180-degree flip)
            T_b2c_alt = flip_transform_180(T_b2c_known)

            # Calculate world poses for alternative PnP result
            T_b2w_votes_alt = T_c2w_known @ T_b2c_alt

            # For each vote determine which (original or alternative) is closer to the stable ref
            q_votes = quaternion_from_matrix(T_b2w_votes)
            q_votes_alt = quaternion_from_matrix(T_b2w_votes_alt)

            # Calculate angular distance to the reference
            dist_original = quaternion_distance(q_votes, q_ref)
            dist_alt = quaternion_distance(q_votes_alt, q_ref)

            # Choose the best pose for each camera view
            use_alt_mask = dist_alt < dist_original
            T_b2w_votes = xp.where(use_alt_mask[:, None, None], T_b2w_votes_alt, T_b2w_votes)

            nb_corrected = xp.sum(use_alt_mask)
            if nb_corrected > 0:
                logger.debug(f"[FLIP_CORRECTED] Corrected {nb_corrected} PnP results using stable reference.")

        # Averaging and quality control
        r_stack, t_stack = decompose_transform_matrix(T_b2w_votes)
        q_stack = quaternion_from_vector(r_stack)
        rt_stack = xp.concatenate([q_stack, t_stack], axis=1)

        q_avg, t_avg, success = average_qtposes(
            qt_stack=rt_stack,
            thresh_radians=self._angular_thresh_rad,
            thresh_distance=self._translational_thresh
        )

        if not success:
            logger.debug(
                f"[CONSENSUS_FAIL] Frame rejected. Could not find a consistent board pose among {rt_stack.shape[0]} views.")
            self._latest_board_pose_w = None  # invalidate the single-frame pose
            return

        # Update state with the new good pose
        T_b2w = compose_transform_matrix(vector_from_quaternion(q_avg), t_avg)
        self._latest_board_pose_w = T_b2w
        self._board_pose_history.append(T_b2w)

        # Calculate new camera extrinsics based on the optimised board pose
        T_c2b_all = invert_transform(T_b2c_all)
        T_c2w_new = T_b2w @ T_c2b_all

        world_pts = (T_b2w @ self._board_pts_hom.T).T[:, :3]

        # Get world-to-camera transforms for projection
        T_w2c_new = invert_transform(T_c2w_new)
        K_batch = self._K[cam_indices]
        D_batch = self._D[cam_indices]

        reproj_pts, reproj_mask = project_to_cameras(
            world_pts,
            T_w2c_new,
            K_batch,
            D_batch,
            distortion_model=self._distortion_model
        )

        effective_visibility = visibility_mask * reproj_mask

        errors_dict = reprojection_errors(gt_points_padded, reproj_pts, effective_visibility)
        frame_rms_euclidean = errors_dict['rms_euclidean']

        FRAME_ERROR_THRESHOLD = 5.0
        if frame_rms_euclidean > FRAME_ERROR_THRESHOLD:
            logger.debug(f"[QUALITY_REJECT] Frame rejected. High Euclidean RMS Error: {frame_rms_euclidean:.2f}px")

            # if the frame is bad, we should not have added it to the history. So we dump it. TODO: that's a bit suboptimal but that'll do for now
            if len(self._board_pose_history) > 0 and xp.all(self._board_pose_history[-1] == T_b2w):
                self._board_pose_history.pop()

            self._latest_board_pose_w = None

            return

        logger.debug(f"[ACCEPTED] Frame Euclidean RMS: {frame_rms_euclidean:.2f} px.")

        # Commit the new extrinsics to the main state for all cameras in this frame
        for i, cam_idx in enumerate(cam_indices):
            if cam_idx != self.origin_idx:  # Never update the origin camera
                self._T_c2w_all = set_at(self._T_c2w_all, cam_idx, T_c2w_new[i])

            self._has_extrinsics[cam_idx] = True

        self.ba_samples.append(entries)

    def register(self, cam_idx: int, detection: DetectionPayload):

        if detection.pointsIDs is None or detection.points2D is None:
            return

        if len(detection.pointsIDs) < 4:
            return

        # Reestimate the board-to-camera pose and validate it
        success, T_b2c, errors_dict = solve_pnp_robust(
            points3d=self._object_points[detection.pointsIDs],
            points2d=detection.points2D,
            K=self._K[cam_idx],
            D=self._D[cam_idx]
        )

        # if PnP fails, return
        if not success:
            return

        # From here on T_b2c should be sane

        f = detection.frame
        self._last_frame[cam_idx] = f
        self._detection_buffer[cam_idx][f] = (T_b2c, detection.points2D, detection.pointsIDs)

        # The origin camera's extrinsics are fixed at identity, so its flag is always true
        # This only needs to be set once
        if not self._has_extrinsics[self.origin_idx]:
            self._has_extrinsics[self.origin_idx] = True

        self._flush_frames()

    def refine_all(self) -> bool:
        """
        Performs a global, three-stage bundle adjustment (BA) over all collected samples
        (Sort of graduated non-convexity process)

        - Stage 1: Solves for a stable global geometry with shared intrinsics and no distortion
        - Stage 2: Refines per-camera intrinsics (still no distortion)
        - Stage 3: Performs a full refinement with all parameters (including distortion)
        """

        if not all(self._has_extrinsics):
            logger.error("[BA] Initial extrinsics have not been estimated yet.")
            return False

        P = self.ba_sample_count
        if P < self._min_detections:
            logger.error(f"[BA] Not enough samples for bundle adjustment. Have {P}, need {self._min_detections}.")
            return False

        logger.debug(f"[BA] Starting 3-stage Bundle Adjustment with {P} samples.")

        C = self.nb_cameras
        N = self._object_points.shape[0]

        ba_succeeded = False
        final_results = None

        # Priors weights to prevent the BA from overfittign
        priors_stage1 = {
            'intrinsics': {
                'focal_length': 0.1,  # weak, just to prevent the *average* focal length from drifting into nonsense
                'principal_point': 5.0,  # This can be quite strong, most modern lenses have it very close to the centre
                'distortion': 0.0
            },
            'extrinsics': {
                'rotation': 0.0,
                'translation': 0.0
            }
        }
        priors_stage2 = {
            'intrinsics': {
                'focal_length': 1.0,    # quite strong. Keeps each camera's focal length from deviating from the average found in Stage 1
                'principal_point': 0.1, # weak, but still here to keep the principal point near the image center
                'distortion': 0.5       # medium, keeps the initial distortion terms small and well-behaved
            },
            'extrinsics': {  # extrinsics priors off
                'rotation': 0.0,
                'translation': 0.0
            }
        }
        priors_stage3 = {
            'intrinsics': {
                'focal_length': 1.0,    # still strong. This is critical to avoid overfitting. TODO: Could be stronger maybe?
                'principal_point': 0.1, # same as in stage 2. Modern cameras with modern lenses should be pretty centered...
                'distortion': 0.1       # Relaxed from stage 2. We want to refine these a bit more.
            },
            'extrinsics': { # We assume by then the geometry is pretty good, so we set priors on the extrinsics
                            # This prevents a single camera with poor visibility in some frames from drifting

                # A weight of ~ 700 on radians is comparable to a weight of 0.1 on mm for a target tolerance of 0.5 deg / 1.0 mm
                # This keeps camera poses very stable, allowing only tiny final adjustments
                # TODO: Maybe we want to do this scaling inside the bundle_adjustment module and only expose normalised weights here?
                'rotation': 700,
                'translation': 0.1
            }
        }

        # The try except loop is a little safeguard to avoid filling up the RAM because of the jacobian
        # (it grows quadratically with the nb of samples)
        current_P = self.ba_sample_count
        while current_P >= self._min_detections:
            try:
                logger.info(f"[BA] Attempting Bundle Adjustment with {current_P} samples.")

                current_samples = list(self.ba_samples)[-current_P:]

                pts2d_buf = np.zeros((C, current_P, N, 2), dtype=np.float32)
                vis_buf = np.zeros((C, current_P, N), dtype=bool)

                for p_idx, entries in enumerate(current_samples):
                    for cam_idx, _, pts2D, ids in entries:
                        pts2d_buf[cam_idx, p_idx, ids, :] = pts2D
                        vis_buf[cam_idx, p_idx, ids] = True

                # Initial guess for board poses (from online estimation)
                T_board_w_list = []

                for p_idx, entries in enumerate(current_samples):

                    cam_indices_in_frame = xp.array([c for c, _, _, _ in entries])

                    T_b2c_in_frame = xp.stack([T_b2c for _, T_b2c, _, _ in entries])
                    T_c2w_in_frame = self._T_c2w_all[cam_indices_in_frame]

                    T_b2w_votes = T_c2w_in_frame @ T_b2c_in_frame

                    r_stack, t_stack = decompose_transform_matrix(T_b2w_votes)
                    q_stack = quaternion_from_vector(r_stack)

                    # Simple average for BA initialisation
                    # (Because the spread of this cluster is a direct result of the accumulated errors
                    # during online camera pose estimates - which are unavoidable!!
                    # The hardcore filter used online would likely jusyt eliminate everyone here)
                    q_avg = quaternion_average(q_stack)
                    t_avg = xp.median(t_stack, axis=0)

                    T_board_w_list.append(compose_transform_matrix(vector_from_quaternion(q_avg), t_avg))

                # Prepare initial matrices
                poses_T_initial = xp.stack(T_board_w_list)  # (P, 4, 4)
                cam_T_online = self._T_c2w_all  # (C, 4, 4)

                K_online = self._K
                D_online = self._D

                pts2d_buf = xp.asarray(pts2d_buf)
                vis_buf = xp.asarray(vis_buf)

                self._points2d, self._visibility_mask = pts2d_buf, vis_buf  # store points for this run

                # STAGE 1: Ideal pinhole world (shared intrinsics, no distortion)
                # ---------------------------------------------------------------
                # Here we care only about the overall camera layout and the average 3D structure of the scene
                #
                logger.debug(f"[BA] >>> STAGE 1: Consolidating cameras position with {current_P} frames...")
                success_s1, results_s1 = bundle_adjustment.run_bundle_adjustment(
                    K_initial=K_online,
                    D_initial=D_online,
                    cam_poses_initial=cam_T_online,

                    images_sizes_hw=self._images_sizes_hw,

                    image_points=pts2d_buf,
                    visibility_mask=vis_buf,

                    object_points_initial=self._object_points,
                    object_poses_initial=poses_T_initial,

                    fix_intrinsics=False,
                    fix_extrinsics=False,
                    fix_object_points=True,  # The board's shape is known and rigid
                    fix_poses=False,  # The board's pose is being optimized

                    # Stage 1 specific flags
                    shared_intrinsics=True,     # Forces a single camera model for all views
                    fix_aspect_ratio=True,      # Assume fx = fy
                    distortion_model='none',    # No distortion in stage 1
                    priors=priors_stage1,

                    origin_idx=self.origin_idx,
                    radial_penalty=0.0,  # for fisrst stage we want to consider all points, even at the edge

                    stage=1
                )
                if not success_s1:
                    raise RuntimeError("BA Stage 1 failed.")

                # STAGE 2: Per-camera pinhole world (shared intrinsics, simple distortion)
                # ------------------------------------------------------------------------
                # Here we relax the shared model and start refining the per-camera details, but we use priors
                # to keep them from deviating wildly from the stable average we found in Stage 1
                #
                logger.debug(f"[BA] >>> STAGE 2: Consolidating per-camera intrinsics with {current_P} frames...")

                K_s2_init = results_s1['K_opt']
                D_s2_init = results_s1['D_opt']
                cam_T_s2_init = results_s1['cam_poses_opt']
                poses_T_s2_init = results_s1['object_poses_opt']

                success_s2, results_s2 = bundle_adjustment.run_bundle_adjustment(
                    K_initial=K_s2_init,
                    D_initial=D_s2_init,
                    cam_poses_initial=cam_T_s2_init,

                    images_sizes_hw=self._images_sizes_hw,

                    image_points=pts2d_buf,
                    visibility_mask=vis_buf,

                    object_points_initial=self._object_points,
                    object_poses_initial=poses_T_s2_init,

                    # BA logic control flags
                    fix_intrinsics=False,
                    fix_extrinsics=False,
                    fix_object_points=True,
                    fix_poses=False,

                    # Stage 2 specific flags
                    shared_intrinsics=False,    # We now optimize per-camera intrinsics
                    fix_aspect_ratio=False,     # We relax the aspect ratio constraint
                    distortion_model='simple',  # start optimising distortion

                    priors=priors_stage2,
                    radial_penalty=2.0, # for second stage we want to start penalising points too far from the working volume

                    stage=2
                )
                if not success_s2:
                    raise RuntimeError("BA Stage 2 failed.")

                # STAGE 3: Real world (Full extrinsics + intrinsics refinement with distortion)
                # -----------------------------------------------------------------------------
                # Everything should be close to the correct solution. We enable the most complex distortion models
                # (like full or rational) and let all parameters adjust simultaneously for the final polish
                #
                logger.debug(f"[BA] >>> STAGE 3: Full refinement with {current_P} frames...")

                K_s3_init = results_s2['K_opt']
                D_s3_init = results_s2['D_opt']
                cam_T_s3_init = results_s2['cam_poses_opt']
                poses_T_s3_init = results_s2['object_poses_opt']

                success_s3, final_results_attempt = bundle_adjustment.run_bundle_adjustment(
                    K_initial=K_s3_init,
                    D_initial=D_s3_init,
                    cam_poses_initial=cam_T_s3_init,

                    images_sizes_hw=self._images_sizes_hw,

                    image_points=pts2d_buf,
                    visibility_mask=vis_buf,

                    object_points_initial=self._object_points,
                    object_poses_initial=poses_T_s3_init,

                    # BA logic control flags
                    fix_intrinsics=False,
                    fix_extrinsics=False,
                    fix_object_points=True,
                    fix_poses=False,

                    # Stage 3 specific flags
                    shared_intrinsics=False,
                    fix_aspect_ratio=False,
                    distortion_model=self._distortion_model,    # Use the desired model

                    priors=priors_stage3,  # Priors are mega important at this stage

                    radial_penalty=4.0,  # now we kinda want to ignore the points far from the working volume

                    stage=1
                )
                if not success_s3:
                    raise RuntimeError("BA Stage 3 failed.")

                # If we reach here, all stages were successful
                ba_succeeded = True
                final_results = final_results_attempt

                break

            except MemoryError:
                gc.collect()
                mem = psutil.virtual_memory()
                logger.warning(
                    f"[BA] Memory error encountered with {current_P} samples. "
                    f"RAM usage: {mem.percent}% ({mem.used / 1e9:.2f}/{mem.total / 1e9:.2f} GB). "
                    f"Reducing sample count and retrying."
                )

                # Reduce sample count by 10% for the next attempt
                current_P = int(current_P * 0.9)
                continue

            except RuntimeError as e:
                logger.error(f"[BA] {e}. Could not converge even with {current_P} samples. Aborting.")
                return False

        if ba_succeeded and final_results is not None:
            logger.info(f"Bundle adjustment complete using {current_P} samples. Storing refined parameters.")

            # Store globally optimised results
            self._refined_intrinsics = (final_results['K_opt'], final_results['D_opt'])
            self._refined_extrinsics = final_results['cam_poses_opt']  # (C, 4, 4)
            self._refined_board_poses = final_results['object_poses_opt']  # (P, 4, 4)

            self._refined = True
            self.volume_of_trust()
            self.ba_samples.clear()
            return True
        else:
            logger.error(f"[BA] Failed to complete bundle adjustment. "
                         f"Minimum sample requirement is {self._min_detections}, but failed even after reducing to {current_P}.")
            return False

    @property
    def intrinsics(self) -> Tuple[xp.ndarray, xp.ndarray]:
        return self._K, self._D

    @property
    def extrinsics(self) -> xp.ndarray:
        # Returns (C, 4, 4) matrix
        return self._T_c2w_all

    @property
    def is_estimated(self) -> bool:
        return all(self._has_extrinsics)

    @property
    def current_board_pose(self) -> Optional[xp.ndarray]:
        return self._latest_board_pose_w

    @property
    def refined_intrinsics(self) -> Tuple[xp.ndarray, xp.ndarray]:
        return self._refined_intrinsics

    @property
    def refined_extrinsics(self) -> xp.ndarray:
        return self._refined_extrinsics

    @property
    def refined_board_poses(self) -> xp.ndarray:
        return self._refined_board_poses

    @property
    def image_points(self):
        return self._points2d, self._visibility_mask

    def volume_of_trust(
            self,
            threshold: float = 1.0,
            iqr_factor: float = 1.5
        ) -> Optional[Dict[str, Tuple[float, float]]]:

        if self._refined:
            # calculate the 3D world coordinates of all point instances using the refined poses
            # Board Poses: (P, 4, 4)
            # Board Points: (N, 4) (Homogeneous)
            T_b2w_all_opt = self.refined_board_poses
            world_pts_all_instances = xp.einsum('pij,nj->pni', T_b2w_all_opt, self._board_pts_hom)[:, :, :3]

            # Reprojection and error calculation
            observed_pts2d, visibility_mask = self.image_points

            # Get world-to-camera transforms
            T_c2w = self.refined_extrinsics  # (C, 4, 4)
            T_w2c = invert_transform(T_c2w)

            # Project all 3D points into all cameras
            reprojected_pts, valid_depth_mask = project_to_cameras_multi(
                world_pts_all_instances,
                T_w2c,
                *self._refined_intrinsics,
                distortion_model=self._distortion_model
            )

            effective_visibility = visibility_mask * valid_depth_mask

            errors_dict = reprojection_errors(
                observed_pts2d,
                reprojected_pts,
                effective_visibility,
                per_point_errors=True   # to get the distances for volume of trust calculation
            )
            # Extract per-point Euclidean distances
            points_errors = errors_dict['mre_per_point']

            # And compute the reliable bounding box using the world points and their errors
            volume_of_trust = compute_bounds(
                world_pts_all_instances,
                points_errors,
                error_threshold=threshold,
                iqr_factor=iqr_factor
            )

            # Convert back to floats to save
            volume_of_trust = {k: (float(v[0]), float(v[1])) for k, v in volume_of_trust.items()}

            if volume_of_trust:
                print("--- Volume of Trust ---")
                print(f"X range: {volume_of_trust['x'][0]:.2f} to {volume_of_trust['x'][1]:.2f} mm")
                print(f"Y range: {volume_of_trust['y'][0]:.2f} to {volume_of_trust['y'][1]:.2f} mm")
                print(f"Z range: {volume_of_trust['z'][0]:.2f} to {volume_of_trust['z'][1]:.2f} mm")

                self._volume_of_trust = volume_of_trust

            return self._volume_of_trust

    @property
    def ba_sample_count(self) -> int:
        return len(self.ba_samples)

    @property
    def is_refined(self) -> bool:
        return self._refined