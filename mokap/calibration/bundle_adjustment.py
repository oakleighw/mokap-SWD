import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix, csr_matrix
from typing import Tuple, Dict, Optional
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from functools import partial
from mokap.utils import CallbackOutputStream
from mokap.utils.datatypes import DistortionModel
from alive_progress import alive_bar

from mokap.utils.geometry.projective import project_object_views_batched, project_to_multiple_cameras, \
    project_multiple_to_multiple
from mokap.utils.geometry.transforms import invert_rtvecs

DIST_MODEL_MAP = {'none': 0, 'simple': 4, 'standard': 5, 'full': 8, 'rational': 8}

# TODO: Kinda want to test Deepmind's Optax solvers here instead of scipy...


def _get_parameter_spec(
        nb_cams:            int,
        nb_frames:          int,
        origin_idx:         int,
        fix_camera_matrix:  bool,
        fix_distortion:     bool,
        fix_extrinsics:     bool,
        fix_board_poses:    bool,
        fix_aspect_ratio:   bool,
        fix_scene:          bool,
        nb_scene_points:    int,
        shared_intrinsics:  bool,
        distortion_model:   DistortionModel
) -> Dict:
    """
    Defines the structure of the optimization vector X
    (the size and offset for each block of parameters to optimise)
    """
    spec = {'config': locals()}
    spec['blocks'] = {}
    current_offset = 0

    is_shared = shared_intrinsics and nb_cams > 1
    num_intr_sets = 1 if is_shared else nb_cams

    # Focal length and principal point
    if not fix_camera_matrix:
        size_per_set = 3 if fix_aspect_ratio else 4
        size = size_per_set * num_intr_sets
        spec['blocks']['cam_mat'] = {'offset': current_offset, 'size': size}
        current_offset += size

    # Distortion coefficients
    n_d = DIST_MODEL_MAP[distortion_model]
    spec['config']['n_d'] = n_d
    if not fix_distortion and n_d > 0:
        size = n_d * num_intr_sets
        spec['blocks']['distortion'] = {'offset': current_offset, 'size': size}
        current_offset += size

    # Camera extrinsics
    if not fix_extrinsics:
        # We optimize for all cameras except the origin camera
        num_optim_cams = nb_cams - 1
        size = 6 * num_optim_cams  # 6 params (3 rvec, 3 tvec) per camera
        spec['blocks']['extrinsics'] = {'offset': current_offset, 'size': size}
        current_offset += size

    # Calibration object poses
    if not fix_board_poses and nb_frames > 0:
        size = 6 * nb_frames
        spec['blocks']['board_poses'] = {'offset': current_offset, 'size': size}
        current_offset += size

    # Unknown 3D scene points (can be per-frame)
    if not fix_scene and nb_scene_points > 0:
        size = 3 * nb_scene_points
        spec['blocks']['scene_points'] = {'offset': current_offset, 'size': size}
        current_offset += size

    spec['total_size'] = current_offset
    return spec

def _get_parameter_scales(
        spec: Dict,
        initial_params: Dict,
        images_sizes_wh: ArrayLike
) -> np.ndarray:
    """ Computes characteristic scales for each optimization variable """

    cfg = spec['config']
    C = cfg['nb_cams']
    P = cfg['nb_frames']
    is_shared = cfg['shared_intrinsics'] and C > 1
    num_intr_sets = 1 if is_shared else C

    scales = np.ones(spec['total_size'], dtype=np.float64)

    # Cameras intrinsics scales
    if 'cam_mat' in spec['blocks']:
        info = spec['blocks']['cam_mat']
        size_per_set = info['size'] // num_intr_sets
        for i in range(num_intr_sets):
            cam_idx = 0 if is_shared else i
            w, h = images_sizes_wh[cam_idx]
            offset = info['offset'] + i * size_per_set

            if cfg['fix_aspect_ratio']:
                # params are [f, cx, cy]
                scales[offset:offset + 3] = [1000.0, w, h]
            else:
                # params are [fx, fy, cx, cy]
                scales[offset:offset + 4] = [1000.0, 1000.0, w, h]

    if 'distortion' in spec['blocks']:
        info = spec['blocks']['distortion']
        # For distortion params a scale of 1.0 is a good default
        scales[info['offset']:info['offset'] + info['size']] = 1.0

    # Cameras extrinsics scales
    if 'extrinsics' in spec['blocks']:
        info = spec['blocks']['extrinsics']
        num_optim_cams = C - 1

        # Scale for rotation vectors (radians)
        r_scales = np.full(3 * num_optim_cams, 1.0)

        # For translation vectors, the std of their initial values is a good heuristic
        origin_idx = cfg.get('origin_idx', 0)
        cam_mask = np.arange(C) != origin_idx
        tvecs_to_optim = np.asarray(initial_params['cam_tvecs'][cam_mask])
        t_std = np.std(tvecs_to_optim, axis=0)
        t_std[t_std < 1e-6] = 1.0
        t_scales = np.tile(t_std, num_optim_cams)

        scales[info['offset']:info['offset'] + info['size']] = np.concatenate([r_scales, t_scales])

    # Calibration object scales
    if 'board_poses' in spec['blocks']:
        info = spec['blocks']['board_poses']

        # Scale for rotation vectors
        r_scales = np.full(3 * P, 1.0)

        # Scale for translation vectors
        tvecs_to_optim = np.asarray(initial_params['board_tvecs'])
        t_mean = np.mean(np.linalg.norm(tvecs_to_optim, axis=1))
        t_mean = 1.0 if t_mean < 1e-6 else t_mean
        t_scales = np.full(3 * P, t_mean)

        scales[info['offset']:info['offset'] + info['size']] = np.concatenate([r_scales, t_scales])

    # Unknown scene scale
    if 'scene_points' in spec['blocks']:
        info = spec['blocks']['scene_points']

        # mean distance of points to the origin is a good heuristic probably
        if initial_params.get('scene_points') is not None:
            pts3d = np.asarray(initial_params['scene_points'])
            p_mean = np.mean(np.linalg.norm(pts3d.reshape(-1, 3), axis=1))
            p_mean = 1.0 if p_mean < 1e-6 else p_mean
            scales[info['offset']:info['offset'] + info['size']] = p_mean

    return scales


def _get_bounds(
        spec: Dict,
        images_sizes_wh: ArrayLike
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Computes lower and upper bounds for the optimization variables based on the spec
    """
    cfg = spec['config']
    C = cfg['nb_cams']
    is_shared = cfg['shared_intrinsics'] and C > 1
    num_intr_sets = 1 if is_shared else C

    # initialize with -inf, +inf for all parameters
    lower_bounds = np.full(spec['total_size'], -np.inf, dtype=np.float64)
    upper_bounds = np.full(spec['total_size'], np.inf, dtype=np.float64)

    # Set bounds for camera intrinsics
    if 'cam_mat' in spec['blocks'] or 'distortion' in spec['blocks']:
        for i in range(num_intr_sets):
            cam_idx = 0 if is_shared else i
            w, h = images_sizes_wh[cam_idx]

            # Focal Length and Principal Point
            if 'cam_mat' in spec['blocks']:
                info = spec['blocks']['cam_mat']
                size_per_set = info['size'] // num_intr_sets
                offset = info['offset'] + i * size_per_set

                f_lo, f_hi = 100.0, 100000.0

                cx_lo, cx_hi = 0.0, w
                cy_lo, cy_hi = 0.0, h

                if cfg['fix_aspect_ratio']:
                    # params are [f, cx, cy]
                    lower_bounds[offset:offset + 3] = [f_lo, cx_lo, cy_lo]
                    upper_bounds[offset:offset + 3] = [f_hi, cx_hi, cy_hi]
                else:
                    # params are [fx, fy, cx, cy]
                    lower_bounds[offset:offset + 4] = [f_lo, f_lo, cx_lo, cy_lo]
                    upper_bounds[offset:offset + 4] = [f_hi, f_hi, cx_hi, cy_hi]

            # Distortion coefficients
            if 'distortion' in spec['blocks']:
                info = spec['blocks']['distortion']
                n_d = cfg['n_d']
                offset = info['offset'] + i * n_d

                # Define bounds for all 8 potential coefficients
                # k_lo, k_hi = -1.5, 1.5
                # p_lo, p_hi = -0.5, 0.5
                # k_higher_order_lo, k_higher_order_hi = -0.5, 0.5  # Tighter bounds for higher order
                # TODO: Bounds dict passed from the outside with lens types presets!!!!!
                k_lo, k_hi = -0.1, 0.1
                p_lo, p_hi = -0.005, 0.005
                k_higher_order_lo, k_higher_order_hi = -0.05, 0.05

                dist_bounds_map = [
                    (k_lo, k_hi), (k_lo, k_hi),  # k1, k2
                    (p_lo, p_hi), (p_lo, p_hi),  # p1, p2
                    (k_lo, k_hi),                # k3
                    (k_higher_order_lo, k_higher_order_hi),  # k4
                    (k_higher_order_lo, k_higher_order_hi),  # k5
                    (k_higher_order_lo, k_higher_order_hi)   # k6
                ]

                lb_dist = [b[0] for b in dist_bounds_map[:n_d]]
                ub_dist = [b[1] for b in dist_bounds_map[:n_d]]

                lower_bounds[offset:offset + n_d] = lb_dist
                upper_bounds[offset:offset + n_d] = ub_dist

    # Extrinsics, board poses and scene points are left unbounded
    return jnp.array(lower_bounds), jnp.array(upper_bounds)

def _pack_params(
        camera_matrices:  jnp.ndarray,
        dist_coeffs:      jnp.ndarray,
        cam_rvecs:        jnp.ndarray,
        cam_tvecs:        jnp.ndarray,
        board_rvecs:      Optional[jnp.ndarray],
        board_tvecs:      Optional[jnp.ndarray],
        scene_points3d:   Optional[jnp.ndarray],
        spec:             Dict
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """ Packs parameters into an optimization vector X and a fixed_params dict """

    optim_parts = []
    fixed_params = {}
    cfg = spec['config']
    is_shared = cfg['shared_intrinsics'] and cfg['nb_cams'] > 1

    # Always store initial intrinsics for priors
    fixed_params['K_init'] = camera_matrices
    fixed_params['D_init'] = dist_coeffs

    # Cameras intrinsics
    # Camera matrix  packing
    if 'cam_mat' in spec['blocks']:
        if cfg['fix_aspect_ratio']:
            f = (camera_matrices[:, 0, 0] + camera_matrices[:, 1, 1]) * 0.5
            fp_block = jnp.column_stack([f, camera_matrices[:, 0, 2], camera_matrices[:, 1, 2]])
        else:
            fp_block = jnp.column_stack([camera_matrices[:, 0, 0], camera_matrices[:, 1, 1], camera_matrices[:, 0, 2],
                                         camera_matrices[:, 1, 2]])
        optim_parts.append(jnp.mean(fp_block, axis=0) if is_shared else fp_block.ravel())
    else:
        fixed_params['K'] = camera_matrices

    # distortion packing
    if 'distortion' in spec['blocks']:
        n_d = cfg['n_d']
        d_block = dist_coeffs[:, :n_d]
        optim_parts.append(jnp.mean(d_block, axis=0) if is_shared else d_block.ravel())
    else:
        fixed_params['D'] = dist_coeffs

    # Cameras extrinsics
    # Store initial extrinsics for priors
    fixed_params['cam_r_init'] = cam_rvecs
    fixed_params['cam_t_init'] = cam_tvecs

    # Always store the full initial arrays in fixed_params so _unpack_params always has a reference for shape and fixed values
    fixed_params['cam_r'] = cam_rvecs
    fixed_params['cam_t'] = cam_tvecs

    if 'extrinsics' in spec['blocks']:
        # If we are optimizing, add relevant parts to optim_parts list
        origin_idx = spec['config'].get('origin_idx', 0)
        cam_mask = jnp.arange(cfg['nb_cams']) != origin_idx

        optim_parts.append(cam_rvecs[cam_mask].ravel())
        optim_parts.append(cam_tvecs[cam_mask].ravel())

        # Also store fixed origin pose separately for convenience in unpacking
        fixed_params['origin_r'] = cam_rvecs[origin_idx]
        fixed_params['origin_t'] = cam_tvecs[origin_idx]
    # if fix_extrinsics=True the full arrays remain in fixed_params, all good

    # Calibration object poses
    if 'board_poses' in spec['blocks']:
        optim_parts.append(board_rvecs.ravel())
        optim_parts.append(board_tvecs.ravel())
    elif board_rvecs is not None and board_tvecs is not None:
        fixed_params['board_r'] = board_rvecs
        fixed_params['board_t'] = board_tvecs

    # Unknown 3D scene points
    if 'scene_points' in spec['blocks']:
        optim_parts.append(scene_points3d.ravel())
    elif scene_points3d is not None:
        fixed_params['scene_points'] = scene_points3d

    x0 = jnp.concatenate(optim_parts) if optim_parts else jnp.array([])
    return x0, fixed_params


def _unpack_params(
        x: jnp.ndarray,
        fixed_params: Dict[str, jnp.ndarray],
        spec: Dict
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]]:
    """ Reconstructs all parameters from the optimization vector X and fixed_params """

    cfg = spec['config']
    C, P = cfg['nb_cams'], cfg['nb_frames']
    is_shared = cfg['shared_intrinsics'] and C > 1

    # Cameras intrinsics
    K_out = fixed_params.get('K', jnp.zeros((C, 3, 3), dtype=x.dtype))

    if 'cam_mat' in spec['blocks']:
        info = spec['blocks']['cam_mat']
        fp_flat = x[info['offset']: info['offset'] + info['size']]

        size_per_set = 3 if cfg['fix_aspect_ratio'] else 4
        fp_block = fp_flat.reshape(-1, size_per_set)
        if is_shared:
            fp_block = jnp.tile(fp_block, (C, 1))

        if cfg['fix_aspect_ratio']:
            K_out = K_out.at[:, 0, 0].set(fp_block[:, 0])
            K_out = K_out.at[:, 1, 1].set(fp_block[:, 0])
            K_out = K_out.at[:, 0, 2].set(fp_block[:, 1])
            K_out = K_out.at[:, 1, 2].set(fp_block[:, 2])
        else:
            K_out = K_out.at[:, 0, 0].set(fp_block[:, 0])
            K_out = K_out.at[:, 1, 1].set(fp_block[:, 1])
            K_out = K_out.at[:, 0, 2].set(fp_block[:, 2])
            K_out = K_out.at[:, 1, 2].set(fp_block[:, 3])
        K_out = K_out.at[:, 2, 2].set(1.0)

    if 'distortion' in spec['blocks']:
        info = spec['blocks']['distortion']
        n_d = cfg['n_d']
        d_flat = x[info['offset']: info['offset'] + info['size']]
        d_block = d_flat.reshape(-1, n_d)
        if is_shared:
            d_block = jnp.tile(d_block, (C, 1))

        D_out = jnp.zeros((C, 8), dtype=x.dtype)
        D_out = D_out.at[:, :n_d].set(d_block)
    else:
        D_out = fixed_params.get('D', jnp.zeros((C, 8), dtype=x.dtype))

    # Cameras extrinsics
    cam_r_out = fixed_params['cam_r']
    cam_t_out = fixed_params['cam_t']

    if 'extrinsics' in spec['blocks']:
        origin_idx = cfg.get('origin_idx', 0)
        info = spec['blocks']['extrinsics']
        num_optim_cams = C - 1

        extr_flat = x[info['offset']: info['offset'] + info['size']]
        r_optim = extr_flat[:3 * num_optim_cams].reshape(num_optim_cams, 3)
        t_optim = extr_flat[3 * num_optim_cams:].reshape(num_optim_cams, 3)

        # Get all camera indices except the origin camera
        optim_cam_indices = jnp.delete(jnp.arange(C), origin_idx)
        cam_r_out = cam_r_out.at[optim_cam_indices].set(r_optim)
        cam_t_out = cam_t_out.at[optim_cam_indices].set(t_optim)

        # The origin camera is still set from fixed_params
        cam_r_out = cam_r_out.at[origin_idx].set(fixed_params['origin_r'])
        cam_t_out = cam_t_out.at[origin_idx].set(fixed_params['origin_t'])

    # Calibration object poses
    if 'board_poses' in spec['blocks']:
        info = spec['blocks']['board_poses']
        board_flat = x[info['offset']: info['offset'] + info['size']]
        board_r_out = board_flat[:3 * P].reshape(P, 3)
        board_t_out = board_flat[3 * P:].reshape(P, 3)
    else:
        board_r_out = fixed_params.get('board_r', jnp.empty((0, 3)))
        board_t_out = fixed_params.get('board_t', jnp.empty((0, 3)))

    # Unknown 3D scene points
    scene_points_out = None
    if 'scene_points' in spec['blocks']:
        info = spec['blocks']['scene_points']
        scene_flat = x[info['offset']: info['offset'] + info['size']]
        scene_points_out = scene_flat.reshape(-1, 3)
    elif 'scene_points' in fixed_params:
        scene_points_out = fixed_params['scene_points']

    return K_out, D_out, cam_r_out, cam_t_out, board_r_out, board_t_out, scene_points_out

def residual_weights(
        points2d:                jnp.ndarray,  # (C, P, N, 2)
        visibility_mask:         jnp.ndarray,  # (C, P, N)
        camera_matrices:         jnp.ndarray,  # (C, 3, 3)
        reproj_error:            Optional[jnp.ndarray] = None,  # (C, P, N) or None
        distance_falloff_gamma:  float = 2.0
) -> jnp.ndarray:
    """
    Compute per-observation weights for BA residuals based on
        - Visibility
        - Distance from image center
        - Number of views per point
        - Reprojection error (optional)
    """

    C, P, N, _ = points2d.shape

    # Distance to center
    cx = camera_matrices[:, 0, 2][:, None, None]  # (C, 1, 1)
    cy = camera_matrices[:, 1, 2][:, None, None]
    center = jnp.stack([cx, cy], axis=-1)  # (C, 1, 1, 2)

    dists = jnp.linalg.norm(points2d - center, axis=-1)  # (C, P, N)
    max_dist = jnp.sqrt(cx[:, 0, 0] ** 2 + cy[:, 0, 0] ** 2)[:, None, None] + 1e-8  # (C,)

    dist_weight = 1.0 / (1.0 + (dists / max_dist) ** distance_falloff_gamma)  # (C, P, N)

    # Weighting with the number of cameras seeing each point
    nb_views = jnp.sum(visibility_mask, axis=0)  # (P, N)
    nb_views_weight = jnp.tile(nb_views[None, :, :], (C, 1, 1))  # (C, P, N)
    nb_views_weight = nb_views_weight / (1.0 + nb_views_weight)

    # Optional reprojection weight
    if reproj_error is not None:
        reproj_weight = 1.0 / (1.0 + reproj_error)
        reproj_weight = jnp.clip(reproj_weight, 0.1, 1.0)
    else:
        reproj_weight = jnp.ones_like(visibility_mask, dtype=jnp.float32)

    # combine
    weights = (
            visibility_mask.astype(np.float32) *
            dist_weight *
            nb_views_weight *
            reproj_weight
    )
    # norm
    weights /= (jnp.max(weights) + 1e-8)

    return weights


def make_jacobian_sparsity(
        spec: Dict,
        use_extrinsics_prior: bool,
        use_intrinsics_prior: bool
) -> csr_matrix:
    """
    Creates the Jacobian sparsity matrix based on the optimization parameter specification
    """

    cfg = spec['config']
    C, P = cfg['nb_cams'], cfg['nb_frames']
    N_board = cfg.get('nb_board_points', 0)
    N_scene = cfg.get('nb_scene_points_per_frame', 0)
    origin_idx = cfg.get('origin_idx', 0)

    # Total number of residuals includes both board and per-frame scene points
    num_residuals = 2 * P * C * (N_board + N_scene)
    num_params = spec['total_size']
    S = lil_matrix((num_residuals, num_params), dtype=bool)

    is_shared = cfg['shared_intrinsics'] and C > 1
    num_intr_sets = 1 if is_shared else C
    optim_cam_indices = np.delete(np.arange(C), origin_idx)
    cam_idx_to_optim_pos = {cam_idx: pos for pos, cam_idx in enumerate(optim_cam_indices)}

    # Reprojection error dependencies
    for p in range(P):
        for c in range(C):
            intr_set_idx = 0 if is_shared else c

            # Dependencies for board points
            for n_board in range(N_board):
                # Row index for the current observation (camera c, frame p, board point n)
                row = 2 * (p * C * N_board + c * N_board + n_board)

                # Each observation (p, c, n) depends on:
                # - intrinsics of camera c
                if 'cam_mat' in spec['blocks']:
                    info = spec['blocks']['cam_mat']
                    size_per_set = info['size'] // num_intr_sets
                    col = info['offset'] + intr_set_idx * size_per_set
                    S[row:row + 2, col:col + size_per_set] = 1

                if 'distortion' in spec['blocks']:
                    info = spec['blocks']['distortion']
                    size_per_set = info['size'] // num_intr_sets
                    col = info['offset'] + intr_set_idx * size_per_set
                    S[row:row + 2, col:col + size_per_set] = 1

                # - extrinsics of camera c
                if 'extrinsics' in spec['blocks'] and c != origin_idx:
                    optim_pos = cam_idx_to_optim_pos[c]
                    info = spec['blocks']['extrinsics']
                    num_optim_cams = C - 1
                    r_col = info['offset'] + optim_pos * 3
                    t_col = info['offset'] + (num_optim_cams * 3) + optim_pos * 3
                    S[row:row + 2, r_col:r_col + 3] = 1
                    S[row:row + 2, t_col:t_col + 3] = 1

                # - pose of board p
                if 'board_poses' in spec['blocks']:
                    info = spec['blocks']['board_poses']
                    r_col = info['offset'] + p * 3
                    t_col = info['offset'] + (P * 3) + p * 3
                    S[row:row + 2, r_col:r_col + 3] = 1
                    S[row:row + 2, t_col:t_col + 3] = 1

            # Dependencies for unknown per-frame scene points
            row_offset_scene = 2 * P * C * N_board

            for n_scene in range(N_scene):
                # Row index for obs (camera c, frame p, scene point n)
                row = row_offset_scene + 2 * (p * C * N_scene + c * N_scene + n_scene)

                # Each observation depends on:
                # - intrinsics of camera c
                if 'cam_mat' in spec['blocks']:
                    info = spec['blocks']['cam_mat']
                    size_per_set = info['size'] // num_intr_sets
                    col = info['offset'] + intr_set_idx * size_per_set
                    S[row:row + 2, col:col + size_per_set] = 1

                if 'distortion' in spec['blocks']:
                    info = spec['blocks']['distortion']
                    size_per_set = info['size'] // num_intr_sets
                    col = info['offset'] + intr_set_idx * size_per_set
                    S[row:row + 2, col:col + size_per_set] = 1

                # - extrinsics of camera c
                if 'extrinsics' in spec['blocks'] and c != origin_idx:
                    optim_pos = cam_idx_to_optim_pos[c]
                    info = spec['blocks']['extrinsics']
                    num_optim_cams = C - 1
                    r_col = info['offset'] + optim_pos * 3
                    t_col = info['offset'] + (num_optim_cams * 3) + optim_pos * 3
                    S[row:row + 2, r_col:r_col + 3] = 1
                    S[row:row + 2, t_col:t_col + 3] = 1

                # 3D coordinates of the n-th scene point for the p-th frame
                if 'scene_points' in spec['blocks']:
                    info = spec['blocks']['scene_points']
                    point_overall_index = p * N_scene + n_scene
                    col = info['offset'] + point_overall_index * 3
                    S[row:row + 2, col:col + 3] = 1

    # Add sparsity for all priors
    num_prior_residuals = 0
    if use_extrinsics_prior and 'extrinsics' in spec['blocks']:
        num_prior_residuals += 6 * (C - 1)  # only for optimizable cameras

    if use_intrinsics_prior:
        if 'cam_mat' in spec['blocks']:
            num_prior_residuals += spec['blocks']['cam_mat']['size']
        if 'distortion' in spec['blocks']:
            num_prior_residuals += spec['blocks']['distortion']['size']

    # if num_prior_residuals > 0:
    #     S_new = lil_matrix((num_residuals + num_prior_residuals, num_params), dtype=bool)
    #     S_new[:num_residuals, :] = S
    #     S = S_new

    if num_prior_residuals > 0:
        S.resize(num_residuals + num_prior_residuals, num_params)

    current_prior_row = num_residuals

    # Cameras extrinsics priors
    if use_extrinsics_prior and 'extrinsics' in spec['blocks']:
        info = spec['blocks']['extrinsics']
        num_optim_cams = C - 1

        # Priors are applied to the flattened block of rvecs then tvecs
        r_cols_start = info['offset']
        t_cols_start = info['offset'] + 3 * num_optim_cams

        # Add identity blocks for rotation and translation priors
        S[
            current_prior_row: current_prior_row + 3 * num_optim_cams, r_cols_start: r_cols_start + 3 * num_optim_cams] = np.eye(
            3 * num_optim_cams, dtype=bool)
        current_prior_row += 3 * num_optim_cams
        S[
            current_prior_row: current_prior_row + 3 * num_optim_cams, t_cols_start: t_cols_start + 3 * num_optim_cams] = np.eye(
            3 * num_optim_cams, dtype=bool)
        current_prior_row += 3 * num_optim_cams

    # Cameras intrinsics priors
    if use_intrinsics_prior:

        if 'cam_mat' in spec['blocks']:
            info = spec['blocks']['cam_mat']
            S[current_prior_row:current_prior_row + info['size'], info['offset']:info['offset'] + info[
                'size']] = np.eye(info['size'], dtype=bool)
            current_prior_row += info['size']

        if 'distortion' in spec['blocks']:
            info = spec['blocks']['distortion']
            S[current_prior_row:current_prior_row + info['size'], info['offset']:info['offset'] + info[
                'size']] = np.eye(info['size'], dtype=bool)
            current_prior_row += info['size']

    return S.tocsr()


def cost_function(
        params:                 jnp.ndarray,  # The 1D optimization vector
        fixed_params:           Dict,
        spec:                   Dict,
        board_points2d:         Optional[jnp.ndarray],
        board_points3d:         Optional[jnp.ndarray],
        board_points_weights:   Optional[jnp.ndarray],
        scene_points2d:         Optional[jnp.ndarray], # (C, P, N_static, 2)
        scene_points_weights:   Optional[jnp.ndarray], # (C, P, N_static)
        distortion_model:       DistortionModel,
        prior_weight_r:         float,
        prior_weight_t:         float,
        prior_weight_f:         float,
        prior_weight_c:         float,
        prior_weight_d:         float
) -> jnp.ndarray:
    Ks, Ds, cam_r, cam_t, board_r, board_t, scene_points = _unpack_params(params, fixed_params, spec)
    all_residuals, all_sq_errors, all_weighted_points = [], [], []
    r_w2c, t_w2c = invert_rtvecs(cam_r, cam_t)

    # Reprojection residuals for moving board
    if board_points3d is not None and board_points2d is not None and board_points_weights is not None:
        reproj, valid_depth = project_object_views_batched(
            board_points3d, r_w2c, t_w2c, board_r, board_t, Ks, Ds, distortion_model)
        reproj = jnp.nan_to_num(reproj)
        resid = reproj - board_points2d

        # Combine the pre-computed weights with the dynamic depth-validity weight
        effective_weights = board_points_weights * valid_depth

        # Apply the combined weights to the residual
        weighted_resid = resid * effective_weights[..., None]

        all_residuals.append(weighted_resid.ravel())
        all_sq_errors.append(jnp.sum(jnp.square(weighted_resid)))
        all_weighted_points.append(jnp.sum(effective_weights > 0))

        # Reprojection residuals for scene points (mode-dependent logic)
    if scene_points is not None and scene_points2d is not None and scene_points_weights is not None:
        is_sparse_mode = spec['config']['is_sparse_mode']

        if is_sparse_mode:
            # Independent Points / Scaffolding mode
            reproj_all, valid_depth_flat = project_to_multiple_cameras(
                scene_points, r_w2c, t_w2c, Ks, Ds, distortion_model
            )  # (C, N_total, 2) and (C, N_total)

            reproj_expanded = jnp.expand_dims(reproj_all, axis=1)  # (C, 1, N_total, 2)
            valid_depth = jnp.expand_dims(valid_depth_flat, axis=1)  # (C, 1, N_total)

            # Broadcasting handles the sparse subtraction
            effective_weights = scene_points_weights * valid_depth
            resid = jnp.where(effective_weights[..., None] > 0, jnp.nan_to_num(reproj_expanded) - scene_points2d, 0.0)

        else:
            # Temporally-consistent / Pose refinement mode
            P = scene_points2d.shape[1]
            N_per_frame = scene_points2d.shape[2]

            # Reshape the flattened 3D points back to be per-frame (P, N, 3)
            per_frame_points3d = scene_points.reshape((P, N_per_frame, 3))

            reproj_all, valid_depth = project_multiple_to_multiple(
                per_frame_points3d, r_w2c, t_w2c, Ks, Ds, distortion_model
            )  # (C, P, N_per_frame, 2) and (C, P, N_per_frame)

            effective_weights = scene_points_weights * valid_depth
            resid = jnp.where(effective_weights[..., None] > 0, jnp.nan_to_num(reproj_all) - scene_points2d, 0.0)

        # Common logic to append results
        weighted_resid = resid * effective_weights[..., None]
        all_residuals.append(weighted_resid.ravel())
        all_sq_errors.append(jnp.sum(jnp.square(weighted_resid)))
        all_weighted_points.append(jnp.sum(effective_weights > 0))

    # RMS Error Reporting
    total_sum_sq_err = jnp.sum(jnp.array(all_sq_errors))
    total_num_points = jnp.sum(jnp.array(all_weighted_points))

    rms_error = jnp.sqrt(total_sum_sq_err / jnp.maximum(1, 2 * total_num_points))
    jax.debug.print("Mean Reprojection Error (RMS): {x:.3f}px", x=rms_error)

    # Prior residuals

    # Cameras extrinsics priors
    if prior_weight_r > 0.0 or prior_weight_t > 0.0:

        # Only penalize deviation for cameras that are being optimized
        origin_idx = spec['config']['origin_idx']
        cam_mask = jnp.arange(spec['config']['nb_cams']) != origin_idx

        # Calculate difference with init parameters for all cameras
        # (jnp.where to zero-out the residual for the fixed origin camera)
        rvec_diff = jnp.where(cam_mask[:, None], cam_r - fixed_params['cam_r_init'], 0.0)
        tvec_diff = jnp.where(cam_mask[:, None], cam_t - fixed_params['cam_t_init'], 0.0)

        # final residual has a static shape, and the zeroed-out rows have no effect on the optimization
        rvec_resid = rvec_diff.ravel() * prior_weight_r
        tvec_resid = tvec_diff.ravel() * prior_weight_t

        all_residuals.extend([rvec_resid, tvec_resid])

    # Cameras intrinsics priors
    is_shared = spec['config']['shared_intrinsics'] and spec['config']['nb_cams'] > 1

    # Get initial values (mean if shared)
    K_init = jnp.mean(fixed_params['K_init'], 0, keepdims=True) if is_shared else fixed_params['K_init']
    D_init = jnp.mean(fixed_params['D_init'], 0, keepdims=True) if is_shared else fixed_params['D_init']

    # Get optimized values (broadcast if shared)
    Ks_opt = jnp.mean(Ks, 0, keepdims=True) if is_shared else Ks
    Ds_opt = jnp.mean(Ds, 0, keepdims=True) if is_shared else Ds

    if prior_weight_f > 0.0:
        f_init_x = K_init[:, 0, 0]
        f_init_y = K_init[:, 1, 1]
        f_opt_x = Ks_opt[:, 0, 0]
        f_opt_y = Ks_opt[:, 1, 1]
        f_resid = jnp.concatenate([(f_opt_x - f_init_x), (f_opt_y - f_init_y)]) * prior_weight_f
        all_residuals.append(f_resid)

    if prior_weight_c > 0.0:
        pp_init = K_init[:, :2, 2]
        pp_opt = Ks_opt[:, :2, 2]
        pp_resid = (pp_opt - pp_init).ravel() * prior_weight_c
        all_residuals.append(pp_resid)

    if prior_weight_d > 0.0 and 'distortion' in spec['blocks']:
        n_d = spec['config']['n_d']
        dist_init = D_init[:, :n_d]
        dist_opt = Ds_opt[:, :n_d]
        dist_resid = (dist_opt - dist_init).ravel() * prior_weight_d
        all_residuals.append(dist_resid)

    return jnp.concatenate(all_residuals)


def run_bundle_adjustment(
        camera_matrices:        jnp.ndarray,
        distortion_coeffs:      jnp.ndarray,
        cam_rvecs:              jnp.ndarray,
        cam_tvecs:              jnp.ndarray,
        images_sizes_wh:        ArrayLike,
        board_rvecs:            Optional[jnp.ndarray] = None,
        board_tvecs:            Optional[jnp.ndarray] = None,
        board_points2d:         Optional[jnp.ndarray] = None,
        board_visibility_mask:  Optional[jnp.ndarray] = None,
        board_points3d:         Optional[jnp.ndarray] = None,
        scene_points3d_initial: Optional[jnp.ndarray] = None,
        scene_points2d:         Optional[jnp.ndarray] = None,
        scene_visibility_mask:  Optional[jnp.ndarray] = None,
        fix_scene:              bool = True,
        origin_idx:             int = 0,
        priors:                 Optional[Dict] = None,
        radial_penalty:         float = 2.0,
        fix_camera_matrix:      bool = False,
        fix_distortion:         bool = False,
        fix_extrinsics:         bool = False,
        fix_board_poses:        bool = False,
        fix_aspect_ratio:       bool = False,
        shared_intrinsics:      bool = False,
        distortion_model:       DistortionModel = 'standard',
        max_frames:             Optional[int] = None,
        tolerance:              float = 1e-8,
        max_nfev:               int = 500
) -> Tuple[bool, Dict]:

    # Determine problem dimensions
    C = camera_matrices.shape[0]
    P, N_board, N_per_frame = 0, 0, 0
    is_sparse_mode = False

    # Get dimensions from board data if available
    if board_visibility_mask is not None:
        _, P_board, N_board = board_visibility_mask.shape
        P = max(P, P_board)

    # Determine number of scene points
    total_scene_points = scene_points3d_initial.shape[0] if scene_points3d_initial is not None else 0

    if scene_visibility_mask is not None:
        _, P_scene, N_mask_dim = scene_visibility_mask.shape
        P = max(P, P_scene)

        # Sparse mode: The last dimension of the visibility mask is the total number of points
        # Dense mode: The last dimension is the number of points per frame
        if N_mask_dim == total_scene_points and P > 0:
            is_sparse_mode = True
            N_per_frame = total_scene_points // P
            if total_scene_points % P != 0:
                raise ValueError("In sparse mode, total points must be divisible by num frames.")
        else:
            is_sparse_mode = False
            N_per_frame = N_mask_dim

    if max_frames is not None and P > 0:
        P = max_frames

    images_sizes_wh = np.atleast_2d(images_sizes_wh)

    spec = _get_parameter_spec(
        nb_cams=C,
        nb_frames=P,
        origin_idx=origin_idx,
        fix_camera_matrix=fix_camera_matrix,
        fix_distortion=fix_distortion,
        fix_extrinsics=fix_extrinsics,
        fix_board_poses=fix_board_poses,
        fix_scene=fix_scene,
        nb_scene_points=total_scene_points,
        fix_aspect_ratio=fix_aspect_ratio,
        shared_intrinsics=shared_intrinsics,
        distortion_model=distortion_model
    )

    # Add run mode and nb of points to spec for jacobian sparsity calculation
    spec['config']['is_sparse_mode'] = is_sparse_mode
    spec['config']['nb_board_points'] = N_board
    spec['config']['nb_scene_points_per_frame'] = N_per_frame

    # Unpack priors dictionary into floats for jax
    # TODO: maybe a dual API with simple and advanced control for the priors would be cool?
    priors = priors if priors is not None else {}
    intr_priors = priors.get('intrinsics', {})
    extr_priors = priors.get('extrinsics', {})

    prior_weight_f = float(intr_priors.get('focal_length', 0.0))
    prior_weight_c = float(intr_priors.get('principal_point', 0.0))
    prior_weight_d = float(intr_priors.get('distortion', 0.0))
    prior_weight_r = float(extr_priors.get('rotation', 0.0))
    prior_weight_t = float(extr_priors.get('translation', 0.0))

    use_extrinsics_prior = prior_weight_r > 0.0 or prior_weight_t > 0.0
    use_intrinsics_prior = prior_weight_f > 0.0 or prior_weight_c > 0.0 or prior_weight_d > 0.0

    # Prepare and pad distortion coefficients if necessary
    if 'distortion' in spec['blocks']:
        n_d = spec['config']['n_d']
        current_d = distortion_coeffs.shape[1]
        if current_d < n_d:
            padding = ((0, 0), (0, n_d - current_d))
            distortion_coeffs = jnp.pad(distortion_coeffs, padding, mode='constant')

    x0, fixed_params = _pack_params(
        camera_matrices,
        distortion_coeffs,
        cam_rvecs,
        cam_tvecs,
        board_rvecs[:P] if board_rvecs is not None else None,
        board_tvecs[:P] if board_tvecs is not None else None,
        scene_points3d_initial,
        spec
    )

    # Bounds and scaling
    lb, ub = _get_bounds(spec, images_sizes_wh)
    x0 = jnp.clip(x0, lb, ub)

    # Generate per-parameter scales for the optimizer
    initial_params_for_scaling = {
        'cam_tvecs': cam_tvecs,
        'board_tvecs': board_tvecs[:P] if board_tvecs is not None else None,
        'scene_points': scene_points3d_initial
    }
    x_scales_np = _get_parameter_scales(spec, initial_params_for_scaling, images_sizes_wh)

    # Setup weights for residual function
    board_weights, scene_weights = None, None

    if board_points2d is not None and board_visibility_mask is not None:
        board_weights = residual_weights(points2d=board_points2d[:, :P],
                                         visibility_mask=board_visibility_mask[:, :P],
                                         camera_matrices=camera_matrices,
                                         reproj_error=None,
                                         distance_falloff_gamma=radial_penalty)     # (C, P, N)

    if scene_points2d is not None and scene_visibility_mask is not None:
        scene_weights = residual_weights(points2d=scene_points2d[:, :P],
                                             visibility_mask=scene_visibility_mask[:, :P],
                                             camera_matrices=camera_matrices,
                                             reproj_error=None,
                                             distance_falloff_gamma=radial_penalty)    # (C, P, N)

    # Create the partial function, baking in all static data
    residuals_fn_partial = partial(
        cost_function,
        fixed_params=fixed_params,
        spec=spec,

        board_points2d=board_points2d[:, :P] if board_points2d is not None else None,
        board_points3d=board_points3d,
        board_points_weights=board_weights,
        scene_points2d=scene_points2d[:, :P] if scene_points2d is not None else None,
        scene_points_weights=scene_weights,
        distortion_model=distortion_model,
        prior_weight_r=prior_weight_r,
        prior_weight_t=prior_weight_t,
        prior_weight_f=prior_weight_f,
        prior_weight_c=prior_weight_c,
        prior_weight_d=prior_weight_d
    )

    jitted_cost_func = jax.jit(residuals_fn_partial)
    jitted_jac_func = jax.jit(jax.jacfwd(residuals_fn_partial))

    # Prepare for scipy, create wrappers
    def scipy_cost_wrapper(p):
        return np.asarray(jitted_cost_func(jnp.asarray(p))).copy()

    def scipy_jac_wrapper(p):
        return np.asarray(jitted_jac_func(jnp.asarray(p))).copy()

    # Pass prior flags to sparsity function
    jac_sparsity = make_jacobian_sparsity(spec, use_extrinsics_prior, use_intrinsics_prior)

    # Call the scipy solver
    # with alive_bar(title='Bundle adjustment...', length=20, force_tty=True) as bar:
    #     with CallbackOutputStream(bar, keep_stdout=False):
    result = least_squares(
        scipy_cost_wrapper,
        np.asarray(x0),
        jac=scipy_jac_wrapper,
        jac_sparsity=jac_sparsity,
        bounds=(np.asarray(lb), np.asarray(ub)),
        x_scale=x_scales_np,
        method='trf',
        loss='cauchy',
        f_scale=2.5,
        ftol=tolerance,
        xtol=tolerance,
        gtol=tolerance,
        max_nfev=max_nfev,
        verbose=2
    )

    # Unpack results
    K_opt, D_opt, cam_r_opt, cam_t_opt, board_r_opt, board_t_opt, struct_pts_opt = _unpack_params(
        jnp.asarray(result.x), fixed_params, spec
    )


    if struct_pts_opt is not None:
        struct_pts_opt = struct_pts_opt.reshape((total_scene_points, 3))

    ret_vals = {
        'K_opt': K_opt, 'D_opt': D_opt,
        'cam_r_opt': cam_r_opt, 'cam_t_opt': cam_t_opt,
        'board_r_opt': board_r_opt, 'board_t_opt': board_t_opt,
        'scene_points_opt': struct_pts_opt
    }

    return result.success, ret_vals