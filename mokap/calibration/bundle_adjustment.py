import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix, csr_matrix
from typing import Tuple, Dict, Optional

from mokap.utils.geometry import USE_JAX
if not USE_JAX:
    print('[WARNING] Mokap math backend set to NumPy. Enabling JAX for the Bundle Adjustment module only.')

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from functools import partial
from mokap.utils import CallbackOutputStream
from mokap.utils.datatypes import DistortionModel
# from alive_progress import alive_bar

from mokap.utils.geometry.projective import project_object_views_batched, project_multiple_to_multiple
from mokap.utils.geometry.transforms import invert_rtvecs

DIST_MODEL_MAP = {'none': 0, 'simple': 4, 'standard': 5, 'full': 8, 'rational': 8}


def _get_parameter_spec(
        nb_cams:                  int,
        nb_frames:                int,
        nb_points:                int,
        nb_points_to_optim:       int,
        origin_idx:               int,
        fix_cameras_intrinsics:   bool,
        fix_cameras_extrinsics:   bool,
        fix_object_points:        bool,
        fix_poses:                bool,
        fix_aspect_ratio:         bool,
        time_independent_points:  bool,
        shared_intrinsics:        bool,
        distortion_model:         DistortionModel
) -> Dict:
    """
    Defines the structure of the optimization vector X
    (the size and offset for each block of parameters to optimise)
    """
    spec = {'config': locals()}
    spec['blocks'] = {}
    current_offset = 0

    is_shared = shared_intrinsics and nb_cams > 1
    nb_intr_sets = 1 if is_shared else nb_cams

    # Cameras intrinsics
    if not fix_cameras_intrinsics:
        # Camera matrix params (f, cx, cy) or (fx, fy, cx, cy)
        size_per_set = 3 if fix_aspect_ratio else 4
        size_cam_mat = size_per_set * nb_intr_sets
        spec['blocks']['cam_mat'] = {'offset': current_offset, 'size': size_cam_mat}
        current_offset += size_cam_mat

        # Distortion coefficients
        n_d = DIST_MODEL_MAP[distortion_model]
        spec['config']['n_d'] = n_d
        if n_d > 0:
            size_dist = n_d * nb_intr_sets
            spec['blocks']['distortion'] = {'offset': current_offset, 'size': size_dist}
            current_offset += size_dist

    # Camera extrinsics
    if not fix_cameras_extrinsics:
        # We optimize for all cameras except the origin camera
        nb_optim_cams = nb_cams - 1
        size = 6 * nb_optim_cams  # 6 params (3 rvec, 3 tvec) per camera
        spec['blocks']['extrinsics'] = {'offset': current_offset, 'size': size}
        current_offset += size

    # Poses of the 3D structure (only for rigid objects like calibration boards)
    if not fix_poses and nb_frames > 0:
        size = 6 * nb_frames
        spec['blocks']['poses'] = {'offset': current_offset, 'size': size}
        current_offset += size

    # 3D structure points
    if not fix_object_points and nb_points_to_optim > 0:
        size = 3 * nb_points_to_optim
        spec['blocks']['object_points'] = {'offset': current_offset, 'size': size}
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
    nb_intr_sets = 1 if is_shared else C

    scales = np.ones(spec['total_size'], dtype=np.float64)

    # Cameras intrinsics scales
    if 'cam_mat' in spec['blocks']:
        info = spec['blocks']['cam_mat']
        size_per_set = info['size'] // nb_intr_sets
        for i in range(nb_intr_sets):
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
        nb_optim_cams = C - 1

        # Scale for rotation vectors (radians)
        r_scales = np.full(3 * nb_optim_cams, 1.0)

        # For translation vectors, the std of their initial values is a good heuristic
        origin_idx = cfg.get('origin_idx', 0)
        cam_mask = np.arange(C) != origin_idx
        tvecs_to_optim = np.asarray(initial_params['cam_tvecs'][cam_mask])
        t_std = np.std(tvecs_to_optim, axis=0)
        t_std[t_std < 1e-6] = 1.0
        t_scales = np.tile(t_std, nb_optim_cams)

        scales[info['offset']:info['offset'] + info['size']] = np.concatenate([r_scales, t_scales])

    # Poses scales
    if 'poses' in spec['blocks']:
        info = spec['blocks']['poses']

        # Scale for rotation vectors
        r_scales = np.full(3 * P, 1.0)

        # Scale for translation vectors
        tvecs_to_optim = np.asarray(initial_params['poses_tvecs'])
        t_mean = np.mean(np.linalg.norm(tvecs_to_optim, axis=1))
        t_mean = 1.0 if t_mean < 1e-6 else t_mean
        t_scales = np.full(3 * P, t_mean)

        scales[info['offset']:info['offset'] + info['size']] = np.concatenate([r_scales, t_scales])

    # 3D points scale
    if 'object_points' in spec['blocks']:
        info = spec['blocks']['object_points']

        # mean distance of points to the origin is a good heuristic probably
        if initial_params.get('object_points') is not None:
            pts3d = np.asarray(initial_params['object_points'])
            p_mean = np.mean(np.linalg.norm(pts3d.reshape(-1, 3), axis=1))
            p_mean = 1.0 if p_mean < 1e-6 else p_mean
            scales[info['offset']:info['offset'] + info['size']] = p_mean

    return scales


def _get_bounds(
        spec: Dict,
        images_sizes_wh: ArrayLike
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes lower and upper bounds for the optimization variables based on the spec
    """
    cfg = spec['config']
    C = cfg['nb_cams']
    is_shared = cfg['shared_intrinsics'] and C > 1
    nb_intr_sets = 1 if is_shared else C

    # initialize with -inf, +inf for all parameters
    lower_bounds = np.full(spec['total_size'], -np.inf, dtype=np.float64)
    upper_bounds = np.full(spec['total_size'], np.inf, dtype=np.float64)

    # Set bounds for camera intrinsics
    if not cfg['fix_cameras_intrinsics']:
        for i in range(nb_intr_sets):
            cam_idx = 0 if is_shared else i
            w, h = images_sizes_wh[cam_idx]

            # Focal Length and Principal Point
            if 'cam_mat' in spec['blocks']:
                info = spec['blocks']['cam_mat']
                size_per_set = info['size'] // nb_intr_sets
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

    # Extrinsics, poses and points are left unbounded
    return lower_bounds, upper_bounds


def _pack_params(
        camera_matrices:    jnp.ndarray,
        dist_coeffs:        jnp.ndarray,
        cam_rvecs:          jnp.ndarray,
        cam_tvecs:          jnp.ndarray,
        object_points:      Optional[jnp.ndarray],
        poses_rvecs:        Optional[jnp.ndarray],
        poses_tvecs:        Optional[jnp.ndarray],
        spec:               Dict
) -> Tuple[jnp.ndarray,     Dict[str, jnp.ndarray]]:
    """ Packs parameters into an optimization vector X and a fixed_params dict """

    optim_parts = []
    fixed_params = {}
    cfg = spec['config']
    is_shared = cfg['shared_intrinsics'] and cfg['nb_cams'] > 1

    # Cameras intrinsics
    fixed_params['K_init'] = camera_matrices
    fixed_params['D_init'] = dist_coeffs
    if not cfg['fix_cameras_intrinsics']:
        if cfg['fix_aspect_ratio']:
            f = (camera_matrices[:, 0, 0] + camera_matrices[:, 1, 1]) * 0.5
            fp_block = jnp.column_stack([f, camera_matrices[:, 0, 2], camera_matrices[:, 1, 2]])
        else:
            fp_block = jnp.column_stack([camera_matrices[:, 0, 0], camera_matrices[:, 1, 1], camera_matrices[:, 0, 2],
                                         camera_matrices[:, 1, 2]])
        optim_parts.append(jnp.mean(fp_block, axis=0) if is_shared else fp_block.ravel())

        if 'distortion' in spec['blocks']:
            n_d = cfg['n_d']
            d_block = dist_coeffs[:, :n_d]
            optim_parts.append(jnp.mean(d_block, axis=0) if is_shared else d_block.ravel())
    else:
        fixed_params['K'] = camera_matrices
        fixed_params['D'] = dist_coeffs

    # Cameras extrinsics
    # Store initial extrinsics for priors
    fixed_params['cam_r_init'] = cam_rvecs
    fixed_params['cam_t_init'] = cam_tvecs

    # Always store the full initial arrays in fixed_params so _unpack_params always has a reference for shape and fixed values
    fixed_params['cam_r'] = cam_rvecs
    fixed_params['cam_t'] = cam_tvecs

    if 'extrinsics' in spec['blocks']:
        origin_idx = cfg.get('origin_idx', 0)
        cam_mask = jnp.arange(cfg['nb_cams']) != origin_idx

        optim_parts.append(cam_rvecs[cam_mask].ravel())  # [r1, r2, r3, r4, ...]
        optim_parts.append(cam_tvecs[cam_mask].ravel())  # [t1, t2, t3, t4, ...]
        # TODO: This is different from how poses are packed (they alternate r,t per pose), consider uniformising

        # Also store fixed origin pose separately for convenience in unpacking
        fixed_params['origin_r'] = cam_rvecs[origin_idx]
        fixed_params['origin_t'] = cam_tvecs[origin_idx]

    # Poses
    if 'poses' in spec['blocks']:
        optim_parts.append(poses_rvecs.ravel())
        optim_parts.append(poses_tvecs.ravel())
    elif poses_rvecs is not None and poses_tvecs is not None:
        fixed_params['poses_r'] = poses_rvecs
        fixed_params['poses_t'] = poses_tvecs

    # 3D points
    if 'object_points' in spec['blocks']:
        optim_parts.append(object_points.ravel())
    elif object_points is not None:
        fixed_params['object_points'] = object_points

    x0 = jnp.concatenate(optim_parts) if optim_parts else jnp.array([])
    return x0, fixed_params


def _unpack_params(
        x: jnp.ndarray,
        fixed_params: Dict[str, jnp.ndarray],
        spec: Dict
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray,
            Optional[jnp.ndarray], Optional[jnp.ndarray], Optional[jnp.ndarray]
]:
    """ Reconstructs all parameters from the optimization vector X and fixed_params """

    cfg = spec['config']
    C, P = cfg['nb_cams'], cfg['nb_frames']
    is_shared = cfg['shared_intrinsics'] and C > 1

    # Cameras intrinsics
    K_out = fixed_params.get('K', jnp.zeros((C, 3, 3), dtype=x.dtype))
    D_out = fixed_params.get('D', jnp.zeros((C, 8), dtype=x.dtype))
    if not cfg['fix_cameras_intrinsics']:
        info_mat = spec['blocks']['cam_mat']
        fp_flat = x[info_mat['offset']: info_mat['offset'] + info_mat['size']]

        size_per_set = 3 if cfg['fix_aspect_ratio'] else 4
        fp_block = fp_flat.reshape(-1, size_per_set)

        if is_shared:
            fp_block = jnp.tile(fp_block, (C, 1))

        if cfg['fix_aspect_ratio']:
            K_out = K_out.at[:, 0, 0].set(fp_block[:, 0])  # f
            K_out = K_out.at[:, 1, 1].set(fp_block[:, 0])  # f
            K_out = K_out.at[:, 0, 2].set(fp_block[:, 1])  # cx
            K_out = K_out.at[:, 1, 2].set(fp_block[:, 2])  # cy
        else:
            K_out = K_out.at[:, 0, 0].set(fp_block[:, 0])  # fx
            K_out = K_out.at[:, 1, 1].set(fp_block[:, 1])  # fy
            K_out = K_out.at[:, 0, 2].set(fp_block[:, 2])  # cx
            K_out = K_out.at[:, 1, 2].set(fp_block[:, 3])  # cy
        K_out = K_out.at[:, 2, 2].set(1.0)

        # Unpack distortion
        if 'distortion' in spec['blocks']:
            info_dist = spec['blocks']['distortion']
            n_d = cfg['n_d']
            d_flat = x[info_dist['offset']: info_dist['offset'] + info_dist['size']]
            d_block = d_flat.reshape(-1, n_d)
            if is_shared:
                d_block = jnp.tile(d_block, (C, 1))
            D_out = D_out.at[:, :n_d].set(d_block)

    # Cameras extrinsics
    cam_r_out = fixed_params['cam_r']
    cam_t_out = fixed_params['cam_t']

    if 'extrinsics' in spec['blocks']:
        origin_idx = cfg.get('origin_idx', 0)
        info = spec['blocks']['extrinsics']
        nb_optim_cams = C - 1

        extr_flat = x[info['offset']: info['offset'] + info['size']]
        r_optim = extr_flat[:3 * nb_optim_cams].reshape(nb_optim_cams, 3)
        t_optim = extr_flat[3 * nb_optim_cams:].reshape(nb_optim_cams, 3)

        # Get all camera indices except the origin camera
        optim_cam_indices = jnp.delete(jnp.arange(C), origin_idx)
        cam_r_out = cam_r_out.at[optim_cam_indices].set(r_optim)
        cam_t_out = cam_t_out.at[optim_cam_indices].set(t_optim)

        # The origin camera is still set from fixed_params
        cam_r_out = cam_r_out.at[origin_idx].set(fixed_params['origin_r'])
        cam_t_out = cam_t_out.at[origin_idx].set(fixed_params['origin_t'])

    # Poses
    poses_r_out, poses_t_out = None, None
    if 'poses' in spec['blocks']:
        info = spec['blocks']['poses']
        poses_flat = x[info['offset']: info['offset'] + info['size']]
        poses_r_out = poses_flat[:3 * P].reshape(P, 3)
        poses_t_out = poses_flat[3 * P:].reshape(P, 3)

    elif 'poses_r' in fixed_params:
        poses_r_out = fixed_params['poses_r']
        poses_t_out = fixed_params['poses_t']

    # 3D points
    object_points_out = None
    if 'object_points' in spec['blocks']:
        info = spec['blocks']['object_points']
        points_flat = x[info['offset']: info['offset'] + info['size']]
        object_points_out = points_flat.reshape(-1, 3)
    elif 'object_points' in fixed_params:
        object_points_out = fixed_params['object_points']

    return K_out, D_out, cam_r_out, cam_t_out, object_points_out, poses_r_out, poses_t_out


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
    center = jnp.stack([cx, cy], axis=-1)
    dists = jnp.linalg.norm(points2d - center, axis=-1)
    max_dist = jnp.sqrt(cx[:, 0, 0] ** 2 + cy[:, 0, 0] ** 2)[:, None, None] + 1e-8
    dist_weight = 1.0 / (1.0 + (dists / max_dist) ** distance_falloff_gamma)

    nb_views = jnp.sum(visibility_mask, axis=0)  # (P, N)
    nb_views_weight = nb_views[None, :, :]  # (1, P, N)
    nb_views_weight = nb_views_weight / (1.0 + nb_views_weight)

    # Optional reprojection weight
    if reproj_error is not None:
        reproj_weight = 1.0 / (1.0 + reproj_error)
        reproj_weight = jnp.clip(reproj_weight, 0.1, 1.0)
    else:
        reproj_weight = jnp.ones_like(visibility_mask, dtype=jnp.float32)

    weights = (visibility_mask.astype(jnp.float32) * dist_weight * nb_views_weight * reproj_weight)
    # Scale weights so the median is around 1.0
    median_weight = jnp.median(weights[weights > 0])

    return weights / (median_weight + 1e-8)


def make_jacobian_sparsity(
        spec: Dict,
        use_extrinsics_prior: bool,
        use_intrinsics_prior: bool
) -> csr_matrix:
    """
    Creates the Jacobian sparsity matrix based on the optimization parameter specification
    """

    cfg = spec['config']
    C, P, N = cfg['nb_cams'], cfg['nb_frames'], cfg['nb_points']
    origin_idx = cfg.get('origin_idx', 0)

    nb_residuals = 2 * P * C * N
    nb_params = spec['total_size']
    S = lil_matrix((nb_residuals, nb_params), dtype=bool)

    is_shared = cfg['shared_intrinsics'] and C > 1
    nb_intr_sets = 1 if is_shared else C
    optim_cam_indices = np.delete(np.arange(C), origin_idx)
    cam_idx_to_optim_pos = {cam_idx: pos for pos, cam_idx in enumerate(optim_cam_indices)}

    # Reprojection error dependencies
    for p in range(P):
        for c in range(C):
            for n in range(N):
                row = 2 * (p * C * N + c * N + n)
                intr_set_idx = 0 if is_shared else c

                # Dependency on camera intrinsics
                if 'cam_mat' in spec['blocks']:
                    info = spec['blocks']['cam_mat']
                    size_per_set = info['size'] // nb_intr_sets
                    col = info['offset'] + intr_set_idx * size_per_set
                    S[row:row + 2, col:col + size_per_set] = 1

                if 'distortion' in spec['blocks']:
                    info = spec['blocks']['distortion']
                    size_per_set = info['size'] // nb_intr_sets
                    col = info['offset'] + intr_set_idx * size_per_set
                    S[row:row + 2, col:col + size_per_set] = 1

                # Dependency on camera extrinsics
                if 'extrinsics' in spec['blocks'] and c != origin_idx:
                    optim_pos = cam_idx_to_optim_pos[c]
                    info = spec['blocks']['extrinsics']
                    nb_optim_cams = C - 1
                    r_col = info['offset'] + optim_pos * 3
                    t_col = info['offset'] + (nb_optim_cams * 3) + optim_pos * 3
                    S[row:row + 2, r_col:r_col + 3] = 1
                    S[row:row + 2, t_col:t_col + 3] = 1

                # Dependency on structure poses
                if 'poses' in spec['blocks']:
                    info = spec['blocks']['poses']
                    r_col = info['offset'] + p * 3
                    t_col = info['offset'] + (P * 3) + p * 3
                    S[row:row + 2, r_col:r_col + 3] = 1
                    S[row:row + 2, t_col:t_col + 3] = 1

                # Dependency on 3D points
                if 'object_points' in spec['blocks']:
                    info = spec['blocks']['object_points']

                    if not cfg['fix_poses']:
                        # Rigid object (calibration board) whose poses are optimized: all frames refer to same N points
                        point_idx_in_optim_vector = n
                    else:
                        # This covers both scaffolding and non-rigid objects (animal): ach frame has a unique set of N points
                        point_idx_in_optim_vector = p * N + n

                    col = info['offset'] + point_idx_in_optim_vector * 3
                    S[row:row + 2, col:col + 3] = 1

    # Add sparsity for all priors
    nb_prior_residuals = 0
    if use_extrinsics_prior and 'extrinsics' in spec['blocks']:
        nb_prior_residuals += 6 * (C - 1)  # only for optimizable cameras

    if use_intrinsics_prior:
        if 'cam_mat' in spec['blocks']:
            nb_prior_residuals += spec['blocks']['cam_mat']['size']
        if 'distortion' in spec['blocks']:
            nb_prior_residuals += spec['blocks']['distortion']['size']

    if nb_prior_residuals > 0:
        S.resize(nb_residuals + nb_prior_residuals, nb_params)

    curr_prior_row = nb_residuals

    # Cameras extrinsics priors
    if use_extrinsics_prior and 'extrinsics' in spec['blocks']:
        info = spec['blocks']['extrinsics']
        nb_optim_cams = C - 1

        # Priors are applied to the flattened block of rvecs then tvecs
        r_cols = info['offset']
        t_cols = info['offset'] + 3 * nb_optim_cams

        # Add identity blocks for rotation and translation priors
        S[curr_prior_row:curr_prior_row + 3 * nb_optim_cams, r_cols:r_cols + 3 * nb_optim_cams] = np.eye(
            3 * nb_optim_cams, dtype=bool)
        curr_prior_row += 3 * nb_optim_cams
        S[curr_prior_row:curr_prior_row + 3 * nb_optim_cams, t_cols:t_cols + 3 * nb_optim_cams] = np.eye(
            3 * nb_optim_cams, dtype=bool)
        curr_prior_row += 3 * nb_optim_cams

    # Cameras intrinsics priors
    if use_intrinsics_prior:

        if 'cam_mat' in spec['blocks']:
            info = spec['blocks']['cam_mat']
            S[curr_prior_row:curr_prior_row + info['size'], info['offset']:info['offset'] + info[
                'size']] = np.eye(info['size'], dtype=bool)
            curr_prior_row += info['size']

        if 'distortion' in spec['blocks']:
            info = spec['blocks']['distortion']
            S[curr_prior_row:curr_prior_row + info['size'], info['offset']:info['offset'] + info[
                'size']] = np.eye(info['size'], dtype=bool)
            curr_prior_row += info['size']

    return S.tocsr()


def cost_function(
        params:             jnp.ndarray,  # The 1D optimization vector
        fixed_params:       Dict,
        spec:               Dict,
        image_points:       jnp.ndarray,
        weights:            jnp.ndarray,
        distortion_model:   DistortionModel,
        prior_weight_r:     float,
        prior_weight_t:     float,
        prior_weight_f:     float,
        prior_weight_c:     float,
        prior_weight_d:     float
) -> jnp.ndarray:
    """Cost function calculating reprojection and prior residuals."""

    Ks, Ds, cam_r, cam_t, object_points, poses_r, poses_t = _unpack_params(params, fixed_params, spec)
    all_residuals = []
    r_w2c, t_w2c = invert_rtvecs(cam_r, cam_t)

    cfg = spec['config']
    C, P, N = cfg['nb_cams'], cfg['nb_frames'], cfg['nb_points']

    # Workflow-dependent reprojection
    is_rigid_object = not cfg['fix_poses']

    if is_rigid_object:
        # Workflow: Rigid object (calibration board)
        # Projects a single set of N points for each of the P poses
        # object_points shape: (N, 3), poses_r/t shape: (P, 3)
        reproj, valid_depth = project_object_views_batched(
            object_points, r_w2c, t_w2c, poses_r, poses_t, Ks, Ds, distortion_model
        )
    else:
        # Workflows: Scaffolding or Temporally-consistent non-rigid (animal)
        # Both use the same multiple-to-multiple projection logic (their 3D point structure is the same)
        object_points_per_frame = object_points.reshape(P, N, 3)
        reproj, valid_depth = project_multiple_to_multiple(
            object_points_per_frame, r_w2c, t_w2c, Ks, Ds, distortion_model
        )

    # Residuals calculation
    resid = reproj - image_points
    effective_weights = weights * valid_depth
    weighted_resid = resid * effective_weights[..., None]
    all_residuals.append(weighted_resid.ravel())

    # RMS Error
    total_nb_points = jnp.sum(effective_weights > 0)  # Number of visible observations
    # total_sum_sq_err = jnp.sum(jnp.square(weighted_resid))  # Sum of all (x, y) squared errors
    # rms_error = jnp.sqrt(total_sum_sq_err / jnp.maximum(1, 2 * total_nb_points))
    # jax.debug.print("Mean Reprojection Error (RMS): {x:.3f}px", x=rms_error)

    # Actual pixel error (unweighted)
    unweighted_sq_err = jnp.sum(jnp.square(resid) * (effective_weights[..., None] > 0))
    rms_error_px = jnp.sqrt(unweighted_sq_err / jnp.maximum(1, 2 * total_nb_points))
    jax.debug.print("Mean Reprojection Error (RMS): {x:.3f}px", x=rms_error_px)

    # Prior residuals

    # Cameras extrinsics priors
    if prior_weight_r > 0.0 or prior_weight_t > 0.0:
        origin_idx = spec['config']['origin_idx']

        # Explicit integer indexing (deletion) instead of boolean masking to ensure concrete shapes for JITting
        all_indices = jnp.arange(spec['config']['nb_cams'])
        optim_indices = jnp.delete(all_indices, origin_idx)

        rvec_resid = (cam_r[optim_indices] - fixed_params['cam_r_init'][optim_indices]).ravel() * prior_weight_r
        tvec_resid = (cam_t[optim_indices] - fixed_params['cam_t_init'][optim_indices]).ravel() * prior_weight_t

        all_residuals.extend([rvec_resid, tvec_resid])

    # Cameras intrinsics priors
    if not cfg['fix_cameras_intrinsics']:
        is_shared = cfg['shared_intrinsics'] and cfg['nb_cams'] > 1
        if cfg['shared_intrinsics'] and cfg['nb_cams'] > 1:
            # When shared, we optimized a single set - compare to mean of initial values
            K_init = jnp.mean(fixed_params['K_init'], axis=0, keepdims=True)  # (1, 3, 3)
            D_init = jnp.mean(fixed_params['D_init'], axis=0, keepdims=True)  # (1, n_d)
            # Ks and Ds are already all identical due to shared optimization, just take first
            Ks_opt = Ks[:1]  # (1, 3, 3)
            Ds_opt = Ds[:1]  # (1, n_d)
        else:
            K_init = fixed_params['K_init']
            D_init = fixed_params['D_init']
            Ks_opt = Ks
            Ds_opt = Ds

        if prior_weight_f > 0.0:
            if cfg['fix_aspect_ratio']:
                f_init = (K_init[:, 0, 0] + K_init[:, 1, 1]) * 0.5
                f_opt = Ks_opt[:, 0, 0]
                f_resid = (f_opt - f_init) * prior_weight_f
            else:
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


def _validate_inputs(**kwargs):
    """
    Validates the shapes and consistency of all inputs to run_bundle_adjustment.
    This enforces the API contract and provides clear, early error messages.
    """

    for key, value in kwargs.items():
        locals()[key] = value

    # Basic shape and consistency checks
    if image_points.ndim != 4:
        raise ValueError(
            f"image_points must be a 4D array of shape (C, P, N, 2), but got {image_points.ndim} dimensions.")

    C, P, N, _ = image_points.shape

    if visibility_mask.shape != (C, P, N):
        raise ValueError(
            f"Shape mismatch: visibility_mask should be ({C}, {P}, {N}), but got {visibility_mask.shape}.")

    if camera_matrices_initial.shape != (C, 3, 3):
        raise ValueError(
            f"Shape mismatch: camera_matrices_initial should be ({C}, 3, 3), but got {camera_matrices_initial.shape}.")

    if cam_rvecs_initial.shape != (C, 3):
        raise ValueError(
            f"Shape mismatch: cam_rvecs_initial should be ({C}, 3) to match the number of cameras, but got {cam_rvecs_initial.shape}.")

    if cam_tvecs_initial.shape != (C, 3):
        raise ValueError(
            f"Shape mismatch: cam_tvecs_initial should be ({C}, 3) to match the number of cameras, but got {cam_tvecs_initial.shape}.")

    if distortion_coeffs_initial.ndim != 2 or distortion_coeffs_initial.shape[0] != C:
        raise ValueError(
            f"Shape mismatch: distortion_coeffs_initial should be a 2D array of shape (C, nb_coeffs), i.e., ({C}, k), "
            f"but got shape {distortion_coeffs_initial.shape}."
        )

    if not 0 <= origin_idx < C:
        raise ValueError(f"origin_idx must be between 0 and {C - 1}, but got {origin_idx}.")

    # Workflow-specific checks

    if not fix_poses:
        # Workflow: Rigid object (calibration board)
        if poses_rvecs_initial is None or poses_tvecs_initial is None:
            raise ValueError("For rigid object optimization (fix_poses=False), initial poses must be provided.")
        if poses_rvecs_initial.shape != (P, 3) or poses_tvecs_initial.shape != (P, 3):
            raise ValueError(
                f"Shape mismatch: initial poses must be ({P}, 3), but got {poses_rvecs_initial.shape} and {poses_tvecs_initial.shape}.")
        if object_points_initial is None:
            raise ValueError("For rigid object optimization, object_points_initial must be provided.")
        if object_points_initial.ndim != 2 or object_points_initial.shape[0] != N or object_points_initial.shape[
            1] != 3:
            raise ValueError(
                f"Shape mismatch: For rigid objects, object_points_initial must be of shape (N, 3), i.e. ({N}, 3), but got {object_points_initial.shape}.")

    else:  # fix_poses is True
        # Workflow: Non-rigid (animal) or Scaffolding
        if object_points_initial is None and not fix_object_points:
            raise ValueError(
                "An initial guess for 3D points must be provided when optimizing them (fix_object_points=False).")

        if object_points_initial is not None:
            total_points = P * N
            if object_points_initial.shape != (total_points, 3):
                raise ValueError(
                    f"Shape mismatch: For non-rigid/scaffolding workflows, object_points_initial must be a flattened array "
                    f"of shape (P*N, 3), i.e., ({total_points}, 3), but got {object_points_initial.shape}."
                )

    print("BA input validation successful.")


def run_bundle_adjustment(
        camera_matrices_initial:    jnp.ndarray,
        distortion_coeffs_initial:  jnp.ndarray,
        cam_rvecs_initial:          jnp.ndarray,
        cam_tvecs_initial:          jnp.ndarray,
        images_sizes_wh:            ArrayLike,
        image_points:               jnp.ndarray,
        visibility_mask:            jnp.ndarray,
        object_points_initial:      Optional[jnp.ndarray] = None,
        poses_rvecs_initial:        Optional[jnp.ndarray] = None,
        poses_tvecs_initial:        Optional[jnp.ndarray] = None,
        fix_cameras_intrinsics:     bool = False,
        fix_cameras_extrinsics:     bool = False,
        fix_object_points:          bool = False,
        fix_poses:                  bool = True,
        fix_aspect_ratio:           bool = False,
        time_independent_points:    bool = False,
        origin_idx:                 int = 0,
        priors:                     Optional[Dict] = None,
        radial_penalty:             float = 2.0,
        shared_intrinsics:          bool = False,
        distortion_model:           DistortionModel = 'standard',
        tolerance:                  float = 1e-8,
        max_nfev:                   int = 500
) -> Tuple[bool, Dict]:

    # _validate_inputs(**locals())

    C, P, N, _ = image_points.shape
    images_sizes_wh = np.atleast_2d(images_sizes_wh)

    nb_points_to_optim = 0
    if not fix_object_points and object_points_initial is not None:
        nb_points_to_optim = object_points_initial.shape[0]

    spec = _get_parameter_spec(
        nb_cams=C,
        nb_frames=P,
        nb_points=N,  # N is the number of points *per frame*
        nb_points_to_optim=nb_points_to_optim,  # This is the total number of 3D variables
        origin_idx=origin_idx,
        fix_cameras_intrinsics=fix_cameras_intrinsics,
        fix_cameras_extrinsics=fix_cameras_extrinsics,
        fix_object_points=fix_object_points,
        fix_poses=fix_poses,
        fix_aspect_ratio=fix_aspect_ratio,
        time_independent_points=time_independent_points,
        shared_intrinsics=shared_intrinsics,
        distortion_model=distortion_model
    )

    # Unpack priors dictionary
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
    if not fix_cameras_intrinsics:
        n_d = spec['config']['n_d']
        current_d = distortion_coeffs_initial.shape[1]
        if current_d < n_d:
            padding = ((0, 0), (0, n_d - current_d))
            distortion_coeffs_initial = jnp.pad(distortion_coeffs_initial, padding, mode='constant')

    x0, fixed_params = _pack_params(
        camera_matrices_initial, distortion_coeffs_initial,
        cam_rvecs_initial, cam_tvecs_initial,
        object_points_initial, poses_rvecs_initial, poses_tvecs_initial,
        spec
    )

    # Bounds and scaling
    lb, ub = _get_bounds(spec, images_sizes_wh)
    x0 = np.clip(x0, lb, ub)

    # Generate per-parameter scales for the optimizer
    initial_params_for_scaling = {
        'cam_tvecs': cam_tvecs_initial,
        'poses_tvecs': poses_tvecs_initial,
        'object_points': object_points_initial
    }
    x_scales_np = _get_parameter_scales(spec, initial_params_for_scaling, images_sizes_wh)

    # Setup weights for residual function
    weights = residual_weights(
        points2d=image_points,
        visibility_mask=visibility_mask,
        camera_matrices=camera_matrices_initial,
        distance_falloff_gamma=radial_penalty
    )

    # Create the partial function, baking in all static data
    residuals_fn_partial = partial(
        cost_function,
        fixed_params=fixed_params,
        spec=spec,
        image_points=image_points,
        weights=weights,
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
        scipy_cost_wrapper, np.asarray(x0),
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
    K_opt, D_opt, cam_r_opt, cam_t_opt, object_points_opt, poses_r_opt, poses_t_opt = _unpack_params(
        jnp.asarray(result.x), fixed_params, spec
    )

    ret_vals = {
        'K_opt': K_opt, 'D_opt': D_opt,
        'cam_r_opt': cam_r_opt, 'cam_t_opt': cam_t_opt,
        'object_points_opt': object_points_opt,
        'poses_r_opt': poses_r_opt, 'poses_t_opt': poses_t_opt
    }

    return result.success, ret_vals