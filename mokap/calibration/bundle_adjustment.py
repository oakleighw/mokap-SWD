import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix, csr_matrix
from typing import Tuple, Dict, Optional
from functools import partial

from mokap.utils.datatypes import DistortionModel
from mokap.geometry.backend import USE_JAX
import mokap.geometry as geom

from alive_progress import alive_bar

if not USE_JAX:
    try:
        import jax
        import jax.numpy as jnp
        from jax import jit
        from jax.typing import ArrayLike
    except ImportError:
        raise ImportError(f'Bundle Adjustment requires JAX to be installed.')
    print(f'[INFO] Mokap math backend is set to NumPy. JAX will be enabled for Bundle Adjustment only.')


# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Pure JAX geometry kernel
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# We define specific JAX geometric primitives here to avoid dependency on the global backend state of 'mokap.geometry'

_eps = 1e-5
_tiny = 1e-12


def _jax_rotation_matrix(rvec):
    """Pure JAX Rodrigues formula."""
    theta_sq = jnp.sum(rvec ** 2, axis=-1, keepdims=True)
    theta = jnp.sqrt(theta_sq)

    kx, ky, kz = rvec[..., 0], rvec[..., 1], rvec[..., 2]
    zeros = jnp.zeros_like(kx)

    K = jnp.stack([
        jnp.stack([zeros, -kz, ky], axis=-1),
        jnp.stack([kz, zeros, -kx], axis=-1),
        jnp.stack([-ky, kx, zeros], axis=-1),
    ], axis=-2)

    I = jnp.eye(3)

    use_taylor = theta_sq < 1e-6
    # Taylor expansion for small angles to maintain gradient flow
    a_taylor = 1.0 - theta_sq / 6.0
    b_taylor = 0.5 - theta_sq / 24.0

    theta_safe = jnp.maximum(theta, 1e-12)
    a_normal = jnp.sin(theta_safe) / theta_safe
    b_normal = (1.0 - jnp.cos(theta_safe)) / (theta_safe ** 2)

    a = jnp.where(use_taylor, a_taylor, a_normal)
    b = jnp.where(use_taylor, b_taylor, b_normal)

    return I + a[..., None] * K + b[..., None] * (K @ K)


def _jax_compose_transform(rvec, tvec):
    """Pure JAX T mat (..., 4, 4) composition."""
    R = _jax_rotation_matrix(rvec)
    t = tvec[..., None]
    top = jnp.concatenate([R, t], axis=-1)
    bottom = jnp.broadcast_to(jnp.array([0., 0., 0., 1.]), top.shape[:-2] + (1, 4))
    return jnp.concatenate([top, bottom], axis=-2)


def _jax_distort(points_norm, D, model_idx):
    """Pure JAX application of distortion to normalised coordinates."""
    x, y = points_norm[..., 0], points_norm[..., 1]
    r2 = x * x + y * y
    r = jnp.sqrt(r2 + _tiny)

    k1, k2 = D[..., 0], D[..., 1]
    p1, p2 = D[..., 2], D[..., 3]

    out_x, out_y = x, y

    # Models
    if model_idx == 4:  # Simple / Fisheye-polynomial
        radial = 1.0 + k1 * r2 + k2 * r2 * r2
        out_x = x * radial
        out_y = y * radial

    elif model_idx == 5:  # Standard (k1, k2, p1, p2, k3)
        k3 = D[..., 4]
        radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        out_x = x * radial
        out_y = y * radial

    elif model_idx == 8:  # Rational
        k3, k4, k5, k6 = D[..., 4], D[..., 5], D[..., 6], D[..., 7]
        num = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        den = 1.0 + k4 * r2 + k5 * r2 * r2 + k6 * r2 * r2 * r2
        radial = num / (den + _tiny)
        out_x = x * radial
        out_y = y * radial

    elif model_idx == 99:  # Fisheye (Equidistant)
        theta = jnp.arctan(r)
        theta2 = theta * theta
        k1, k2, k3, k4 = D[..., 0], D[..., 1], D[..., 2], D[..., 3]
        scale = theta * (1.0 + k1 * theta2 + k2 * theta2 * theta2 + k3 * theta2 ** 3 + k4 * theta2 ** 4)
        factor = jnp.where(r < _tiny, 1.0, scale / r)
        return points_norm * factor[..., None]

    # Tangential (for standard/rational)
    if model_idx in [5, 8]:
        xy = x * y
        x2 = x * x
        y2 = y * y
        dx = 2.0 * p1 * xy + p2 * (r2 + 2.0 * x2)
        dy = 2.0 * p2 * xy + p1 * (r2 + 2.0 * y2)
        out_x = out_x + dx
        out_y = out_y + dy

    return jnp.stack([out_x, out_y], axis=-1)


def _jax_project(points3d, T_w2c, K, D, dist_model_idx):
    """
    Pure JAX projection.
    Returns:
        uv: Projected 2D coordinates
        z_depth: The actual Z depth in camera frame (for barrier loss)
    """
    R = T_w2c[..., :3, :3]
    t = T_w2c[..., :3, 3]

    Xc = jnp.einsum('...ij,...nj->...ni', R, points3d) + t[..., None, :]

    z = Xc[..., 2]

    # Important: We avoid division by zero, but we do *not* clamp strictly to 1.0
    # We allow the math to proceed (even if negative) so gradients exist,
    # and rely on the cost function to heavily penalize negative Z values
    z_safe = jnp.where(jnp.abs(z) < _tiny, _tiny, z)

    xy_norm = Xc[..., :2] / z_safe[..., None]

    if dist_model_idx > 0:
        D_b = D[..., None, :]
        xy_dist = _jax_distort(xy_norm, D_b, dist_model_idx)
    else:
        xy_dist = xy_norm

    fx, fy = K[..., 0, 0], K[..., 1, 1]
    cx, cy = K[..., 0, 2], K[..., 1, 2]

    fx, fy = fx[..., None], fy[..., None]
    cx, cy = cx[..., None], cy[..., None]

    u = xy_dist[..., 0] * fx + cx
    v = xy_dist[..., 1] * fy + cy

    # returns the actual Z for the cost function to inspect
    return jnp.stack([u, v], axis=-1), z


def _jax_residual_weights(points2d, visibility_mask, K, gamma=2.0):
    """
    Pure JAX computation of weights for BA residuals, based on points visibility mask,
    radial distance from principal point (downweight edges) and view count weighting
    to trust multi-view features more.
    """
    cx = K[:, 0, 2][:, None, None]
    cy = K[:, 1, 2][:, None, None]

    # Radial weighting
    dist_sq = jnp.square(points2d[..., 0] - cx) + jnp.square(points2d[..., 1] - cy)
    dist = jnp.sqrt(dist_sq)
    max_dist = jnp.sqrt(jnp.square(cx) + jnp.square(cy)) + 1e-8
    radial_weight = 1.0 / (1.0 + (dist / max_dist) ** gamma)

    # View count weighting
    # visibility_mask is (C, P, N) or (C, N), sum across cameras
    nb_views = jnp.sum(visibility_mask, axis=0)

    # Broadcast back to (C, P, N)
    # (nb_views / (1 + nb_views)) scales from ~0.5 (2 views) to ~1.0 (all views)
    if visibility_mask.ndim == 3: # need to handle the shapes depending on if P exists
        view_weight = nb_views[None, :, :]
    else:
        view_weight = nb_views[None, :]

    # Sigmoidal scaling for view count
    view_weight_factor = view_weight / (1.0 + view_weight)

    # Combine
    weights = visibility_mask.astype(jnp.float32) * radial_weight * view_weight_factor

    # Normalise median to ~1.0 to keep gradients in a good range
    w_masked = jnp.where(weights > _tiny, weights, jnp.nan)
    median_w = jnp.nanmedian(w_masked)
    safe_median = jnp.where(jnp.isnan(median_w), 1.0, median_w)
    safe_median = jnp.where(safe_median > _tiny, safe_median, 1.0)

    return weights / safe_median


# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Helpers and specs config (NumPy)
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

DIST_MODEL_MAP = {'none': 0, 'simple': 4, 'standard': 5, 'rational': 8, 'thinprism': 12, 'tilted': 14, 'fisheye': 99}


def _get_parameter_spec(nb_cams, nb_frames, nb_points, nb_points_to_optim,
                        origin_idx, fix_intrinsics, fix_extrinsics,
                        fix_object_points, fix_poses, fix_aspect_ratio,
                        shared_intrinsics, distortion_model):
    """Pure NumPy. Defines the structure of the optimization vector X."""

    spec = {'config': locals()}
    spec['blocks'] = {}
    current_offset = 0

    is_shared = shared_intrinsics and nb_cams > 1
    nb_intr_sets = 1 if is_shared else nb_cams

    if not fix_intrinsics:
        size_per_set = 3 if fix_aspect_ratio else 4
        size_cam_mat = size_per_set * nb_intr_sets
        spec['blocks']['K'] = {'offset': current_offset, 'size': size_cam_mat}
        current_offset += size_cam_mat

        n_d = DIST_MODEL_MAP.get(distortion_model, 0)
        n_d_size = 4 if distortion_model == 'fisheye' else n_d

        spec['config']['n_d'] = n_d
        spec['config']['n_d_size'] = n_d_size
        spec['config']['dist_model_idx'] = n_d

        if n_d > 0:
            size_dist = n_d_size * nb_intr_sets
            spec['blocks']['D'] = {'offset': current_offset, 'size': size_dist}
            current_offset += size_dist
    else:
        # Fallback for when intrinsics are fixed
        spec['config']['n_d'] = DIST_MODEL_MAP.get(distortion_model, 0)
        spec['config']['n_d_size'] = 4 if distortion_model == 'fisheye' else spec['config']['n_d']
        spec['config']['dist_model_idx'] = spec['config']['n_d']

    if not fix_extrinsics:
        nb_optim_cams = nb_cams - 1
        spec['blocks']['extrinsics'] = {'offset': current_offset, 'size': 6 * nb_optim_cams}
        current_offset += 6 * nb_optim_cams

    if not fix_poses and nb_frames > 0:
        spec['blocks']['poses'] = {'offset': current_offset, 'size': 6 * nb_frames}
        current_offset += 6 * nb_frames

    if not fix_object_points and nb_points_to_optim > 0:
        spec['blocks']['object_points'] = {'offset': current_offset, 'size': 3 * nb_points_to_optim}
        current_offset += 3 * nb_points_to_optim

    spec['total_size'] = current_offset
    return spec


def _get_parameter_scales(spec, initial_params, images_sizes_hw):
    """Pure NumPy. Computes characteristic scales for optimiser."""

    cfg = spec['config']
    C = cfg['nb_cams']
    P = cfg['nb_frames']

    is_shared = cfg['shared_intrinsics'] and C > 1
    nb_intr_sets = 1 if is_shared else C
    scales = np.ones(spec['total_size'], dtype=np.float64)

    if 'K' in spec['blocks']:
        info = spec['blocks']['K']
        size_per_set = info['size'] // nb_intr_sets
        K_init = initial_params.get('K', None)

        for i in range(nb_intr_sets):
            cam_idx = 0 if is_shared else i
            h, w = images_sizes_hw[cam_idx]
            offset = info['offset'] + i * size_per_set
            f_val = (K_init[cam_idx, 0, 0] + K_init[cam_idx, 1, 1]) / 2.0 if K_init is not None else 1000.0
            f_scale = max(100.0, float(f_val))

            if cfg['fix_aspect_ratio']:
                scales[offset:offset + 3] = [f_scale, w, h]
            else:
                scales[offset:offset + 4] = [f_scale, f_scale, w, h]

    if 'extrinsics' in spec['blocks']:
        info = spec['blocks']['extrinsics']
        nb_optim_cams = C - 1
        t_scales = np.ones(3 * nb_optim_cams)

        # Try to guess scale from initial translation magnitude
        if 'cam_tvecs' in initial_params:
            t_std = np.std(initial_params['cam_tvecs'], axis=0)
            t_avg = np.mean(t_std) if np.mean(t_std) > 1e-4 else 1.0
            t_scales[:] = t_avg

        scales[info['offset']:info['offset'] + info['size']] = np.concatenate(
            [np.full(3 * nb_optim_cams, 0.1), t_scales])

    if 'poses' in spec['blocks']:
        info = spec['blocks']['poses']
        scales[info['offset']:info['offset'] + info['size']] = np.concatenate([np.full(3 * P, 0.1),
                                                                               np.ones(3 * P)])

    return scales


def _get_bounds(spec, images_sizes_hw):
    """
    Pure NumPy. Computes bounds for optimiser.
    """
    cfg = spec['config']
    lb = np.full(spec['total_size'], -np.inf)
    ub = np.full(spec['total_size'], np.inf)

    if 'K' in spec['blocks']:
        info = spec['blocks']['K']
        is_shared = cfg['shared_intrinsics']
        nb_sets = 1 if is_shared else cfg['nb_cams']
        size_per_set = info['size'] // nb_sets

        for i in range(nb_sets):
            offset = info['offset'] + i * size_per_set
            h, w = images_sizes_hw[0 if is_shared else i]

            # Intrinsic bounds (fx, fy, cx, cy)
            if cfg['fix_aspect_ratio']:
                lb[offset:offset + 3] = [10.0, -w, -h]
                ub[offset:offset + 3] = [100000.0, 2 * w, 2 * h]
            else:
                lb[offset:offset + 4] = [10.0, 10.0, -w, -h]
                ub[offset:offset + 4] = [100000.0, 100000.0, 2 * w, 2 * h]

    # Distortion bounds
    if 'D' in spec['blocks']:
        info = spec['blocks']['D']
        n_d = cfg['n_d_size']
        nb_sets = 1 if cfg['shared_intrinsics'] else cfg['nb_cams']

        # Create bounds

        # TODO: These should be configurable based on type of lens

        # This works well for our Basler cameras + Basler 50 mm lenses + 10 mm extender rings
        k_lim = 0.1
        p_lim = 0.005
        k_higher_order_lim = 0.05

        # These ones should be a bit more generic
        # k_lim = 0.5
        # p_lim = 0.05
        # k_higher_order_lim = 0.05

        dist_bounds_map = [
            (-k_lim, k_lim),    # k1
            (-k_lim, k_lim),    # k2
            (-p_lim, p_lim),    # p1
            (-p_lim, p_lim),    # p2
            (-k_lim, k_lim),    # k3
            (-k_higher_order_lim, k_higher_order_lim),  # k4
            (-k_higher_order_lim, k_higher_order_lim),  # k5
            (-k_higher_order_lim, k_higher_order_lim)   # k6
        ]

        # TODO: Fisheye map is different: k1, k2, k3, k4

        dist_lb = [b[0] for b in dist_bounds_map[:n_d]]
        dist_ub = [b[1] for b in dist_bounds_map[:n_d]]

        # Tile for all cameras
        dist_lb = np.tile(dist_lb, nb_sets)
        dist_ub = np.tile(dist_ub, nb_sets)

        lb[info['offset']:info['offset'] + info['size']] = dist_lb
        ub[info['offset']:info['offset'] + info['size']] = dist_ub

    return lb, ub


# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Packing and unpacking (JAX/NumPy compatible)
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

def _unpack_priors_dict(priors_dict: Optional[Dict[str, float]] = None) -> Tuple[float, float, float, float, float]:
    """
    Unpacks the priors weight dict.
    This runs once at setup.
    """

    if priors_dict is None:
        priors_dict = {}

    intr_priors = priors_dict.get('intrinsics', {})
    extr_priors = priors_dict.get('extrinsics', {})

    weight_f = float(intr_priors.get('focal_length', 0.0))
    weight_c = float(intr_priors.get('principal_point', 0.0))
    weight_d = float(intr_priors.get('D', 0.0))
    weight_r = float(extr_priors.get('rotation', 0.0))
    weight_t = float(extr_priors.get('translation', 0.0))

    return weight_f, weight_c, weight_d, weight_r, weight_t


def _pack_params_numpy(K, D, cam_rvecs, cam_tvecs, object_points, poses_rvecs, poses_tvecs, spec):
    """
    Packs initial parameters into the optimization vector x0 using NumPy.
    This runs once at setup.
    """
    cfg = spec['config']
    parts = []
    is_shared = cfg['shared_intrinsics'] and cfg['nb_cams'] > 1

    if 'K' in spec['blocks']:
        # Extract f, cx, cy
        if cfg['fix_aspect_ratio']:
            f = (K[:, 0, 0] + K[:, 1, 1]) * 0.5
            block = np.column_stack([f, K[:, 0, 2], K[:, 1, 2]])
        else:
            block = np.column_stack([K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2]])

        parts.append(np.mean(block, axis=0) if is_shared else block.ravel())

    if 'D' in spec['blocks']:
        n_d = cfg['n_d_size']
        block = D[:, :n_d]
        parts.append(np.mean(block, axis=0) if is_shared else block.ravel())

    if 'extrinsics' in spec['blocks']:
        origin = cfg['origin_idx']
        mask = np.arange(cfg['nb_cams']) != origin
        parts.append(cam_rvecs[mask].ravel())
        parts.append(cam_tvecs[mask].ravel())

    if 'poses' in spec['blocks']:
        parts.append(poses_rvecs.ravel())
        parts.append(poses_tvecs.ravel())

    if 'object_points' in spec['blocks']:
        parts.append(object_points.ravel())

    return np.concatenate(parts) if parts else np.array([])


def _unpack_params_jax(x, fixed_params, spec):
    """Unpacks parameters inside the cost function using JAX."""

    cfg = spec['config']
    C, P = cfg['nb_cams'], cfg['nb_frames']
    is_shared = cfg['shared_intrinsics'] and C > 1

    # Start with fixed defaults
    K = fixed_params['K_init']  # (C, 3, 3)
    D = fixed_params['D_init']  # (C, n_d)

    # Intrinsics
    if 'K' in spec['blocks']:
        info = spec['blocks']['K']
        vals = x[info['offset']:info['offset'] + info['size']]

        size_set = 3 if cfg['fix_aspect_ratio'] else 4
        block = vals.reshape(-1, size_set)  # (sets, params)

        if is_shared:
            block = jnp.tile(block, (C, 1))

        if cfg['fix_aspect_ratio']:
            K = K.at[:, 0, 0].set(block[:, 0])
            K = K.at[:, 1, 1].set(block[:, 0])
            K = K.at[:, 0, 2].set(block[:, 1])
            K = K.at[:, 1, 2].set(block[:, 2])
        else:
            K = K.at[:, 0, 0].set(block[:, 0])
            K = K.at[:, 1, 1].set(block[:, 1])
            K = K.at[:, 0, 2].set(block[:, 2])
            K = K.at[:, 1, 2].set(block[:, 3])

    if 'D' in spec['blocks']:
        info = spec['blocks']['D']
        vals = x[info['offset']:info['offset'] + info['size']]
        n_d = cfg['n_d_size']
        block = vals.reshape(-1, n_d)
        if is_shared:
            block = jnp.tile(block, (C, 1))
        D = D.at[:, :n_d].set(block)

    # Extrinsics
    cam_r = fixed_params['cam_r']
    cam_t = fixed_params['cam_t']

    if 'extrinsics' in spec['blocks']:
        info = spec['blocks']['extrinsics']
        vals = x[info['offset']:info['offset'] + info['size']]
        nb_opt = C - 1
        r_opt = vals[:3 * nb_opt].reshape(nb_opt, 3)
        t_opt = vals[3 * nb_opt:].reshape(nb_opt, 3)

        optim_indices = jnp.delete(jnp.arange(C), cfg['origin_idx'])
        cam_r = cam_r.at[optim_indices].set(r_opt)
        cam_t = cam_t.at[optim_indices].set(t_opt)

    # Poses
    poses_r = fixed_params.get('poses_r')
    poses_t = fixed_params.get('poses_t')

    if 'poses' in spec['blocks']:
        info = spec['blocks']['poses']
        vals = x[info['offset']:info['offset'] + info['size']]
        poses_r = vals[:3 * P].reshape(P, 3)
        poses_t = vals[3 * P:].reshape(P, 3)

    # Points
    obj_pts = fixed_params.get('object_points')
    if 'object_points' in spec['blocks']:
        info = spec['blocks']['object_points']
        vals = x[info['offset']:info['offset'] + info['size']]
        obj_pts = vals.reshape(-1, 3)

    return K, D, cam_r, cam_t, obj_pts, poses_r, poses_t


# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Cost function (pure JAX)
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

def cost_function(params, fixed_params, spec, image_points, points_weights,
                  prior_w_r, prior_w_t, prior_w_f, prior_w_c, prior_w_d):
    """Pure JAX. Computes weighted residuals using the JAX geometry kernels."""

    K, D, cam_r, cam_t, obj_pts, poses_r, poses_t = _unpack_params_jax(params, fixed_params, spec)

    all_residuals = []

    cfg = spec['config']
    dist_model = cfg['dist_model_idx']

    T_c2w = _jax_compose_transform(cam_r, cam_t)

    # Invert
    R_c2w = T_c2w[..., :3, :3]
    t_c2w = T_c2w[..., :3, 3]
    R_w2c = jnp.swapaxes(R_c2w, -1, -2)
    t_w2c = -jnp.einsum('...ij,...j->...i', R_w2c, t_c2w)

    T_w2c = jnp.concatenate([R_w2c, t_w2c[..., None]], axis=-1)
    bottom = jnp.broadcast_to(jnp.array([0., 0., 0., 1.]), T_w2c.shape[:-2] + (1, 4))
    T_w2c = jnp.concatenate([T_w2c, bottom], axis=-2)

    # Project
    if not cfg['fix_poses']:
        # Rigid object workflow: Points(N, 3) transformed by Poses(P, 4, 4) to world
        T_o2w = _jax_compose_transform(poses_r, poses_t)

        T_total = jnp.matmul(T_w2c[:, None, ...], T_o2w[None, :, ...])

        C, P = T_total.shape[:2]
        T_flat = T_total.reshape(C * P, 4, 4)

        # K and D also need broadcasting
        K_flat = jnp.repeat(K, P, axis=0)  # (C*P, 3, 3)
        D_flat = jnp.repeat(D, P, axis=0)

        reproj_flat, z_flat = _jax_project(obj_pts, T_flat, K_flat, D_flat, dist_model)

        # reshape back
        reproj = reproj_flat.reshape(C, P, -1, 2)
        z_depths = z_flat.reshape(C, P, -1)

    else:
        # Scaffolding / Non-rigid object workflow: obj_pts is (P*N, 3) or just (N, 3) shared
        # This path assumes obj_pts matches structure directly (world coordinates)
        pts_reshaped = obj_pts.reshape(cfg['nb_frames'], cfg['nb_points'], 3)

        reproj, z_depths = _jax_project(
            pts_reshaped[None, ...],
            T_w2c[:, None, ...],
            K[:, None, ...],
            D[:, None, ...],
            dist_model
        )

    # Reprojection residuals
    diff = reproj - image_points

    valid_mask = z_depths > _eps

    # If point is not visible, we zero it out here *but* we add the barrier cost below
    w_reproj = jnp.where(valid_mask, points_weights, 0.0)

    res = diff * w_reproj[..., None]
    all_residuals.append(res.ravel())

    # Depth barrier
    # If Z < near_plane, add a cost (near_plane - Z): this pushes points in front of the camera
    near_plane = fixed_params['near_plane']
    depth_violation = jnp.maximum(0.0, near_plane - z_depths)

    # Weight this heavily so the optimizer prioritizes validity over pixel error
    depth_penalty = depth_violation * 100.0 * points_weights

    # We only care about this for points that *should* be visible
    # (points_weights already contains the visibility mask from inputs)
    all_residuals.append(depth_penalty.ravel())

    # ──────────────────────────────────────────────────────────────────────────────
    # DEBUG: Print RMS Errors
    # ──────────────────────────────────────────────────────────────────────────────

    # Mask and count
    mask = w_reproj > 0
    N = jnp.maximum(jnp.sum(mask), 1.0)

    # Squared Euclidean distance per point
    # (zero out invalid points immediately so they don't affect sums)
    sq_dist_per_point = jnp.sum(jnp.square(diff), axis=-1) * mask

    # Component RMS: sqrt( Sum(dx^2 + dy^2) / 2N )
    total_sse = jnp.sum(sq_dist_per_point)
    comp_rms = jnp.sqrt(total_sse / (2.0 * N))

    # MRE: Sum( sqrt(dx^2 + dy^2) ) / N
    # Note: sqrt(0) is 0, so invalid points add nothing to the sum
    mre = jnp.sum(jnp.sqrt(sq_dist_per_point)) / N

    jax.debug.print("Component RMS: {x:.4f} px, MRE: {y:.4f} px", x=comp_rms, y=mre)

    # ──────────────────────────────────────────────────────────────────────────────


    # Priors: Extrinsics
    if prior_w_r > 0.0 or prior_w_t > 0.0:
        origin_idx = spec['config']['origin_idx']
        all_indices = jnp.arange(spec['config']['nb_cams'])
        optim_indices = jnp.delete(all_indices, origin_idx)

        # Note: fixed_params['cam_r_init'] contains the initial guess we want to stay close to
        rvec_resid = (cam_r[optim_indices] - fixed_params['cam_r_init'][optim_indices]).ravel() * prior_w_r
        tvec_resid = (cam_t[optim_indices] - fixed_params['cam_t_init'][optim_indices]).ravel() * prior_w_t
        all_residuals.extend([rvec_resid, tvec_resid])

    # Priors: Intrinsics
    if not spec['config']['fix_intrinsics']:
        is_shared = spec['config']['shared_intrinsics'] and spec['config']['nb_cams'] > 1
        if is_shared:
            K_init = jnp.mean(fixed_params['K_init'], axis=0, keepdims=True)
            D_init = jnp.mean(fixed_params['D_init'], axis=0, keepdims=True)
            Ks_opt = K[:1]
            Ds_opt = D[:1]
        else:
            K_init = fixed_params['K_init']
            D_init = fixed_params['D_init']
            Ks_opt = K
            Ds_opt = D

        if prior_w_f > 0.0:
            if spec['config']['fix_aspect_ratio']:
                f_init = (K_init[:, 0, 0] + K_init[:, 1, 1]) * 0.5
                f_opt = Ks_opt[:, 0, 0]
                f_resid = (f_opt - f_init) * prior_w_f
            else:
                f_init_x = K_init[:, 0, 0]
                f_init_y = K_init[:, 1, 1]
                f_opt_x = Ks_opt[:, 0, 0]
                f_opt_y = Ks_opt[:, 1, 1]
                f_resid = jnp.concatenate([(f_opt_x - f_init_x), (f_opt_y - f_init_y)]) * prior_w_f
            all_residuals.append(f_resid)

        if prior_w_c > 0.0:
            pp_init = K_init[:, :2, 2]
            pp_opt = Ks_opt[:, :2, 2]
            pp_resid = (pp_opt - pp_init).ravel() * prior_w_c
            all_residuals.append(pp_resid)

        if prior_w_d > 0.0 and 'D' in spec['blocks']:
            n_d = spec['config']['n_d']
            dist_init = D_init[:, :n_d]
            dist_opt = Ds_opt[:, :n_d]
            dist_resid = (dist_opt - dist_init).ravel() * prior_w_d
            all_residuals.append(dist_resid)

    return jnp.concatenate(all_residuals)


# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Sparsity (pure NumPy)
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────


def make_jacobian_sparsity(spec, use_extrinsics_prior: bool, use_intrinsics_prior: bool) -> csr_matrix:
    """
    Pure NumPy. Creates the Jacobian sparsity matrix.
    This runs once at setup.
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
    for c in range(C):
        for p in range(P):
            for n in range(N):
                row = 2 * (c * P * N + p * N + n)
                intr_set_idx = 0 if is_shared else c

                # Dependency on camera intrinsics
                if 'K' in spec['blocks']:
                    info = spec['blocks']['K']
                    size_per_set = info['size'] // nb_intr_sets
                    col = info['offset'] + intr_set_idx * size_per_set
                    S[row:row + 2, col:col + size_per_set] = 1

                if 'D' in spec['blocks']:
                    info = spec['blocks']['D']
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
                        # For both scaffolding and non-rigid objects (animal): each frame has a unique set of N points
                        point_idx_in_optim_vector = p * N + n

                    col = info['offset'] + point_idx_in_optim_vector * 3
                    S[row:row + 2, col:col + 3] = 1

    # Add sparsity for all priors
    nb_prior_residuals = 0
    if use_extrinsics_prior and 'extrinsics' in spec['blocks']:
        nb_prior_residuals += 6 * (C - 1)  # only for optimizable cameras

    if use_intrinsics_prior:
        if 'K' in spec['blocks']:
            nb_prior_residuals += spec['blocks']['K']['size']

        if 'D' in spec['blocks']:
            nb_prior_residuals += spec['blocks']['D']['size']

    if nb_prior_residuals > 0:
        S.resize(nb_residuals + nb_prior_residuals, nb_params)

    curr_prior_row = nb_residuals

    # Cameras extrinsics priors
    if use_extrinsics_prior and 'extrinsics' in spec['blocks']:
        info = spec['blocks']['extrinsics']
        nb_optim_cams = C - 1

        S[curr_prior_row:curr_prior_row + 3 * nb_optim_cams, info['offset']:info['offset'] + 3 * nb_optim_cams] = True
        curr_prior_row += 3 * nb_optim_cams

        t_off = info['offset'] + 3 * nb_optim_cams

        S[curr_prior_row:curr_prior_row + 3 * nb_optim_cams, t_off:t_off + 3 * nb_optim_cams] = True
        curr_prior_row += 3 * nb_optim_cams

    # Cameras intrinsics priors
    if use_intrinsics_prior:

        if 'K' in spec['blocks']:
            info = spec['blocks']['K']
            S[curr_prior_row:curr_prior_row + info['size'], info['offset']:info['offset'] + info['size']] = True
            curr_prior_row += info['size']

        if 'D' in spec['blocks']:
            info = spec['blocks']['D']
            S[curr_prior_row:curr_prior_row + info['size'], info['offset']:info['offset'] + info['size']] = True

    return S.tocsr()


# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Main run
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

def run_bundle_adjustment(
        K_initial: jnp.ndarray,
        D_initial: jnp.ndarray,
        cam_poses_initial: jnp.ndarray,
        images_sizes_hw: ArrayLike,
        image_points: jnp.ndarray,
        visibility_mask: jnp.ndarray,
        object_points_initial: Optional[jnp.ndarray] = None,
        object_poses_initial: Optional[jnp.ndarray] = None,
        fix_intrinsics: bool = False,
        fix_extrinsics: bool = False,
        fix_object_points: bool = False,
        fix_poses: bool = True,
        fix_aspect_ratio: bool = False,
        origin_idx: int = 0,
        priors: Optional[Dict] = None,
        radial_penalty: float = 2.0,
        shared_intrinsics: bool = False,
        distortion_model: DistortionModel = 'standard',
        tolerance: float = 1e-8,
        max_nfev: int = 500
) -> Tuple[bool, Dict]:
    # TODO: Maybe get rid of priors and only rely in covariance matrices...?

    # Validation and setup (NumPy)
    C, P, N, _ = image_points.shape
    cam_r_init, cam_t_init = geom.decompose_transform_matrix(cam_poses_initial)

    poses_r_init, poses_t_init = None, None
    if object_poses_initial is not None:
        poses_r_init, poses_t_init = geom.decompose_transform_matrix(object_poses_initial)

    nb_pts_opt = object_points_initial.shape[0] if object_points_initial is not None and not fix_object_points else 0

    spec = _get_parameter_spec(
        C, P, N, nb_pts_opt, origin_idx, fix_intrinsics, fix_extrinsics,
        fix_object_points, fix_poses, fix_aspect_ratio, shared_intrinsics, distortion_model
    )

    prior_f, prior_c, prior_d, prior_r, prior_t = _unpack_priors_dict(priors)

    # Parameters packing (NumPy)
    n_d = spec['config']['n_d_size']
    if D_initial.shape[1] < n_d:
        D_initial = np.pad(D_initial, ((0, 0), (0, n_d - D_initial.shape[1])))

    x0 = _pack_params_numpy(K_initial, D_initial, cam_r_init, cam_t_init,
                            object_points_initial, poses_r_init, poses_t_init, spec)

    scales = _get_parameter_scales(spec, {'K': K_initial, 'cam_tvecs': cam_t_init}, images_sizes_hw)
    lb, ub = _get_bounds(spec, images_sizes_hw)
    x0 = np.clip(x0, lb, ub)

    # Prepare fixed params and weights (JAX-ready)
    points_weights_np = (visibility_mask > 0).astype(np.float32)

    # Calculate advanced weights using the JAX kernel
    compute_weights_fun = jit(_jax_residual_weights)

    points_weights_jax = compute_weights_fun(
        jnp.array(image_points),
        jnp.array(points_weights_np),
        jnp.array(K_initial),
        radial_penalty
    )
    img_pts_jax = jnp.array(image_points)

    # Compute near_plane for barrier loss: median Z depth of the initial points relative to the origin camera
    near_plane = 1e-3
    if object_points_initial is not None:
        try:
            # Get world -> camera transform for the origin camera
            T_w2c_origin = geom.invert_transform(cam_poses_initial[origin_idx])

            # Project a subset of points into this camera's frame
            pts_sub = object_points_initial[:min(100, len(object_points_initial))]
            pts_cam = geom.transform_points(pts_sub, T_w2c_origin)

            # Compute median Z
            median_z = float(np.median(pts_cam[:, 2]))

            if median_z > 1e-6:
                near_plane = median_z * 1e-3  # 0.1% of scene scale
        except Exception:
            # use default if anything fails
            pass

    fixed_params = {
        'K_init': jnp.array(K_initial),
        'D_init': jnp.array(D_initial),
        'cam_r': jnp.array(cam_r_init),
        'cam_t': jnp.array(cam_t_init),
        'cam_r_init': jnp.array(cam_r_init.copy()),  # store separate copy for priors reference
        'cam_t_init': jnp.array(cam_t_init.copy()),  # same
        'poses_r': jnp.array(poses_r_init) if poses_r_init is not None else None,
        'poses_t': jnp.array(poses_t_init) if poses_t_init is not None else None,
        'object_points': jnp.array(object_points_initial) if object_points_initial is not None else None,
        'near_plane': near_plane
    }

    # JIT compilation
    # The static dictionary 'spec' and fixed data are baked in
    cost_fn_jitted = jit(partial(cost_function,
                                 fixed_params=fixed_params,
                                 spec=spec,
                                 image_points=img_pts_jax,
                                 points_weights=points_weights_jax,
                                 prior_w_r=prior_r,
                                 prior_w_t=prior_t,
                                 prior_w_f=prior_f,
                                 prior_w_c=prior_c,
                                 prior_w_d=prior_d))

    jac_fn_jitted = jit(jax.jacfwd(partial(cost_function,
                                           fixed_params=fixed_params,
                                           spec=spec,
                                           image_points=img_pts_jax,
                                           points_weights=points_weights_jax,
                                           prior_w_r=prior_r,
                                           prior_w_t=prior_t,
                                           prior_w_f=prior_f,
                                           prior_w_c=prior_c,
                                           prior_w_d=prior_d)))

    # Scipy wrappers (they bridge the NumPy -> JAX -> NumPy merry-go-round)
    def fun_wrapped(x):
        return np.array(cost_fn_jitted(x))

    def jac_wrapped(x):
        return np.array(jac_fn_jitted(x))

    # Optimize
    S = make_jacobian_sparsity(
        spec,
        use_extrinsics_prior=prior_r > 0.0 or prior_t > 0.0,
        use_intrinsics_prior=prior_f > 0.0 or prior_c > 0.0 or prior_d > 0.0
    )

    # with alive_bar(title='Bundle adjustment...', length=20, force_tty=True) as bar:
    #     with CallbackOutputStream(bar, keep_stdout=False):
    res = least_squares(
        fun_wrapped, np.asarray(x0),
        jac=jac_wrapped,
        jac_sparsity=S,
        bounds=(np.asarray(lb), np.asarray(ub)),
        x_scale=scales,
        method='trf',
        loss='cauchy',
        f_scale=2.5,
        ftol=tolerance,
        xtol=tolerance,
        gtol=tolerance,
        max_nfev=max_nfev,
        verbose=2
    )

    # Unpack (NumPy)
    # Note: unpacking assumes x is a JAX array so we cast res.x just one time, it's fine
    opt_params = _unpack_params_jax(jnp.array(res.x), fixed_params, spec)

    K_opt, D_opt, cam_r, cam_t, obj_pts, poses_r, poses_t = [np.array(p) if p is not None else None for p in opt_params]

    # Recompose matrices using standard geometry lib
    cam_poses_opt = geom.compose_transform_matrix(cam_r, cam_t)
    obj_poses_opt = None
    if poses_r is not None:
        obj_poses_opt = geom.compose_transform_matrix(poses_r, poses_t)

    return res.success, {
        'K_opt': K_opt,
        'D_opt': D_opt,
        'cam_poses_opt': cam_poses_opt,
        'object_poses_opt': obj_poses_opt,
        'object_points_opt': obj_pts
    }