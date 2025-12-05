import warnings
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix, csr_matrix
from scipy.linalg import cholesky
from typing import Tuple, Dict, Optional, Any
from functools import partial
from dataclasses import dataclass

from mokap.utils import CallbackOutputStream
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
    if model_idx == 4:  # Simple
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
    if model_idx in [4, 5, 8]:
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
    if visibility_mask.ndim == 3:  # need to handle the shapes depending on if P exists
        view_weight = nb_views[None, :, :]
    else:
        view_weight = nb_views[None, :]

    # Sigmoidal scaling for view count
    view_weight_factor = view_weight / (1.0 + view_weight)

    # Combine
    weights = visibility_mask.astype(jnp.float32) * radial_weight * view_weight_factor
    return weights

jax_residual_weights = jit(_jax_residual_weights)


# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Covariance and Information matrix helpers
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

@dataclass
class PriorInfo:
    """
    Container for prior information matrices and their square roots.

    The information matrix (inverse covariance) is used in the cost function.
    The square root information matrix (SRIM) is used for residual weighting:
        residual = SRIM @ (x - x0)

    This ensures the residual contribution is ||SRIM @ (x - x0)||^2 = (x - x0)^T @ Info @ (x - x0)
    """

    # Intrinsics priors
    intrinsics_srim: Optional[jnp.ndarray] = None  # (nb_intr_sets, n_K + n_D, n_K + n_D), or None
    intrinsics_mean: Optional[jnp.ndarray] = None  # (nb_intr_sets, n_K + n_D)

    # Extrinsics priors (per-camera, excluding origin)
    extrinsics_srim: Optional[jnp.ndarray] = None  # (C-1, 6, 6), or None
    extrinsics_mean: Optional[jnp.ndarray] = None  # (C-1, 6), rvec, tvec concatenated


# TODO: This function should probably not live in this file
def covariance_from_std(
        nb_cameras: int,
        nb_dist_coeffs: int,
        fix_aspect_ratio: bool,
        shared_intrinsics: bool,
        sigma_f: float = 100.0,     # TODO: tune these for our cameras + lenses combinations
        sigma_c: float = 10.0,
        sigma_d: float = 0.1,
        sigma_r: float = 0.1,
        sigma_t: float = 10.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates diagonal covariance matrices from scalar standard deviations.
    This is useful for the first BA run when no prior covariance is available.

    TODO: the weights correspond to 1/sigma^2 in the old, scalar priors system: multiview.py needs update

    Args:
        nb_cameras: Number of cameras
        nb_dist_coeffs: Number of distortion coefficients
        fix_aspect_ratio: Whether aspect ratio is fixed (affects K size)
        shared_intrinsics: Whether intrinsics are shared across cameras
        sigma_f: Standard deviation for focal length (in pixels)
        sigma_c: Standard deviation for principal point (in pixels)
        sigma_d: Standard deviation for distortion coefficients
        sigma_r: Standard deviation for rotation (in radians)
        sigma_t: Standard deviation for translation (in scene units)

    Returns:
        Extrinsics covariance and Intrinsics covariance arrays
    """

    nb_optim_cams = nb_cameras - 1
    nb_intr_sets = 1 if shared_intrinsics else nb_cameras

    # Tiny epsilon to avoid singular matrices if any sigma passed is 0
    eps = 1e-9

    # Extrinsics: (C-1, 6, 6) diagonal (each camera has rvec, tvec)
    extr_variances = np.array([sigma_r ** 2 + eps] * 3 + [sigma_t ** 2 + eps] * 3)
    extrinsics_cov = np.zeros((nb_optim_cams, 6, 6))
    for i in range(nb_optim_cams):
        extrinsics_cov[i] = np.diag(extr_variances)

    # Intrinsics: (nb_intr_sets, n_K + n_D, n_K + n_D)
    nb_K = 3 if fix_aspect_ratio else 4  # f or (fx, fy), cx, cy
    intrinsics_dim = nb_K + nb_dist_coeffs

    # Build variance vector for one intrinsics set
    if fix_aspect_ratio:
        intr_variances = np.array([sigma_f ** 2 + eps, sigma_c ** 2 + eps, sigma_c ** 2 + eps])
    else:
        intr_variances = np.array([sigma_f ** 2 + eps, sigma_f ** 2 + eps, sigma_c ** 2 + eps, sigma_c ** 2 + eps])

    # Add distortion variances
    intr_variances = np.concatenate([intr_variances, np.full(nb_dist_coeffs, sigma_d ** 2 + eps)])

    intrinsics_cov = np.zeros((nb_intr_sets, intrinsics_dim, intrinsics_dim))
    for i in range(nb_intr_sets):
        intrinsics_cov[i] = np.diag(intr_variances)

    return extrinsics_cov, intrinsics_cov


def covariance_to_information(cov: np.ndarray, regularization: float = 1e-8) -> np.ndarray:
    """
    Converts covariance matrix to information matrix (inverse).

    Args:
        regularization: Small regularization added to diagonal for numerical stability.
    """
    reg_cov = cov.copy()

    if reg_cov.ndim == 2:
        reg_cov += np.eye(reg_cov.shape[0]) * regularization
        return np.linalg.inv(reg_cov)

    else:
        batch_shape = reg_cov.shape[:-2]
        n = reg_cov.shape[-1]
        reg_cov = reg_cov + np.eye(n) * regularization

        # flatten batch dims, invert, reshape
        flat = reg_cov.reshape(-1, n, n)
        inv_flat = np.linalg.inv(flat)
        return inv_flat.reshape(batch_shape + (n, n))


def compute_SRIM(info: np.ndarray) -> np.ndarray:
    """
    Computes the square root information matrix (SRIM) via Cholesky decomposition.

    The SRIM L satisfies: L @ L.T = Info
    So: residual = L @ (x - x0) gives ||residual||^2 = (x-x0)^T @ Info @ (x-x0)

    Handles batched inputs: (..., N, N) -> (..., N, N)
    """
    if info.ndim == 2:
        return cholesky(info, lower=True)
    else:
        batch_shape = info.shape[:-2]
        n = info.shape[-1]
        flat = info.reshape(-1, n, n)

        srim_flat = np.zeros_like(flat)
        for i in range(flat.shape[0]):
            srim_flat[i] = cholesky(flat[i], lower=True)

        return srim_flat.reshape(batch_shape + (n, n))


def prepare_prior_info(
        spec: Dict,
        initial_params: Dict[str, np.ndarray],
        covariance_intrinsics: Optional[np.ndarray] = None,
        covariance_extrinsics: Optional[np.ndarray]= None
) -> Optional[PriorInfo]:
    """
    Prepares the prior information for use in the cost function.

    Args:
        spec: Parameter specification dictionary
        initial_params: Dictionary with initial parameter values (for means)
        covariance_intrinsics: Covariance matrix for intrinsics. Shape depends on distortion model
        covariance_extrinsics: Covariance matrix for extrinsics. Shape (C-1, 6, 6) because origin camera is fixed

    Returns:
        PriorInfo object with SRIMs and means, or None if no priors
    """
    if covariance_intrinsics is None and covariance_extrinsics is None:
        return None

    cfg = spec['config']
    origin_cam_idx = cfg['origin_cam_idx']

    extr_srim = None
    extr_mean = None
    intr_srim = None
    intr_mean = None

    # Extrinsics priors
    if covariance_extrinsics is not None and 'extrinsics' in spec['blocks']:
        extr_info = covariance_to_information(covariance_extrinsics)
        extr_srim = jnp.array(compute_SRIM(extr_info))

        # Build mean from initial rvecs/tvecs (excluding origin)
        camera_rvecs = initial_params['camera_rvecs']
        camera_tvecs = initial_params['camera_tvecs']
        mask = np.arange(cfg['nb_cams']) != origin_cam_idx

        # Concatenate rvec and tvec for each camera
        extr_mean = jnp.concatenate([
            jnp.array(camera_rvecs[mask]),
            jnp.array(camera_tvecs[mask])
        ], axis=-1)  # (C-1, 6)

    # Intrinsics priors
    if covariance_intrinsics is not None:
        has_K = 'K' in spec['blocks']
        has_D = 'D' in spec['blocks']

        if has_K or has_D:
            intr_info = covariance_to_information(covariance_intrinsics)
            intr_srim = jnp.array(compute_SRIM(intr_info))

            # Build mean from initial K and D
            is_shared = cfg['shared_intrinsics'] and cfg['nb_cams'] > 1
            K_init = initial_params['K']
            D_init = initial_params['D']

            if is_shared:
                K_init = np.mean(K_init, axis=0, keepdims=True)
                D_init = np.mean(D_init, axis=0, keepdims=True)

            # Extract the relevant parameters based on spec
            mean_parts = []

            if has_K:
                if cfg['fix_aspect_ratio']:
                    f = (K_init[:, 0, 0] + K_init[:, 1, 1]) * 0.5
                    mean_parts.append(np.column_stack([f, K_init[:, 0, 2], K_init[:, 1, 2]]))
                else:
                    mean_parts.append(np.column_stack([
                        K_init[:, 0, 0], K_init[:, 1, 1],
                        K_init[:, 0, 2], K_init[:, 1, 2]
                    ]))

            if has_D:
                nb_dist_coeffs = cfg['n_d_size']
                mean_parts.append(D_init[:, :nb_dist_coeffs])

            intr_mean_np = np.concatenate(mean_parts, axis=-1)
            expected_dim = intr_mean_np.shape[-1]
            prior_dim = covariance_intrinsics.shape[-1]

            # Check for dimension mismatch
            if expected_dim != prior_dim:
                warnings.warn(
                    f"[Bundle Adjustment] Ignoring intrinsics prior: Dimension mismatch. "
                    f"Current config requires {expected_dim} params (fix_aspect_ratio={cfg['fix_aspect_ratio']}), "
                    f"but covariance matrix has {prior_dim}."
                )
                intr_srim = None
                intr_mean = None
            else:
                intr_info = covariance_to_information(covariance_intrinsics)
                intr_srim = jnp.array(compute_SRIM(intr_info))
                intr_mean = jnp.array(intr_mean_np)

    if extr_srim is None and intr_srim is None:
        return None

    return PriorInfo(
        intrinsics_srim=intr_srim,
        intrinsics_mean=intr_mean,
        extrinsics_srim=extr_srim,
        extrinsics_mean=extr_mean
    )


# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Helpers and specs config (NumPy)
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

DIST_MODEL_MAP = {'none': 0, 'simple': 4, 'standard': 5, 'rational': 8, 'thinprism': 12, 'tilted': 14, 'fisheye': 99}


def _get_parameter_spec(
        nb_cams, nb_frames, nb_points,
        nb_points_to_optim, origin_cam_idx, distortion_model,
        fix_intrinsics, fix_extrinsics, fix_object_points,
        fix_object_poses, fix_aspect_ratio, shared_intrinsics
    ):
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

        nb_dist_coeffs = DIST_MODEL_MAP.get(distortion_model, 0)
        n_d_size = 4 if distortion_model == 'fisheye' else nb_dist_coeffs

        spec['config']['nb_dist_coeffs'] = nb_dist_coeffs
        spec['config']['n_d_size'] = n_d_size
        spec['config']['dist_model_idx'] = nb_dist_coeffs

        if nb_dist_coeffs > 0:
            size_dist = n_d_size * nb_intr_sets
            spec['blocks']['D'] = {'offset': current_offset, 'size': size_dist}
            current_offset += size_dist
    else:
        # Fallback for when intrinsics are fixed
        spec['config']['nb_dist_coeffs'] = DIST_MODEL_MAP.get(distortion_model, 0)
        spec['config']['n_d_size'] = 4 if distortion_model == 'fisheye' else spec['config']['nb_dist_coeffs']
        spec['config']['dist_model_idx'] = spec['config']['nb_dist_coeffs']

    if not fix_extrinsics:
        nb_optim_cams = nb_cams - 1
        spec['blocks']['extrinsics'] = {'offset': current_offset, 'size': 6 * nb_optim_cams}
        current_offset += 6 * nb_optim_cams

    if not fix_object_poses and nb_frames > 0:
        spec['blocks']['object_poses'] = {'offset': current_offset, 'size': 6 * nb_frames}
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
        if 'camera_tvecs' in initial_params:
            t_std = np.std(initial_params['camera_tvecs'], axis=0)
            t_avg = np.mean(t_std) if np.mean(t_std) > 1e-4 else 1.0
            t_scales[:] = t_avg

        scales[info['offset']:info['offset'] + info['size']] = np.concatenate(
            [np.full(3 * nb_optim_cams, 0.1), t_scales])

    if 'object_poses' in spec['blocks']:
        info = spec['blocks']['object_poses']
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
        nb_dist_coeffs = cfg['n_d_size']
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
            (-k_lim, k_lim),  # k1
            (-k_lim, k_lim),  # k2
            (-p_lim, p_lim),  # p1
            (-p_lim, p_lim),  # p2
            (-k_lim, k_lim),  # k3
            (-k_higher_order_lim, k_higher_order_lim),  # k4
            (-k_higher_order_lim, k_higher_order_lim),  # k5
            (-k_higher_order_lim, k_higher_order_lim)   # k6
        ]

        # TODO: Fisheye map is different: k1, k2, k3, k4

        dist_lb = [b[0] for b in dist_bounds_map[:nb_dist_coeffs]]
        dist_ub = [b[1] for b in dist_bounds_map[:nb_dist_coeffs]]

        # Tile for all cameras
        dist_lb = np.tile(dist_lb, nb_sets)
        dist_ub = np.tile(dist_ub, nb_sets)

        lb[info['offset']:info['offset'] + info['size']] = dist_lb
        ub[info['offset']:info['offset'] + info['size']] = dist_ub

    return lb, ub


# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Packing and unpacking (JAX/NumPy compatible)
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

def _pack_params_numpy(spec, K, D, camera_rvecs, camera_tvecs, object_points, object_rvecs, object_tvecs):
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
        nb_dist_coeffs = cfg['n_d_size']
        block = D[:, :nb_dist_coeffs]
        parts.append(np.mean(block, axis=0) if is_shared else block.ravel())

    if 'extrinsics' in spec['blocks']:
        origin = cfg['origin_cam_idx']
        mask = np.arange(cfg['nb_cams']) != origin
        parts.append(camera_rvecs[mask].ravel())
        parts.append(camera_tvecs[mask].ravel())

    if 'object_poses' in spec['blocks']:
        parts.append(object_rvecs.ravel())
        parts.append(object_tvecs.ravel())

    if 'object_points' in spec['blocks']:
        parts.append(object_points.ravel())

    return np.concatenate(parts) if parts else np.array([])


def _unpack_params_jax(spec, x, fixed_params):
    """Unpacks parameters inside the cost function using JAX."""

    cfg = spec['config']
    C, P = cfg['nb_cams'], cfg['nb_frames']
    is_shared = cfg['shared_intrinsics'] and C > 1

    # Start with fixed defaults
    K = fixed_params['K']  # (C, 3, 3)
    D = fixed_params['D']  # (C, nb_dist_coeffs)

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
        nb_dist_coeffs = cfg['n_d_size']
        block = vals.reshape(-1, nb_dist_coeffs)
        if is_shared:
            block = jnp.tile(block, (C, 1))
        D = D.at[:, :nb_dist_coeffs].set(block)

    # Extrinsics
    camera_rvecs = fixed_params['camera_rvecs']
    camera_tvecs = fixed_params['camera_tvecs']

    if 'extrinsics' in spec['blocks']:
        info = spec['blocks']['extrinsics']
        vals = x[info['offset']:info['offset'] + info['size']]
        nb_opt = C - 1
        r_opt = vals[:3 * nb_opt].reshape(nb_opt, 3)
        t_opt = vals[3 * nb_opt:].reshape(nb_opt, 3)

        optim_indices = jnp.delete(jnp.arange(C), cfg['origin_cam_idx'])
        camera_rvecs = camera_rvecs.at[optim_indices].set(r_opt)
        camera_tvecs = camera_tvecs.at[optim_indices].set(t_opt)

    # Object poses
    object_rvecs = fixed_params.get('object_rvecs')
    object_tvecs = fixed_params.get('object_tvecs')

    if 'object_poses' in spec['blocks']:
        info = spec['blocks']['object_poses']
        vals = x[info['offset']:info['offset'] + info['size']]
        object_rvecs = vals[:3 * P].reshape(P, 3)
        object_tvecs = vals[3 * P:].reshape(P, 3)

    # Points
    object_points = fixed_params.get('object_points')
    if 'object_points' in spec['blocks']:
        info = spec['blocks']['object_points']
        vals = x[info['offset']:info['offset'] + info['size']]
        object_points = vals.reshape(-1, 3)

    return K, D, camera_rvecs, camera_tvecs, object_points, object_rvecs, object_tvecs


def _extract_current_intrinsics_vector(K, D, spec):
    """
    Extracts the current intrinsics as a flat vector matching the prior mean format.
    Used for computing intrinsics prior residuals.
    """
    cfg = spec['config']
    is_shared = cfg['shared_intrinsics'] and cfg['nb_cams'] > 1

    if is_shared:
        K_use = K[:1]
        D_use = D[:1]
    else:
        K_use = K
        D_use = D

    parts = []

    if 'K' in spec['blocks']:
        if cfg['fix_aspect_ratio']:
            f = K_use[:, 0, 0]
            parts.append(jnp.column_stack([f, K_use[:, 0, 2], K_use[:, 1, 2]]))
        else:
            parts.append(jnp.column_stack([
                K_use[:, 0, 0], K_use[:, 1, 1],
                K_use[:, 0, 2], K_use[:, 1, 2]
            ]))

    if 'D' in spec['blocks']:
        nb_dist_coeffs = cfg['n_d_size']
        parts.append(D_use[:, :nb_dist_coeffs])

    return jnp.concatenate(parts, axis=-1)  # (nb_sets, dim)


# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Cost function (pure JAX)
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

def cost_function(params, fixed_params, image_points, points_weights, spec, prior_info: Optional[PriorInfo]):
    """Pure JAX. Computes weighted residuals using the JAX geometry kernels and covariance-based priors."""

    K, D, camera_rvecs, camera_tvecs, object_points, object_rvecs, object_tvecs = _unpack_params_jax(
        spec=spec,
        x=params,
        fixed_params=fixed_params
    )

    all_residuals = []

    cfg = spec['config']
    dist_model = cfg['dist_model_idx']

    T_c2w = _jax_compose_transform(camera_rvecs, camera_tvecs)

    # Invert
    R_c2w = T_c2w[..., :3, :3]
    t_c2w = T_c2w[..., :3, 3]
    R_w2c = jnp.swapaxes(R_c2w, -1, -2)
    t_w2c = -jnp.einsum('...ij,...j->...i', R_w2c, t_c2w)

    T_w2c = jnp.concatenate([R_w2c, t_w2c[..., None]], axis=-1)
    bottom = jnp.broadcast_to(jnp.array([0., 0., 0., 1.]), T_w2c.shape[:-2] + (1, 4))
    T_w2c = jnp.concatenate([T_w2c, bottom], axis=-2)

    # Project
    if not cfg['fix_object_poses']:
        # Rigid object workflow: Points(N, 3) transformed by Poses(P, 4, 4) to world
        T_o2w = _jax_compose_transform(object_rvecs, object_tvecs)

        T_total = jnp.matmul(T_w2c[:, None, ...], T_o2w[None, :, ...])

        C, P = T_total.shape[:2]
        T_flat = T_total.reshape(C * P, 4, 4)

        # K and D also need broadcasting
        K_flat = jnp.repeat(K, P, axis=0)  # (C*P, 3, 3)
        D_flat = jnp.repeat(D, P, axis=0)

        reproj_flat, z_flat = _jax_project(object_points, T_flat, K_flat, D_flat, dist_model)

        # reshape back
        reproj = reproj_flat.reshape(C, P, -1, 2)
        z_depths = z_flat.reshape(C, P, -1)

    else:
        # Scaffolding / Non-rigid object workflow: object_points is (P*N, 3) or just (N, 3) shared
        # This path assumes object_points matches structure directly (world coordinates)
        pts_reshaped = object_points.reshape(cfg['nb_frames'], cfg['nb_points'], 3)

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

    jax.debug.print("Component RMS: {x:.3f} px, MRE: {y:.3f} px", x=comp_rms, y=mre)

    # ──────────────────────────────────────────────────────────────────────────────
    # Priors with covariance matrices
    # ──────────────────────────────────────────────────────────────────────────────

    if prior_info is not None:
        origin_cam_idx = cfg['origin_cam_idx']

        # Extrinsics priors using SRIM
        if prior_info.extrinsics_srim is not None:
            all_indices = jnp.arange(cfg['nb_cams'])
            optim_indices = jnp.delete(all_indices, origin_cam_idx)

            # Current extrinsics for optimizable cameras: (C-1, 6)
            current_extr = jnp.concatenate([camera_rvecs[optim_indices], camera_tvecs[optim_indices]], axis=-1)

            # Deviation from prior mean
            delta = current_extr - prior_info.extrinsics_mean

            # Apply SRIM: residual = SRIM @ delta for each camera
            # extrinsics_srim is (C-1, 6, 6), delta is (C-1, 6)
            extr_residuals = jnp.einsum('cij,cj->ci', prior_info.extrinsics_srim, delta)
            all_residuals.append(extr_residuals.ravel())

        # Intrinsics priors using SRIM
        if prior_info.intrinsics_srim is not None:
            # Extract current intrinsics as vector matching the mean format
            current_intr = _extract_current_intrinsics_vector(K, D, spec)

            # Deviation from prior mean
            delta = current_intr - prior_info.intrinsics_mean

            # Apply SRIM: residual = SRIM @ delta for each intrinsics set
            # intrinsics_srim is (nb_sets, dim, dim), delta is (nb_sets, dim)
            intr_residuals = jnp.einsum('sij,sj->si', prior_info.intrinsics_srim, delta)
            all_residuals.append(intr_residuals.ravel())

    return jnp.concatenate(all_residuals)


# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Jacobian sparsity creation, and Covariance from Jacobian (pure NumPy)
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────


def jacobian_sparsity(spec, prior_info: Optional[PriorInfo]) -> csr_matrix:
    """
    Pure NumPy. Creates the Jacobian sparsity matrix.
    This runs once at setup.

    When covariance priors are used, the prior blocks are fully dense (not diagonal)
    since the SRIM couples all parameters within each block.
    """
    cfg = spec['config']
    C, P, N = cfg['nb_cams'], cfg['nb_frames'], cfg['nb_points']
    origin_cam_idx = cfg.get('origin_cam_idx', 0)

    nb_residuals = 2 * P * C * N
    nb_params = spec['total_size']
    S = lil_matrix((nb_residuals, nb_params), dtype=bool)

    is_shared = cfg['shared_intrinsics'] and C > 1
    nb_intr_sets = 1 if is_shared else C
    optim_cam_indices = np.delete(np.arange(C), origin_cam_idx)
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
                if 'extrinsics' in spec['blocks'] and c != origin_cam_idx:
                    optim_pos = cam_idx_to_optim_pos[c]
                    info = spec['blocks']['extrinsics']
                    nb_optim_cams = C - 1
                    r_col = info['offset'] + optim_pos * 3
                    t_col = info['offset'] + (nb_optim_cams * 3) + optim_pos * 3
                    S[row:row + 2, r_col:r_col + 3] = 1
                    S[row:row + 2, t_col:t_col + 3] = 1

                # Dependency on structure poses
                if 'object_poses' in spec['blocks']:
                    info = spec['blocks']['object_poses']
                    r_col = info['offset'] + p * 3
                    t_col = info['offset'] + (P * 3) + p * 3
                    S[row:row + 2, r_col:r_col + 3] = 1
                    S[row:row + 2, t_col:t_col + 3] = 1

                # Dependency on 3D points
                if 'object_points' in spec['blocks']:
                    info = spec['blocks']['object_points']

                    if not cfg['fix_object_poses']:
                        # Rigid object (calibration board) whose poses are optimized: all frames refer to same N points
                        point_idx_in_optim_vector = n

                    else:
                        # For both scaffolding and non-rigid objects (animal): each frame has a unique set of N points
                        point_idx_in_optim_vector = p * N + n

                    col = info['offset'] + point_idx_in_optim_vector * 3
                    S[row:row + 2, col:col + 3] = 1

    # Add depth barrier residuals (same sparsity as reprojection)
    nb_depth_residuals = P * C * N
    S.resize(nb_residuals + nb_depth_residuals, nb_params)

    for c in range(C):
        for p in range(P):
            for n in range(N):
                reproj_row = 2 * (c * P * N + p * N + n)
                depth_row = nb_residuals + (c * P * N + p * N + n)
                # Copy sparsity pattern from reprojection row
                S[depth_row, :] = S[reproj_row, :]

    curr_row = nb_residuals + nb_depth_residuals

    # Add sparsity for covariance-based priors
    if prior_info is not None:
        # Extrinsics priors: each camera's 6 residuals depend on all 6 extrinsics params
        # But with SRIM, each residual element depends on the full 6-vector
        if prior_info.extrinsics_srim is not None and 'extrinsics' in spec['blocks']:
            nb_optim_cams = C - 1
            info = spec['blocks']['extrinsics']

            # For each optimizable camera, we have 6 residuals
            # Each residual depends on ALL 6 parameters of that camera (dense coupling from SRIM)
            S.resize(curr_row + 6 * nb_optim_cams, nb_params)

            for cam_pos in range(nb_optim_cams):
                # Residual rows for this camera
                res_start = curr_row + cam_pos * 6

                # Parameter columns for this camera's rvec and tvec
                r_col = info['offset'] + cam_pos * 3
                t_col = info['offset'] + nb_optim_cams * 3 + cam_pos * 3

                # Dense block: all 6 residuals depend on all 6 params
                S[res_start:res_start + 6, r_col:r_col + 3] = True
                S[res_start:res_start + 6, t_col:t_col + 3] = True

            curr_row += 6 * nb_optim_cams

        # Intrinsics priors: each set's residuals depend on all its params
        if prior_info.intrinsics_srim is not None:
            # Determine total intrinsics dimension per set
            nb_K = 0
            nb_D = 0

            if 'K' in spec['blocks']:
                nb_K = 3 if cfg['fix_aspect_ratio'] else 4

            if 'D' in spec['blocks']:
                nb_D = cfg['n_d_size']

            intr_dim = nb_K + nb_D

            S.resize(curr_row + nb_intr_sets * intr_dim, nb_params)

            for set_idx in range(nb_intr_sets):
                res_start = curr_row + set_idx * intr_dim

                # All residuals for this set depend on all params for this set
                if 'K' in spec['blocks']:
                    info_K = spec['blocks']['K']
                    size_per_set_K = info_K['size'] // nb_intr_sets
                    col_K = info_K['offset'] + set_idx * size_per_set_K
                    S[res_start:res_start + intr_dim, col_K:col_K + size_per_set_K] = True

                if 'D' in spec['blocks']:
                    info_D = spec['blocks']['D']
                    size_per_set_D = info_D['size'] // nb_intr_sets
                    col_D = info_D['offset'] + set_idx * size_per_set_D
                    S[res_start:res_start + intr_dim, col_D:col_D + size_per_set_D] = True

    return S.tocsr()


def covariance_from_jacobian(
        result,
        jac_fn,
        spec: Dict,
        residual_variance: Optional[float] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Estimates parameter covariance from the Jacobian at the solution.

    The covariance is computed as: Cov = sigma^2 * (J^T @ J)^{-1}
    where sigma^2 is the residual variance.

    Args:
        result: scipy.optimize.OptimizeResult from least_squares
        jac_fn: Function to compute the Jacobian at a point
        spec: Parameter specification dictionary
        residual_variance: If None, estimated from residuals. Otherwise use this value.

    Returns:
        full_covariance: Full parameter covariance matrix
        covariance_intrinsics: Optional per-set intrinsics covariance (nb_sets, dim, dim), or None
        covariance_extrinsics: Optional per-camera extrinsics covariance (C-1, 6, 6), or None
    """
    cfg = spec['config']

    # Compute Jacobian at solution
    J = np.array(jac_fn(result.x))

    # Compute J^T @ J (the Hessian approximation)
    JtJ = J.T @ J

    # Estimate residual variance if not provided
    if residual_variance is None:
        n_residuals = len(result.fun)
        n_params = len(result.x)
        dof = max(1, n_residuals - n_params)
        residual_variance = np.sum(result.fun ** 2) / dof

    # Regularize and invert
    reg = 1e-8 * np.trace(JtJ) / JtJ.shape[0]  # TODO: this will likely be slow or memory error for large nb of parameters
    JtJ_reg = JtJ + reg * np.eye(JtJ.shape[0])

    try:
        full_cov = residual_variance * np.linalg.inv(JtJ_reg)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse
        full_cov = residual_variance * np.linalg.pinv(JtJ_reg)

    # Extract block covariances
    C = cfg['nb_cams']
    is_shared = cfg['shared_intrinsics'] and C > 1
    nb_intr_sets = 1 if is_shared else C

    # Extrinsics covariance
    if 'extrinsics' in spec['blocks']:
        info = spec['blocks']['extrinsics']
        nb_optim_cams = C - 1

        extr_cov = np.zeros((nb_optim_cams, 6, 6))

        for cam_pos in range(nb_optim_cams):
            # rvec indices
            r_start = info['offset'] + cam_pos * 3
            r_end = r_start + 3

            # tvec indices
            t_start = info['offset'] + nb_optim_cams * 3 + cam_pos * 3
            t_end = t_start + 3

            # Build 6x6 covariance for this camera
            indices = list(range(r_start, r_end)) + list(range(t_start, t_end))
            extr_cov[cam_pos] = full_cov[np.ix_(indices, indices)]
    else:
        extr_cov = None

    # Intrinsics covariance
    if 'K' in spec['blocks'] or 'D' in spec['blocks']:
        nb_K = 0
        nb_D = 0

        if 'K' in spec['blocks']:
            nb_K = 3 if cfg['fix_aspect_ratio'] else 4
        if 'D' in spec['blocks']:
            nb_D = cfg['n_d_size']

        intr_dim = nb_K + nb_D
        intr_cov = np.zeros((nb_intr_sets, intr_dim, intr_dim))

        for set_idx in range(nb_intr_sets):
            indices = []

            if 'K' in spec['blocks']:
                info_K = spec['blocks']['K']
                size_per_set_K = info_K['size'] // nb_intr_sets
                start_K = info_K['offset'] + set_idx * size_per_set_K
                indices.extend(range(start_K, start_K + size_per_set_K))

            if 'D' in spec['blocks']:
                info_D = spec['blocks']['D']
                size_per_set_D = info_D['size'] // nb_intr_sets
                start_D = info_D['offset'] + set_idx * size_per_set_D
                indices.extend(range(start_D, start_D + size_per_set_D))

            intr_cov[set_idx] = full_cov[np.ix_(indices, indices)]
    else:
        intr_cov = None

    return full_cov, intr_cov, extr_cov


# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Main run
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

def run_bundle_adjustment(
        K: jnp.ndarray,
        D: jnp.ndarray,
        cameras_poses: jnp.ndarray,
        images_sizes_hw: ArrayLike,
        points2d_observed: jnp.ndarray,
        visibility_mask: jnp.ndarray,
        object_points: Optional[jnp.ndarray] = None,
        object_poses: Optional[jnp.ndarray] = None,
        origin_cam_idx: int = 0,
        distortion_model: DistortionModel = 'standard',
        fix_intrinsics: bool = False,
        fix_extrinsics: bool = False,
        fix_object_points: bool = False,
        fix_object_poses: bool = True,
        fix_aspect_ratio: bool = False,
        shared_intrinsics: bool = False,
        radial_penalty: float = 2.0,
        covariance_intrinsics: Optional[np.ndarray] = None,
        covariance_extrinsics: Optional[np.ndarray] = None,
        tolerance: float = 1e-8,
        max_nfev: int = 500,
        stage: Optional[Any] = None,
) -> Tuple[bool, Dict]:
    """
    Run bundle adjustment with optional covariance-based priors.

    Args:
        K: Initial camera matrices (C, 3, 3)
        D: Initial distortion coefficients (C, nb_dist_coeffs)
        cameras_poses: Initial camera poses as T matrices (C, 4, 4)
        images_sizes_hw: Image sizes (C, 2) or (2,) if shared
        points2d_observed: Observed 2D points (C, P, N, 2)
        visibility_mask: Visibility mask (C, P, N)
        object_points: Initial 3D points (N, 3), or None
        object_poses: Initial object poses as T matrices (P, 4, 4), or None
        origin_cam_idx: Index of camera used as world origin (fixed pose)
        distortion_model: Distortion model type
        fix_intrinsics: If True, don't optimize intrinsics
        fix_extrinsics: If True, don't optimize extrinsics
        fix_object_points: If True, don't optimize 3D points
        fix_object_poses: If True, don't optimize object poses
        fix_aspect_ratio: If True, fix fx = fy
        shared_intrinsics: If True, share intrinsics across cameras
        radial_penalty: Weight for radial distance penalty (radial distance from cx, cy)
        covariance_intrinsics: Optional covariance matrix for intrinsics. If None, no priors are applied.
        covariance_extrinsics: Optional covariance matrix for extrinsics. If None, no priors are applied.
        tolerance: Convergence tolerance
        max_nfev: Maximum function evaluations
        stage: Optional BA run number or label, for progress report only

    Returns:
        (success, results_dict) where results_dict contains:
        - K_opt, D_opt: Optimized intrinsics
        - camera_poses_opt: Optimized camera poses (C, 4, 4)
        - object_poses_opt: Optimized object poses (P, 4, 4), or None
        - object_points_opt: Optimized 3D points, or None
        - covariance_intrinsics: Posterior extrinsics covariance
        - covariance_extrinsics: Posterior intrinsics covariance
    """

    # Setup (NumPy)
    C, P, N, _ = points2d_observed.shape
    
    camera_rvecs, camera_tvecs = geom.decompose_transform_matrix(cameras_poses)
    object_rvecs, object_tvecs = geom.decompose_transform_matrix(object_poses) if object_poses is not None else (None, None)

    nb_pts_opt = object_points.shape[0] if object_points is not None and not fix_object_points else 0
    
    # Create initial params dict
    initial_params = {
        'K': np.array(K).copy(),
        'D': np.array(D).copy(),
        'camera_rvecs': np.array(camera_rvecs).copy(),
        'camera_tvecs': np.array(camera_tvecs).copy(),
        'object_rvecs': np.array(object_rvecs).copy(),
        'object_tvecs': np.array(object_tvecs).copy()
    }
    
    # Create specs dict
    spec = _get_parameter_spec(
        nb_cams=C,
        nb_frames=P,
        nb_points=N,
        nb_points_to_optim=nb_pts_opt,
        origin_cam_idx=origin_cam_idx,
        distortion_model=distortion_model,
        fix_intrinsics=fix_intrinsics,
        fix_extrinsics=fix_extrinsics,
        fix_object_points=fix_object_points,
        fix_object_poses=fix_object_poses,
        fix_aspect_ratio=fix_aspect_ratio,
        shared_intrinsics=shared_intrinsics
    ) 

    # Parameters packing (NumPy)
    nb_dist_coeffs = spec['config']['n_d_size']
    if D.shape[1] < nb_dist_coeffs:
        D = np.pad(D, ((0, 0), (0, nb_dist_coeffs - D.shape[1])))

    x0 = _pack_params_numpy(
        spec=spec,
        K=K,
        D=D,
        camera_rvecs=camera_rvecs,
        camera_tvecs=camera_tvecs,
        object_points=object_points,
        object_rvecs=object_rvecs,
        object_tvecs=object_tvecs
    )

    # Determine parameters scales
    scales = _get_parameter_scales(
        spec=spec,
        initial_params=initial_params,
        images_sizes_hw=images_sizes_hw
    )
    
    lb, ub = _get_bounds(
        spec=spec,
        images_sizes_hw=images_sizes_hw
    )
    x0 = np.clip(x0, lb, ub)

    # Prepare fixed params and weights (JAX-ready)
    points_weights_np = (visibility_mask > 0).astype(np.float32)

    # Calculate advanced weights using the JAX kernel
    img_pts_jax = jnp.array(points2d_observed)
    points_weights_jax = jax_residual_weights(
        points2d=img_pts_jax,
        visibility_mask=jnp.array(points_weights_np),
        K=jnp.array(K),
        gamma=radial_penalty
    )

    # Compute near_plane for barrier loss: median Z depth of the initial points relative to the origin camera
    near_plane = 1e-3
    if object_points is not None:
        try:
            # Get world -> camera transform for the origin camera
            T_w2c_origin = geom.invert_transform(cameras_poses[origin_cam_idx])

            # Project a subset of points into this camera's frame
            pts_sub = object_points[:min(100, len(object_points))]
            pts_cam = geom.transform_points(pts_sub, T_w2c_origin)

            # Compute median Z
            median_z = float(np.median(pts_cam[:, 2]))

            if median_z > 1e-6:
                near_plane = median_z * 1e-3  # 0.1% of scene scale
        except Exception:
            # use default if anything fails
            pass

    fixed_params = {
        'K': jnp.array(K),
        'D': jnp.array(D),
        'camera_rvecs': jnp.array(camera_rvecs),
        'camera_tvecs': jnp.array(camera_tvecs),
        'object_rvecs': jnp.array(object_rvecs) if object_rvecs is not None else None,
        'object_tvecs': jnp.array(object_tvecs) if object_tvecs is not None else None,
        'object_points': jnp.array(object_points) if object_points is not None else None,
        'near_plane': near_plane
    }

    # Prepare prior information from covariances
    prior_info = prepare_prior_info(
        spec=spec,
        initial_params=initial_params,
        covariance_intrinsics=covariance_intrinsics,
        covariance_extrinsics=covariance_extrinsics,
    )

    # Make jacobian sparsity matrix
    S = jacobian_sparsity(spec, prior_info)

    # JIT compilation
    # The static dictionary 'spec' and fixed data are baked in
    cost_fn_jitted = jit(partial(cost_function,
                                 fixed_params=fixed_params,
                                 image_points=img_pts_jax,
                                 points_weights=points_weights_jax,
                                 spec=spec,
                                 prior_info=prior_info))

    jac_fn_jitted = jit(jax.jacfwd(partial(cost_function,
                                           fixed_params=fixed_params,
                                           image_points=img_pts_jax,
                                           points_weights=points_weights_jax,
                                           spec=spec,
                                           prior_info=prior_info)))

    # Scipy wrappers (they bridge the NumPy -> JAX -> NumPy merry-go-round)
    def fun_wrapped(x):
        return np.array(cost_fn_jitted(x))

    def jac_wrapped(x):
        return np.array(jac_fn_jitted(x))

    # Setup pretty progress report
    def status_formatter(match):
        rms, mre = match.groups()
        return f"| Component RMS: {rms} px | MRE: {mre} px"
    title = f'Stage: {stage}' if stage else 'Bundle Adjustment'
    with alive_bar(title=title, length=20, force_tty=True, stats=False, monitor=False, spinner=None) as bar:
        with CallbackOutputStream(bar,
                pattern=r"Component RMS: ([0-9\.]+) px, MRE: ([0-9\.]+) px",
                refresh_rate=10,    # update text every 10 evals
                formatter=status_formatter
        ):

            # Optimize
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
    K_opt, D_opt, cam_rvecs_opt, cam_tvecs_opt, obj_points_opt, obj_rvecs_opt, obj_tvecs_opt = _unpack_params_jax(
        spec, 
        jnp.array(res.x),   # unpacking assumes x is a JAX array so we cast res.x just one time, it's fine
        fixed_params
    )

    # Recompose matrices using standard geometry lib
    cam_poses_opt = geom.compose_transform_matrix(cam_rvecs_opt, cam_tvecs_opt)
    obj_poses_opt = geom.compose_transform_matrix(obj_rvecs_opt, obj_tvecs_opt) if obj_rvecs_opt is not None else None

    results = {
        'K_opt': np.asarray(K_opt),
        'D_opt': np.asarray(D_opt),
        'camera_poses_opt': np.asarray(cam_poses_opt),
        'object_poses_opt': np.asarray(obj_poses_opt),
        'object_points_opt': np.asarray(obj_points_opt)
    }

    # Estimate posterior covariance
    try:
        full_cov, cov_intr_final, cov_extr_final = covariance_from_jacobian(res, jac_wrapped, spec)
        
        if cov_intr_final is not None:
            results['covariance_intrinsics'] = cov_intr_final
        
        if cov_extr_final is not None:
            results['covariance_extrinsics'] = cov_extr_final

    except Exception as e:
        warnings.warn(f"Failed to estimate covariance: {e}")

    return res.success, results