from functools import partial
from typing import Tuple, Dict
from .backend import USE_JAX, xp, jit, lax, vmap, _tiny
from .transforms import quaternions_angular_distance, rodrigues, inverse_rodrigues


@jit
def weighted_median(data: xp.ndarray, weights: xp.ndarray) -> xp.ndarray:
    """
    Computes the weighted median of data.
    Works for both continuous weights and binary masks (0/1)

    Args:
        data: (N,) values
        weights: (N,) weights or mask

    Returns:
        The weighted median value
    """

    sort_idx = xp.argsort(data)
    sorted_data = data[sort_idx]
    sorted_weights = weights[sort_idx]

    cumsum = xp.cumsum(sorted_weights)
    total_weight = cumsum[-1]

    # We want the first index where cumulative weight >= 0.5 * total
    median_idx = xp.searchsorted(cumsum, 0.5 * total_weight)

    # Clip to ensure valid index (handles empty/zero-weight cases)
    median_idx = xp.clip(median_idx, 0, data.shape[0] - 1)

    return sorted_data[median_idx]


@jit
def find_rigid_transform(
        points_A: xp.ndarray,
        points_B: xp.ndarray
) -> Tuple[xp.ndarray, xp.ndarray]:
    """
    Estimates the rigid transformation (rotation R, translation t) between two
    sets of corresponding 3D points using the Kabsch algorithm.
    Finds T such that: B ~ T(A) = R @ A + t

    Args:
        points_A: Source points (..., N, 3)
        points_B: Destination points (..., N, 3)

    Returns:
        R: Rotation matrix (..., 3, 3)
        t: Translation vector (..., 3)
    """

    # Compute centroids (N dim)
    centroid_A = xp.mean(points_A, axis=-2, keepdims=True)
    centroid_B = xp.mean(points_B, axis=-2, keepdims=True)

    # Center the points
    A_centered = points_A - centroid_A
    B_centered = points_B - centroid_B

    # Compute Covariance Matrix H
    # swap axes to perform the transpose of A_centered: (..., 3, N) @ (..., N, 3) -> (..., 3, 3)
    H = xp.matmul(xp.swapaxes(A_centered, -1, -2), B_centered)

    # Find the rotation using SVD
    U, S, Vt = xp.linalg.svd(H)

    # Compute Rotation R = Vt.T @ U.T
    # Note: V from svd is V^T (Vt), so we need Vt.T @ U.T
    R = xp.matmul(xp.swapaxes(Vt, -1, -2), xp.swapaxes(U, -1, -2))

    # Reflection Correction (if det(R) < 0)
    det_R = xp.linalg.det(R)

    # Need to scale the last column of U by sign(det_R) * 1.0 (since det is -1 or 1 usually)
    ones = xp.ones_like(det_R)
    s = xp.stack([ones, ones, det_R], axis=-1)  # (..., 3)

    # Apply scaling to U before recomputing R
    # U is (..., 3, 3), we want to scale the 3rd column
    s_mat = s[..., None, :]
    U_corrected = U * s_mat

    R_corrected = xp.matmul(xp.swapaxes(Vt, -1, -2), xp.swapaxes(U_corrected, -1, -2))

    # Compute translation
    # t = centroid_B - R @ centroid_A
    cA_t = xp.swapaxes(centroid_A, -1, -2)
    cB_t = xp.swapaxes(centroid_B, -1, -2)

    t_t = cB_t - xp.matmul(R_corrected, cA_t)
    t = t_t.squeeze(-1)  # (..., 3)

    return R_corrected, t


@jit
def find_affine_transform(
        points_A: xp.ndarray,
        points_B: xp.ndarray
) -> Tuple[xp.ndarray, xp.ndarray]:
    """
    Estimates the affine transformation (A, t) between two sets of corresponding 3D points.
    Finds T such that: B ~ T(A) = A @ A.T + t. Uses linear least squares.

    Args:
        points_A: Source points (..., M, 3)
        points_B: Destination points (..., M, 3)

    Returns:
        A_mat: The affine transformation matrix (..., 3, 3)
        t_vec: The translation vector (..., 3)
    """

    # Construct Design Matrix X: [x, y, z, 1]
    ones = xp.ones(points_A.shape[:-1] + (1,), dtype=points_A.dtype)
    X = xp.concatenate([points_A, ones], axis=-1)  # (..., M, 4)

    # Solve T = (X^T X)^-1 X^T B
    XT = xp.swapaxes(X, -1, -2)  # (..., 4, M)

    XTX = xp.matmul(XT, X)
    XTB = xp.matmul(XT, points_B)

    # Solve (..., 4, 4) * T = (..., 4, 3)
    eye_reg = xp.eye(4) * 1e-6  # small regularization to diagonal for stability
    T = xp.linalg.solve(XTX + eye_reg, XTB)  # (..., 4, 3)

    # Extract A and t
    # T is [ A.T ]
    #      [  t  ]
    A_mat_T = T[..., :3, :]  # (..., 3, 3)
    t_vec = T[..., 3, :]  # (..., 3)

    A_mat = xp.swapaxes(A_mat_T, -1, -2)

    return A_mat, t_vec


@jit
def interpolate3d(
        points3d: xp.ndarray,
        visibility_mask: xp.ndarray,
        points3d_theoretical: xp.ndarray
) -> xp.ndarray:
    """
    Fills in missing points in a cloud using a theoretical template.
    Uses Weighted Least Squares (via Normal Equations) to align the template to visible points.

    Args:
        points3d: Observed points (N, 3)
        visibility_mask: Visibility (N,)
        points3d_theoretical: Template points (N, 3)

    Returns:
        Completed point cloud (N, 3)
    """

    N = points3d.shape[0]
    mask = visibility_mask.astype(xp.float32)

    # Design matrix [X_th | 1]
    ones = xp.ones((N, 1), dtype=points3d.dtype)
    A = xp.concatenate([points3d_theoretical, ones], axis=1)  # (N, 4)

    # We want to solve A * T = Y, but weighted by mask
    # W is diagonal of mask
    # Normal eq: (A.T W A) T = A.T W Y

    # Multiply A and Y by sqrt(weights) (which is just mask (0 or 1) here)
    # Broadcasting mask: (N, 1)
    mask_col = mask[:, None]

    A_weighted = A * mask_col
    Y_weighted = points3d * mask_col

    # Solve
    AT_W_A = A_weighted.T @ A_weighted
    AT_W_Y = A_weighted.T @ Y_weighted

    # Regularize slightly
    eps = 1e-6 * xp.eye(4)
    T = xp.linalg.solve(AT_W_A + eps, AT_W_Y)  # (4, 3)

    # Predict all
    filled_all = A @ T

    # Combine original and filled
    return xp.where(mask_col.astype(bool), points3d, filled_all)


@jit
def huber_weight(residual_norm: xp.ndarray, delta: float = 1.0) -> xp.ndarray:
    """
    Compute Huber weight for each ||error||

        w = 1                   if ||error|| <= delta
        w = delta / ||error||   if ||error|| > delta
    """
    return xp.where(residual_norm <= delta, 1.0, delta / (residual_norm + 1e-12))


@jit
def translation_average(
        t_samples: xp.ndarray,  # (M, 3)
        num_iters: int = 3,
        delta: float = 1.0
) -> xp.ndarray:
    """
    Computes Huber-weighted mean of translations using IRLS.
    """
    M = t_samples.shape[0]

    # Handle empty input safely
    if M == 0:
        return xp.zeros((3,), dtype=t_samples.dtype)

    # Initial guess: Median
    t0 = xp.median(t_samples, axis=0)

    def body_fn(t_curr):
        res = t_samples - t_curr  # (M, 3)
        norms = xp.linalg.norm(res, axis=1)  # (M,)
        w = huber_weight(norms, delta)  # (M,)

        sum_w = xp.sum(w)
        # Normalise weights safely
        w_norm = w / (sum_w + 1e-12)

        t_next = xp.sum(w_norm[:, None] * t_samples, axis=0)
        return t_next

    # Using the shim for the loop
    def loop_body(i, t_val):
        return body_fn(t_val)

    t_final = lax.fori_loop(0, num_iters, loop_body, t0)
    return t_final


@jit
def quaternion_average(quats: xp.ndarray, weights: xp.ndarray = None) -> xp.ndarray:
    """
    Averages quaternions using Eigen-decomposition (Markley's method).
    Handles q = -q ambiguity.
    """
    if weights is None:
        weights = xp.ones(quats.shape[0], dtype=quats.dtype)

    # Normalise weights
    weights = weights / (xp.sum(weights) + 1e-8)

    # Align quaternions to handle q = -q ambiguity
    # Pick the one with largest weight as reference
    idx_ref = xp.argmax(weights)
    q_ref = quats[idx_ref]

    # Dot product with reference
    dots = xp.sum(q_ref * quats, axis=-1)
    flip = xp.sign(dots)
    # if dot is 0, sign is 0, which destroys data, so default to 1
    flip = xp.where(flip == 0, 1.0, flip)

    quats_aligned = quats * flip[:, None]

    # Build M = sum(w_i * q_i * q_i.T)
    # Weighted sum over N
    q_outer = quats_aligned[..., :, None] * quats_aligned[..., None, :]  # (N, 4, 4)
    M = xp.sum(weights[:, None, None] * q_outer, axis=0)

    # Eigen decomposition
    vals, vecs = xp.linalg.eigh(M)  # eigh returns eigenvalues in ascending order

    # The eigenvector with largest eigenvalue is the average
    avg_quat = vecs[:, -1]

    # Ensure positive w for canonical representation
    s = xp.sign(avg_quat[0])
    s = xp.where(s < 0, -1.0, 1.0)

    return avg_quat * s


def filter_rt_samples(
        rt_stack: xp.ndarray,  # (N, 7) [quat, trans]
        ang_thresh: float = xp.pi / 6.0,
        trans_thresh: float = 1.0,
        num_iters: int = 3
) -> Tuple[xp.ndarray, xp.ndarray, bool]:
    """
    Robustly averages poses (quaternion, translation) using IRLS and outlier rejection.
    """

    # Clean data
    finite_mask = xp.all(xp.isfinite(rt_stack), axis=1)
    weights_init = finite_mask.astype(xp.float32)

    quats = rt_stack[:, :4]
    trans = rt_stack[:, 4:]

    # Check if we have any valid data
    count = xp.sum(weights_init)

    # Initial guess

    # Rotation: Eigen-analysis (weighted average equivalent for quats)
    q_curr = quaternion_average(quats, weights_init)

    # Translation: Masked median (dimension-wise)
    # JAX vmap to apply the 1D median function to each column (x, y, z)
    # axis 0 is batch (N), axis 1 is coords (3), so we move coords to front
    trans_T = trans.T  # (3, N)

    def _med(d):
        return weighted_median(d, weights_init)

    if USE_JAX:
        # JAX vmap magic
        t_curr = vmap(_med)(trans_T)
    else:
        # numpy fallback
        t_curr = xp.array([_med(trans_T[0]), _med(trans_T[1]), _med(trans_T[2])])

    # Fallback if median fails (e.g. all weights were 0): we use [0, 0, 0]
    t_curr = xp.where(count > 0, t_curr, xp.zeros(3))

    def body_fn(state):
        q_c, t_c, _ = state

        # Calculate errors
        q_c_broad = xp.broadcast_to(q_c, quats.shape)
        ang_errs = quaternions_angular_distance(quats, q_c_broad)

        trans_errs = xp.linalg.norm(trans - t_c, axis=1)

        # Determine inliers
        is_inlier = (ang_errs <= ang_thresh) & (trans_errs <= trans_thresh)
        new_weights = is_inlier.astype(xp.float32) * weights_init

        total_w = xp.sum(new_weights)
        has_data = total_w > 1e-4

        # If we lost all data, keep previous estimate

        # Update Q
        q_new = quaternion_average(quats, new_weights)
        q_next = xp.where(has_data, q_new, q_c)

        # Update T
        w_norm = new_weights / (total_w + 1e-12)
        t_new = xp.sum(w_norm[:, None] * trans, axis=0)
        t_next = xp.where(has_data, t_new, t_c)

        return q_next, t_next, has_data

    def loop_wrapper(i, state):
        return body_fn(state)

    # Initial state
    init_state = (q_curr, t_curr, count > 0)

    q_final, t_final, valid = lax.fori_loop(0, num_iters, loop_wrapper, init_state)

    return q_final, t_final, valid


@jit
def rays_intersection_3d(
        ray_origins: xp.ndarray,  # (C, 3)
        ray_directions: xp.ndarray  # (C, 3)
) -> xp.ndarray:
    """
    Finds the 3D point that minimizes the sum of squared distances to a set of rays.
    """

    # Per-ray projection matrix onto the plane orthogonal to the direction
    I = xp.eye(3)
    # Projections: I - v v.T
    P_i = I - xp.einsum('ci,cj->cij', ray_directions, ray_directions)

    # Sum of matrices: A = sum(P_i)
    A = xp.sum(P_i, axis=0)  # (3, 3)

    # RHS: b = sum(P_i @ origin)
    b_vecs = xp.einsum('cij,cj->ci', P_i, ray_origins)
    b = xp.sum(b_vecs, axis=0)  # (3,)

    # Solve 3x3
    # lstsq makes it robust against parallel rays (singular A)
    pt, _, _, _ = xp.linalg.lstsq(A, b, rcond=None)
    return pt


@jit
def ray_intersection_AABB(
        ray_origin: xp.ndarray,
        ray_dir: xp.ndarray,
        aabb_min: xp.ndarray,
        aabb_max: xp.ndarray
) -> Tuple[xp.ndarray, xp.ndarray, xp.ndarray]:
    """
    Slab method for AABB intersection. Supports batched rays.

    Args:
        ray_origin: Ray origins (..., 3)
        ray_dir: Ray directions (..., 3)
        aabb_min: Box min corner (3,)
        aabb_max: Box max corner (3,)

    Returns:
        p_near: Entry points (..., 3)
        p_far: Exit points (..., 3)
        has_intersection: Boolean mask
    """
    # Safe inverse direction
    is_zero = xp.abs(ray_dir) < _tiny
    safe_dir = xp.where(is_zero, _tiny, ray_dir)
    dir_inv = 1.0 / safe_dir

    # Broadcast AABB to rays
    t1 = (aabb_min - ray_origin) * dir_inv
    t2 = (aabb_max - ray_origin) * dir_inv

    # Find entering and exiting planes
    t_min = xp.max(xp.minimum(t1, t2), axis=-1)
    t_max = xp.min(xp.maximum(t1, t2), axis=-1)

    has_intersection = t_min < t_max

    # Compute points
    p_near = ray_origin + t_min[..., None] * ray_dir
    p_far = ray_origin + t_max[..., None] * ray_dir

    # Masking
    nan_v = xp.full_like(p_near, xp.nan)
    mask = has_intersection[..., None]

    p_near = xp.where(mask, p_near, nan_v)
    p_far = xp.where(mask, p_far, nan_v)

    return p_near, p_far, has_intersection


@partial(jit, static_argnames=['error_threshold_px', 'percentile'])
def reliability_bounds_3d(
        world_points: xp.ndarray,
        all_errors: xp.ndarray,
        error_threshold_px: float = 1.0,
        percentile: float = 1.0
) -> Dict[str, Tuple[float, float]]:
    """
    Computes a bounding box in world coordinates from a cloud of 3D points.
    Reliability is determined by the mean reprojection error.

    Args:
        world_points: Cloud of 3D points (P, N, 3)
        all_errors: Error values for each point (C, P, N)
        error_threshold_px: Max mean error for a point to be considered reliable
        percentile: Percentile to clip outliers when computing bounds

    Returns:
        Dict with 'x', 'y', 'z' keys containing (min, max) bounds
    """

    # Mean error per point instance
    mean_error = xp.nanmean(all_errors, axis=0)  # (P, N)

    # Mask
    reliable_mask = mean_error < error_threshold_px
    count = xp.sum(reliable_mask)

    # Filter points (set unreliable to NaN)
    pts_clean = xp.where(reliable_mask[..., None], world_points, xp.nan)

    # Compute bounds (percentiles ignore NaNs)
    q_low = percentile
    q_high = 100.0 - percentile

    x_min = xp.nanpercentile(pts_clean[..., 0], q_low)
    x_max = xp.nanpercentile(pts_clean[..., 0], q_high)
    y_min = xp.nanpercentile(pts_clean[..., 1], q_low)
    y_max = xp.nanpercentile(pts_clean[..., 1], q_high)
    z_min = xp.nanpercentile(pts_clean[..., 2], q_low)
    z_max = xp.nanpercentile(pts_clean[..., 2], q_high)

    # Check reliability (returns NaN if count too low)
    is_valid = count >= 3

    def gate(val):
        return xp.where(is_valid, val, xp.nan)

    return {
        'x': (gate(x_min), gate(x_max)),
        'y': (gate(y_min), gate(y_max)),
        'z': (gate(z_min), gate(z_max))
    }


@partial(jit, static_argnames=['error_threshold_px', 'iqr_factor'])
def reliability_bounds_3d_iqr(
        world_points: xp.ndarray,
        all_errors: xp.ndarray,
        error_threshold_px: float = 1.0,
        iqr_factor: float = 1.5
) -> Dict[str, Tuple[float, float]]:
    """
    Computes a robust bounding box using the Interquartile Range method.

    Args:
        world_points: Cloud of 3D points (P, N, 3)
        all_errors: Error values (C, P, N)
        error_threshold_px: Max mean error
        iqr_factor: Multiplier for IQR (default 1.5)

    Returns:
        Dict with 'x', 'y', 'z' bounds
    """

    # Average errors across observations (cameras) for each point instance
    mean_error = xp.nanmean(all_errors, axis=0)

    reliable_mask = mean_error < error_threshold_px
    count = xp.sum(reliable_mask)

    pts_clean = xp.where(reliable_mask[..., None], world_points, xp.nan)

    # Quartiles
    qs = xp.array([25.0, 75.0])

    qx = xp.nanpercentile(pts_clean[..., 0], qs)
    qy = xp.nanpercentile(pts_clean[..., 1], qs)
    qz = xp.nanpercentile(pts_clean[..., 2], qs)

    iqr_x = qx[1] - qx[0]
    iqr_y = qy[1] - qy[0]
    iqr_z = qz[1] - qz[0]

    x_min, x_max = qx[0] - iqr_factor * iqr_x, qx[1] + iqr_factor * iqr_x
    y_min, y_max = qy[0] - iqr_factor * iqr_y, qy[1] + iqr_factor * iqr_y
    z_min, z_max = qz[0] - iqr_factor * iqr_z, qz[1] + iqr_factor * iqr_z

    # need at least 4 points for quartiles to be meaningful
    is_valid = count >= 4

    def gate(val):
        return xp.where(is_valid, val, xp.nan)

    return {
        'x': (gate(x_min), gate(x_max)),
        'y': (gate(y_min), gate(y_max)),
        'z': (gate(z_min), gate(z_max))
    }


@jit
def generate_ambiguous_pose(
        rvec_o2c: xp.ndarray,
        tvec_o2c: xp.ndarray
) -> Tuple[xp.ndarray, xp.ndarray]:
    """
    Generates the second, ambiguous solution for a planar PnP problem.
    Corresponds to a 180-degree rotation around the object's X-axis.
    """
    R_b2c = rodrigues(rvec_o2c)  # (..., 3, 3)

    #                 [1, 0, 0]
    # Flip matrix is  [0,-1, 0]
    #                 [0, 0,-1]
    diag = xp.array([1.0, -1.0, -1.0], dtype=rvec_o2c.dtype)
    R_flip = xp.diag(diag)  # (3, 3)

    # R_alt = R_b2c @ R_flip
    R_alt = xp.matmul(R_b2c, R_flip)
    rvec_alt = inverse_rodrigues(R_alt)

    tvec_alt = tvec_o2c

    return rvec_alt, tvec_alt


@jit
def point_to_segment_distance(
        p: xp.ndarray,
        a: xp.ndarray,
        b: xp.ndarray
) -> xp.ndarray:
    """
    Computes distance from point p to segment [a, b].
    """
    ab = b - a
    ap = p - a

    # dot product (..., 3) . (..., 3) -> (...)
    num = xp.sum(ap * ab, axis=-1)
    den = xp.sum(ab * ab, axis=-1) + 1e-9

    t = num / den
    t_clamped = xp.clip(t, 0.0, 1.0)

    closest = a + t_clamped[..., None] * ab

    return xp.linalg.norm(p - closest, axis=-1)