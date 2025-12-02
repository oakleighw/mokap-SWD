from functools import partial
from typing import Tuple, Dict, Optional
try:
    from .backend import USE_JAX, xp, jit, lax, vmap, _tiny, align_batch_dims
    from .transforms import quaternion_distance, rotation_matrix, rotation_vector, homogenize
except ImportError:
    from mokap.geometry.backend import USE_JAX, xp, jit, lax, vmap, _tiny, align_batch_dims
    from mokap.geometry.transforms import quaternion_distance, rotation_matrix, rotation_vector, homogenize


@jit
def weighted_median(
        data: xp.ndarray,
        weights: xp.ndarray
) -> xp.ndarray:
    """
    Computes the weighted median of data.
    Works for both continuous weights and binary masks (0/1)

    Args:
        data: (N,) values
        weights: (N,) weights or mask

    Returns:
        The weighted median value
    """

    # Sort data and weights based on data
    sort_idx = xp.argsort(data, axis=-1)
    sorted_data = xp.take_along_axis(data, sort_idx, axis=-1)
    sorted_weights = xp.take_along_axis(weights, sort_idx, axis=-1)

    # Compute cumulative weights
    cumsum = xp.cumsum(sorted_weights, axis=-1)

    # Handle total weight safely (keepdims to broadcast later)
    total_weight = xp.take(cumsum, xp.array([-1]), axis=-1) # take last element of cumsum
    target = 0.5 * total_weight

    # Find insertion point
    # Already sorted, so we can use a comparison mask

    # Find first index where cumsum >= target
    condition = cumsum >= target
    median_idx = xp.argmax(condition, axis=-1)  # index of first True

    # Expand dims and take_along_axis
    median_idx_expanded = median_idx[..., None]
    result = xp.take_along_axis(sorted_data, median_idx_expanded, axis=-1)

    return result.squeeze(-1)


@jit
def align_rigid(
        points3d_A: xp.ndarray,
        points3d_B: xp.ndarray
) -> Tuple[xp.ndarray, xp.ndarray]:
    """
    Estimates the rigid transformation (rotation R, translation t) between two
    sets of corresponding 3D points using the Kabsch algorithm.
    Finds T such that: B ~ T(A) = R @ A + t

    Args:
        points3d_A: Source points (..., N, 3)
        points3d_B: Destination points (..., N, 3)

    Returns:
        R: Rotation matrix (..., 3, 3)
        t: Translation vector (..., 3)
    """

    # Compute centroids (N dim)
    centroid_A = xp.mean(points3d_A, axis=-2, keepdims=True)
    centroid_B = xp.mean(points3d_B, axis=-2, keepdims=True)

    # Center the points
    A_centered = points3d_A - centroid_A
    B_centered = points3d_B - centroid_B

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
    s = xp.stack([ones, ones, xp.sign(det_R)], axis=-1)  # (..., 3)

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
def align_affine(
        points3d_A: xp.ndarray,
        points3d_B: xp.ndarray
) -> Tuple[xp.ndarray, xp.ndarray]:
    """
    Estimates the affine transformation (A, t) between two sets of corresponding 3D points.
    Finds T such that: B ~ T(A) = A @ A.T + t. Uses linear least squares.

    Args:
        points3d_A: Source points (..., M, 3)
        points3d_B: Destination points (..., M, 3)

    Returns:
        A_mat: The affine transformation matrix (..., 3, 3)
        t_vec: The translation vector (..., 3)
    """

    # Construct Design Matrix X: [x, y, z, 1]
    X = homogenize(points3d_A)  # (..., M, 4)

    # Solve T = (X^T X)^-1 X^T B
    XT = xp.swapaxes(X, -1, -2)  # (..., 4, M)

    XTX = xp.matmul(XT, X)
    XTB = xp.matmul(XT, points3d_B)

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
def fill_missing_points(
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
    A = homogenize(points3d_theoretical)  # (N, 4)

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


@partial(jit, static_argnames=['iters', 'delta'])
def translation_average(
        tvec_stack: xp.ndarray,
        iters: int = 3,
        delta: float = 1.0
) -> xp.ndarray:
    """
    Computes the robust average of translation vectors using Iteratively Reweighted Least Squares (IRLS).
    Uses Huber weights to suppress outliers.

    Args:
        tvec_stack: Input translation vectors (M, 3)
        iters: Number of IRLS iterations
        delta: Huber loss threshold

    Returns:
        The robustly averaged translation vector (3,)
    """

    M = tvec_stack.shape[0]

    # Handle empty input safely
    if M == 0:
        return xp.zeros((3,), dtype=tvec_stack.dtype)

    # Initial guess: Median
    t0 = xp.median(tvec_stack, axis=0)

    def body_fn(t_curr):
        res = tvec_stack - t_curr  # (M, 3)
        norms = xp.linalg.norm(res, axis=1)  # (M,)

        # Huber weight
        w = xp.where(norms <= delta, 1.0, delta / (norms + 1e-12))  # (M,)

        sum_w = xp.sum(w)
        # Normalise weights safely
        w_norm = w / (sum_w + 1e-12)

        t_next = xp.sum(w_norm[:, None] * tvec_stack, axis=0)
        return t_next

    # Using the shim for the loop
    def loop_body(i, t_val):
        return body_fn(t_val)

    t_final = lax.fori_loop(0, iters, loop_body, t0)
    return t_final


@jit
def quaternion_average(
        q_stack: xp.ndarray,
        weights: Optional[xp.ndarray] = None
) -> xp.ndarray:
    """
    Computes the weighted average of quaternions using Markley's Eigen-decomposition method.
    Robustly handles the antipodal ambiguity (q == -q).

    Args:
        q_stack: Input quaternions (N, 4)
        weights: Optional weights for each quaternion (N,). Defaults to uniform.

    Returns:
        The averaged unit quaternion (4,)
    """

    if weights is None:
        weights = xp.ones(q_stack.shape[0], dtype=q_stack.dtype)

    # Normalise weights
    weights = weights / (xp.sum(weights) + 1e-8)

    # Align quaternions to handle q = -q ambiguity
    # Pick the one with largest weight as reference
    idx_ref = xp.argmax(weights)
    q_ref = q_stack[idx_ref]

    # Dot product with reference
    dots = xp.sum(q_ref * q_stack, axis=-1)
    flip = xp.sign(dots)
    # if dot is 0, sign is 0, which destroys data, so default to 1
    flip = xp.where(flip == 0, 1.0, flip)

    quats_aligned = q_stack * flip[:, None]

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


@partial(jit, static_argnames=['thresh_radians', 'thresh_distance', 'iters'])
def average_qtposes(
        qt_stack: xp.ndarray,
        thresh_radians: float = xp.pi / 6.0,
        thresh_distance: float = 1.0,
        iters: int = 3
) -> Tuple[xp.ndarray, xp.ndarray, bool]:
    """
    Robustly averages a stack of pose hypotheses (quaternions and translations).
    Filters outliers based on angular and Euclidean distance from the current estimate.

    Args:
        qt_stack: Stack of poses (N, 7), where each row is [qx, qy, qz, qw, tx, ty, tz]
        thresh_radians: Max angular distance (radians) for a sample to be considered an inlier
        thresh_distance: Max Euclidean distance for a sample to be considered an inlier
        iters: Number of IRLS iterations

    Returns:
        q_final: Robust average quaternion (4,)
        t_final: Robust average translation (3,)
        valid: Boolean indicating if enough inliers were found to form a valid estimate
    """

    # Clean data
    finite_mask = xp.all(xp.isfinite(qt_stack), axis=1)
    weights_init = finite_mask.astype(xp.float32)

    quats = qt_stack[:, :4]
    trans = qt_stack[:, 4:]

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
        ang_errs = quaternion_distance(quats, q_c_broad)

        trans_errs = xp.linalg.norm(trans - t_c, axis=1)

        # Determine inliers
        is_inlier = (ang_errs <= thresh_radians) & (trans_errs <= thresh_distance)
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

    q_final, t_final, valid = lax.fori_loop(0, iters, loop_wrapper, init_state)

    return q_final, t_final, valid


@jit
def intersect_rays(
        ray_origins: xp.ndarray,
        ray_directions: xp.ndarray
) -> xp.ndarray:
    """
    Finds the single 3D point that minimizes the sum of squared distances to a set of rays.
    Mathematically equivalent to the least-squares intersection.

    Args:
        ray_origins: Origins of the rays (C, 3)
        ray_directions: Normalized directions of the rays (C, 3)

    Returns:
        The optimal intersection point (3,)
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
def intersect_aabb(
        ray_origins: xp.ndarray,
        ray_directions: xp.ndarray,
        aabb_min: xp.ndarray,
        aabb_max: xp.ndarray
) -> Tuple[xp.ndarray, xp.ndarray, xp.ndarray]:
    """
    Slab method for AABB intersection. Supports batched rays.

    Args:
        ray_origins: Ray origins (..., 3)
        ray_directions: Ray directions (..., 3)
        aabb_min: Box min corner (3,)
        aabb_max: Box max corner (3,)

    Returns:
        p_near: Entry points (..., 3)
        p_far: Exit points (..., 3)
        has_intersection: Boolean mask
    """
    # Safe inverse direction
    is_zero = xp.abs(ray_directions) < _tiny
    safe_dir = xp.where(is_zero, _tiny, ray_directions)
    dir_inv = 1.0 / safe_dir

    # Broadcast AABB to rays
    t1 = (aabb_min - ray_origins) * dir_inv
    t2 = (aabb_max - ray_origins) * dir_inv

    # Find entering and exiting planes
    t_min = xp.max(xp.minimum(t1, t2), axis=-1)
    t_max = xp.min(xp.maximum(t1, t2), axis=-1)

    has_intersection = t_min < t_max

    # Compute points
    p_near = ray_origins + t_min[..., None] * ray_directions
    p_far = ray_origins + t_max[..., None] * ray_directions

    # Masking
    nan_v = xp.full_like(p_near, xp.nan)
    mask = has_intersection[..., None]

    p_near = xp.where(mask, p_near, nan_v)
    p_far = xp.where(mask, p_far, nan_v)

    return p_near, p_far, has_intersection


@partial(jit, static_argnames=['method'])
def compute_bounds(
        points3d: xp.ndarray,
        all_errors: xp.ndarray,
        error_threshold_px: float = 1.0,
        method: str = 'iqr',  # 'iqr' or 'percentile'
        percentile: float = 1.0,
        iqr_factor: float = 1.5
) -> Dict[str, Tuple[float, float]]:
    """
    Computes a bounding box in world coordinates from a cloud of 3D points.
    Uses Interquartile Range method. Reliability is determined by the mean reprojection error.

    Args:
        points3d: Cloud of 3D points (P, N, 3)
        all_errors: Error values (C, P, N)
        error_threshold_px: Max mean error
        method: Which method to use, IQR or percentile
        percentile: Percentile to clip outliers when computing bounds
        iqr_factor: Multiplier for IQR (default 1.5)

    Returns:
        Dict with 'x', 'y', 'z' bounds
    """

    # Reliability masking
    mean_error = xp.nanmean(all_errors, axis=0)  # (P, N) or (N,)
    reliable_mask = mean_error < error_threshold_px

    # Check if enough points (threshold of 4 for IQR)
    count = xp.sum(reliable_mask.astype(xp.float32))
    is_valid = count >= 4

    # Filter points
    # Ensure proper broadcasting if points3d has temporal dim but mask does not
    pts_clean = xp.where(reliable_mask[..., None], points3d, xp.nan)

    # Compute bounds
    if method.lower() == 'percentile':
        q_low, q_high = percentile, 100.0 - percentile
        bounds = xp.nanpercentile(pts_clean, xp.array([q_low, q_high]), axis=(0, 1))
        # Result is (2, 3) -> [min_x, min_y, min_z], [max_x, max_y, max_z]
        mins, maxs = bounds[0], bounds[1]

    elif method.lower() == 'iqr':
        # Compute 25/75 quartiles
        qs = xp.nanpercentile(pts_clean, xp.array([25.0, 75.0]), axis=(0, 1))  # (2, 3)
        q25, q75 = qs[0], qs[1]
        iqr = q75 - q25

        mins = q25 - iqr_factor * iqr
        maxs = q75 + iqr_factor * iqr

    else:
        # Fallback to simple min/max
        mins = xp.nanmin(pts_clean, axis=(0, 1))
        maxs = xp.nanmax(pts_clean, axis=(0, 1))

    # Gate output
    def get_dim(idx):
        return (xp.where(is_valid, mins[idx], xp.nan),
                xp.where(is_valid, maxs[idx], xp.nan))

    return {
        'x': get_dim(0),
        'y': get_dim(1),
        'z': get_dim(2)
    }


@jit
def flip_rotation_180(
        R: xp.ndarray
) -> xp.ndarray:
    """
    Generates the ambiguous "flipped" rotation matrix for a planar object.
    This corresponds to a 180-degree rotation of the object around its own X-axis.

    Args:
        R: Original rotation matrix (..., 3, 3)

    Returns:
        R_alt: Flipped rotation matrix (..., 3, 3)

    """

    #                 [1, 0, 0]
    # Flip matrix is  [0,-1, 0]
    #                 [0, 0,-1]
    diag = xp.array([1.0, -1.0, -1.0], dtype=R.dtype)
    R_flip = xp.diag(diag)  # (3, 3)

    # R_alt = R @ R_flip
    return xp.matmul(R, R_flip)


@jit
def flip_transform_180(
        T_o2c: xp.ndarray
) -> xp.ndarray:
    """
    Generates the ambiguous "flipped" transform solution often found in planar PnP problems.
    This corresponds to a 180-degree rotation of the object around its own X-axis.

    Args:
        T_o2c: Original transform matrix (..., 4, 4)

    Returns:
        T_alt: Flipped transform matrix (..., 4, 4)
    """
    R = T_o2c[..., :3, :3]
    t = T_o2c[..., :3, 3]

    R_alt = flip_rotation_180(R)

    # Build output matrix
    batch_shape = T_o2c.shape[:-2]
    extmat = xp.concatenate([R_alt, t[..., None]], axis=-1)
    bottom_row = xp.broadcast_to(
        xp.array([0., 0., 0., 1.], dtype=T_o2c.dtype),
        batch_shape + (1, 4)
    )
    return xp.concatenate([extmat, bottom_row], axis=-2)


@jit
def segment_distance(
        points3d: xp.ndarray,
        segments: xp.ndarray,
) -> xp.ndarray:
    """
    Computes the shortest Euclidean distance from point(s) P to line segment(s) AB.

    Args:
        points3d: Point(s) to query (..., 3)
        segments: Start and end point(s) of the segment(s) (..., 2, 3)

    Returns:
        Distance(s) (..., )
    """

    # Determine target batch rank
    pts_batch_rank = points3d.ndim - 1  # points data_dims=1
    seg_batch_rank = segments.ndim - 2  # segments data_dims=2 (start/end + coords)

    target_rank = max(pts_batch_rank, seg_batch_rank)

    # Align ranks (inserts singleton dims at the start of batch dims)
    points3d = align_batch_dims(target_rank, points3d, data_dims=1)
    segments = align_batch_dims(target_rank, segments, data_dims=2)

    a = segments[..., 0, :]
    b = segments[..., 1, :]

    ab = b - a
    ap = points3d - a

    dot_ap_ab = xp.sum(ap * ab, axis=-1)
    dot_ab_ab = xp.sum(ab * ab, axis=-1)

    t = dot_ap_ab / (dot_ab_ab + _tiny)
    t = xp.clip(t, 0.0, 1.0)

    closest = a + t[..., None] * ab

    return xp.linalg.norm(points3d - closest, axis=-1)


@jit
def fit_plane(
        points3d: xp.ndarray
) -> Tuple[xp.ndarray, xp.ndarray]:
    """
    Fits a plane to a set of 3D points using SVD.
    Equation: normal . (x - centroid) = 0

    Args:
        points3d: Input 3D points (..., N, 3)

    Returns:
        centroid: (..., 3)
        normal: Unit normal vector (..., 3)
    """

    centroid = xp.mean(points3d, axis=-2)
    centered = points3d - centroid[..., None, :]

    # last row of Vh is the smallest singular value
    _, _, Vh = xp.linalg.svd(centered)

    normal = Vh[..., -1, :]

    return centroid, normal