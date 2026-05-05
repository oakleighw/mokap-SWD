from functools import partial
from typing import Tuple, Union, Optional

from mokap.geometry.backend import xp, jit, lax, _tiny, align_batch_dims
from mokap.geometry import projection_matrix, invert_intrinsics, homogenize, dehomogenize

@jit
def normalize_pixel_coordinates(
        points2d: xp.ndarray,
        K: xp.ndarray
) -> xp.ndarray:
    """
    Normalises pixel coordinates to the image plane (Z=1).
    Inverse of applying intrinsics: (uv - c) / f
    """

    # Align params
    target_ndim = points2d.ndim - 1
    K = align_batch_dims(target_ndim, K, data_dims=2)

    # Extract intrinsics
    # f = [fx, fy], c = [cx, cy]
    f = xp.stack([K[..., 0, 0], K[..., 1, 1]], axis=-1)
    c = K[..., :2, 2]

    # Normalise
    return (points2d - c) / (f + _tiny)


@partial(jit, static_argnames=['distortion_model'])
def distort(
        points2d_normalised: xp.ndarray,
        D: xp.ndarray,
        distortion_model: str = 'standard'
) -> xp.ndarray:
    """
    Computes radial and tangential distortion terms for normalised coordinates.
    Follows OpenCV conventions.

    Args:
        points2d_normalised: Normalised 2D points (x/z, y/z) shape (..., 2)
        D: Distortion coefficients (..., N)
        distortion_model: 'none', 'simple', 'standard', 'rational', 'thinprism', 'fisheye'

    Returns:
        radial: Radial scaling factor (..., 1) or matching shape
        tangential: Tangential (and prism) offset vector (..., 2)
    """

    if distortion_model == 'none':
        return points2d_normalised

    req_len = {
        'simple': 4,        # k1, k2, p1, p2
        'fisheye': 4,       # k1, k2, k3, k4
        'standard': 5,      # k1, k2, p1, p2, k3
        'rational': 8,      # k1, k2, p1, p2, k3, k4, k5, k6
        'thinprism': 12,    # k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4
        'tilted': 14        # k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, tauX, tauY
    }.get(distortion_model, 5)

    if D.shape[-1] < req_len:
        D = xp.pad(D, [(0, 0)] * (D.ndim - 1) + [(0, req_len - D.shape[-1])])

    D = align_batch_dims(points2d_normalised.ndim - 1, D, data_dims=1)

    # Handle Fisheye (Kannala-Brandt) separately as it uses a completely different model (theta-based)
    if distortion_model == 'fisheye':

        k1 = D[..., 0:1]
        k2 = D[..., 1:2]
        k3 = D[..., 2:3]
        k4 = D[..., 3:4]

        r2 = xp.sum(xp.square(points2d_normalised), axis=-1, keepdims=True)
        r = xp.sqrt(r2)
        theta = xp.arctan(r)
        theta2 = theta * theta
        theta4 = theta2 * theta2
        theta6 = theta4 * theta2
        theta8 = theta4 * theta4

        scale = 1.0 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8
        theta_dist = theta * scale

        # Safe division for r -> 0
        safe_r = xp.where(r < _tiny, 1.0, r)
        factor = xp.where(r < _tiny, 1.0, theta_dist / safe_r)

        return points2d_normalised * factor

    # Coordinate transformation (Tilted model)
    # Tilted rotates the sensor plane coordinates *before* applying radial distortion
    coords = points2d_normalised

    if distortion_model == 'tilted':

        tauX = D[..., 12:13]
        tauY = D[..., 13:14]

        cx, sx = xp.cos(tauX), xp.sin(tauX)
        cy, sy = xp.cos(tauY), xp.sin(tauY)

        x = coords[..., 0:1]
        y = coords[..., 1:2]

        # Rotate around X
        y1 = y * cx - sx
        z1 = y * sx + cx
        # Rotate around Y
        x2 = x * cy + z1 * sy
        y2 = y1
        z2 = -x * sy + z1 * cy

        # Re-normalise
        safe_z2 = xp.where(xp.abs(z2) < _tiny, 1.0, z2)
        coords = xp.concatenate([x2 / safe_z2, y2 / safe_z2], axis=-1)

    # Standard coefficients
    k1 = D[..., 0:1]
    k2 = D[..., 1:2]
    p1 = D[..., 2:3]
    p2 = D[..., 3:4]

    # Precompute radii (using potentially tilted coordinates)
    r2 = xp.sum(xp.square(points2d_normalised), axis=-1, keepdims=True)
    r4 = r2 * r2
    r6 = r4 * r2

    # Radial component
    if distortion_model == 'simple':
        radial = 1.0 + k1 * r2 + k2 * r4

    elif distortion_model in ('rational', 'thinprism', 'tilted'):
        k3 = D[..., 4:5]
        k4 = D[..., 5:6]
        k5 = D[..., 6:7]
        k6 = D[..., 7:8]

        num = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
        den = 1.0 + k4 * r2 + k5 * r4 + k6 * r6

        safe_den = xp.where(xp.abs(den) < _tiny, 1.0, den)
        radial = num / safe_den

    else: # standard
        k3 = D[..., 4:5]
        radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6

    # Tangential
    x = coords[..., 0:1]
    y = coords[..., 1:2]

    xy = x * y
    x2 = x * x
    y2 = y * y

    dx = 2.0 * p1 * xy + p2 * (r2 + 2.0 * x2)
    dy = 2.0 * p2 * xy + p1 * (r2 + 2.0 * y2)

    tangential = xp.concatenate([dx, dy], axis=-1)

    # Thin prism component
    if distortion_model == 'thinprism':
        s1 = D[..., 8:9]
        s2 = D[..., 9:10]
        s3 = D[..., 10:11]
        s4 = D[..., 11:12]

        # OpenCV thin prism:
        # x_dist += s1*r^2 + s2*r^4
        # y_dist += s3*r^2 + s4*r^4
        prism_x = s1 * r2 + s2 * r4
        prism_y = s3 * r2 + s4 * r4

        tangential = tangential + xp.concatenate([prism_x, prism_y], axis=-1)

    return coords * radial + tangential


@partial(jit, static_argnames=['distortion_model', 'iters'])
def undistort(
        points2d: xp.ndarray,
        K: xp.ndarray,
        D: xp.ndarray,
        R: Optional[xp.ndarray] = None,
        P: Optional[xp.ndarray] = None,
        distortion_model: str = 'standard',
        iters: int = 10
) -> xp.ndarray:
    """
    Invert distortion & reprojection using Newton-Raphson iteration.
    (equivalent to cv2.undistortPoints).

    Args:
        points2d: Distorted 2D points (..., 2)
        K: Intrinsics (3, 3)
        D: Distortion coefficients (..., D)
        R: Optional, rectification matrix (3, 3)
        P: Optional, new camera matrix (3, 3)
        iters: Max iterations for the solver

    Returns:
        Undistorted points (..., 2)
    """

    target_ndim = points2d.ndim - 1

    # Normalise to get target distorted points on the image plane: (uv - c) / f
    uv_distorted = normalize_pixel_coordinates(points2d, K)

    # Iterative solver
    # We want to find uv such that distort(uv) == uv_distorted
    # Guess: start with the distorted point (identity assumption)

    def loop_body(i, uv_guess):
        # Forward pass
        uv_pred = distort(uv_guess, D, distortion_model)
        # Error
        error = uv_distorted - uv_pred
        # Update (Simple residual addition works because slope ~ 1)
        return uv_guess + error

    uv_undistorted = lax.fori_loop(0, iters, loop_body, uv_distorted)

    # Rectification (rotation)
    pts_h = homogenize(uv_undistorted)  # (..., 3)

    if R is not None:
        R = align_batch_dims(target_ndim, R, data_dims=2)
        pts_rectified = xp.einsum('...ij,...j->...i', R, pts_h)
    else:
        pts_rectified = pts_h

    # Project to new camera (P)
    P_eff = K if P is not None else K   # default to original K if P is not provided
    P_eff = align_batch_dims(target_ndim, P_eff, data_dims=2)

    new_f = xp.stack([P_eff[..., 0, 0], P_eff[..., 1, 1]], axis=-1)
    new_c = P_eff[..., :2, 2]

    # Extract Z coordinate
    z_rect = pts_rectified[..., 2:3]

    # Check for points behind the camera or at z=0 (singularities)
    is_valid_z = z_rect > 1e-6  # slightly larger epsilon than _tiny for stability in division

    # Avoid division by zero for invalid points (they are masked later anyways)
    safe_z = xp.where(is_valid_z, z_rect, 1.0)

    # Project back to normalised plane
    uv_rect = pts_rectified[..., :2] / safe_z

    # Apply new intrinsics
    result = uv_rect * new_f + new_c

    # Mask invalid points (behind camera)
    result = xp.where(is_valid_z, result, xp.nan)

    return result


@jit
def project_to_normalized(
        points3d: xp.ndarray,
        T: xp.ndarray
) -> Tuple[xp.ndarray, xp.ndarray]:
    """
    Transforms 3D points to camera frame and projects them to the normalised plane.
    """
    target_batch_dim = points3d.ndim - 1
    T = align_batch_dims(target_batch_dim, T, data_dims=2)

    # R @ P + t
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    Xc = xp.einsum('...ij,...j->...i', R, points3d) + t

    z = Xc[..., 2]

    # Mask logic
    valid_mask_bool = z > 1e-4
    valid_mask = valid_mask_bool.astype(xp.float32)

    # Project
    z_safe = xp.where(valid_mask_bool, z, 1.0)
    x_norm_raw = Xc[..., :2] / z_safe[..., None]

    # Zero out invalid points
    x_norm = xp.where(valid_mask_bool[..., None], x_norm_raw, 0.0)

    return x_norm, valid_mask


@partial(jit, static_argnames=['distortion_model'])
def project(
        points3d: xp.ndarray,
        T: xp.ndarray,
        K: xp.ndarray,
        D: xp.ndarray,
        distortion_model: str = 'standard'
) -> Tuple[xp.ndarray, xp.ndarray]:
    """
    Projects points from a source coordinate system into the image plane of a camera.

    Args:
        points3d: Points in the source coordinate system (..., 3)
        T: Transform matrix (source-to-camera) (..., 4, 4) or (..., 3, 4)
        K: Camera intrinsics matrix (..., 3, 3)
        D: Camera distortion coefficients (..., D)
        distortion_model: Distortion model string

    Returns:
        image_points: Projected 2D points in the image plane (..., 2)
        valid_mask: Boolean mask indicating points strictly in front of the camera (Z > 0)
    """

    # Transform and project to normalised plane
    x_norm, valid_mask = project_to_normalized(points3d, T)

    # Apply Distortion
    # dist_coeffs alignment happens inside here based on x.ndim
    points_distorted = distort(x_norm, D, distortion_model)

    # Apply intrinsics (denormalise)
    target_batch_dim = points3d.ndim - 1
    K = align_batch_dims(target_batch_dim, K, data_dims=2)

    f = xp.stack([K[..., 0, 0], K[..., 1, 1]], axis=-1)
    c = K[..., :2, 2]

    image_points = points_distorted * f + c

    return image_points, valid_mask


def project_to_cameras(
        points3d: xp.ndarray,
        T_w2c: xp.ndarray,
        K: xp.ndarray,
        D: xp.ndarray,
        distortion_model: str = 'standard'
):
    """
    Project points (N, 3) to C different cameras.

    Args:
        points3d: Points in world coordinates (N, 3)
        T_w2c: World-to-camera transforms (C, 4, 4)
        K: Camera matrices (C, 3, 3)
        D: Distortion coefficients (C, D)

    Returns:
        points2d: (C, N, 2)
        valid_mask: (C, N)
    """
    obj_exp = points3d[None, ...]        # (1, N, 3)
    T_exp = T_w2c[:, None, :, :]         # (C, 1, 4, 4)
    return project(obj_exp, T_exp, K, D, distortion_model)


def project_to_cameras_multi(
        points3d: xp.ndarray,
        T_w2c: xp.ndarray,
        K: xp.ndarray,
        D: xp.ndarray,
        distortion_model: str = 'standard'
) -> Tuple[xp.ndarray, xp.ndarray]:
    """
    Projects P temporal snapshots of a point cloud into C camera views.

    Args:
        points3d: Points in world coordinates (P, N, 3) or (N, 3)
        T_w2c: World-to-camera transforms (C, 4, 4)
        K: Camera matrices (C, 3, 3)
        D: Distortion coefficients (C, D)

    Returns:
        points2d: (C, P, N, 2)
        valid_mask: (C, P, N)
    """
    if points3d.ndim == 2:
        obj_exp = points3d[None, None, ...]  # (1, 1, N, 3)
    else:
        obj_exp = points3d[None, ...]        # (1, P, N, 3)

    T_exp = T_w2c[:, None, None, :, :]       # (C, 1, 1, 4, 4)
    K_exp = K[:, None, None, :, :]           # (C, 1, 1, 3, 3)
    D_exp = D[:, None, None, :]              # (C, 1, 1, D)

    return project(obj_exp, T_exp, K_exp, D_exp, distortion_model)


@partial(jit, static_argnames=['distortion_model'])
def project_with_object_pose(
        object_points3d: xp.ndarray,
        T_w2c: xp.ndarray,
        T_o2w: xp.ndarray,
        K: xp.ndarray,
        D: xp.ndarray,
        distortion_model: str = 'standard'
) -> Tuple[xp.ndarray, xp.ndarray]:
    """
    Projects 3D points from an object's local frame into a camera view by
    composing object-to-world and world-to-camera transforms.

    T_o2c = T_w2c @ T_o2w

    Args:
        object_points3d: Points in object coordinates (..., 3)
        T_w2c: World-to-camera transform (..., 4, 4)
        T_o2w: Object-to-world transform (..., 4, 4)
        K: Camera intrinsics (..., 3, 3)
        D: Distortion coefficients (..., D)

    Returns:
        image_points: (..., 2)
        valid_mask: (...)
    """
    target_dim = object_points3d.ndim - 1
    T_w2c = align_batch_dims(target_dim, T_w2c, 2)
    T_o2w = align_batch_dims(target_dim, T_o2w, 2)

    T_o2c = T_w2c @ T_o2w
    return project(object_points3d, T_o2c, K, D, distortion_model)


def project_object_to_cameras(
        object_points3d: xp.ndarray,
        T_w2c: xp.ndarray,
        T_o2w: xp.ndarray,
        K: xp.ndarray,
        D: xp.ndarray,
        distortion_model: str = 'standard'
):
    """
    Projects N object points through P object poses into C cameras.

    Args:
        object_points3d: 3D points in object coordinates (N, 3)
        T_w2c: Camera poses (world -> cam), (C, 4, 4)
        T_o2w: Object poses (object -> world), (P, 4, 4)
        K: Camera intrinsics (C, 3, 3)
        D: Distortion coefficients (C, D)

    Returns:
        points2d: (C, P, N, 2)
        valid_mask: (C, P, N)
    """
    # Expand for broadcasting: (C, 1, 4, 4) @ (1, P, 4, 4) -> (C, P, 4, 4)
    T_w2c_exp = T_w2c[:, None, :, :]
    T_o2w_exp = T_o2w[None, :, :, :]
    T_net = T_w2c_exp @ T_o2w_exp

    # Expand for points dimension
    obj_exp = object_points3d[None, None, :, :]  # (1, 1, N, 3)
    T_net_exp = T_net[:, :, None, :, :]          # (C, P, 1, 4, 4)

    if K.ndim == 3:
        K_exp = K[:, None, None, :, :]
    else:
        K_exp = K

    if D.ndim == 2:
        D_exp = D[:, None, None, :]
    else:
        D_exp = D

    return project(obj_exp, T_net_exp, K_exp, D_exp, distortion_model)


@partial(jit, static_argnames=['distortion_model'])
def unproject(
        points2d: xp.ndarray,
        depth: Union[float, xp.ndarray],
        K: xp.ndarray,
        T_c2w: xp.ndarray,
        D: Optional[xp.ndarray] = None,
        distortion_model: str = 'standard'
) -> xp.ndarray:
    """
    Back-project 2D points into 3D world coords at given depth.

    Args:
        points2d: (..., 2)
        depth: Scalar or (..., )
        K: (..., 3, 3)
        T_c2w: Camera-to-world transform (..., 4, 4) or Extrinsics (..., 3, 4).
        D: Optional coeffs to undistort points first

    Returns:
        World points (..., 3)
    """

    # Align batch dimensions for numpy compatibility
    # points2d is (..., 2), so batch dims are ndim - 1
    target_batch_dim = points2d.ndim - 1

    # Align matrix inputs (data_dims=2 for 3x3 or 4x4 matrices)
    K = align_batch_dims(target_batch_dim, K, data_dims=2)
    T_c2w = align_batch_dims(target_batch_dim, T_c2w, data_dims=2)

    # Align depth input (data_dims=0 for scalars)
    depth_arr = xp.asarray(depth)
    if depth_arr.ndim > 0:
        depth_arr = align_batch_dims(target_batch_dim, depth_arr, data_dims=0)

    # Undistort if needed
    if D is not None:
        points2d = undistort(
            points2d,
            K=K,
            D=D,
            distortion_model=distortion_model,
            R=xp.eye(3),
            P=K,
        )

    hom2d = homogenize(points2d)

    K_inv = invert_intrinsics(K)

    # invK @ hom2d -> (..., 1, 3, 3) @ (..., N, 3) -> (..., N, 3)
    cam_dirs = xp.einsum('...ij,...j->...i', K_inv, hom2d)

    # Depth broadcast
    if depth_arr.ndim == 0:
        cam_pts = cam_dirs * depth_arr
    else:
        # depth_arr is aligned (e.g. 5, 1), we extend to (5, 1, 1) to broadcast against (5, 6, 3)
        cam_pts = cam_dirs * depth_arr[..., None]

    hom_cam = homogenize(cam_pts)
    world_pts = xp.einsum('...ij,...j->...i', T_c2w[..., :3, :], hom_cam)

    return world_pts


@partial(jit, static_argnames=['lambda_reg'])
def triangulate_from_projections(
        points2d: xp.ndarray,
        P: xp.ndarray,
        weights: Optional[xp.ndarray] = None,
        lambda_reg: float = 0.0
) -> xp.ndarray:
    """
    Triangulates N 3D points from C 2D observations using SVD.

    Args:
        points2d: 2D observations (C, N, 2)
        P: Projection matrices (C, 3, 4)
        weights: Optional confidence weights (C, N)
        lambda_reg: Tikhonov regularization term

    Returns:
        points3d: Triangulated points (N, 3)
    """
    valid_observations = xp.isfinite(points2d[..., 0])
    n_obs = xp.sum(valid_observations, axis=0)

    u = xp.where(valid_observations, points2d[..., 0], 0.0)
    v = xp.where(valid_observations, points2d[..., 1], 0.0)

    if weights is None:
        w = valid_observations.astype(points2d.dtype)
    else:
        weights = xp.asarray(weights)
        w = xp.where(valid_observations, weights, 0.0)

    # P_mats (C, 3, 4)
    P0 = P[:, None, 0, :]
    P1 = P[:, None, 1, :]
    P2 = P[:, None, 2, :]

    u_exp = u[..., None]
    v_exp = v[..., None]
    w_exp = w[..., None]

    r1 = (u_exp * P2 - P0) * w_exp
    r2 = (v_exp * P2 - P1) * w_exp

    A_stacked = xp.stack([r1, r2], axis=1)  # (C, 2, N, 4)

    # Group by point
    A_transposed = A_stacked.transpose(2, 0, 1, 3)  # (N, C, 2, 4)

    # Reshape for SVD
    A = A_transposed.reshape((points2d.shape[1], -1, 4))

    if lambda_reg != 0.0:
        # build A^T A + lambda I for each point
        ATA = xp.einsum('pni,pnj->pij', A, A) + lambda_reg * xp.eye(4)
        # TODO: does svd still crash on Apple silicon with JAX-Metal??
        _, _, Vh = xp.linalg.svd(ATA, full_matrices=False)
    else:
        _, _, Vh = xp.linalg.svd(A, full_matrices=False)

    # Dehomogenize
    Xh = Vh[:, -1, :]
    points3d = dehomogenize(Xh)

    reliable = (n_obs >= 2)[:, None]
    return xp.where(reliable, points3d, xp.nan)


def triangulate(
        points2d: xp.ndarray,
        T_w2c: xp.ndarray,
        K: xp.ndarray,
        D: xp.ndarray,
        weights: Optional[xp.ndarray] = None,
        distortion_model: str = 'standard',
):
    """
    High-level triangulation wrapper.
    Undistorts points, computes projection matrices, and solves 3D positions.

    Args:
        points2d: Observed points (C, N, 2)
        T_w2c: World-to-camera transforms (C, 4, 4)
        K: Intrinsics (C, 3, 3)
        D: Distortion coeffs
        weights: Optional weights (C, N)

    Returns:
        points3d: (N, 3)
    """

    pts2d_ud = undistort(points2d, K, D, distortion_model=distortion_model)

    if weights is None:
        # undistortion might also introduce NaNs
        weights = xp.isfinite(points2d[..., 0]).astype(points2d.dtype)
    weights = weights.astype(xp.float32)

    P_mats = projection_matrix(K, T_w2c)

    return triangulate_from_projections(pts2d_ud, P_mats, weights=weights)


@jit
def pixels_to_rays(
        points2d: xp.ndarray,
        K: xp.ndarray
) -> xp.ndarray:
    """
    Converts 2D pixel coordinates to normalized 3D direction vectors in the camera frame.

    Args:
        points2d: 2D point coordinates (..., N) or (...)
        K: Camera intrinsics matrices (..., 3, 3)

    Returns:
        Unit direction vectors (..., 3)
    """

    # Align K to match points batch dims
    target_ndim = points2d.ndim - 1
    K = align_batch_dims(target_ndim, K, data_dims=2)

    # Extract f and c
    f = xp.stack([K[..., 0, 0], K[..., 1, 1]], axis=-1)  # (..., 2)
    c = K[..., :2, 2]  # (..., 2)

    # Normalised coordinates
    xy = (points2d - c) / (f + _tiny)
    dirs_unscaled = homogenize(xy)
    norm = xp.linalg.norm(dirs_unscaled, axis=-1, keepdims=True)
    return dirs_unscaled / (norm + _tiny)