from functools import partial
from typing import Tuple, Union, Optional, Dict
from .backend import xp, jit, lax, _eps, _tiny
from .transforms import rodrigues, extrinsics_matrix, projection_matrix, extmat_to_rtvecs, invert_intrinsics_matrix


def _align_batch_dims(target_ndim: int, arr: xp.ndarray, data_dims: int = 1) -> xp.ndarray:
    """
    Helper to ensure 'arr' broadcasts correctly against a data array with 'target_ndim' batch dimensions.
    It inserts singleton dimensions between the array's batch dims and its data dims.

    Args:
        target_ndim: The number of batch dimensions in the reference data (e.g. points)
        arr: The parameter array (e.g. K, D, rvec)
        data_dims: How many dimensions at the end of 'arr' are data (1 for vec, 2 for matrix)
    """
    arr = xp.asarray(arr)
    arr_batch_ndim = arr.ndim - data_dims
    pad_needed = target_ndim - arr_batch_ndim
    if pad_needed > 0:
        # Handle data_dims=0 case where slicing with [:-0] returns empty tuple
        if data_dims == 0:
            # For scalars/1D arrays, we append 1s at the end
            # e.g. (5,) -> (5, 1) to match (5, 6)
            new_shape = arr.shape + (1,) * pad_needed
        else:
            # Insert 1s before the data dimensions
            # e.g. (5, 3, 3) -> (5, 1, 3, 3) to match (5, 6, ...)
            new_shape = arr.shape[:-data_dims] + (1,) * pad_needed + arr.shape[-data_dims:]

        return arr.reshape(new_shape)
    return arr


@partial(jit, static_argnames=['distortion_model'])
def distortion(
        x: xp.ndarray,
        y: xp.ndarray,
        dist_coeffs: xp.ndarray,
        distortion_model: str = 'standard'
) -> Tuple[xp.ndarray, xp.ndarray, xp.ndarray]:
    """
    Computes tangential and radial distortion factors given normalised coordinates.

    Args:
        x: Normalised x coordinates
        y: Normalised y coordinates
        dist_coeffs: Distortion coefficients (..., D)
        distortion_model: The distortion model to apply

    Returns:
        radial: Radial distortion factor
        dx: Tangential distortion in x
        dy: Tangential distortion in y
    """

    # TODO: The distortion models should be standardised across mokap (bundle adjustment, here, and unreleased new dataclasses for camera parameters
    D = dist_coeffs.shape[-1]
    if D < 8:
        pad_width = [(0, 0)] * (dist_coeffs.ndim - 1) + [(0, 8 - D)]
        dist_coeffs = xp.pad(dist_coeffs, pad_width)

    dist_coeffs = _align_batch_dims(x.ndim, dist_coeffs, data_dims=1)

    k1 = dist_coeffs[..., 0]
    k2 = dist_coeffs[..., 1]
    p1 = dist_coeffs[..., 2]
    p2 = dist_coeffs[..., 3]
    k3 = dist_coeffs[..., 4]
    k4 = dist_coeffs[..., 5]
    k5 = dist_coeffs[..., 6]
    k6 = dist_coeffs[..., 7]

    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2

    # TODO: No need for all these branches. Should just do the same thing but with zeros. Except for Fisheye?
    if distortion_model == 'rational':
        # Rational model
        numerator = 1 + k1 * r2 + k2 * r4 + k3 * r6
        denominator = 1 + k4 * r2 + k5 * r4 + k6 * r6
        # Clip the denominator to be strictly positive to avoid division by zero/negative
        safe_denominator = xp.maximum(denominator, _eps)
        radial = numerator / safe_denominator
    elif distortion_model == 'full':
        # 8-parameter polynomial model
        radial = 1 + k1 * r2 + k2 * r4 + k3 * r6 + k4 * r2 * r6 + k5 * r4 * r6 + k6 * r6 * r6
    elif distortion_model == 'simple':
        # Simple 4-parameter model (k1, k2, p1, p2)
        radial = 1 + k1 * r2 + k2 * r4
    elif distortion_model == 'standard':
        # Standard 5-parameter model (k1, k2, p1, p2, k3)
        radial = 1 + k1 * r2 + k2 * r4 + k3 * r6
    else:  # 'none'
        radial = xp.ones_like(x)

    # Tangential distortion is the same for all models (except 'none')
    dx, dy = 0.0, 0.0
    if distortion_model != 'none':
        dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y

    return radial, dx, dy


@partial(jit, static_argnames=['distortion_model'])
def project_points(
        object_points: xp.ndarray,
        rvec: xp.ndarray,
        tvec: xp.ndarray,
        camera_matrix: xp.ndarray,
        dist_coeffs: xp.ndarray,
        distortion_model: str = 'standard'
) -> Tuple[xp.ndarray, xp.ndarray]:
    """
    Fundamental projection function (equivalent to cv2.projectPoints).
    Projects points from a source coordinate system into the image plane of a camera.

    Args:
        object_points: Points in the source coordinate system (..., 3)
        rvec: Rotation vector for transform from source to camera (..., 3)
        tvec: Translation vector for transform from source to camera (..., 3)
        camera_matrix: Camera intrinsics matrix K (..., 3, 3)
        dist_coeffs: Camera distortion coefficients (..., D)
        distortion_model: Distortion model string

    Returns:
        image_points: Projected 2D points in the image plane (..., 2)
        valid_mask: Boolean mask indicating points strictly in front of the camera (Z > 0)
    """

    # object_points is (..., 3), so batch dims are ndim - 1
    target_batch_dim = object_points.ndim - 1

    rvec = _align_batch_dims(target_batch_dim, rvec, data_dims=1)
    tvec = _align_batch_dims(target_batch_dim, tvec, data_dims=1)
    # Note: dist_coeffs alignment happens in the distortion() call

    # camera_matrix is matrix, so data_dims=2
    camera_matrix = _align_batch_dims(target_batch_dim, camera_matrix, data_dims=2)

    R = rodrigues(rvec)  # (..., 3, 3)

    # R @ P + t
    Xc = xp.einsum('...ij,...j->...i', R, object_points) + tvec
    z = Xc[..., 2]

    valid_mask = (z > 1e-4).astype(xp.float32)  # small positive threshold for safety

    # Project invalid points to (0, 0), but their mask will be False
    z_safe = xp.where(valid_mask, z, 1.0) # we do NOT use _eps here because 1e-4 is small enough to cause overflow in x/z
    x = Xc[..., 0] / z_safe
    y = Xc[..., 1] / z_safe

    # dist_coeffs alignment happens inside here based on x.ndim
    radial, dx, dy = distortion(x, y, dist_coeffs, distortion_model)
    x_d = x * radial + dx
    y_d = y * radial + dy

    fx = camera_matrix[..., 0, 0]
    fy = camera_matrix[..., 1, 1]
    cx = camera_matrix[..., 0, 2]
    cy = camera_matrix[..., 1, 2]

    u = fx * x_d + cx
    v = fy * y_d + cy

    image_points = xp.stack([u, v], axis=-1)
    return image_points, valid_mask


def project_multiple_poses(
        object_points: xp.ndarray,
        rvec: xp.ndarray,
        tvec: xp.ndarray,
        camera_matrix: xp.ndarray,
        dist_coeffs: xp.ndarray,
        distortion_model: str = 'standard'
):
    """
    Wrapper for project_points that projects 1 set of points (N, 3) using P poses (P, 3).
    Returns (P, N, 2).
    """
    obj_exp = object_points[None, ...]  # (1, N, 3)
    rvec_exp = rvec[:, None, :]  # (P, 1, 3)
    tvec_exp = tvec[:, None, :]  # (P, 1, 3)
    return project_points(obj_exp, rvec_exp, tvec_exp, camera_matrix, dist_coeffs, distortion_model)


def project_to_multiple_cameras(
        object_points: xp.ndarray,
        rvec: xp.ndarray,
        tvec: xp.ndarray,
        camera_matrix: xp.ndarray,
        dist_coeffs: xp.ndarray,
        distortion_model: str = 'standard'
):
    """
    Wrapper for project_points to project points (N, 3) to C different cameras.
    Where each camera has its own Intrinsics (K, D) and its own pose (rvec, tvec).
    Returns (C, N, 2).
    """
    # Points: (N, 3) -> (1, N, 3)
    obj_exp = object_points[None, ...]
    # Poses: (C, 3) -> (C, 1, 3)
    rvec_exp = rvec[:, None, :]
    tvec_exp = tvec[:, None, :]
    # project_points will align intrinsics to (C, 1, 3, 3) because 'obj_exp' has 2 batch dimensions
    return project_points(obj_exp, rvec_exp, tvec_exp, camera_matrix, dist_coeffs, distortion_model)


def project_multiple_to_multiple(
        object_points: xp.ndarray,
        rvecs: xp.ndarray,
        tvecs: xp.ndarray,
        Ks: xp.ndarray,
        Ds: xp.ndarray,
        distortion_model: str = 'standard'
) -> Tuple[xp.ndarray, xp.ndarray]:
    """
    Projects P temporal snapshots of a point cloud into C camera views.
    Computes the Cartesian product of (C cameras) x (P frames).

    Args:
        object_points: Points in world coordinates for each frame (P, N, 3)
        rvecs: World-to-camera rotation vectors (C, 3)
        tvecs: World-to-camera translation vectors (C, 3)
        Ks: Camera matrice (C, 3, 3) or (3, 3)
        Ds: Distortion coefficients (C, D) or (D,)

    Returns:
        points2d: (C, P, N, 2)
        valid_mask: (C, P, N)
    """

    # Explicit tiling to ensure robust broadcasting for JAX einsum
    # Target shape: (C, P, N, ...)

    P, N, _ = object_points.shape
    C = rvecs.shape[0]

    # Expand Points: (P, N, 3) -> (1, P, N, 3) -> Tile to (C, P, N, 3)
    obj_exp = object_points[None, ...].repeat(C, axis=0)

    # Expand Extrinsics: (C, 3) -> (C, 1, 1, 3) -> Tile to (C, P, 1, 3)
    # N stays as singleton to let project_points broadcast the matrix mult
    rvec_exp = rvecs[:, None, None, :].repeat(P, axis=1)
    tvec_exp = tvecs[:, None, None, :].repeat(P, axis=1)

    # Expand Intrinsics
    if Ks.ndim == 3:  # (C, 3, 3)
        K_exp = Ks[:, None, None, :, :].repeat(P, axis=1)
    else:  # (3, 3)
        K_exp = Ks[None, None, None, :, :].repeat(C, axis=0).repeat(P, axis=1)

    if Ds.ndim == 2:  # (C, D)
        D_exp = Ds[:, None, None, :].repeat(P, axis=1)
    else:  # (D,)
        D_exp = Ds[None, None, None, :].repeat(C, axis=0).repeat(P, axis=1)

    # All inputs have batch shape (C, P, N) or (C, P, 1)
    # project_points handles the final broadcast over N
    return project_points(obj_exp, rvec_exp, tvec_exp, K_exp, D_exp, distortion_model)


@partial(jit, static_argnames=['distortion_model'])
def project_object_to_camera(
        object_points: xp.ndarray,
        r_w2c: xp.ndarray,
        t_w2c: xp.ndarray,
        r_o2w: xp.ndarray,
        t_o2w: xp.ndarray,
        camera_matrix: xp.ndarray,
        dist_coeffs: xp.ndarray,
        distortion_model: str = 'standard'
) -> Tuple[xp.ndarray, xp.ndarray]:
    """
    Projects 3D points from an object's local frame into a camera view by
    composing object-to-world and world-to-camera poses.

    T_o2c = T_w2c @ T_o2w
    """

    # Target batch dim based on points
    target_dim = object_points.ndim - 1

    r_w2c = _align_batch_dims(target_dim, r_w2c, 1)
    t_w2c = _align_batch_dims(target_dim, t_w2c, 1)
    r_o2w = _align_batch_dims(target_dim, r_o2w, 1)
    t_o2w = _align_batch_dims(target_dim, t_o2w, 1)

    # Compose poses: world -> camera and object -> world  ==>  object -> camera
    T_w2c = extrinsics_matrix(r_w2c, t_w2c)
    T_o2w = extrinsics_matrix(r_o2w, t_o2w)
    T_o2c = T_w2c @ T_o2w

    r_o2c, t_o2c = extmat_to_rtvecs(T_o2c)
    return project_points(object_points, r_o2c, t_o2c, camera_matrix, dist_coeffs, distortion_model)


def project_object_views_batched(
        object_points: xp.ndarray,
        r_w2c: xp.ndarray,
        t_w2c: xp.ndarray,
        r_o2w: xp.ndarray,
        t_o2w: xp.ndarray,
        camera_matrices: xp.ndarray,
        dist_coeffs: xp.ndarray,
        distortion_model: str = 'standard'
):
    """
    Projects N object points through P object poses into C cameras.

    Inputs:
        object_points: 3D points in object coordinates (N, 3)
        r_w2c, t_w2c: Camera poses (world -> cam), (C, 3)
        r_o2w, t_o2w: Object poses (object -> world), (P, 3)

    Returns:
        Projected points (C, P, N, 2)
    """

    # Cameras (C, ...) -> (C, 1, ...)
    r_w2c_exp = r_w2c[:, None, :]
    t_w2c_exp = t_w2c[:, None, :]

    # Object Poses (P, ...) -> (1, P, ...)
    r_o2w_exp = r_o2w[None, :, :]
    t_o2w_exp = t_o2w[None, :, :]

    # Compute net Extrinsics (C, P, 4, 4)
    T_w2c = extrinsics_matrix(r_w2c_exp, t_w2c_exp)  # (C, 1, 4, 4)
    T_o2w = extrinsics_matrix(r_o2w_exp, t_o2w_exp)  # (1, P, 4, 4)

    # Matmul broadcasts (C, 1, 4, 4) @ (1, P, 4, 4) -> (C, P, 4, 4)
    T_net = T_w2c @ T_o2w

    # Convert back to rvec/tvec (C, P, 3)
    r_net, t_net = extmat_to_rtvecs(T_net)

    # Project
    # Poses (C, P, 3), ok
    # Points (N, 3) -> Need (1, 1, N, 3)
    # Intrinsics (C, 3, 3) -> Need (C, 1, 1, 3, 3)

    obj_exp = object_points[None, None, :, :]

    r_net = r_net[:, :, None, :]  # (C, P, 1, 3)
    t_net = t_net[:, :, None, :]

    if camera_matrices.ndim == 3:
        K_exp = camera_matrices[:, None, None, :, :]
    else:
        K_exp = camera_matrices

    if dist_coeffs.ndim == 2:
        D_exp = dist_coeffs[:, None, None, :]
    else:
        D_exp = dist_coeffs

    return project_points(obj_exp, r_net, t_net, K_exp, D_exp, distortion_model)


@partial(jit, static_argnames=['distortion_model', 'max_iter'])
def undistort_points(
        points2d: xp.ndarray,
        camera_matrix: xp.ndarray,
        dist_coeffs: xp.ndarray,
        R: Optional[xp.ndarray] = None,
        P: Optional[xp.ndarray] = None,
        distortion_model: str = 'standard',
        max_iter: int = 5
) -> xp.ndarray:
    """
    Invert distortion & reprojection using Newton-Raphson iteration.
    (equivalent to cv2.undistortPoints).

    Args:
        points2d: Distorted 2D points (..., 2)
        camera_matrix: Intrinsics (3, 3)
        dist_coeffs: Distortion coefficients (..., D)
        R: Optional, rectification matrix (3, 3)
        P: Optional, new camera matrix (3, 3)
        max_iter: Max iterations for the solver

    Returns:
        Undistorted points (..., 2)
    """

    # Alignment
    target_dim = points2d.ndim - 1
    camera_matrix = _align_batch_dims(target_dim, camera_matrix, 2)
    dist_coeffs = _align_batch_dims(target_dim, dist_coeffs, 1)

    fx = camera_matrix[..., 0, 0]
    fy = camera_matrix[..., 1, 1]
    cx = camera_matrix[..., 0, 2]
    cy = camera_matrix[..., 1, 2]

    # Normalise distorted points
    x_d = (points2d[..., 0] - cx) / fx
    y_d = (points2d[..., 1] - cy) / fy

    # Initial guess for undistorted points is the distorted points
    x_u, y_u = x_d, y_d

    # Newton-Raphson iteration to find the undistorted normalised coordinates
    def newton(i, uv):
        x, y = uv
        radial, dx, dy = distortion(x, y, dist_coeffs, distortion_model)
        safe_radial = xp.where(xp.abs(radial) < _tiny, _tiny, radial)
        return ((x_d - dx) / safe_radial,
                (y_d - dy) / safe_radial)

    # This shim runs a python loop in numpy, or unrolled XLA loop in JAX. Both are fine for 5 iters.
    x_u, y_u = lax.fori_loop(0, max_iter, newton, (x_u, y_u))

    # (x_u, y_u) are undistorted, normalised coordinates (on the z=1 plane)
    ones = xp.ones_like(x_u)
    pts_h = xp.stack([x_u, y_u, ones], axis=-1) # homogeneous coordinates

    # Optional rectification and reprojection: mimics cv2.undistortPoints' R and P arguments

    if R is not None:   # if R is provided, apply rectification rotation
        R = _align_batch_dims(target_dim, R, 2)
        R_T = xp.swapaxes(R, -1, -2)
        pts_rectified = xp.einsum('...ij,...j->...i', R, pts_h)
    else:
        pts_rectified = pts_h

    # If P is provided, project using the new camera matrix
    # Otherwise, use the original camera matrix to return to pixel coordinates
    if P is not None:
        P = _align_batch_dims(target_dim, P, 2)
        new_fx, new_fy = P[..., 0, 0], P[..., 1, 1]
        new_cx, new_cy = P[..., 0, 2], P[..., 1, 2]
    else:
        new_fx, new_fy = fx, fy
        new_cx, new_cy = cx, cy

    # Project to pixel coordinates
    # Note: We use the components of the rectified point
    u_new = pts_rectified[..., 0] * new_fx + new_cx
    v_new = pts_rectified[..., 1] * new_fy + new_cy

    return xp.stack([u_new, v_new], axis=-1)


@partial(jit, static_argnames=['distortion_model'])
def back_projection(
        points2d: xp.ndarray,
        depth: Union[float, xp.ndarray],
        camera_matrix: xp.ndarray,
        T_c2w: xp.ndarray,
        dist_coeffs: Optional[xp.ndarray] = None,
        distortion_model: str = 'standard'
) -> xp.ndarray:
    """
    Back-project 2D points into 3D world coords at given depth.

    Args:
        points2d: (..., 2)
        depth: Scalar or (..., )
        camera_matrix: (..., 3, 3)
        T_c2w: Camera-to-World transform (..., 4, 4) or Extrinsics (..., 3, 4).
        dist_coeffs: Optional coeffs to undistort points first

    Returns:
        World points (..., 3)
    """

    # Align batch dimensions for numpy compatibility
    # points2d is (..., 2), so batch dims are ndim - 1
    target_batch_dim = points2d.ndim - 1

    # Align matrix inputs (data_dims=2 for 3x3 or 4x4 matrices)
    camera_matrix = _align_batch_dims(target_batch_dim, camera_matrix, data_dims=2)
    T_c2w = _align_batch_dims(target_batch_dim, T_c2w, data_dims=2)

    # Align depth input (data_dims=0 for scalars)
    depth_arr = xp.asarray(depth)
    if depth_arr.ndim > 0:
        depth_arr = _align_batch_dims(target_batch_dim, depth_arr, data_dims=0)

    # Undistort if needed
    if dist_coeffs is not None:
        points2d = undistort_points(
            points2d,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            distortion_model=distortion_model,
            R=xp.eye(3),
            P=camera_matrix,
        )

    ones = xp.ones_like(points2d[..., :1])
    hom2d = xp.concatenate([points2d, ones], axis=-1)

    invK = invert_intrinsics_matrix(camera_matrix)

    # invK @ hom2d -> (..., 1, 3, 3) @ (..., N, 3) -> (..., N, 3)
    cam_dirs = xp.einsum('...ij,...j->...i', invK, hom2d)

    # Depth broadcast
    if depth_arr.ndim == 0:
        cam_pts = cam_dirs * depth_arr
    else:
        # depth_arr is aligned (e.g. 5, 1), we extend to (5, 1, 1) to broadcast against (5, 6, 3)
        cam_pts = cam_dirs * depth_arr[..., None]

    hom_cam = xp.concatenate([cam_pts, ones], axis=-1)
    world_pts = xp.einsum('...ij,...j->...i', T_c2w[..., :3, :], hom_cam)

    return world_pts


@partial(jit, static_argnames=['per_point_errors'])
def reprojection_errors(
        points_2d_observed: xp.ndarray,
        points_2d_reprojected: xp.ndarray,
        visibility_mask: Optional[xp.ndarray] = None,
        per_point_errors: bool = False
) -> Dict[str, Union[float, xp.ndarray]]:
    """
    Calculates various reprojection error metrics.

    Args:
        points_2d_observed: Ground truth 2D points (..., N, 2)
        points_2d_reprojected: Reprojected 2D points (..., N, 2)
        visibility_mask: Boolean mask of visible points (..., N)
        per_point_errors: If True, include 'mre_per_point' in output

    Returns:
        Dictionary with 'rms', 'mre', 'opencv_rms', and optionally 'mre_per_point'
    """

    sq_diff = xp.square(points_2d_observed - points_2d_reprojected)

    if visibility_mask is not None:
        # Use where to avoid nans in gradients if they were to be used
        sq_diff_masked = xp.where(visibility_mask[..., None], sq_diff, 0.0)
        num_visible_points = xp.sum(visibility_mask.astype(xp.float32))
    else:
        sq_diff_masked = sq_diff
        num_visible_points = points_2d_observed.size // 2 # last dimension is 2, so number of points is total size / 2

    # Metric calculations
    # True RMS Error (of all 2*N coordinates)
    total_sum_sq_err = xp.sum(sq_diff_masked)
    rms_error = xp.sqrt(total_sum_sq_err / xp.maximum(2 * num_visible_points, 1))

    # Mean Reprojection Error (MRE - mean of per-point distances)
    distances = xp.sqrt(xp.sum(sq_diff, axis=-1))  # unmasked distances for per-point analysis

    if visibility_mask is not None:
        dist_masked = xp.where(visibility_mask, distances, 0.0)
    else:
        dist_masked = distances
    mre_error = xp.sum(dist_masked) / xp.maximum(num_visible_points, 1)

    # OpenCV 'calibrateCamera'-style RMS
    mean_sq_per_coord = xp.sum(sq_diff_masked, axis=-2) / xp.maximum(num_visible_points, 1)
    opencv_rms_error = xp.sqrt(xp.sum(mean_sq_per_coord))

    results = {'rms': rms_error, 'mre': mre_error, 'opencv_rms': opencv_rms_error}

    if per_point_errors:
        results['mre_per_point'] = xp.where(visibility_mask, distances,
                                            xp.nan) if visibility_mask is not None else distances
    return results


@jit
def triangulate_points_from_projections(
        points2d: xp.ndarray,  # (C, N, 2)
        P_mats: xp.ndarray,  # (C, 3, 4)
        weights: Optional[xp.ndarray] = None,
        lambda_reg: float = 0.0
) -> xp.ndarray:
    """
    Triangulates N 3D points from C 2D observations using SVD.

    Args:
        points2d: 2D observations (C, N, 2)
        P_mats: Projection matrices (C, 3, 4)
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
    P0 = P_mats[:, None, 0, :]
    P1 = P_mats[:, None, 1, :]
    P2 = P_mats[:, None, 2, :]

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
    w_coord = Xh[:, 3:4]
    safe_w = xp.where(xp.abs(w_coord) < _tiny, _tiny, w_coord)
    points3d = Xh[:, :3] / safe_w

    reliable = (n_obs >= 2)[:, None]
    return xp.where(reliable, points3d, xp.nan)


def triangulate(
        points2d: xp.ndarray,
        camera_matrices: xp.ndarray,
        dist_coeffs: xp.ndarray,
        rvecs_w2c: xp.ndarray,
        tvecs_w2c: xp.ndarray,
        weights: Optional[xp.ndarray] = None,
        distortion_model: str = 'standard',
):
    """
    High-level triangulation wrapper.
    Undistorts points, computes projection matrices, and solves 3D positions.

    Args:
        points2d: Observed points (C, N, 2)
        camera_matrices: Intrinsics (C, 3, 3)
        dist_coeffs: Distortion coeffs
        rvecs_w2c: Camera rotations (C, 3)
        tvecs_w2c: Camera translations (C, 3)
        weights: Optional weights (C, N)

    Returns:
        points3d: (N, 3)
    """

    pts2d_ud = undistort_points(points2d, camera_matrices, dist_coeffs, distortion_model=distortion_model)

    # if no mask is provided, infer it
    if weights is None:
        # undistortion might also introduce NaNs
        weights = xp.isfinite(points2d[..., 0]).astype(points2d.dtype)
    weights = weights.astype(xp.float32)

    T_w2c = extrinsics_matrix(rvecs_w2c, tvecs_w2c)
    P_mats = projection_matrix(camera_matrices, T_w2c)

    return triangulate_points_from_projections(pts2d_ud, P_mats, weights=weights)