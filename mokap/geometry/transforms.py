from typing import Tuple
try:
    from .backend import xp, jit, _eps, _tiny
except ImportError:
    from mokap.geometry.backend import xp, jit, _eps, _tiny


def homogenize(
        points: xp.ndarray
) -> xp.ndarray:
    """
    Converts Euclidean points to homogeneous coordinates by appending 1.

    Args:
        points: Input points (..., D)

    Returns:
        Homogeneous points (..., D+1)
    """
    return xp.concatenate([points, xp.ones_like(points[..., :1])], axis=-1)


def dehomogenize(
        points: xp.ndarray
) -> xp.ndarray:
    """
    Converts homogeneous coordinates to Euclidean by dividing by the last component.

    Args:
        points: Homogeneous points (..., D+1)

    Returns:
        Euclidean points (..., D)
    """
    w_coord = points[..., -1:]
    safe_w = xp.where(xp.abs(w_coord) < _tiny, _tiny, w_coord)

    # Divide the coordinate components (all but the last) by w
    return points[..., :-1] / safe_w


@jit
def normalize_vector(
        v: xp.ndarray,
        axis: int = -1
) -> xp.ndarray:
    """
    Normalizes vectors to unit length. Handles zero-vectors safely.

    Args:
        v: Input vectors (..., D)
        axis: Axis to normalize along

    Returns:
        Unit vectors (..., D)
    """
    norm = xp.linalg.norm(v, axis=axis, keepdims=True)
    safe_norm = xp.where(norm < _tiny, 1.0, norm)
    return v / safe_norm


@jit
def skew_symmetric(v: xp.ndarray) -> xp.ndarray:
    """
    Returns the skew-symmetric cross-product matrix of vector v.
    [0 -z  y]
    [z  0 -x]
    [-y x  0]

    Args:
        v: Input vector (..., 3)

    Returns:
        Skew-symmetric cross-product matrix (..., 3, 3)
    """
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    zeros = xp.zeros_like(x)

    return xp.stack([
        xp.stack([zeros, -z, y], axis=-1),
        xp.stack([z, zeros, -x], axis=-1),
        xp.stack([-y, x, zeros], axis=-1),
    ], axis=-2)


@jit
def rotation_matrix(
        rvec: xp.ndarray
) -> xp.ndarray:
    """
    Converts rotation vector(s) to rotation matrix(ces).
    Uses Taylor expansion for small angles to ensure numerical stability.

    Args:
        rvec: Rotation vectors (..., 3)

    Returns:
        Rotation matrices (..., 3, 3)
    """

    # Magnitude of each vector
    theta_sq = xp.sum(rvec ** 2, axis=-1, keepdims=True)  # (..., 1)

    # When theta is small, we use the Taylor expansion of R = I + [r_x] to avoid division by theta
    is_zero = theta_sq < _tiny
    theta_sq_safe = xp.where(is_zero, 1.0, theta_sq)
    theta_safe = xp.sqrt(theta_sq_safe)

    is_small_angle = theta_sq < (_eps * _eps)

    # Path A: Normal angle
    # if rvec is 0, theta_safe is 1, k is 0. Gradients are fine.
    k = rvec / theta_safe  # (..., 3)
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]

    zeros = xp.zeros_like(kx)  # kx is (...)

    K = xp.stack([
        xp.stack([zeros, -kz, ky], axis=-1),
        xp.stack([kz, zeros, -kx], axis=-1),
        xp.stack([-ky, kx, zeros], axis=-1),
    ], axis=-2)

    sin_t = xp.sin(theta_safe)[..., None]
    cos_t = xp.cos(theta_safe)[..., None]

    I = xp.eye(3)  # broadcasts to (..., 3, 3)

    R_normal = I + sin_t * K + (1 - cos_t) * (K @ K)

    # Path B: Small angle (Taylor expansion)
    R_skew = skew_symmetric(rvec)
    R_small = I + R_skew

    return xp.where(is_small_angle[..., None], R_small, R_normal)


@jit
def rotation_vector(
        R: xp.ndarray
) -> xp.ndarray:
    """
    Converts rotation matrix(ces) to rotation vector(s).
    Handles small angles (Taylor) and 180-degree singularities (Eigen-analysis).

    Args:
        R: Rotation matrices (..., 3, 3)

    Returns:
        Rotation vectors (..., 3)
    """

    # Calculate angle
    trace = xp.trace(R, axis1=-2, axis2=-1)
    costheta = (trace - 1.0) / 2.0
    safe_costheta = xp.clip(costheta, -1.0 + _eps, 1.0 - _eps)
    theta = xp.arccos(safe_costheta)

    # Extract skew components (matrix -> vector)
    rv_unscaled = xp.stack([
        R[..., 2, 1] - R[..., 1, 2],
        R[..., 0, 2] - R[..., 2, 0],
        R[..., 1, 0] - R[..., 0, 1]
    ], axis=-1)

    # Define conditions
    is_small = theta < 1e-2
    # trace near -1 implies theta near pi (180 deg)
    is_pi = (trace + 1.0) < 1e-2

    # Path A: Normal angle
    # rvec = rv_unscaled * (theta / 2sin(theta))
    sin_t = xp.sin(theta)
    # Add tiny to denominator to prevent div/0 in the is_small/is_pi branches
    scale_normal = theta / (2.0 * sin_t + _tiny)
    rvec_normal = rv_unscaled * scale_normal[..., None]

    # Path B: Small angle (Taylor expansion)
    # theta / 2sin(theta) approx 0.5 * (1 + theta^2/6)
    rvec_small = rv_unscaled * (0.5 * (1.0 + theta[..., None] ** 2 / 6.0))

    # Path C: 180 deg singularity
    # R + I = 2 * v * v.T
    # The columns of (R+I) are parallel to the axis.
    # -> pick the column corresponding to the largest diagonal element to avoid numerical issues

    # Identify which column to take (max diagonal index)
    diag = xp.stack([R[..., 0, 0], R[..., 1, 1], R[..., 2, 2]], axis=-1)
    k = xp.argmax(diag, axis=-1)

    # Mask to extract column k
    mask = xp.eye(3, dtype=R.dtype)[k]

    # Extract the column: (R+I) @ mask
    I = xp.eye(3, dtype=R.dtype)
    R_plus_I = R + I

    # Matmul: (..., 3, 3) @ (..., 3, 1) -> (..., 3, 1) -> squeeze to (..., 3)
    col_k = xp.matmul(R_plus_I, mask[..., None]).squeeze(-1)

    # Normalise to get unit vector then scale by pi
    axis_pi = normalize_vector(col_k)
    rvec_pi = axis_pi * xp.pi

    # Select result
    # Priority small angle, then pi singularity, then normal
    rvec = xp.where(is_small[..., None], rvec_small, rvec_normal)
    rvec = xp.where(is_pi[..., None], rvec_pi, rvec)

    return rvec


@jit
def compose_transform_matrix(
        rvec: xp.ndarray,
        tvec: xp.ndarray
) -> xp.ndarray:
    """
    Combines rotation and translation vectors into 4x4 Homogeneous Transform matrices (T).

                     R_mat                      t
              [  r00  r01  r02   |    t0    ]   v
    T_mat  =  [  r10  r11  r12   |    t1    ]   e
              [  r20  r21  r22   |    t2    ]   c
              [ -----------------+--------- ]
              [   0    0    0    |     1    ]

    Args:
        rvec: Rotation vectors (..., 3)
        tvec: Translation vectors (..., 3)

    Returns:
        T: Homogeneous Transform matrices (..., 4, 4)
    """

    R = rotation_matrix(rvec)  # (..., 3, 3)
    t = tvec[..., None]  # (..., 3, 1)

    # Concatenate R and t into the extrinsics matrix (..., 3, 4)
    extmat = xp.concatenate([R, t], axis=-1)

    # Append the homogeneous coordinates row to make it a T mat
    batch_shape = extmat.shape[:-2]
    bottom_row = xp.array([0.0, 0.0, 0.0, 1.0], dtype=extmat.dtype)
    bottom_row = xp.broadcast_to(bottom_row, batch_shape + (1, 4))

    return xp.concatenate([extmat, bottom_row], axis=-2)


@jit
def decompose_transform_matrix(
        T: xp.ndarray
) -> Tuple[xp.ndarray, xp.ndarray]:
    """
    Decomposes a Transform matrix (T) into rotation and translation vectors.

    Args:
        T: Homogeneous transform matrices (..., 4, 4), or extrinsics matrix (..., 3, 4)

    Returns:
        rvec: Rotation vectors (..., 3)
        tvec: Translation vectors (..., 3)
    """
    R = T[..., :3, :3]  # (..., 3, 3)
    tvec = T[..., :3, 3]  # (..., 3)
    rvec = rotation_vector(R)  # (..., 3)
    return rvec, tvec


@jit
def transform_points(
        points3d: xp.ndarray,
        transform: xp.ndarray
) -> xp.ndarray:
    """
    Applies a 4x4 rigid transform to 3D points.
    Handles the homogeneous coordinate conversion internally.

    Args:
        points3d: (..., N, 3)
        transform: (..., 4, 4)
    """
    points_h = homogenize(points3d)
    transformed_h = xp.einsum('...ij,...nj->...ni', transform, points_h)
    return transformed_h[..., :3]


@jit
def transform_vectors(
        vectors: xp.ndarray,
        transform: xp.ndarray
) -> xp.ndarray:
    """
    Applies a 4x4 Homogeneous Transform to 3D vectors.
    This ignores the translation component of the transform because, well, vectors.

    Args:
        vectors: Input vectors (..., N, 3)
        transform: Homogeneous transform matrices (..., 4, 4)

    Returns:
        Rotated vectors (..., N, 3)
    """
    # We slice only the Rotation block (top-left 3x3)
    R = transform[..., :3, :3]

    # We do NOT add the translation vector (transform[..., :3, 3])
    return xp.einsum('...ij,...nj->...ni', R, vectors)


@jit
def projection_matrix(
        K: xp.ndarray,
        T: xp.ndarray
) -> xp.ndarray:
    """
    Computes projection matrices P = K @ [R|t].
    The projection matrix maps 3D points in homogeneous world coordinates (X, Y, Z, 1)
     to 2D points in homogeneous image coordinates (u, v, 1).

    2d_point      matrix_K          matrix_ext       3d_point
                (intrinsics)       (extrinsics)

    [ u ]     [ fx, 0, cx ]   [ r00, r01, r02, t0 ]   [ X ]
    [ v ]  =  [ 0, fy, cy ] . [ r10, r11, r12, t1 ] . [ Y ]
    [ 1 ]     [ 0,  0,  1 ]   [ r20, r21, r22, t2 ]   [ Z ]
                                                      [ 1 ]
    Args:
        K: Camera intrinsics matrices (..., 3, 3)
        T: Homogeneous transform matrices (..., 4, 4), or extrinsics matrix (..., 3, 4)

    Returns:
        P: Projection matrices (..., 3, 4)
    """

    extmat = T[..., :3, :]
    return xp.einsum('...ij,...jk->...ik', K, extmat)


@jit
def invert_intrinsics(
        K: xp.ndarray
) -> xp.ndarray:
    """
    Computes the analytical inverse of camera matrix K.

              [ 1/fx,   0,  -cx/fx ]
    inv_K  =  [  0,   1/fy,  cy/fy ]
              [  0,     0,     1   ]

    Args:
        K: Camera matrices (..., 3, 3)

    Returns:
        Inverse camera matrices (..., 3, 3)
    """
    fx = K[..., 0, 0]
    fy = K[..., 1, 1]
    cx = K[..., 0, 2]
    cy = K[..., 1, 2]

    zeros = xp.zeros_like(fx)
    ones = xp.ones_like(fx)

    invK = xp.stack([
        xp.stack([ones / fx, zeros, -cx / fx], axis=-1),
        xp.stack([zeros, ones / fy, -cy / fy], axis=-1),
        xp.stack([zeros, zeros, ones], axis=-1)
    ], axis=-2)
    return invK


@jit
def fundamental_matrix(
        K_pair: Tuple[xp.ndarray, xp.ndarray],
        rvecs_w2c_pair: Tuple[xp.ndarray, xp.ndarray],
        tvecs_w2c_pair: Tuple[xp.ndarray, xp.ndarray],
) -> xp.ndarray:
    """
    Computes the Fundamental Matrix (F) between two cameras.

    Uses the Essential Matrix formulation E = [t]_x R, then F = K2^-T @ E @ K1^-1.
    Enforces rank-2 constraint on F via SVD.

    Args:
        K_pair: Tuple of (K1, K2), each (..., 3, 3)
        rvecs_w2c_pair: Tuple of (rvec1, rvec2), each (..., 3)
        tvecs_w2c_pair: Tuple of (tvec1, tvec2), each (..., 3)

    Returns:
        F: Fundamental matrices (..., 3, 3).
    """
    K1, K2 = K_pair
    r1, r2 = rvecs_w2c_pair
    t1, t2 = tvecs_w2c_pair

    R1 = rotation_matrix(r1)  # world to camera 1 rotation
    R2 = rotation_matrix(r2)  # world to camera 2 rotation

    # The relative transformation from camera 1's coordinate system to camera 2's is
    # T_c1_c2 = T_w_c2 * inv(T_w_c1)

    R_c1_c2 = R2 @ xp.swapaxes(R1, -1, -2)

    Rt1 = xp.einsum('...ij,...j->...i', R_c1_c2, t1)
    t_c1_c2 = t2 - Rt1

    # Construct skew matrix
    t_skew = skew_symmetric(t_c1_c2)

    # The essential matrix E relates a point x1 in cam 1 to a point x2 in cam 2 via: x2^T * E * x1 = 0
    E_mat = t_skew @ R_c1_c2

    invK2_T = xp.swapaxes(invert_intrinsics(K2), -1, -2)
    invK1 = invert_intrinsics(K1)
    F = invK2_T @ E_mat @ invK1

    # Enforce rank-2
    U, S, Vt = xp.linalg.svd(F)
    mask = xp.array([1.0, 1.0, 0.0], dtype=S.dtype)
    S_new = S * mask

    # Recompose U @ diag(S) @ Vt
    F_corrected = (U * S_new[..., None, :]) @ Vt

    norm = xp.linalg.norm(F_corrected, axis=(-1, -2), keepdims=True)
    F_normalized = F_corrected / (norm + _tiny)

    return F_normalized


@jit
def essential_from_fundamental(
        F: xp.ndarray,
        K1: xp.ndarray,
        K2: xp.ndarray
) -> xp.ndarray:
    """
    Computes the Essential Matrix E from the Fundamental Matrix F.
    E = K2.T @ F @ K1

    Also enforces rank-2 constraint (singular values 1, 1, 0).

    Args:
        F: Fundamental matrices (..., 3, 3)
        K1: Intrinsics of camera 1 (..., 3, 3)
        K2: Intrinsics of camera 2 (..., 3, 3)

    Returns:
        Essential matrices (..., 3, 3)
    """

    K2_T = xp.swapaxes(K2, -1, -2)
    E_temp = xp.matmul(K2_T, xp.matmul(F, K1))

    # Essential Matrix constraints: singular values must be [s, s, 0]
    # (we enforce [1, 1, 0] by reconstructing using only the first 2 components)
    U, _, Vt = xp.linalg.svd(E_temp)

    # E = U @ diag(1,1,0) @ Vt
    # Reconstructed by summing the outer products of the first two singular vectors
    # U[..., :, 0:1] is (..., 3, 1)
    # Vt[..., 0:1, :] is (..., 1, 3)
    E = xp.matmul(U[..., :, 0:1], Vt[..., 0:1, :]) + xp.matmul(U[..., :, 1:2], Vt[..., 1:2, :])

    return E


@jit
def invert_transform(
        T: xp.ndarray
) -> xp.ndarray:
    """
    Inverts a 4x4 Homogeneous Transform Matrix (T).
    Exploits the structure of [R|t] to invert efficiently without general matrix inversion.

    Args:
        T: Transform matrices (..., 4, 4)

    Returns:
        Inverse transform matrices (..., 4, 4)
    """
    R = T[..., :3, :3]
    t = T[..., :3, 3]

    R_inv = xp.swapaxes(R, -1, -2)
    t_inv = -xp.einsum('...ij,...j->...i', R_inv, t)

    extmat = xp.concatenate([R_inv, t_inv[..., None]], axis=-1)

    batch_shape = T.shape[:-2]
    bottom_row = xp.broadcast_to(
        xp.array([0., 0., 0., 1.], dtype=T.dtype),
        batch_shape + (1, 4)
    )
    return xp.concatenate([extmat, bottom_row], axis=-2)


@jit
def invert_vectors(
        rvec: xp.ndarray,
        tvec: xp.ndarray
) -> Tuple[xp.ndarray, xp.ndarray]:
    """
    Inverts a pose defined by a rotation vector and translation vector.
    Computes (R^T, -R^T @ t).

    Args:
        rvec: Rotation vectors (..., 3)
        tvec: Translation vectors (..., 3)

    Returns:
        rvec_inv: Inverse rotation vectors (..., 3)
        tvec_inv: Inverse translation vectors (..., 3)
    """
    T = compose_transform_matrix(rvec, tvec)
    T_inv = invert_transform(T)
    return decompose_transform_matrix(T_inv)


@jit
def matrix_from_axis_angle(
        angle_degrees: xp.ndarray,
        axis: xp.ndarray
) -> xp.ndarray:
    """
    Creates rotation matrices from angle-axis representation.

    Args:
        angle_degrees: Rotation angle in degrees (scalar or (N,))
        axis: Rotation axis (3,) or (N, 3), will be normalised

    Returns:
        Rotation matrices (..., 3, 3)
    """
    angle_degrees = xp.asarray(angle_degrees)
    axis = xp.asarray(axis)

    theta = xp.deg2rad(angle_degrees)

    # Broadcast theta against axis if necessary
    if theta.ndim == axis.ndim - 1:
        theta = theta[..., None]

    # Normalise axis
    axis_norm = xp.linalg.norm(axis, axis=-1, keepdims=True)
    axis_u = axis / (axis_norm + _tiny)

    rvec = axis_u * theta
    return rotation_matrix(rvec)


@jit
def rotate_points(
        points3d: xp.ndarray,
        angle_degrees: xp.ndarray,
        axis: xp.ndarray,
) -> xp.ndarray:
    """
    Rotates 3D points around a specific axis.

     Args:
        points3d: 3D points to rotate (..., 3)
        angle_degrees: Rotation angle in degrees (scalar or (N,))
        axis: Rotation axis (3,) or (N, 3), will be normalised

    Returns:
        The rotated points (..., 3)
    """

    R = matrix_from_axis_angle(angle_degrees, axis)
    return xp.einsum('...ij,...j->...i', R, points3d)


@jit
def rotate_pose(
        T: xp.ndarray,
        angle_degrees: xp.ndarray,
        axis: xp.ndarray,
) -> xp.ndarray:
    """
    Rotates a pose matrix T by a global rotation defined by an axis and angle.

    Args:
        T: Homogeneous transform matrices (..., 4, 4)
        angle_degrees: Rotation angle in degrees (scalar or N)
        axis: Rotation axis (3,) or (N, 3)

    Returns:
        Rotated transform matrices (..., 4, 4)
    """
    Rg = matrix_from_axis_angle(angle_degrees, axis)

    R = T[..., :3, :3]
    t = T[..., :3, 3]

    R_new = Rg @ R
    t_new = xp.einsum('...ij,...j->...i', Rg, t)

    extmat_new = xp.concatenate([R_new, t_new[..., None]], axis=-1)

    batch_shape = T.shape[:-2]
    bottom_row = xp.broadcast_to(
        xp.array([0., 0., 0., 1.], dtype=T.dtype),
        batch_shape + (1, 4)
    )
    return xp.concatenate([extmat_new, bottom_row], axis=-2)


@jit
def translate_pose(
        T: xp.ndarray,
        translation: xp.ndarray
) -> xp.ndarray:
    """
    Applies a translation to an existing pose matrix T in its local frame.
    Equivalent to T_new = T + translation (applied to position column).

    Args:
        T: Homogeneous transform matrices (..., 4, 4)
        translation: Translation vectors (..., 3)

    Returns:
        Translated transform matrices (..., 4, 4)
    """

    R = T[..., :3, :3]
    t = T[..., :3, 3]

    t_new = t + translation

    extmat_new = xp.concatenate([R, t_new[..., None]], axis=-1)

    batch_shape = T.shape[:-2]
    bottom_row = xp.broadcast_to(
        xp.array([0., 0., 0., 1.], dtype=T.dtype),
        batch_shape + (1, 4)
    )
    return xp.concatenate([extmat_new, bottom_row], axis=-2)


@jit
def compose_transforms(
        pose: xp.ndarray,
        modifier: xp.ndarray
) -> xp.ndarray:
    """
    Just a mini wrapper for clarity.
    Applies a rigid transformation (modifier) to an existing pose.

    Args:
        pose: The target pose (..., 4, 4)
        modifier: The transformation to apply (..., 4, 4)

    Returns:
        pose_new = modifier @ pose
    """
    return modifier @ pose


@jit
def quaternion_from_vector(
        rvec: xp.ndarray
) -> xp.ndarray:
    """
    Convert axis-angle (Rodrigues) vectors rvec to unit quaternions [w, x, y, z].

    Args:
        rvec: Rotation vectors (..., 3).

    Returns:
        Quaternions (..., 4).
    """
    theta_sq = xp.sum(rvec ** 2, axis=-1, keepdims=True)

    # Safe theta for division
    is_zero = theta_sq < _tiny
    theta_sq_safe = xp.where(is_zero, 1.0, theta_sq)
    theta_safe = xp.sqrt(theta_sq_safe)

    is_small = theta_sq < (_eps * _eps)

    # Path A: Normal case
    half = 0.5 * theta_safe
    w = xp.cos(half)

    axis = rvec / theta_safe
    xyz = axis * xp.sin(half)

    q_normal = xp.concatenate([w, xyz], axis=-1)

    # Path B: Small angle (identity quat)
    q_identity = xp.array([1.0, 0.0, 0.0, 0.0], dtype=rvec.dtype)
    q_small = xp.broadcast_to(q_identity, rvec.shape[:-1] + (4,))

    return xp.where(is_small, q_small, q_normal)


@jit
def vector_from_quaternion(
        q: xp.ndarray
) -> xp.ndarray:
    """
    Convert unit quaternions [w, x, y, z] to axis-angle rvec.

    Args:
        q: Quaternions (..., 4).

    Returns:
        rvec: Rotation vectors (..., 3).
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    w_clamped = xp.clip(w, -1.0 + _eps, 1.0 - _eps)  # clip w to avoid NaN gradients in arccos
    theta = 2.0 * xp.arccos(w_clamped)

    s2 = 1.0 - w_clamped * w_clamped

    # Avoid div/0 and sqrt(0) gradients
    is_small = s2 < (_eps * _eps)
    safe_s2 = xp.where(is_small, 1.0, s2)
    s = xp.sqrt(safe_s2)

    # Path A: Normal case
    xyz = xp.stack([x, y, z], axis=-1)  # (..., 3)
    axis = xyz / s[..., None]
    normal_res = axis * theta[..., None]

    # Path B: Small angle case
    small_res = xp.zeros_like(xyz)

    return xp.where(is_small[..., None], small_res, normal_res)


@jit
def matrix_from_quaternion(
        q: xp.ndarray
) -> xp.ndarray:
    """
    Converts unit quaternions [w, x, y, z] to 3x3 rotation matrices.

    Args:
        q: Unit quaternions (..., 4)

    Returns:
        Rotation matrices (..., 3, 3)
    """

    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    x2, y2, z2 = x * 2, y * 2, z * 2
    wx, wy, wz = w * x2, w * y2, w * z2
    xx, xy, xz = x * x2, x * y2, x * z2
    yy, yz, zz = y * y2, y * z2, z * z2

    row0 = xp.stack([1 - (yy + zz), xy - wz, xz + wy], axis=-1)
    row1 = xp.stack([xy + wz, 1 - (xx + zz), yz - wx], axis=-1)
    row2 = xp.stack([xz - wy, yz + wx, 1 - (xx + yy)], axis=-1)

    return xp.stack([row0, row1, row2], axis=-2)


@jit
def quaternion_from_matrix(
        R: xp.ndarray
) -> xp.ndarray:
    """
    Converts 3x3 rotation matrices to unit quaternions [w, x, y, z].
    Shepperd's algorithm, to handle numerical instability near trace=0 or singularities.

    Args:
        R: Rotation matrices (..., 3, 3)

    Returns:
        Quaternions (..., 4)
    """

    # 4 candidate squared terms for the denominator
    # t = 1 + trace
    t = 1.0 + R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]

    # Candidates for 4 * q_w^2, 4 * q_x^2, etc
    c0 = t
    c1 = 1.0 + R[..., 0, 0] - R[..., 1, 1] - R[..., 2, 2]
    c2 = 1.0 - R[..., 0, 0] + R[..., 1, 1] - R[..., 2, 2]
    c3 = 1.0 - R[..., 0, 0] - R[..., 1, 1] + R[..., 2, 2]

    # Stack to find the largest candidate index (..., 4)
    candidates = xp.stack([c0, c1, c2, c3], axis=-1)
    best_idx = xp.argmax(candidates, axis=-1)

    # Get largest denominator
    largest = xp.max(candidates, axis=-1)
    scale = 0.5 * xp.sqrt(xp.maximum(largest, _tiny))
    inv_scale = 0.25 / scale

    # Compute all 4 potential quaternion sets
    # Case 0: w is dominant
    q0 = xp.stack([
        scale,
        (R[..., 2, 1] - R[..., 1, 2]) * inv_scale,
        (R[..., 0, 2] - R[..., 2, 0]) * inv_scale,
        (R[..., 1, 0] - R[..., 0, 1]) * inv_scale
    ], axis=-1)

    # Case 1: x is dominant
    q1 = xp.stack([
        (R[..., 2, 1] - R[..., 1, 2]) * inv_scale,
        scale,
        (R[..., 0, 1] + R[..., 1, 0]) * inv_scale,
        (R[..., 0, 2] + R[..., 2, 0]) * inv_scale
    ], axis=-1)

    # Case 2: y is dominant
    q2 = xp.stack([
        (R[..., 0, 2] - R[..., 2, 0]) * inv_scale,
        (R[..., 0, 1] + R[..., 1, 0]) * inv_scale,
        scale,
        (R[..., 1, 2] + R[..., 2, 1]) * inv_scale
    ], axis=-1)

    # Case 3: z is dominant
    q3 = xp.stack([
        (R[..., 1, 0] - R[..., 0, 1]) * inv_scale,
        (R[..., 0, 2] + R[..., 2, 0]) * inv_scale,
        (R[..., 1, 2] + R[..., 2, 1]) * inv_scale,
        scale
    ], axis=-1)

    # Select result based on best_idx
    m0 = (best_idx == 0)[..., None]
    m1 = (best_idx == 1)[..., None]
    m2 = (best_idx == 2)[..., None]
    m3 = (best_idx == 3)[..., None]

    return q0 * m0 + q1 * m1 + q2 * m2 + q3 * m3


@jit
def invert_quaternion(
        q: xp.ndarray
) -> xp.ndarray:
    """
    Inverts unit quaternions. q_inv = [w, -x, -y, -z].
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return xp.stack([w, -x, -y, -z], axis=-1)


@jit
def multiply_quaternions(q1: xp.ndarray, q2: xp.ndarray) -> xp.ndarray:
    """
    Computes the Hamilton product of two sets of quaternions.

    Args:
        q1: Left quaternions (..., 4)
        q2: Right quaternions (..., 4)

    Returns:
        Product quaternions (..., 4)
    """

    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return xp.stack([w, x, y, z], axis=-1)


@jit
def apply_quaternion(
        q: xp.ndarray,
        v: xp.ndarray
) -> xp.ndarray:
    """
    Rotates 3D vectors v by unit quaternions q.
    Implements the operation v' = q * v * q_inverse.

    Args:
        q: Unit quaternions (..., 4)
        v: 3D vectors (..., 3)

    Returns:
        Rotated vectors (..., 3)
    """

    # Pad v (..., 3) to quaternion [0, v] (..., 4)
    zeros = xp.zeros_like(v[..., :1])
    v_quat = xp.concatenate([zeros, v], axis=-1)

    q_inv = invert_quaternion(q)

    # q * v * q_inv
    temp = multiply_quaternions(v_quat, q_inv)
    v_rot_quat = multiply_quaternions(q, temp)

    return v_rot_quat[..., 1:]


@jit
def quaternion_distance(
        q1: xp.ndarray,
        q2: xp.ndarray
) -> xp.ndarray:
    """
    Computes the geodesic angular distance (in radians) between two unit quaternions.
    Defined as 2 * acos(|<q1, q2>|).

    Args:
        q1: First set of quaternions (..., 4)
        q2: Second set of quaternions (..., 4)

    Returns:
        Angular distance in radians (..., )
    """
    d = xp.abs(xp.sum(q1 * q2, axis=-1))
    d = xp.clip(d, -1.0 + _eps, 1.0 - _eps)  # clip for gradient safety
    return 2.0 * xp.arccos(d)


@jit
def angular_distance(
        R1: xp.ndarray,
        R2: xp.ndarray
) -> xp.ndarray:
    """
    Computes the element-wise geodesic angular distance between corresponding
    rotation matrices in two arrays.

    Args:
        R1: First set of rotation matrices (..., 3, 3)
        R2: Second set of rotation matrices (..., 3, 3)

    Returns:
        Angles in radians (..., )
    """

    # trace(R1 @ R2.T) is the sum of element-wise multiplication
    trace = xp.sum(R1 * R2, axis=(-1, -2))

    cos_theta = (trace - 1.0) / 2.0
    cos_theta = xp.clip(cos_theta, -1.0 + _eps, 1.0 - _eps)

    return xp.arccos(cos_theta)


@jit
def pairwise_angular_distance(
        R_A: xp.ndarray,
        R_B: xp.ndarray
) -> xp.ndarray:
    """
    Computes the angular distance between EVERY matrix in R_A and EVERY matrix in R_B.
    Useful for comparing a trajectory against ground truth or finding nearest neighbors.

    Args:
        R_A: Set A of rotation matrices (N, 3, 3)
        R_B: Set B of rotation matrices (M, 3, 3)

    Returns:
        Matrix of angles in radians (N, M)
    """

    # Einstein summation to compute trace(Ra @ Rb.T) for all pairs
    trace = xp.einsum('nij,mij->nm', R_A, R_B)

    cos_theta = (trace - 1.0) / 2.0
    cos_theta = xp.clip(cos_theta, -1.0 + _eps, 1.0 - _eps)

    return xp.arccos(cos_theta)