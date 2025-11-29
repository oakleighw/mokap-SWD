from typing import Tuple
from .backend import xp, jit, _eps, _tiny


@jit
def rodrigues(rvec: xp.ndarray) -> xp.ndarray:
    """
    Converts rotation vector(s) to rotation matrix(ces).
    Uses Taylor expansion for small angles to ensure numerical stability.

    Args:
        rvec: Rotation vectors (..., 3)

    Returns:
        Rotation matrices (..., 3, 3)
    """

    # Magnitude of each vector
    theta_sq = xp.sum(rvec**2, axis=-1, keepdims=True)  # (..., 1)

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
    rx, ry, rz = rvec[..., 0], rvec[..., 1], rvec[..., 2]
    zeros_r = xp.zeros_like(rx)

    # Skew-symmetric matrix of the rvec
    R_skew = xp.stack([
        xp.stack([zeros_r, -rz, ry], axis=-1),
        xp.stack([rz, zeros_r, -rx], axis=-1),
        xp.stack([-ry, rx, zeros_r], axis=-1),
    ], axis=-2)

    R_small = I + R_skew

    return xp.where(is_small_angle[..., None], R_small, R_normal)


@jit
def inverse_rodrigues(Rmat: xp.ndarray) -> xp.ndarray:
    """
    Converts rotation matrix(ces) to rotation vector(s).
    Handles small angles by using the skew-symmetric approximation to avoid instability.

    Args:
        Rmat: Rotation matrices (..., 3, 3)

    Returns:
        Rotation vectors (..., 3)
    """

    trace = xp.trace(Rmat, axis1=-2, axis2=-1)
    costheta = (trace - 1) / 2

    # Clip to (-1+tiny, 1-tiny) to ensure finite gradients for arccos
    safe_costheta = xp.clip(costheta, -1.0 + _eps, 1.0 - _eps)
    theta_safe = xp.arccos(safe_costheta)

    # Skew part of the matrix
    rv_unscaled = xp.stack([
        Rmat[..., 2, 1] - Rmat[..., 1, 2],
        Rmat[..., 0, 2] - Rmat[..., 2, 0],
        Rmat[..., 1, 0] - Rmat[..., 0, 1]
    ], axis=-1)

    # Condition: theta near 0
    theta_check = xp.arccos(xp.clip(costheta, -1.0, 1.0))
    is_near_zero = theta_check < _eps

    # Path A: Normal angle (scale is theta / (2 * sin(theta)) )
    # sin(theta_safe) is safe because theta_safe is clipped away from 0 and pi
    scale_normal = theta_safe / (2 * xp.sin(theta_safe))

    # Path B: Small angle
    # Taylor expansion of theta / (2 sin theta) ~ 0.5 * (1 + theta^2 / 6)
    # rvec = rv_unscaled * 0.5 * (1 + theta^2/6)
    # First order: 0.5
    rvec_small_angle = 0.5 * rv_unscaled

    rvec_normal_angle = rv_unscaled * scale_normal[..., None]

    return xp.where(is_near_zero[..., None], rvec_small_angle, rvec_normal_angle)


@jit
def extrinsics_matrix(rvec: xp.ndarray, tvec: xp.ndarray) -> xp.ndarray:
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

    R = rodrigues(rvec)  # (..., 3, 3)
    t = tvec[..., None]  # (..., 3, 1)

    # Concatenate R and t into the extrinsics matrix (..., 3, 4)
    extmat = xp.concatenate([R, t], axis=-1)

    # Append the homogeneous coordinates row to make it a T mat
    batch_shape = extmat.shape[:-2]
    bottom_row = xp.array([0.0, 0.0, 0.0, 1.0], dtype=extmat.dtype)
    bottom_row = xp.broadcast_to(bottom_row, batch_shape + (1, 4))

    return xp.concatenate([extmat, bottom_row], axis=-2)


@jit
def extmat_to_rtvecs(T: xp.ndarray) -> Tuple[xp.ndarray, xp.ndarray]:
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
    rvec = inverse_rodrigues(R)  # (..., 3)
    return rvec, tvec


@jit
def projection_matrix(K: xp.ndarray, T: xp.ndarray) -> xp.ndarray:
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
def invert_intrinsics_matrix(K: xp.ndarray) -> xp.ndarray:
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

    R1 = rodrigues(r1)  # world to camera 1 rotation
    R2 = rodrigues(r2)  # world to camera 2 rotation

    # The relative transformation from camera 1's coordinate system to camera 2's is
    # T_c1_c2 = T_w_c2 * inv(T_w_c1)

    R_c1_c2 = R2 @ xp.swapaxes(R1, -1, -2)

    Rt1 = xp.einsum('...ij,...j->...i', R_c1_c2, t1)
    t_c1_c2 = t2 - Rt1

    # Construct skew matrix
    z = xp.zeros_like(t_c1_c2[..., 0])
    tx, ty, tz = t_c1_c2[..., 0], t_c1_c2[..., 1], t_c1_c2[..., 2]

    # The essential matrix E relates a point x1 in cam 1 to a point x2 in cam 2 via: x2^T * E * x1 = 0
    t_skew = xp.stack([
        xp.stack([z, -tz, ty], axis=-1),
        xp.stack([tz, z, -tx], axis=-1),
        xp.stack([-ty, tx, z], axis=-1)
    ], axis=-2)

    E_mat = t_skew @ R_c1_c2

    invK2_T = xp.swapaxes(invert_intrinsics_matrix(K2), -1, -2)
    invK1 = invert_intrinsics_matrix(K1)
    F = invK2_T @ E_mat @ invK1

    # Enforce rank-2
    U, S, Vt = xp.linalg.svd(F)
    mask = xp.array([1.0, 1.0, 0.0], dtype=S.dtype)
    S_new = S * mask

    # Recompose U @ diag(S) @ Vt
    F_corrected = (U * S_new[..., None, :]) @ Vt
    F_normalized = F_corrected / (F_corrected[..., 2:3, 2:3] + _tiny)

    return F_normalized


@jit
def invert_extrinsics_matrix(T: xp.ndarray) -> xp.ndarray:
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
def invert_rtvecs(rvec: xp.ndarray, tvec: xp.ndarray) -> Tuple[xp.ndarray, xp.ndarray]:
    """
    Inverts extrinsics vectors: (r, t) -> (r_inv, t_inv).
    """
    T = extrinsics_matrix(rvec, tvec)
    T_inv = invert_extrinsics_matrix(T)
    return extmat_to_rtvecs(T_inv)


@jit
def Rmat_from_angle(
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
    return rodrigues(rvec)


@jit
def rotate_points3d(
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

    R = Rmat_from_angle(angle_degrees, axis)
    return xp.einsum('...ij,...j->...i', R, points3d)


@jit
def rotate_rtvecs(
        rvecs: xp.ndarray,
        tvecs: xp.ndarray,
        angle_degrees: xp.ndarray,
        axis: xp.ndarray,
) -> tuple[xp.ndarray, xp.ndarray]:
    """
    Rotates a pose (rvec, tvec) by a global rotation defined by angle/axis.

    Args:
        rvecs: Input rotation vectors (..., 3)
        tvecs: Input translation vectors (..., 3)
        angle_degrees: Rotation angle (scalar or N)
        axis: Rotation axis (3,)

    Returns:
        Tuple of rotated rvecs and tvecs.
    """

    Rg = Rmat_from_angle(angle_degrees, axis)
    tvecs_rot = xp.einsum('...ij,...j->...i', Rg, tvecs)

    Rl = rodrigues(rvecs)
    R_comb = xp.matmul(Rg, Rl)  # Rg is (3, 3), it broadcasts to (..., 3, 3)
    rvecs_rot = inverse_rodrigues(R_comb)

    return rvecs_rot, tvecs_rot


@jit
def rotate_extrinsics_matrix(
        T: xp.ndarray,
        angle_degrees: xp.ndarray,
        axis: xp.ndarray,
) -> xp.ndarray:
    """
    Rotates a Transform matrix T by a global rotation defined by angle/axis.

    Args:
        T: Transform matrix (..., 4, 4)
        angle_degrees: Rotation angle
        axis: Rotation axis

    Returns:
        Rotated Transform matrix (..., 4, 4)
    """
    Rg = Rmat_from_angle(angle_degrees, axis)

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
def axisangle_to_quaternion(rvec: xp.ndarray) -> xp.ndarray:
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
def quaternion_to_axisangle(q: xp.ndarray) -> xp.ndarray:
    """
    Convert unit quaternions [w, x, y, z] to axis-angle rvec.

    Args:
        q: Quaternions (..., 4).

    Returns:
        rvec: Rotation vectors (..., 3).
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    w_clamped = xp.clip(w, -1.0 + _eps, 1.0 - _eps) # clip w to avoid NaN gradients in arccos
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
def quaternion_inverse(q: xp.ndarray) -> xp.ndarray:
    """
    Inverts unit quaternions. q_inv = [w, -x, -y, -z].
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return xp.stack([w, -x, -y, -z], axis=-1)


@jit
def quaternion_multiply(q1: xp.ndarray, q2: xp.ndarray) -> xp.ndarray:
    """
    Multiplies two sets of quaternions.
    """

    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return xp.stack([w, x, y, z], axis=-1)


@jit
def rotate_vector_by_quat(q: xp.ndarray, v: xp.ndarray) -> xp.ndarray:
    """
    Rotates 3D vector v by unit-quaternion q using q * v * q_inv.
    """

    # Pad v (..., 3) to quaternion [0, v] (..., 4)
    zeros = xp.zeros_like(v[..., :1])
    v_quat = xp.concatenate([zeros, v], axis=-1)

    q_inv = quaternion_inverse(q)

    # q * v * q_inv
    temp = quaternion_multiply(v_quat, q_inv)
    v_rot_quat = quaternion_multiply(q, temp)

    return v_rot_quat[..., 1:]


@jit
def quaternions_angular_distance(q1: xp.ndarray, q2: xp.ndarray) -> xp.ndarray:
    """
    Computes the geodesic angular distance between two unit quaternions.
    """
    d = xp.abs(xp.sum(q1 * q2, axis=-1))
    d = xp.clip(d, -1.0 + _eps, 1.0 - _eps) # clip for gradient safety
    return 2.0 * xp.arccos(d)