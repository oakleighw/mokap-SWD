import logging
from dataclasses import dataclass, field
import cv2
import numpy as np
from mokap.geometry.backend import xp, ArrayLike
from typing import Tuple, Optional, Literal, Union, Sequence
from mokap.geometry import (
    project,
    reprojection_errors,
    compose_transform_matrix,
    rotation_matrix,
    rotation_vector,
    flip_rotation_180
)
from mokap.utils.datatypes import ChessBoard, CharucoBoard, DistortionModel

logger = logging.getLogger(__name__)


@dataclass
class CalibrateCameraResult:
    """A container for the results of an intrinsic camera calibration."""

    success: bool

    K_new: Optional[np.ndarray] = None
    D_new: Optional[np.ndarray] = None
    poses: Optional[np.ndarray] = None  # Transform matrices (N, 4, 4)

    rms_euclidean: float = np.inf
    rms_per_view: Optional[np.ndarray] = field(default=None, repr=False)

    std_devs_intrinsics: Optional[np.ndarray] = None

    error_message: str = ""


def PnP_wrapper(
        points3d: np.ndarray,
        points2d: np.ndarray,
        K: np.ndarray,
        D: np.ndarray,
        mode: Literal['IPPE', 'SQPNP', 'Iterative']
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Internal wrapper for OpenCV PnP solvers.
    Returns (rvec, tvec) as numpy arrays, or (None, None).
    """
    flags = {
        'sqpnp': cv2.SOLVEPNP_SQPNP,
        'iterative': cv2.SOLVEPNP_ITERATIVE,
        'ippe': cv2.SOLVEPNP_IPPE
    }

    if mode.lower() == 'iterative':
        try:
            success, r, t = cv2.solvePnPGeneric(points3d, points2d, K, D, flags=flags[mode.lower()])
            if success and len(t) > 0:
                t_chk = t[0]
                if t_chk[2] > 0:
                    return r[0], t[0]
        except cv2.error:
            pass

    if mode.lower() in ['sqpnp', 'ippe']:
        try:
            nb, rvecs, tvecs, errs = cv2.solvePnPGeneric(points3d, points2d, K, D, flags=flags[mode.lower()])
            if nb > 0:
                solutions = [{'rvec': r, 'tvec': t, 'error': e[0]} for r, t, e in zip(rvecs, tvecs, errs) if t[2] > 0]
                if solutions:
                    best_solution = min(solutions, key=lambda x: x['error'])
                    return best_solution['rvec'], best_solution['tvec']
        except cv2.error:
            pass

    return None, None


def solve_pnp_robust(
        points3d: ArrayLike,
        points2d: ArrayLike,
        K: Optional[ArrayLike],
        D: Optional[ArrayLike],
        refine_method: Optional[Literal['VVS', 'LM', 'none']] = None
) -> Tuple[bool, Optional[xp.ndarray], Optional[dict]]:
    """
    A robust wrapper for solvePnP that handles the ambiguity of planar targets.
    It returns a single, physically plausible 4x4 pose matrix with the lowest reprojection error.

    Strategy:
        Tries to use the IPPE algorithm which is designed for planar calibration boards
        Falls back to the robust SQPNP algorithm
        Falls back again to the lenient iterative algorithm
        Manually generates and checks ambiguous poses if the solver doesn't
        Optionally refines the final pose
    """

    # OpenCV needs these as numpy arrays, no matter what's mokap's backend
    points3d = np.asarray(points3d, dtype=np.float32)
    points2d = np.asarray(points2d, dtype=np.float32)
    K = np.asarray(K, dtype=np.float32) if K is not None else None
    D = np.asarray(D, dtype=np.float32) if D is not None else None

    # Shape validation
    if points3d.ndim != 2 or points3d.shape[1] != 3:
        raise ValueError(f"Object points must have shape (N, 3), but got {points3d.shape}")

    if points2d.ndim != 2 or points2d.shape[1] != 2:
        raise ValueError(f"Image points must have shape (N, 2), but got {points2d.shape}")

    if points3d.shape[0] != points2d.shape[0]:
        raise ValueError("Mismatch in number of object and image points.")

    if points3d.shape[0] < 4:
        # most PnP methods require at least 4 points
        return False, None, None

    if K.shape != (3, 3):
        raise ValueError(f"Camera matrix must have shape (3, 3), but got {K.shape}")

    # Try IPPE
    best_rvec, best_tvec = PnP_wrapper(points3d, points2d, K, D, 'IPPE')

    # Try SQPNP if IPPE failed
    if best_rvec is None:
        best_rvec, best_tvec = PnP_wrapper(points3d, points2d, K, D, 'SQPNP')

        # Try Iterative if SQPNP failed
        if best_rvec is None:
            best_rvec, best_tvec = PnP_wrapper(points3d, points2d, K, D, 'Iterative')

    if best_rvec is None:
        return False, None, None

    # Disambiguation: determine if the flipped pose is better

    # Pre-convert data to XP for geometry checks
    points3d_xp = xp.asarray(points3d)
    points2d_xp = xp.asarray(points2d)
    K_xp = xp.asarray(K)
    D_xp = xp.asarray(D)

    rvec_xp = xp.asarray(best_rvec).squeeze()
    tvec_xp = xp.asarray(best_tvec).squeeze()

    # Note: tvec does not change during the flip (rotation around object origin)

    rvec_flip_xp = rotation_vector(flip_rotation_180(rotation_matrix(rvec_xp)))

    T = compose_transform_matrix(rvec_xp, tvec_xp)
    T_flip = compose_transform_matrix(rvec_flip_xp, tvec_xp)

    reproj, _ = project(points3d_xp, T, K_xp, D_xp)
    reproj_flip, _ = project(points3d_xp, T_flip, K_xp, D_xp)

    errors_dict = reprojection_errors(points2d_xp, reproj)
    errors_dict_flip = reprojection_errors(points2d_xp, reproj_flip)

    if errors_dict['rms_euclidean'] <= errors_dict_flip['rms_euclidean']:
        best_error = errors_dict
    else:
        best_rvec = np.asarray(rvec_flip_xp)
        best_tvec = np.asarray(tvec_xp)  # tvec is the same
        best_error = errors_dict_flip

    if refine_method and refine_method.lower() != 'none':
        try:
            refine_func_map = {'vvs': cv2.solvePnPRefineVVS, 'lm': cv2.solvePnPRefineLM}
            refine_func = refine_func_map[refine_method.lower()]

            best_rvec, best_tvec = refine_func(
                objectPoints=points3d,
                imagePoints=points2d,
                cameraMatrix=K,
                distCoeffs=D,
                rvec=best_rvec,
                tvec=best_tvec
            )
            # After refinement the calculated error is invalid and must be recalculated
            best_error = None
        except (cv2.error, AttributeError, KeyError):
            pass

    # Final compose
    r_final_xp = xp.asarray(best_rvec).squeeze()
    t_final_xp = xp.asarray(best_tvec).squeeze()
    T_final = compose_transform_matrix(r_final_xp, t_final_xp)

    if best_error is None:
        # Re-calculate error if we refined the pose
        if points2d_xp is None:  # Safety re-cast if references were lost
            points2d_xp = xp.asarray(points2d)
            points3d_xp = xp.asarray(points3d)
            K_xp = xp.asarray(K)
            D_xp = xp.asarray(D)

        final_reproj, _ = project(points3d_xp, T_final, K_xp, D_xp)
        errors_dict_final = reprojection_errors(points2d_xp, final_reproj)
    else:
        errors_dict_final = best_error

    return True, T_final, errors_dict_final


def calibrate_camera_robust(
        board: Union[ChessBoard, CharucoBoard],
        image_points_stack: Sequence[np.ndarray],
        pointsIDs_stack: Sequence[np.ndarray],
        image_size_wh: Sequence[int],
        initial_K: Optional[ArrayLike] = None,
        initial_D: Optional[ArrayLike] = None,
        distortion_model: Union[DistortionModel, str] = 'standard',
        fix_aspect_ratio: bool = False
) -> CalibrateCameraResult:
    """ A convenience wrapper for OpenCV's camera calibration functions """

    # OpenCV needs numpy no matter what
    initial_K = np.asarray(initial_K) if initial_K is not None else None
    initial_D = np.asarray(initial_D) if initial_D is not None else None

    # Build calibration flags
    calib_flags = 0
    if initial_K is not None and initial_D is not None:
        calib_flags |= cv2.CALIB_USE_INTRINSIC_GUESS

        if fix_aspect_ratio:
            calib_flags |= cv2.CALIB_FIX_ASPECT_RATIO

    if distortion_model == 'none':
        calib_flags |= (cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 |
                        cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6 |
                        cv2.CALIB_FIX_TANGENT_DIST)
    elif distortion_model == 'simple':
        calib_flags |= (cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6)
    elif distortion_model == 'rational':
        calib_flags |= cv2.CALIB_RATIONAL_MODEL

    try:
        if board.type == 'charuco':
            (rms, K_new, D_new, rvecs, tvecs,
             std_intr, _, pve_opencv) = cv2.aruco.calibrateCameraCharucoExtended(
                charucoCorners=image_points_stack,
                charucoIds=pointsIDs_stack,
                board=board.to_opencv(),
                imageSize=image_size_wh,
                cameraMatrix=initial_K.copy() if initial_K is not None else None,
                distCoeffs=initial_D.copy() if initial_D is not None else None,
                flags=calib_flags
            )

        elif board.type == 'chessboard':
            object_points_stack = [board.object_points] * len(image_points_stack)
            (rms, K_new, D_new, rvecs, tvecs,
             std_intr, _, pve_opencv) = cv2.calibrateCameraExtended(
                objectPoints=object_points_stack,
                imagePoints=image_points_stack,
                imageSize=image_size_wh,
                cameraMatrix=initial_K.copy() if initial_K is not None else None,
                distCoeffs=initial_D.copy() if initial_D is not None else None,
                flags=calib_flags
            )
        else:
            return CalibrateCameraResult(success=False, error_message=f"Unsupported board type '{board.type}'.")

        # Check for invalid results
        invalid_vals = not (np.isfinite(K_new).all() and np.isfinite(D_new).all())
        negative_K_vals = (K_new < 0).any()
        invalid_central_point = (K_new[:2, 2] >= np.array(image_size_wh)).any()

        if invalid_vals or negative_K_vals or invalid_central_point:
            return CalibrateCameraResult(success=False,
                                         error_message="Calibration resulted in an invalid camera matrix.")

        # Sanity checks for lens distortion (NOT suitable for fisheye) # TODO: support fisheye here
        absurd_distortion = False
        reason = ''
        d_abs = np.abs(D_new)

        if len(d_abs) >= 4 and (d_abs[2] > 0.5 or d_abs[3] > 0.5):
            # Tangential distortion should always be small for a well-centered lens
            # so |p1| > 0.5 or |p2| > 0.5 is almost certainly wrong
            # TODO: This is not true if the image has been cropped...
            absurd_distortion = True
            reason = "Unplausible tangential distortion (p1, p2)"

        if not absurd_distortion and len(d_abs) >= 2:
            # A k1 or k2 value with an absolute magnitude > 2.0 is extremely rare for non-fisheye lenses
            if d_abs[0] > 1.5 or d_abs[1] > 2.0:
                absurd_distortion = True
                reason = "Unplausible radial distortion (k1, k2)"

        if not absurd_distortion and distortion_model in ['full', 'rational']:
            # Check higher-order terms for full or rational models
            if len(d_abs) >= 5 and np.any(d_abs[4:8] > 1.5):
                absurd_distortion = True
                reason = "Unplausible higher-order distortion (k3-k6)"

        if absurd_distortion:
            error_message = f"Calibration resulted in invalid distortion: {reason}. Values: {D_new.round(4)}"
            return CalibrateCameraResult(success=False, error_message=error_message)

        # Error conversion:
        # OpenCV returns the Component RMS (divides by 2N), but we want the Euclidean RMS (divides by N)
        # So: RMS_euclidean = RMS_component * sqrt(2)

        rms_euclidean = rms * np.sqrt(2.0)
        rms_euclidean_per_view = pve_opencv.squeeze() * np.sqrt(2.0)

        # Package results
        rvecs_xp = xp.asarray([r.squeeze() for r in rvecs])
        tvecs_xp = xp.asarray([t.squeeze() for t in tvecs])

        poses_stack = compose_transform_matrix(rvecs_xp, tvecs_xp)

        return CalibrateCameraResult(
            success=True,
            K_new=K_new.squeeze(),
            D_new=D_new.squeeze(),
            poses=poses_stack,
            rms_euclidean=rms_euclidean,
            rms_per_view=rms_euclidean_per_view,
            std_devs_intrinsics=std_intr
        )

    except cv2.error as e:
        error_msg = f"OpenCV Error in calibrateCamera: {e}"
        logger.warning(error_msg)
        return CalibrateCameraResult(success=False, error_message=error_msg)