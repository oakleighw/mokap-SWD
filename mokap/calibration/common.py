import logging
import cv2

import numpy as np
from mokap.geometry.backend import xp, ArrayLike

from typing import Tuple, Optional, Literal, Union, Sequence
from mokap.geometry import project, reprojection_errors, compose_transform_matrix
from mokap.utils.datatypes import ChessBoard, CharucoBoard, CalibrateCameraResult, DistortionModel

logger = logging.getLogger(__name__)


def PnP_wrapper(points3d, points2d, K, D, mode: Literal['IPPE', 'SQPNP', 'Iterative']):

    flags = {
        'sqpnp': cv2.SOLVEPNP_SQPNP,
        'iterative': cv2.SOLVEPNP_ITERATIVE,
        'ippe': cv2.SOLVEPNP_IPPE
    }

    if mode.lower() == 'iterative':
        try:
            success, rvec, tvec = cv2.solvePnPGeneric(points3d, points2d, K, D, flags=flags[mode.lower()])
            if success and tvec[2] > 0:
                return rvec, tvec
        except cv2.error:
            return None, None

    if mode.lower() in ['sqpnp', 'ippe']:
        try:
            nb, rvecs, tvecs, errs = cv2.solvePnPGeneric(points3d, points2d, K, D, flags=flags[mode.lower()])
        except cv2.error:
            return None, None

        if nb > 0:
            solutions = [{'rvec': r, 'tvec': t, 'error': e[0]} for r, t, e in zip(rvecs, tvecs, errs) if t[2] > 0]
            if solutions:
                best_solution = min(solutions, key=lambda x: x['error'])
                return best_solution['rvec'], best_solution['tvec']

    return None, None


def solve_pnp_robust(
        points3d:       ArrayLike,
        points2d:       ArrayLike,
        K:              Optional[ArrayLike],
        D:              Optional[ArrayLike],
        refine_method:  Optional[Literal['VVS', 'LM', 'none']] = None
) -> Tuple[bool, Optional[xp.ndarray], Optional[xp.ndarray], Optional[dict]]:
    """
    A robust wrapper for solvePnP that handles the ambiguity of planar targets
    It returns a single, physically plausible pose with the lowest reprojection error

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
        return False, None, None, None

    if K.shape != (3, 3):
        raise ValueError(f"Camera matrix must have shape (3, 3), but got {K.shape}")

    best_rvec, best_tvec, best_error = None, None, None
    points2d_xp = None
    points3d_xp = None
    K_xp = None
    D_xp = None

    # Try IPPE
    best_rvec, best_tvec = PnP_wrapper(points3d, points2d, K, D, 'IPPE')

    # Try SQPNP
    if best_rvec is None:
        candidate_rvec, candidate_tvec = PnP_wrapper(points3d, points2d, K, D, 'SQPNP')

        # Try Iterative
        if candidate_rvec is None:
            candidate_rvec, candidate_tvec = PnP_wrapper(points3d, points2d, K, D, 'Iterative')

        # Manual disambiguation for the best candidate
        if candidate_rvec is not None and candidate_tvec is not None:

            rvec_xp = xp.asarray(candidate_rvec).squeeze()
            tvec_xp = xp.asarray(candidate_tvec).squeeze()

            rvec_flip_xp, tvec_flip_xp = flip_pose_180(rvec_xp, tvec_xp)

            if tvec_flip_xp[2] <= 0:
                # The ambiguous pose is invalid, so the first candidate is probably correct
                best_rvec, best_tvec = candidate_rvec, candidate_tvec

            else:
                # if both are valid, compare their errors

                points2d_xp = xp.asarray(points2d)
                points3d_xp = xp.asarray(points3d)
                K_xp = xp.asarray(K)
                D_xp = xp.asarray(D)

                T_xp = compose_transform_matrix(rvec_xp, tvec_xp)
                T_flip_xp = compose_transform_matrix(rvec_flip_xp, tvec_flip_xp)

                reproj, _ = project(points3d_xp, T_xp, K_xp, D_xp)
                reproj_flip, _ = project(points3d_xp, T_flip_xp, K_xp, D_xp)

                reproj_errors = reprojection_errors(points2d_xp, reproj)
                reproj_errors_flip = reprojection_errors(points2d_xp, reproj_flip)

                # Compare using the standard RMS error
                if reproj_errors['rms'] <= reproj_errors_flip['rms']:
                    best_rvec, best_tvec = candidate_rvec, candidate_tvec
                    best_error = reproj_errors
                else:
                    best_rvec, best_tvec = rvec_flip_xp.reshape(3, 1), tvec_flip_xp.reshape(3, 1)  # Note [1]: still xp
                    best_error = reproj_errors_flip

    if best_rvec is None or best_tvec is None:
        # all methods failed to produce a valid pose
        return False, None, None, None

    # Optionally refine
    if refine_method and refine_method.lower() != 'none':
        try:
            refine_func_map = {'vvs': cv2.solvePnPRefineVVS, 'lm': cv2.solvePnPRefineLM}
            refine_func = refine_func_map[refine_method.lower()]

            best_rvec, best_tvec = refine_func(
                objectPoints=points3d, imagePoints=points2d, cameraMatrix=K,
                distCoeffs=D, rvec=np.asarray(best_rvec), tvec=np.asarray(best_tvec)   # Re note [1]: convert if needed
            )
            # After refinement, the already-calculated error is invalid and must be recalculated
            best_error = None
        except (cv2.error, AttributeError, KeyError):
            pass

    # No need for OpenCV anymore, these can be moved to GPU if needed
    best_rvec_xp = xp.asarray(best_rvec).squeeze()
    best_tvec_xp = xp.asarray(best_tvec).squeeze()

    if best_error is None:
        if points2d_xp is None or points3d_xp is None or K_xp is None or D_xp is None:
            # TODO: is this possible here?
            points2d_xp = xp.asarray(points2d)
            points3d_xp = xp.asarray(points3d)
            K_xp = xp.asarray(K)
            D_xp = xp.asarray(D)

        T_xp = compose_transform_matrix(best_rvec_xp, best_tvec_xp)
        final_reproj, _ = project(points3d_xp, T_xp, K_xp, D_xp)
        final_errors = reprojection_errors(points2d_xp, final_reproj)
    else:
        # otherwise, the one we stored is the correct one
        final_errors = best_error

    return True, best_rvec_xp, best_tvec_xp, final_errors


def calibrate_camera_robust(
    board:                  Union[ChessBoard, CharucoBoard],
    image_points_stack:     Sequence[np.ndarray],
    image_ids_stack:        Sequence[np.ndarray],
    image_size_wh:          Sequence[int],
    initial_K:              Optional[ArrayLike] = None,
    initial_D:              Optional[ArrayLike] = None,
    distortion_model:       DistortionModel = 'standard',
    fix_aspect_ratio:       bool = False
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

    # Set distortion flags based on the model
    if distortion_model == 'none':
        calib_flags |= (cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 |
                        cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6 |
                        cv2.CALIB_FIX_TANGENT_DIST)

    elif distortion_model == 'simple':
        # Optimize for k1, k2, but fix others
        calib_flags |= (cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6)

    elif distortion_model == 'rational':
        calib_flags |= cv2.CALIB_RATIONAL_MODEL

    # 'standard' and 'full' don't need special flags, they are the default behavior
    # when the corresponding CALIB_FIX_K* flags are not set
    try:
        if board.type == 'charuco':

            # calib_flags |= cv2.CALIB_USE_LU   # TODO: Should we use LU or QR? How 'worse' are they?

            (rms, K_new, D_new, rvecs, tvecs,
             std_intr, _, pve_opencv) = cv2.aruco.calibrateCameraCharucoExtended(
                charucoCorners=image_points_stack,
                charucoIds=image_ids_stack,
                board=board.to_opencv(),
                imageSize=image_size_wh,
                cameraMatrix=initial_K.copy() if initial_K is not None else None,
                distCoeffs=initial_D.copy() if initial_D is not None else None,
                flags=calib_flags
            )

        elif board.type == 'chessboard':

            # For chessboard, it's always all points, so we repeat
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
            return CalibrateCameraResult(success=False, error_message="Calibration resulted in an invalid camera matrix.")

        # TODO: These limits are for standard rectilinear lenses using the Brown-Conrady model
        # They are NOT suitable for fisheye lenses, which use a different model and calibration pipeline (cv2.fisheye.calibrate)
        absurd_distortion = False
        reason = ''
        d_abs = np.abs(D_new)

        if len(d_abs) >= 4 and (d_abs[2] > 0.5 or d_abs[3] > 0.5):
            # Tangential distortion should always be small for a well-centered lens
            # so |p1| > 0.5 or |p2| > 0.5 is almost certainly wrong
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

        # Note:
        # -----
        #
        # The per-view reprojection errors as returned by calibrateCamera() is:
        #   the square root of the sum of the 2 means in x and y of the squared diff
        #       np.sqrt(np.sum(np.mean(sq_diff, axis=0)))
        #
        # These are NOT the same as the per-view RMS errors typically computed after solvePnP():
        #   this one is the square root of the mean of the squared diff over both x and y
        #        np.sqrt(np.mean(sq_diff, axis=(0, 1)))
        #
        # In other words, the first one is larger by a factor √(2)
        #
        # In addition, the global RMS error returned by calibrateCamera() is:
        #       np.sqrt(np.sum([sq_diff for view in stack]) / np.sum([len(view) for view in stack]))
        #

        # ...so we just divide it by √2 and we're consistent with the rest of mokap
        pve_rms = pve_opencv.squeeze() / np.sqrt(2.0)

        # Package into the result dataclass
        return CalibrateCameraResult(
            success=True,
            rms_error=rms,
            K_new=K_new.squeeze(),
            D_new=D_new.squeeze(),
            rvecs=np.array(rvecs).squeeze(),
            tvecs=np.array(tvecs).squeeze(),
            std_devs_intrinsics=std_intr,
            per_view_errors=pve_rms
        )

    except cv2.error as e:
        error_msg = f"OpenCV Error in calibrateCamera: {e}"
        logger.warning(error_msg)
        return CalibrateCameraResult(success=False, error_message=error_msg)