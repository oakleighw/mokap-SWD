from collections import defaultdict
from dataclasses import dataclass, field
from typing import FrozenSet, Dict, Tuple, List
import numpy as np


@dataclass
class SoupData:
    """
    SoA container for 3D reconstructed points and 2D orphan rays.
    Can represent a single frame or an entire video.
    """

    # 3D points
    positions: np.ndarray           # (N, 3) float32
    confidences: np.ndarray         # (N,) float32
    kp_types: np.ndarray            # (N,) int16, maps to keypoint_names
    frame_indices: np.ndarray       # (N,) int32, frame index for batch processing
    camera_masks: np.ndarray        # (N,) uint32, bitmask (bit 0 = cam1, bit 1 = cam2, etc), maps to camera_names

    # 2D orphan rays (single view)
    ray_origins: np.ndarray         # (M, 3) float32, camera center
    ray_directions: np.ndarray      # (M, 3) float32, normalised direction vector
    ray_confidences: np.ndarray     # (M,) float32
    ray_kp_types: np.ndarray        # (M,) int16
    ray_frame_indices: np.ndarray   # (M,) int32

    # Metadata
    # mapping int types back to strings
    camera_names: List[str]
    keypoint_names: List[str]

    @property
    def num_points(self):
        return len(self.positions)

    def get_frame_slice(self, frame_idx: int) -> 'SoupData':
        """
        Zero-copy view of a single frame. Assumes data is sorted by frame_idx.
        """

        # Find range for 3D points
        start_p = np.searchsorted(self.frame_indices, frame_idx, side='left')
        end_p = np.searchsorted(self.frame_indices, frame_idx, side='right')

        # Find range for Rays
        start_r = np.searchsorted(self.ray_frame_indices, frame_idx, side='left')
        end_r = np.searchsorted(self.ray_frame_indices, frame_idx, side='right')

        return SoupData(
            positions=self.positions[start_p:end_p],
            confidences=self.confidences[start_p:end_p],
            kp_types=self.kp_types[start_p:end_p],
            frame_indices=self.frame_indices[start_p:end_p],
            camera_masks=self.camera_masks[start_p:end_p],

            ray_origins=self.ray_origins[start_r:end_r],
            ray_directions=self.ray_directions[start_r:end_r],
            ray_confidences=self.ray_confidences[start_r:end_r],
            ray_kp_types=self.ray_kp_types[start_r:end_r],
            ray_frame_indices=self.ray_frame_indices[start_r:end_r],

            camera_names=self.camera_names,
            keypoint_names=self.keypoint_names
        )

    @staticmethod
    def concatenate(soups: List['SoupData']) -> 'SoupData':
        """Merges a list of SoupData objects into one."""

        if not soups:
            raise ValueError("Cannot concatenate empty list of SoupData")

        # this assumes names match the first item
        cam_names = soups[0].camera_names
        kp_names = soups[0].keypoint_names

        return SoupData(
            positions=np.concatenate([s.positions for s in soups]),
            confidences=np.concatenate([s.confidences for s in soups]),
            kp_types=np.concatenate([s.kp_types for s in soups]),
            frame_indices=np.concatenate([s.frame_indices for s in soups]),
            camera_masks=np.concatenate([s.camera_masks for s in soups]),

            ray_origins=np.concatenate([s.ray_origins for s in soups]),
            ray_directions=np.concatenate([s.ray_directions for s in soups]),
            ray_confidences=np.concatenate([s.ray_confidences for s in soups]),
            ray_kp_types=np.concatenate([s.ray_kp_types for s in soups]),
            ray_frame_indices=np.concatenate([s.ray_frame_indices for s in soups]),

            camera_names=cam_names,
            keypoint_names=kp_names
        )

    @classmethod
    def empty(cls, cam_names: List[str], kp_names: List[str]):
        """Helper to create empty container."""

        return cls(
            positions=np.empty((0, 3), dtype=np.float32),
            confidences=np.empty((0,), dtype=np.float32),
            kp_types=np.empty((0,), dtype=np.int16),
            frame_indices=np.empty((0,), dtype=np.int32),
            camera_masks=np.empty((0,), dtype=np.uint32),
            ray_origins=np.empty((0, 3), dtype=np.float32),
            ray_directions=np.empty((0, 3), dtype=np.float32),
            ray_confidences=np.empty((0,), dtype=np.float32),
            ray_kp_types=np.empty((0,), dtype=np.int16),
            ray_frame_indices=np.empty((0,), dtype=np.int32),
            camera_names=cam_names,
            keypoint_names=kp_names
        )


Bone = FrozenSet[str]


@dataclass
class CandidateSkeleton:
    """Assembler's internal representation for a potential skeleton during the assembly process."""

    nodes: FrozenSet[Tuple[str, int]]
    scale: float
    competition_score: float
    anatomical_score: float


@dataclass
class AssembledSkeleton:
    """Represents a final assembled skeleton for a frame."""

    keypoints: Dict[str, np.ndarray]
    score: float
    scale: float
    point_indices: Dict[str, int] = field(default_factory=dict)
    track_idx: int = -1

    def to_dict(self) -> dict:
        return {
            'keypoints': self.keypoints,
            'score': self.score,
            'scale': self.scale,
            'point_indices': self.point_indices,
            'track_idx': self.track_idx
        }


@dataclass
class TrackletData:
    """Simple container that holds all skeletons and associated metadata for a tracklet."""

    idx: int
    frames: np.ndarray
    skeletons: List[Dict]
    healths: np.ndarray
    integrities: np.ndarray
    uncertainties: np.ndarray
    velocities: np.ndarray

    # The Kalman Filter's prediction of the central keypoint's position at the end of the tracklet
    end_state_prediction: np.ndarray

    @property
    def start(self) -> int:
        return int(self.frames[0])

    @property
    def end(self) -> int:
        return int(self.frames[-1])

    def template_pose(self, end: bool = False, num_frames: int = 3) -> Dict[str, np.ndarray]:
        """Creates an averaged pose from the start or end of the tracklet."""

        skeletons_to_use = self.skeletons[-num_frames:] if end else self.skeletons[:num_frames]

        if not skeletons_to_use:
            return {}

        kp_positions = defaultdict(list)

        for skel in skeletons_to_use:
            for kp_name, pos in skel['keypoints'].items():
                kp_positions[kp_name].append(np.array(pos))

        return {name: np.mean(pos_list, axis=0) for name, pos_list in kp_positions.items()}

    def template_velocity(self, end: bool = False, num_frames: int = 3) -> np.ndarray:
        """Creates an averaged velocity from the start or end of the tracklet."""

        vels_to_use = self.velocities[-num_frames:] if end else self.velocities[:num_frames]
        return np.mean(vels_to_use, axis=0) if vels_to_use.size > 0 else np.zeros(3)

