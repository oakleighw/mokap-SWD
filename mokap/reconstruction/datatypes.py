from collections import defaultdict
from dataclasses import dataclass, field
from typing import FrozenSet, Dict, Tuple, Optional, List
import numpy as np
from jax.typing import ArrayLike


@dataclass
class SoupPoint:
    """ A single reconstructed 3D point in the soup """

    frame_idx: int          # frame index
    idx: int                # keypoint's unique index within the frame
    keypoint_type: str
    position: ArrayLike     # x, y, z
    confidence: float       # aggregated confidence from reconstruction


Bone = FrozenSet[str]


@dataclass
class CandidateSkeleton:
    """ Assembler's internal representation for a potential skeleton during the assembly process """

    nodes: FrozenSet[Tuple[str, int]]
    scale: float
    competition_score: float
    anatomical_score: float
    constituent_indices: Optional[FrozenSet[int]] = None  # to track original fragments


@dataclass
class AssembledSkeleton:
    """ Represents a final assembled skeleton for a frame """

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
    """ Simple container that holds all skeletons and associated metadata for a tracklet """

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
        """ Creates an averaged pose from the start or end of the tracklet """

        skeletons_to_use = self.skeletons[-num_frames:] if end else self.skeletons[:num_frames]

        if not skeletons_to_use:
            return {}

        kp_positions = defaultdict(list)

        for skel in skeletons_to_use:
            for kp_name, pos in skel['keypoints'].items():
                kp_positions[kp_name].append(np.array(pos))

        return {name: np.mean(pos_list, axis=0) for name, pos_list in kp_positions.items()}

    def template_velocity(self, end: bool = False, num_frames: int = 3) -> np.ndarray:
        """ Creates an averaged velocity from the start or end of the tracklet """

        vels_to_use = self.velocities[-num_frames:] if end else self.velocities[:num_frames]
        return np.mean(vels_to_use, axis=0) if vels_to_use.size > 0 else np.zeros(3)

