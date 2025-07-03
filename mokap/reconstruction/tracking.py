import json
import pickle
import re
import logging
from typing import Tuple, Union, Optional,  Dict, List, FrozenSet
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import pandas as pd
import networkx as nx
import numpy as np
from alive_progress import alive_bar
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter
from scipy.linalg import block_diag
from scipy.optimize import linear_sum_assignment
from scipy.stats import median_abs_deviation
from scipy.spatial import cKDTree
from mokap.utils import fileio
from mokap.utils.geometry.fitting import find_rigid_transform


# TODO: Profile the two solvers a bit more
#
# from pyscipopt import Model
# import cProfile
# import pstats
#
# def solve_mwis_SCIP(graph: nx.Graph) -> list[int]:
#     """ Solves the MWIS problem using the SCIP ILP solver """
#
#     model = Model("mwis")
#     model.hideOutput()
#
#     # Create a binary variable for each node in the graph
#     # The variable will be 1 if the node is in the solution, 0 otherwise
#     nodes = list(graph.nodes())
#     variables = {node: model.addVar(vtype="B", name=f"x_{node}") for node in nodes}
#
#     # Set the objective function: Maximize the sum of the weights of the selected nodes
#     objective_terms = [graph.nodes[node]['weight'] * variables[node] for node in nodes]
#     model.setObjective(sum(objective_terms), "maximize")
#
#     # Add constraints: For every edge (u, v) in the conflict graph, the two nodes
#     # cannot be chosen together. This is the "independent set" constraint
#     # x_u + x_v <= 1
#     for u, v in graph.edges():
#         model.addCons(variables[u] + variables[v] <= 1)
#
#     # Solve the model
#     model.optimize()
#
#     # Extract the solution
#     solution_nodes = []
#     if model.getStatus() == "optimal":
#         for node in nodes:
#             # Check if the variable is close to 1 in the solution
#             if model.getVal(variables[node]) > 0.99:
#                 solution_nodes.append(node)
#
#     return solution_nodes


def solve_mwis_networkx(graph: nx.Graph) -> list[int]:
    """ The basic MWIS solver using networkx """

    # TODO: move this to utils bc it is also used by the reconstructor

    if graph.number_of_nodes() == 0:
        return []

    # The MWC of the complement is equivalent to MWIS of the original
    complement_graph = nx.complement(graph)

    # Copy weights to the complement graph
    node_weights = nx.get_node_attributes(graph, 'weight')
    nx.set_node_attributes(complement_graph, node_weights, name='weight')

    winner_indices, _ = nx.algorithms.clique.max_weight_clique(complement_graph, weight='weight')
    return winner_indices


logger = logging.getLogger(__name__)


# Type aliases for clarity and type safety
Bone = FrozenSet[str]
PointNode = Tuple[str, int]


@dataclass
class PointObservation:
    """ Represents a single 3D point observation with its confidence """
    pos: np.ndarray
    conf: float = 1.0

@dataclass
class AssembledSkeleton:
    """ Represents a final assembled skeleton for a frame """
    keypoints: Dict[str, np.ndarray]
    score: float
    scale: float
    point_indices: Dict[str, int] = field(default_factory=dict)
    track_id: int = -1

    def to_dict(self) -> dict:
        return {
            'keypoints': self.keypoints,
            'score': self.score,
            'scale': self.scale,
            'point_indices': self.point_indices,
            'track_id': self.track_id
        }

@dataclass
class CandidateSkeleton:
    """ Assembler's internal representation for a potential skeleton during the assembly process """
    nodes: FrozenSet[PointNode]  # a frozenset of (kp_name, point_index) tuples
    score: float
    scale: float
    original_score: float



CONFIG = {

    # --- Stats Bootstrapper parameters ---
    "BOOTSTRAP_GENERIC_MAD_RATIO": 0.10,    # Generic MAD ratio to apply when bootstrapping from a simple prior or if data-driven MAD is zero
    "BOOTSTRAP_MIN_SAMPLES": 20,            # Min samples needed to calculate data-driven stats for a bone
    "BOOTSTRAP_MAX_BONE_LEN": 3.0,          # Max bone length for the simple greedy assembler used in bootstrapping

    # --- Anatomy Learner parameters ---
    "LEARNER_MIN_SAMPLES_FOR_UPDATE": 100,  # How many new high-quality skeleton measurements before re-calculating anatomy stats
    "LEARNER_MIN_SCORE_FOR_LEARNING": 5.0,  # Minimum score of a tracked skeleton to be used for learning
    "LEARNER_MIN_REF_BONE_LEN": 1.0,        # Sanity check: min plausible length of reference bone for learning
    "LEARNER_MAX_REF_BONE_LEN": 15.0,       # Sanity check: max plausible length of reference bone for learning

    # --- Assembler parameters ---
    "ASSEMBLER_MIN_KPS_FOR_SKELETON": 3,        # Min keypoints to be considered a valid skeleton fragment
    "ASSEMBLER_BONE_SCORE_MAD_THRESH": 5.0,     # How far a bone's length can deviate from expected (in MADs) before its score is zero
    "ASSEMBLER_BONE_SCORE_MAD_EPSILON": 0.05,   # Small constant added to MAD for numerical stability
    "ASSEMBLER_MIN_SANE_SCALE": 0.7,            # Min plausible scale estimate for a skeleton fragment
    "ASSEMBLER_MAX_SANE_SCALE": 1.5,            # Max plausible scale estimate for a skeleton fragment
    "ASSEMBLER_SCORE_DEBT_TOLERANCE": 10.0,     # How much of a score hit is it possible to take to add one more part

    # --- Conflict solver parameters ---
    "CONFLICT_SOLVER_BROAD_RADIUS": 3.0,        # Skeletons with centroids further than this are assumed not to conflict
    "CONFLICT_SOLVER_PROXIMITY_RADIUS": 0.25,   # Max distance to consider two corresponding keypoints 'the same'
    "CONFLICT_SOLVER_JACCARD_THRESHOLD": 0.85,  # Jaccard proximity threshold to consider two skeletons 'clones'

    # --- Tracker parameters ---
    "TRACKER_MAX_AGE": 15,                  # How many frames a track can coast without an update before being deleted
    "TRACKER_MIN_KPS_FOR_INFERENCE": 3,     # Min shared KPs needed to infer a missing central keypoint via alignment
    "TRACKER_SCALE_LEARNING_RATE": 0.25,    # Learning rate for the track's adapting scale estimate
    "TRACKER_ASSOCIATION_RADIUS": 1.0,      # Max distance between a track's prediction and a candidate for association
    "TRACKER_ASSOCIATION_MIN_KPS": 3,       # Min shared keypoints to associate a track with a candidate
    "TRACKER_CONTINUITY_BONUS": 500.0,      # Large bonus to a candidate's score if it matches an existing track
    
    # --- Cost function weights (for final assignment) ---
    "COST_POSE_DISTANCE_WEIGHT": 0.9,       # Weights for the Hungarian algorithm cost matrix. Lower cost = better
    "COST_SKELETON_SCORE_WEIGHT": -0.1,     # A higher intrinsic skeleton score should lower the cost (hence negative)

    # --- Kalman Filter parameters ---
    "KF_PROCESS_NOISE_POS": 0.1,            # Process noise for position (assumes random acceleration). Higher = less smooth
    "KF_PROCESS_NOISE_SCALE": 0.01,         # Process noise for scale
    "KF_MEASUREMENT_NOISE_POS": 5.0,        # Measurement noise for position (reflects 3D reconstruction uncertainty)
    "KF_MEASUREMENT_NOISE_SCALE": 0.25,     # Measurement noise for scale
    "KF_INIT_COV_VEL": 1.0,                 # Initial covariance for velocity
    "KF_INIT_COV_SCALE": 1.0,               # Initial covariance for scale
    "KF_INFERENCE_UNCERTAINTY_FACTOR": 2.0, # Multiplier for measurement noise when a keypoint position is inferred, not measured
}

# TODO: move this to the statsbootstraper
def create_symmetry_map(symmetry_groups: List[Tuple[str, str]]) -> Dict[str, str]:
    """
    Creates a map from a keypoint to a canonical symmetrical name
    eg {'left_eye': 'eye', 'right_eye': 'eye'}
    """
    sym_map = {}
    for group in symmetry_groups:
        if not group: continue
        canonical_name = group[0].replace('left_', '').replace('right_', '')
        for kp_name in group:
            sym_map[kp_name] = canonical_name
    return sym_map

def get_side(kp_name: str) -> Optional[str]:
    """ Returns 'left', 'right', or None if the side is not specified """
    if kp_name.lower().startswith('left'): return 'left'
    if kp_name.lower().startswith('right'): return 'right'
    return None


class StatsBootstrapper:
    """
    Handles the loading, creation, and standardization of anatomical statistics
    """

    def __init__(self,
                 output_path:       Union[str, Path],
                 bones_list:        List[Tuple[str, str]],
                 symmetry_map:      Optional[List[Tuple[str, str]]] = None,
                 prior_stats_path:  Optional[Union[str, Path]] = None,
                 bootstrap_data:    Optional[List[Dict]] = None
                 ):

        self.bones_list: List[Bone] = [frozenset(b) for b in bones_list]

        # Build skeleton graph once for degree calculations
        # TODO: This is also done in the skeleton assembler, that's a bit redundant
        self._skeleton_graph = nx.Graph()
        self._skeleton_graph.add_edges_from([tuple(b) for b in self.bones_list])
        self._degrees = dict(self._skeleton_graph.degree())

        self.output_path = Path(output_path)
        self.prior_path = Path(prior_stats_path) if prior_stats_path else None
        self.bootstrap_data = bootstrap_data

        self.delimiter_regex = re.compile(r'[-;, ]')

        if symmetry_map is not None:
            self.symmetry_map = create_symmetry_map(symmetry_map)
            logger.info(f"Symmetry map created for {len(self.symmetry_map)} keypoints.")

    def get_initial_stats(self) -> Dict:
        """
        Main method to get stats. Tries 3 ways to do it:
        1. Load from a user-provided prior stats file
        2. Load from a previously generated output stats file
        3. Bootstrap from 3D data if provided
        """

        # try loading a user-provided prior file
        if self.prior_path and self.prior_path.exists():
            logger.info(f"Loading stats from provided prior file: '{self.prior_path.name}'")

            return self._load_and_process_file(self.prior_path)

        # check if the designated output file already exists
        if self.output_path.exists():
            logger.info(f"Loading pre-existing stats file: '{self.output_path.name}'")

            with open(self.output_path, 'r') as f:
                stats = json.load(f)

            return self._validate_and_normalize_stats(stats) # always validate pre-existing files

        # try to bootstrap from data
        if self.bootstrap_data:
            logger.info("No stats file provided or found. Bootstrapping from 3D data...")

            stats = self._bootstrap_from_data()
            self._save_stats(stats)

            return stats

        raise ValueError('[ERROR] Could not obtain stats. No prior file provided, no existing stats file found, and no bootstrap data given.')

    def _get_most_stable_bone(self, available_bones: set[Bone]) -> Bone:
        """
        Selects the most stable bone as defined by the sum of the degrees of the two keypoints
        forming it. This favors central well-connected bones.
        """

        if not available_bones:
            raise ValueError("Cannot select a stable bone from an empty set.")

        best_bone = None
        max_score = -1

        for bone in available_bones:
            kp1, kp2 = tuple(bone)
            if kp1 not in self._degrees or kp2 not in self._degrees:
                continue

            # Score is the sum of the degrees of the two keypoints
            score = self._degrees[kp1] + self._degrees[kp2]

            if score > max_score:
                max_score = score
                best_bone = bone

        if best_bone is None:
            # Fallback to the first available bone if no degrees match (disjoint graph)
            return list(available_bones)[0]

        return best_bone

    def _parse_bone_name(self, bone_name: str) -> Bone:
        parts = [str(p).strip() for p in self.delimiter_regex.split(bone_name) if str(p).strip()]

        if len(parts) != 2:
            raise ValueError(f"Could not parse bone name '{bone_name}'. Expected two keypoints separated by a delimiter.")

        return frozenset(parts)

    def _load_and_process_file(self, file_path: Path) -> Dict:
        """ Loads a JSON or CSV file, normalises it, and ensures std dev exists """

        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip().str.lower()

            if not {'bone', 'length'}.issubset(df.columns):
                raise ValueError("CSV file must contain 'bone' and 'length' columns.")

            bone_data = df.set_index('bone')['length'].to_dict()

        elif file_path.suffix == '.json':
            with open(file_path, 'r') as f:
                json_data = json.load(f)

            if 'bones_ratios' in json_data:
                print("[INFO] File is already in final stats format. Validating...")
                return self._validate_and_normalize_stats(json_data)

            bone_data = json_data

        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}. Please use .csv or .json.")

        stats = self._normalize_lengths(bone_data)
        self._save_stats(stats)

        return stats

    def _symmetrise_bone_names(self, bone: Bone) -> Bone:
        """ Normalizes a bone's keypoint names using the symmetry map """

        if not self.symmetry_map:
            return bone

        kp1, kp2 = tuple(bone)

        # Check if both keypoints are part of the symmetry definition
        if kp1 not in self.symmetry_map or kp2 not in self.symmetry_map:
            return bone

        # Check if they belong to the same side (or neither has a side)
        side1, side2 = get_side(kp1), get_side(kp2)

        if side1 is not None and side2 is not None and side1 != side2:
            # This is a cross-body bone (eg left_hip to right_hip), do not normalize it
            return bone

        # Normalize to canonical names
        return frozenset((self.symmetry_map[kp1], self.symmetry_map[kp2]))

    def _normalize_lengths(self, bones_data: Dict[str, float]) -> Dict:
        """ Normalises a dictionary of bone lengths using a reference bone """

        parsed_bones_data = {self._parse_bone_name(name): length for name, length in bones_data.items()}

        if not parsed_bones_data:
            raise ValueError("Provided bones data is empty.")

        ref_bone = self._get_most_stable_bone(set(parsed_bones_data.keys()))
        ref_bone_len = parsed_bones_data[ref_bone]
        
        if ref_bone_len <= 0:
            raise ValueError(f"Reference bone '{' - '.join(ref_bone)}' has a non-positive length ({ref_bone_len}).")

        bones_ratios = {}
        generic_mad_ratio = CONFIG["BOOTSTRAP_GENERIC_MAD_RATIO"]
        for name, length in parsed_bones_data.items():
            median_ratio = length / ref_bone_len
            bone_key = ';'.join(sorted(list(name)))     # we want to serialize with consistent sorting
            bones_ratios[bone_key] = {
                'median_ratio': float(median_ratio),
                'mad_ratio': float(median_ratio * generic_mad_ratio)  # Default MAD
            }

        logger.info(f"Using data-driven reference bone: '{'-'.join(ref_bone)}' (length: {ref_bone_len:.2f})")
        return {
            'reference_bone': sorted(list(ref_bone)),
            'median_reference_length': float(ref_bone_len),
            'bones_ratios': bones_ratios
        }

    def _validate_and_normalize_stats(self, stats: Dict) -> Dict:
        """ Checks a fully-formatted stats dict and adds missing deviation """

        generic_mad_ratio = CONFIG["BOOTSTRAP_GENERIC_MAD_RATIO"]
        updated = False

        # Standardize keys first, in case they come from a file with mixed delimiters
        standardized_ratios = {
            ';'.join(sorted(list(self._parse_bone_name(k)))): v
            for k, v in stats['bones_ratios'].items()
        }
        stats['bones_ratios'] = standardized_ratios

        # Now check for missing MAD values
        for bone_key, bone_stats in stats['bones_ratios'].items():
            if 'mad_ratio' not in bone_stats or not np.isfinite(bone_stats['mad_ratio']) or bone_stats['mad_ratio'] == 0:
                bone_stats['mad_ratio'] = float(bone_stats['median_ratio'] * generic_mad_ratio)
                updated = True

        if updated:
            logger.debug(f"Standardized bone names and/or added missing MAD values using generic {generic_mad_ratio * 100:.1f}% ratio.")
            self._save_stats(stats)

        return stats

    def _greedy_assembler(self, frame_3d_data:  Dict) -> List[dict]:
        """ Simplified assembler for bootstrapping stats. Builds skeletons greedily from the central keypoint """

        if 'points' not in frame_3d_data or not frame_3d_data['points']:
            return []

        all_points = frame_3d_data['points']
        skeletons = []

        # Keep track of used points (kp_name, index) to avoid redundant assemblies
        used_point_nodes = set()

        # Iterate through every point of every type as a potential seed
        for seed_kp, (seed_kp_points, _) in all_points.items():

            for i in range(len(seed_kp_points)):
                if (seed_kp, i) in used_point_nodes:
                    continue

                center_pos = seed_kp_points[i]
                skeleton = {'keypoints': {seed_kp: center_pos}}

                # Greedily find the closest available keypoint of each other type
                for kp_name, (points, _) in all_points.items():
                    if kp_name == seed_kp or not points.size:
                        continue

                    # Find the single closest point of this type to the seed position
                    distances = np.linalg.norm(points - center_pos, axis=1)
                    best_idx = np.argmin(distances)

                    if distances[best_idx] < CONFIG["BOOTSTRAP_MAX_BONE_LEN"]:
                        skeleton['keypoints'][kp_name] = points[best_idx]

                # Only keep reasonably sized fragments
                if len(skeleton['keypoints']) >= 2:
                    skeletons.append(skeleton)

                # Mark all points used in this new skeleton so they aren't used as seeds again
                # (this is a simplification for bootstrapping, it's ok if bones overlap)
                for kp, point_pos in skeleton['keypoints'].items():
                    # TODO: this is inefficient (but fine for now, for a one-off bootstrap)
                    idx = np.where((all_points[kp][0] == point_pos).all(axis=1))[0][0]
                    used_point_nodes.add((kp, idx))

        return skeletons

    def _bootstrap_from_data(self) -> Dict:
        """
        Performs data-driven bootstrapping, using symmetry to create robust stats
        """

        # Gather data under canonical bone names
        canonical_bone_lengths = defaultdict(list)
        canonical_to_original_map = defaultdict(set)

        with alive_bar(title='Gathering bone measurements...', total=len(self.bootstrap_data)) as bar:
            for frame_data in self.bootstrap_data:
                fragments = self._greedy_assembler(frame_data)
                for frag in fragments:
                    for original_bone in self.bones_list:
                        if original_bone.issubset(frag['keypoints']):
                            kp1, kp2 = tuple(original_bone)
                            length = np.linalg.norm(frag['keypoints'][kp1] - frag['keypoints'][kp2])
                            if length > 1e-3:
                                canonical_bone = self._symmetrise_bone_names(original_bone)
                                canonical_bone_lengths[canonical_bone].append(length)
                                # Store the mapping
                                canonical_to_original_map[canonical_bone].add(original_bone)
                bar()

        # Calculate stats for canonical bones
        valid_canonical_bones = {b for b, lengths in canonical_bone_lengths.items() if
                                 len(lengths) >= CONFIG["BOOTSTRAP_MIN_SAMPLES"]}
        median_lengths = {b: np.median(canonical_bone_lengths[b]) for b in valid_canonical_bones}

        if not median_lengths:
            raise ValueError("Bootstrap failed: Not enough valid bone samples found in the data.")

        # We use stability-based selection for the canonical reference
        canonical_graph = nx.Graph()
        canonical_degrees = defaultdict(int)
        for bone in self.bones_list:
            canonical_bone = self._symmetrise_bone_names(bone)
            # Only add edges if the canonical bone is valid (has 2 parts)
            if len(canonical_bone) == 2:
                canonical_graph.add_edge(*tuple(canonical_bone))

        # We need to map original degrees to canonical degrees
        # (a canonical kp's degree is the sum of degrees of its original constituent parts)
        for kp, deg in self._degrees.items():
            if self.symmetry_map and kp in self.symmetry_map:
                canonical_degrees[self.symmetry_map[kp]] += deg
            else:
                canonical_degrees[kp] += deg

        def get_best_canonical_ref(canonical_bones: set[Bone]) -> Bone:
            best_bone, max_score = None, -1
            for bone in canonical_bones:
                kp1, kp2 = tuple(bone)
                score = canonical_degrees.get(kp1, 0) + canonical_degrees.get(kp2, 0)
                if score > max_score:
                    max_score = score
                    best_bone = bone
            return best_bone if best_bone else list(canonical_bones)[0]

        canonical_ref_bone = get_best_canonical_ref(set(median_lengths.keys()))
        ref_len = median_lengths[canonical_ref_bone]

        # Pick any original bone that corresponds to the canonical reference
        reference_bone = list(canonical_to_original_map[canonical_ref_bone])[0]
        logger.info(
            f"Bootstrap determined stability-driven canonical reference '{' - '.join(canonical_ref_bone)}'. "
            f"Saving final reference as '{' - '.join(reference_bone)}' (median length: {ref_len:.2f})"
        )

        # Calculate canonical ratios
        canonical_stats = {}
        for bone, med_len in median_lengths.items():
            ratios_dist = np.array(canonical_bone_lengths[bone]) / ref_len
            canonical_stats[bone] = {
                'median_ratio': float(med_len / ref_len),
                'mad_ratio': float(median_abs_deviation(ratios_dist))
            }

        # Map the canonical stats back to original bone names
        final_bones_ratios = {}
        for canonical_bone, stats in canonical_stats.items():
            original_bones = canonical_to_original_map[canonical_bone]
            for original_bone in original_bones:
                bone_key = ';'.join(sorted(list(original_bone)))
                final_bones_ratios[bone_key] = stats

        final_stats = {
            'reference_bone': sorted(list(reference_bone)),
            'median_reference_length': float(ref_len),
            'bones_ratios': final_bones_ratios
        }
        return self._validate_and_normalize_stats(final_stats)

    def _save_stats(self, stats: Dict):
        """ Saves the generated stats to the specified JSON file """
        print(f"[INFO] Saving processed stats to '{self.output_path}'...")

        with open(self.output_path, 'w') as f:
            json.dump(stats, f, indent=2)


class AnatomyLearner:
    def __init__(self, initial_stats: dict):

        self.reference_bone: Bone = frozenset(initial_stats['reference_bone'])

        # Store raw measurements to re-calculate stats on the fly
        self.ref_lengths = []
        self.bones_ratios: Dict[Bone, List[float]] = defaultdict(list)

        # Keep track of the current stats to avoid re-computing every frame
        self.current_stats = initial_stats
        self.is_stale = True  # Flag to indicate that stats need re-computing

        self.measurements_count = 0

        # Quality filter
        self.min_samples_for_update = CONFIG["LEARNER_MIN_SAMPLES_FOR_UPDATE"]
        self.min_score_for_learning = CONFIG["LEARNER_MIN_SCORE_FOR_LEARNING"]
        self.min_ref_len = CONFIG["LEARNER_MIN_REF_BONE_LEN"]
        self.max_ref_len = CONFIG["LEARNER_MAX_REF_BONE_LEN"]

    def add_sample(self, skeleton: AssembledSkeleton):
        """ Adds a new high-quality skeleton to the measurement pool """

        # Quality gate
        if skeleton.score < self.min_score_for_learning:
            return

        if not self.reference_bone.issubset(skeleton.keypoints):
            return

        # Add measurements
        kp1_name, kp2_name = tuple(self.reference_bone)
        p1, p2 = skeleton.keypoints[kp1_name], skeleton.keypoints[kp2_name]
        ref_len = np.linalg.norm(p1 - p2)

        # sanity check
        if not (self.min_ref_len < ref_len < self.max_ref_len):
            return

        self.ref_lengths.append(ref_len)

        for bone_str in self.current_stats['bones_ratios']:
            kp1, kp2 = bone_str.split(';')
            if kp1 in skeleton.keypoints and kp2 in skeleton.keypoints:
                bone_len = np.linalg.norm(skeleton.keypoints[kp1] - skeleton.keypoints[kp2])
                self.bones_ratios[frozenset((kp1, kp2))].append(bone_len / ref_len)

        self.measurements_count += 1
        if self.measurements_count >= self.min_samples_for_update:
            self.is_stale = True

    def get_stats(self) -> dict:
        """ Returns the current best stats, re-computing them if enough new data has arrived """

        if not self.is_stale:
            return self.current_stats

        # Recompute stats from the full pool of measurements
        new_bones_ratios = {
            ';'.join(sorted(list(b))): {'median_ratio': float(np.median(r)),
                                        'mad_ratio': float(median_abs_deviation(r))}
            for b, r in self.bones_ratios.items() if len(r) > 50    # Only well-observed bones. TODO: maybe this could be added to config dict
        }

        if not new_bones_ratios:  # not enough data yet so we stick to old stats
            self.is_stale = False
            return self.current_stats

        self.current_stats['bones_ratios'].update(new_bones_ratios)

        if self.ref_lengths:
            self.current_stats['median_reference_length'] = float(np.median(self.ref_lengths))

        self.is_stale = False
        self.measurements_count = 0

        logger.debug(f"Refinement complete. New reference length: {self.current_stats['median_reference_length']:.2f}")
        return self.current_stats


class Track:
    """
    Represents a single tracked object (a skeleton)
    Manages state estimation (position, velocity, scale) using a Kalman Filter

    It can predict its future state and be updated with new measurements
    It also includes logic to infer the position of its central keypoint if it's occluded
    """

    def __init__(self,
            track_id:           int,
            initial_skeleton:   AssembledSkeleton,
            frame_idx:          int,
            central_kp:         str):

        self.id = track_id
        self.age = 0
        self.time_since_update = 0
        self.last_update_frame = frame_idx

        self.skeleton: AssembledSkeleton = initial_skeleton
        self.central_kp = central_kp

        # KF inference parameters
        self.inference_uncertainty_factor = CONFIG['KF_INFERENCE_UNCERTAINTY_FACTOR']
        self.min_kps_for_inference = CONFIG['TRACKER_MIN_KPS_FOR_INFERENCE']

        # Kalman Filter for 3D position (x, y, z), 3D velocity (vx, vy, vz), and scale (s)
        # State vector (dim_x = 7): [x, y, z, vx, vy, vz, s]
        # Measurement (dim_z = 4): [x, y, z, s]
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        dt = 1.0  # Time step

        self.kf.F = np.array([[1.0, 0.0, 0.0,  dt, 0.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0, 0.0,  dt, 0.0, 0.0],
                              [0.0, 0.0, 1.0, 0.0, 0.0,  dt, 0.0],
                              [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])  # Constant velocity and scale model

        # Measurement function
        self.kf.H = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])  # We measure position and scale

        # Process noise
        pos_vel_q = Q_discrete_white_noise(dim=2, dt=dt, var=CONFIG["KF_PROCESS_NOISE_POS"], block_size=3)
        scale_q = np.array([[CONFIG["KF_PROCESS_NOISE_SCALE"]]])
        self.kf.Q = block_diag(pos_vel_q, scale_q)

        # Measurement noise
        self.kf.R = np.diag([CONFIG["KF_MEASUREMENT_NOISE_POS"], CONFIG["KF_MEASUREMENT_NOISE_POS"],
                             CONFIG["KF_MEASUREMENT_NOISE_POS"], CONFIG["KF_MEASUREMENT_NOISE_SCALE"]])

        # Initial state covariance
        self.kf.P[3:6, 3:6] *= 1.0
        self.kf.P[6, 6] = 1.0

        # Initial state
        self.kf.x[:3] = self.skeleton.keypoints[self.central_kp].reshape(3, 1)
        self.kf.x[6] = self.skeleton.scale

    def predict(self, current_frame_idx: int):
        """ Predicts the state of the track for the current frame """

        for _ in range(current_frame_idx - self.last_update_frame):
            self.kf.predict()
            self.age += 1
            self.time_since_update += 1

    def update(self, skeleton: AssembledSkeleton, frame_idx: int):
        """
        Updates the track's state with a new skeleton measurement

        If the central keypoint is missing, it attempts to infer its position using
        rigid alignment (Kabsch algorithm) before updating the Kalman Filter
        """
        update_skeleton = skeleton
        inferred = False

        # if the primary keypoint is missing, try to infer it
        if self.central_kp not in skeleton.keypoints:
            inferred_skeleton = self._infer_missing_central_kp(skeleton)
            if inferred_skeleton:
                # Use the completed skeleton for the update
                update_skeleton = inferred_skeleton
                inferred = True
            else:
                # if inference fails (too few points) we can't update the KF
                # We still update the track's skeleton to the partial view, reset its age, and exit
                self.skeleton = skeleton
                self.score = skeleton.score
                self.time_since_update = 0
                self.last_update_frame = frame_idx
                return

        self.skeleton = update_skeleton

        self.time_since_update = 0
        self.last_update_frame = frame_idx

        # Create the measurement vector for the Kalman Filter
        measurement = np.array([*update_skeleton.keypoints[self.central_kp], update_skeleton.scale])

        # If the position was inferred, we are less certain about it...
        # so we tell this to the KF by temporarily increasing its measurement noise
        if inferred:
            original_R = self.kf.R.copy()
            self.kf.R[:3, :3] *= self.inference_uncertainty_factor # only increase position uncertainty
            self.kf.update(measurement)
            self.kf.R = original_R  # and restore for the next update
        else:
            self.kf.update(measurement)

    def _infer_missing_central_kp(self, fragment: AssembledSkeleton) -> Optional[AssembledSkeleton]:
        """
        Infers the position of a missing central keypoint by aligning the last known
        full pose to the currently visible fragment using a rigid transform
        """

        prev_kps, curr_kps = self.skeleton.keypoints, fragment.keypoints
        common_names = list(prev_kps.keys() & curr_kps.keys())

        if len(common_names) < self.min_kps_for_inference:
            return None  # not enough information to infer a stable alignment

        # Point clouds for alignment: current fragment (A) to previous pose (B)
        points_A = np.array([prev_kps[name] for name in common_names])
        points_B = np.array([curr_kps[name] for name in common_names])

        # Use Kabsch algorithm to find R, t such that: B ~ R @ A + t
        R_mat, t_vec = find_rigid_transform(points_A, points_B)

        # Apply the found transform to the last known position of the central keypoint
        inferred_pos = np.array(R_mat) @ prev_kps[self.central_kp] + np.array(t_vec)

        # Return a new object to avoid modifying the input fragment
        completed_skeleton = AssembledSkeleton(
            keypoints=fragment.keypoints.copy(),
            score=fragment.score,
            scale=fragment.scale,
            point_indices=fragment.point_indices
        )
        completed_skeleton.keypoints[self.central_kp] = inferred_pos
        return completed_skeleton

    @property
    def predicted_pose(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Returns a full skeleton pose based on the Kalman Filter's predicted position of the central keypoint
        """
        if self.central_kp not in self.skeleton.keypoints:
            return None

        translation = self.predicted_position - self.skeleton.keypoints[self.central_kp]

        return {kp_name: pos + translation for kp_name, pos in self.skeleton.keypoints.items()}

    @property
    def predicted_position(self) -> np.ndarray:
        """
        Returns the predicted 3D position from the KF state
        """
        return self.kf.x[:3].flatten()

    @property
    def predicted_scale(self) -> float:
        """
        Returns the predicted scale from the KF state
        """
        return self.kf.x[6, 0]


class SkeletonAssembler:
    """
    Assembles skeletons from a 'soup' of 3D reconstructed points for a single frame

    This is a two-stage process:
    1.  generate_candidates(): Generates all plausible skeleton candidates using a
        holistic, score-based growth algorithm. This creates redundancy on purpose
    2.  solve_conflicts(): Resolves the redundancy by building a conflict graph and
        finding the Maximum Weight Independent Set (MWIS), yielding the optimal set
        of non-overlapping skeletons
    """

    def __init__(self, bones_list: list, bone_stats: dict):

        self.bones_list: List[Bone] = [frozenset(bone) for bone in bones_list]
        self.update_bone_stats(bone_stats)

        # Build the graph to determine topology
        skeleton_graph = nx.Graph([tuple(b) for b in self.bones_list])
        if not skeleton_graph.nodes:
            raise ValueError("Cannot assemble skeletons from an empty bone list.")

        # Get keypoints degrees, and determine which is the central, which are anchors
        degrees = dict(skeleton_graph.degree())

        leaf_nodes = {node for node, degree in degrees.items() if degree == 1}
        all_kps = set(degrees.keys())

        self.anchor_keypoints = all_kps - leaf_nodes
        logger.debug(f"Found {len(self.anchor_keypoints)} anchor keypoints: {sorted(list(self.anchor_keypoints))}")

        self.central_kp = max(degrees, key=degrees.get)
        logger.debug(f"Assembler determined central keypoint: '{self.central_kp}'")

    def update_bone_stats(self, new_stats: dict):
        """ Allows the assembler's anatomical model to be updated on the fly """
        self.reference_bone: Bone = frozenset(new_stats['reference_bone'])
        self.median_ref_len = new_stats['median_reference_length']
        self.bones_ratios = {frozenset(k.split(';')): v for k, v in new_stats['bones_ratios'].items()}

    def generate_candidates(self,
            reconstructed_data: dict
    ) -> Tuple[List[CandidateSkeleton], Dict[PointNode, PointObservation]]:
        """
        Generates all plausible skeleton candidates from all valid anchor seeds

        It does *not* consume points, allowing for redundant candidates which are then resolved
        globally by `solve_conflicts`
        """
        if 'points' not in reconstructed_data or not reconstructed_data['points']:
            return [], {}

        # points_map is an internal standardized representation of the point soup
        # that simplifies the greedy growth algorithm. It maps (kp_name, index) to {pos, conf}
        points_map = {
            (kp_name, i): PointObservation(pos=points[i], conf=confs[i])
            for kp_name, (points, confs) in reconstructed_data['points'].items()
            for i in range(len(points))
        }
        if not points_map: return [], {}

        # We only seed skeletons from non-leaf keypoints
        anchor_nodes = [node for node in points_map if node[0] in self.anchor_keypoints]
        candidate_skeletons = []

        # For every anchor node in the frame, try to grow a skeleton
        for point in anchor_nodes:
            candidate = self._grow_skeleton(point, points_map)
            if candidate:
                candidate_skeletons.append(candidate)

        return candidate_skeletons, points_map

    def solve_conflicts(self,
            candidates:     List[CandidateSkeleton],
            points_map:     Dict[PointNode, PointObservation]
    ) -> List[AssembledSkeleton]:
        """ Solves for the best set of non-conflicting skeletons using MWIS """

        if not candidates:
            return []

        # Build conflict graph where nodes are skeletons and edges represent a conflict
        conflict_graph = nx.Graph()
        for i, candidate_skel in enumerate(candidates):
            weight = max(0, int(candidate_skel.score * 100))
            conflict_graph.add_node(i, weight=weight)

        # Precompute centroids and build the KDtree
        centroids = np.array([
            np.mean([points_map[node].pos for node in cand.nodes], axis=0) if cand.nodes else np.array([np.nan] * 3)
            for cand in candidates
        ])
        valid_indices = np.where(~np.isnan(centroids).any(axis=1))[0]

        if len(valid_indices) < 2:  # not enough valid skeletons to have conflicts
            winner_indices = valid_indices
        else:
            tree = cKDTree(centroids[valid_indices])
            potential_pairs = tree.query_pairs(r=CONFIG["CONFLICT_SOLVER_BROAD_RADIUS"], output_type='set')

            # Perform detailed narrow phase checks only on pairs that are close to each other
            for i_local, j_local in potential_pairs:
                i, j = valid_indices[i_local], valid_indices[j_local]
                cand_i, cand_j = candidates[i], candidates[j]

                # Direct point sharing conflict
                if not cand_i.nodes.isdisjoint(cand_j.nodes):

                    logger.debug(
                        f"  - Conflict (Shared Points): Skel {i} vs {j}. "
                        f"Shared: {set(cand_i.nodes) & set(cand_j.nodes)}"
                    )

                    conflict_graph.add_edge(i, j)
                    continue

                # Jaccard proximity conflict
                kps_i = {node[0]: points_map[node].pos for node in cand_i.nodes}
                kps_j = {node[0]: points_map[node].pos for node in cand_j.nodes}
                common = kps_i.keys() & kps_j.keys()
                union = kps_i.keys() | kps_j.keys()

                if not union:
                    continue

                proximal_intersection = sum(
                    1 for name in common if np.linalg.norm(kps_i[name] - kps_j[name]
                                                           ) < CONFIG["CONFLICT_SOLVER_PROXIMITY_RADIUS"])

                if proximal_intersection / len(union) > CONFIG["CONFLICT_SOLVER_JACCARD_THRESHOLD"]:
                    logger.debug(f"  - Conflict (Proximity): Skel {i} vs {j}. Jaccard: {proximal_intersection:.2f}")
                    conflict_graph.add_edge(i, j)

            # The Maximum Weight Independent Set (MWIS) of the conflict graph is the set of
            # non-conflicting skeletons with the maximum total score. This is equivalent to
            # finding the max weight clique in the complement graph

            winner_indices = solve_mwis_networkx(conflict_graph)
            # winner_indices = solve_mwis_SCIP(conflict_graph)

        final_skeletons = [
            AssembledSkeleton(
                keypoints={node[0]: points_map[node].pos for node in candidates[i].nodes},
                point_indices={node[0]: node[1] for node in candidates[i].nodes},
                score=candidates[i].original_score,
                scale=candidates[i].scale
            )
            for i in winner_indices
        ]

        logger.debug(
            f"Conflict resolution complete. Selected {len(winner_indices)} skeletons from {len(candidates)} candidates."
        )

        return final_skeletons

    def _grow_skeleton(self,
            anchor_node:    PointNode,
            points_map:     Dict[PointNode, PointObservation]
    ) -> Optional[CandidateSkeleton]:
        """ Grows a single skeleton candidate from an anchor point using a holistic score """

        # TODO: This method is slow af, needs to be sped up

        score_cache = {}

        def _holistic_score(nodes_to_score: FrozenSet[PointNode]) -> float:

            if nodes_to_score in score_cache:
                return score_cache[nodes_to_score]

            if len(nodes_to_score) < 2:
                return 0.0

            kps_dict = {node[0]: points_map[node].pos for node in nodes_to_score}
            scale = self._get_skeleton_scale(kps_dict)

            if not (CONFIG["ASSEMBLER_MIN_SANE_SCALE"] < scale < CONFIG["ASSEMBLER_MAX_SANE_SCALE"]):
                return -1000  # penalize skeletons with insane scales early

            total_score = 0.0
            num_bones = 0
            for kp1_name, kp2_name in combinations(kps_dict.keys(), 2):
                bone = frozenset((kp1_name, kp2_name))
                if bone in self.bones_list:
                    score = self._score_bone(bone, kps_dict, points_map, nodes_to_score, scale)
                    total_score += score
                    num_bones += 1

            score = total_score / (num_bones + 1e-6)
            score_cache[nodes_to_score] = score
            return score

        current_nodes = {anchor_node}

        while True:
            # The score for comparison must include the size bonus just like the final score
            current_avg_score = _holistic_score(frozenset(current_nodes))
            current_growth_score = current_avg_score * len(current_nodes)

            current_kp_types = {node[0] for node in current_nodes}
            logger.debug(
                f"  [Loop] Nodes: {len(current_nodes)} {current_kp_types}, "
                f"AvgScore: {current_avg_score:.3f}, "
                f"GrowthScore: {current_growth_score:.3f}"
            )

            # Find all valid keypoints that could connect to the current skeleton
            nodes_to_evaluate = {
                cand_node for node in current_nodes
                for cand_node in points_map
                if cand_node not in current_nodes and cand_node[0] not in current_kp_types
                   and frozenset((node[0], cand_node[0])) in self.bones_list
            }
            if not nodes_to_evaluate:
                break

            # Score each potential extension
            potential_extensions = [
                (node, _holistic_score(frozenset(current_nodes | {node})) * (len(current_nodes) + 1))
                for node in nodes_to_evaluate
            ]
            if not potential_extensions:
                logger.debug("  [Break] No more potential extensions.")
                break

            # Find the extension that results in the best new holistic score
            best_node, best_new_score = max(potential_extensions, key=lambda x: x[1])

            # If the best new score is better or not much worse than the current score, grow
            if best_new_score > (current_growth_score - CONFIG["ASSEMBLER_SCORE_DEBT_TOLERANCE"]):
                logger.debug(
                    f"  [OK] Adding {best_node[0]}. "
                    f"New score {best_new_score:.2f} is within tolerance of current {current_growth_score:.2f}."
                )
                current_nodes.add(best_node)
            else:
                logger.debug(
                    f"  [Break] Best new score ({best_new_score:.3f}) is a drop of "
                    f"more than {CONFIG['ASSEMBLER_SCORE_DEBT_TOLERANCE']}. Stopping growth."
                )
                break

        if len(current_nodes) < CONFIG["ASSEMBLER_MIN_KPS_FOR_SKELETON"]:
            return None

        final_kps = {node[0]: points_map[node].pos for node in current_nodes}
        final_score = _holistic_score(frozenset(current_nodes))
        if final_score <= 0:
            return None

        return CandidateSkeleton(
            nodes=frozenset(current_nodes),
            score=final_score * len(current_nodes),
            original_score=final_score * len(current_nodes),
            scale=self._get_skeleton_scale(final_kps)
        )

    def _get_skeleton_scale(self, keypoints: Dict[str, np.ndarray]) -> float:
        """ Estimates the scale of a (possibly partial) skeleton relative to the reference stats """

        scales = []

        # Use primary reference bone first
        if self.reference_bone.issubset(keypoints) and self.median_ref_len > 1e-6:
            kp1, kp2 = tuple(self.reference_bone)
            scales.append(np.linalg.norm(keypoints[kp1] - keypoints[kp2]) / self.median_ref_len)

        # Use all other bones for a robust estimate
        for bone_type, stats in self.bones_ratios.items():
            kp1, kp2 = tuple(bone_type)
            if kp1 in keypoints and kp2 in keypoints:
                expected_len = self.median_ref_len * stats['median_ratio']
                if expected_len > 1e-6:
                    scales.append(np.linalg.norm(keypoints[kp1] - keypoints[kp2]) / expected_len)

        if not scales:
            return 1.0

        # Filter out absurd scale values before taking the median
        sane_scales = [s for s in scales if CONFIG["ASSEMBLER_MIN_SANE_SCALE"] <= s <= CONFIG["ASSEMBLER_MAX_SANE_SCALE"]]

        return float(np.median(sane_scales)) if sane_scales else 1.0

    def _score_bone(self,
            bone:           Bone,
            keypoints:      Dict[str, np.ndarray],
            points_map:     Dict[PointNode, PointObservation],
            nodes:          FrozenSet[PointNode],
            scale:          float
    ) -> float:
        """ Scores a single bone based on its length conformity to the stats (adjusted for scale) """

        if bone not in self.bones_ratios:
            return 0.0

        stats = self.bones_ratios[bone]
        kp1_name, kp2_name = tuple(bone)
        p1 = keypoints[kp1_name]
        p2 = keypoints[kp2_name]

        # Find confidences
        node1 = next(n for n in nodes if n[0] == kp1_name)
        node2 = next(n for n in nodes if n[0] == kp2_name)
        conf1 = points_map[node1].conf
        conf2 = points_map[node2].conf

        expected_length = self.median_ref_len * stats['median_ratio'] * scale

        # Scale the expected deviation
        expected_mad = self.median_ref_len * stats['mad_ratio'] * scale + CONFIG["ASSEMBLER_BONE_SCORE_MAD_EPSILON"]
        distance = np.linalg.norm(p1 - p2)

        # How many std dev (using MAD) the length is from the mean
        num_mads_away = abs(distance - expected_length) / (expected_mad + 1e-6)

        if num_mads_away > 2.0:  # Debug print for bones that are just a bit off
            logger.debug(f"      - Scoring bone {kp1_name}-{kp2_name}:\n"
                         f"          Scale: {scale:.2f}, Measured Dist: {distance:.2f}, Expected: {expected_length:.2f}"
                         f"          MADs away: {num_mads_away:.2f} (Thresh: {CONFIG['ASSEMBLER_BONE_SCORE_MAD_THRESH']})")

        if num_mads_away > CONFIG["ASSEMBLER_BONE_SCORE_MAD_THRESH"]:
            # impossible bone: large negative penalty that will poison the average score
            return -1000.0

        # Gaussian-like fall off for the length score
        length_score = np.exp(-0.5 * num_mads_away ** 2)

        # Factor in the confidence of the 2D detections that created the 3D points
        confidence_score = (conf1 + conf2) / 2.0

        return length_score * confidence_score

class MultiObjectTracker:
    """
    Main class for tracking multiple skeletons over time
    """

    def __init__(self, assembler: SkeletonAssembler):

        self.assembler = assembler
        self.frame_idx = -1
        self.tracks: List[Track] = []
        self.next_track_id = 0
        self.max_age = CONFIG['TRACKER_MAX_AGE']

    def update(self,
            reconstructed_data: dict,
            frame_idx:          int
    ) -> List[Track]:
        """
        Processes a single frame using a 'Guided global assembly' workflow

        By boosting scores of candidates that align with existing tracks, we guide the conflict
        resolution (MWIS) to favor solutions with high temporal continuity
        """

        self.frame_idx = frame_idx

        # Predict
        for track in self.tracks:
            track.predict(self.frame_idx)

        # Generate all candidates
        candidates, points_map = self.assembler.generate_candidates(reconstructed_data)
        if not candidates:
            self.prune_tracks()
            return self.get_active_tracks()

        # Boost scores to guide assembly
        # Bonus for candidates that align well with existing track predictions to bias the conflict resolution
        # (in favor of temporal continuity)
        bonuses = self._calculate_association_bonuses(candidates, points_map)

        for i, candidate_skel in enumerate(candidates):
            # The score attribute is the boosted score used for conflict resolution
            candidate_skel.score += bonuses[i] * CONFIG["TRACKER_CONTINUITY_BONUS"]

        # Conflict resolution
        # Run the conflict solver on the complete (score-boosted) set of candidates
        # This is where clones are detected and eliminated via MWIS (before any assignments are made)
        winning_skeletons = self.assembler.solve_conflicts(candidates, points_map)

        # Final association and update
        if self.tracks and winning_skeletons:

            # here we use the original (non-boosted) scores to ensure matching on pure geometry and anatomy
            cost_matrix = self._build_final_assignment_cost_matrix(self.tracks, winning_skeletons)
            track_inds, winner_inds = linear_sum_assignment(cost_matrix)

            matched_winner_indices = set()
            for t_idx, w_idx in zip(track_inds, winner_inds):

                # association is only made if the cost not infinite
                if cost_matrix[t_idx, w_idx] < 1e9:
                    self.tracks[t_idx].update(skeleton=winning_skeletons[w_idx], frame_idx=self.frame_idx)
                    matched_winner_indices.add(w_idx)
        else:
            matched_winner_indices = set()

        # Any winning skeleton that was not matched to an existing track is a new object
        for i, skel in enumerate(winning_skeletons):
            if i not in matched_winner_indices and self.assembler.central_kp in skel.keypoints:
                new_track = Track(self.next_track_id, skel, self.frame_idx, self.assembler.central_kp)
                self.tracks.append(new_track)
                self.next_track_id += 1

        # Remove tracks that weren't updated in this frame and are too old
        self.prune_tracks()

        return self.get_active_tracks()

    def predict_only(self, frame_idx: int) -> List[Track]:
        """ Handles frames with no detections by only running the prediction step """

        self.frame_idx = frame_idx

        for track in self.tracks:
            track.predict(self.frame_idx)

        self.prune_tracks()

        return self.get_active_tracks()

    def get_active_tracks(self) -> List[Track]:
        """ Returns a list of tracks that have been updated in the current frame """
        return [t for t in self.tracks if t.time_since_update == 0]

    def prune_tracks(self):
        """ Removes tracks that have been lost for too long """
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

    def _calculate_association_bonuses(self,
            candidates: List[CandidateSkeleton],
            points_map: Dict[PointNode, PointObservation]
    ) -> np.ndarray:
        """
        Calculates a score bonus (0 to 1) for each candidate skeleton based on its
        best possible alignment with any existing track's prediction
        """

        if not self.tracks or not candidates:
            return np.zeros(len(candidates))

        bonuses = np.zeros(len(candidates))
        sigma = CONFIG["TRACKER_ASSOCIATION_RADIUS"]

        for j, cand_skel in enumerate(candidates):
            skel_kps = {node[0]: points_map[node].pos for node in cand_skel.nodes}

            if not skel_kps:
                continue

            max_bonus = 0.0
            for track in self.tracks:
                pred_pose = track.predicted_pose

                if not pred_pose:
                    continue

                common_kps = pred_pose.keys() & skel_kps.keys()
                if len(common_kps) < CONFIG["TRACKER_ASSOCIATION_MIN_KPS"]:
                    continue

                # Calculate mean squared distance between common keypoints
                mean_dist_sq = sum(np.sum((pred_pose[kp] - skel_kps[kp])**2) for kp in common_kps) / len(common_kps)

                # Gaussian bonus falls off as distance increases
                bonus = np.exp(-0.5 * mean_dist_sq / (sigma**2))
                if bonus > max_bonus:
                    max_bonus = bonus

            bonuses[j] = max_bonus

        return bonuses

    def _build_final_assignment_cost_matrix(self,
            tracks:     List[Track],
            skeletons:  List[AssembledSkeleton]
    ) -> np.ndarray:
        """ Builds the cost matrix for the final assignment via Hungarian algorithm """

        cost_matrix = np.full((len(tracks), len(skeletons)), 1e9)

        for i, track in enumerate(tracks):
            pred_pose = track.predicted_pose

            if not pred_pose:
                continue

            for j, skel in enumerate(skeletons):
                common_kps = pred_pose.keys() & skel.keypoints.keys()

                if len(common_kps) < CONFIG["TRACKER_ASSOCIATION_MIN_KPS"]:
                    continue

                mean_dist_sq = sum(np.sum((pred_pose[kp] - skel.keypoints[kp])**2)
                                   for kp in common_kps) / len(common_kps)

                if mean_dist_sq > CONFIG["TRACKER_ASSOCIATION_RADIUS"] ** 2:
                    continue

                # The cost is a weighted sum of pose distance and the skeleton's intrinsic score
                # A good skeleton (high score) should reduce the cost
                cost = (CONFIG["COST_POSE_DISTANCE_WEIGHT"] * mean_dist_sq +
                        CONFIG["COST_SKELETON_SCORE_WEIGHT"] * skel.score)
                cost_matrix[i, j] = cost

        return cost_matrix


if __name__ == '__main__':

    folder = Path().home() / 'Desktop' / '3d_ant_data'
    prefix = '240905-1616'

    stats_output_file = 'normalized_bone_stats.json'
    # prior_stats_file = Path().home() / 'Desktop' / 'bone_lengths.csv'
    prior_stats_file = None

    # Data for bootstrapping
    reconstructed_points_file = 'reconstructed_points.pkl'
    skeleton_input_path = folder / prefix / 'inputs' / 'tracking'

    # Load data
    all_reconstructed_points = pickle.load(open(reconstructed_points_file, 'rb'))
    keypoints, bones, symmetry = fileio.load_skeleton_SLEAP(skeleton_input_path, symmetry=True)

    # Get Anatomical stats
    bootstrapper = StatsBootstrapper(
        output_path=stats_output_file,
        bones_list=bones,
        symmetry_map=symmetry,
        prior_stats_path=prior_stats_file,
        bootstrap_data=all_reconstructed_points
    )
    bone_stats = bootstrapper.get_initial_stats()

    # Initialise the tracking pipeline
    anatomy_learner = AnatomyLearner(initial_stats=bone_stats)
    assembler = SkeletonAssembler(bones_list=bones, bone_stats=bone_stats)
    tracker = MultiObjectTracker(assembler=assembler)

    # Run tracking pipeline
    reconstructed_data_map = {item['frame_idx']: item for item in all_reconstructed_points}
    all_frames = sorted(reconstructed_data_map.keys())
    min_frame, max_frame = all_frames[0], all_frames[-1]

    all_tracked_skeletons = []

    with alive_bar(title='Tracking Skeletons...', length=20, total=(max_frame - min_frame + 1), force_tty=True) as bar:
        for frame_idx in range(min_frame, max_frame + 1):

            # The assembler's anatomical model is updated with the latest learned stats
            current_stats = anatomy_learner.get_stats()
            assembler.update_bone_stats(current_stats)

            # Run the tracker for the frame
            if frame_idx in reconstructed_data_map:
                active_tracks = tracker.update(reconstructed_data_map[frame_idx], frame_idx)
            else:
                active_tracks = tracker.predict_only(frame_idx)

            # Feed the results back into the learner
            for track in active_tracks:
                # We only use skeletons from the current frame for learning
                if track.last_update_frame == frame_idx:
                    anatomy_learner.add_sample(track.skeleton)

            # Convert dataclasses back to dicts for serialization
            frame_skeletons = [track.skeleton.to_dict() for track in active_tracks]
            for skel, track in zip(frame_skeletons, active_tracks):
                skel['track_id'] = track.id
            all_tracked_skeletons.append({"frame_idx": frame_idx, "skeletons": frame_skeletons})

            bar()

    print("Tracking complete.")


    # Save final results

    output_file = 'final_tracked_skeletons.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(all_tracked_skeletons, f)

    print(f"Results saved to '{output_file}'")
