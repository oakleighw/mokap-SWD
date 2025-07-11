import pickle
import logging
from typing import Tuple, Optional, Dict, List, FrozenSet
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import networkx as nx
import numpy as np
from alive_progress import alive_bar
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter
from scipy.linalg import block_diag
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from mokap.reconstruction.anatomy import StatsBootstrapper, AnatomyLearner
from mokap.reconstruction.config import AssemblerConfig, TrackerConfig, AnatomyConfig
from mokap.reconstruction.datatypes import Bone, AssembledSkeleton, CandidateSkeleton
from mokap.reconstruction.utils import solve_mwis_networkx
from mokap.utils import fileio
from mokap.utils.geometry.fitting import find_rigid_transform
from mokap.reconstruction.reconstruction import SoupPoint


logger = logging.getLogger(__name__)


class Tracklet:
    """
    Stateful class that represents a single skeleton in a tracklet
    Manages state estimation (position, velocity, scale) using a Kalman Filter
    """

    # TODO: Maybe this class should store its data into a TrackletData object directly...

    def __init__(self,
            track_idx:          int,
            initial_skeleton:   AssembledSkeleton,
            frame_idx:          int,
            central_kp:         str,
            config:             TrackerConfig
        ):

        self.config = config
        self.track_idx = track_idx
        self.age = 0
        self.time_since_update = 0
        self.last_update_frame = frame_idx

        # Tracklet health and score metrics
        self.health = 1.0  # Running confidence metric (1.0 = high confidence)
        self.anatomical_integrity = initial_skeleton.score  # Exponential Moving Average of the skeleton score

        self.skeleton: AssembledSkeleton = initial_skeleton
        self.central_kp = central_kp

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
        pos_vel_q = Q_discrete_white_noise(dim=2, dt=dt, var=self.config.KF_PROCESS_NOISE_POS, block_size=3)
        scale_q = np.array([[self.config.KF_PROCESS_NOISE_SCALE]])
        self.kf.Q = block_diag(pos_vel_q, scale_q)

        # Measurement noise
        self.kf.R = np.diag([self.config.KF_MEASUREMENT_NOISE_POS, self.config.KF_MEASUREMENT_NOISE_POS,
                             self.config.KF_MEASUREMENT_NOISE_POS, self.config.KF_MEASUREMENT_NOISE_SCALE])

        # Initial state covariance
        self.kf.P[3:6, 3:6] *= 1.0
        self.kf.P[6, 6] = 1.0

        # Initial state
        self.kf.x[:3] = self.skeleton.keypoints[self.central_kp].reshape(3, 1)
        self.kf.x[6] = self.skeleton.scale

    def predict(self, current_frame_idx: int):
        """ Predicts the state of the tracklet for the current frame """

        # The number of prediction steps is how many frames we've coasted
        steps_to_predict = current_frame_idx - self.last_update_frame

        for _ in range(steps_to_predict):
            self.kf.predict()
            self.age += 1
            self.time_since_update += 1
            # Decay health for each frame without a measurement update
            self.health *= self.config.HEALTH_DECAY_RATE

    def update(self, skeleton: AssembledSkeleton, frame_idx: int):
        """
        Updates the tracklet's state with a new skeleton measurement

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
                # We still update the tracklet's skeleton to the partial view, reset its age,
                # but do *not* update health or score_ema, as it was not a full KF update
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
            self.kf.R[:3, :3] *= self.config.KF_INFERENCE_UNCERTAINTY_FACTOR  # only increase position uncertainty
            self.kf.update(measurement)
            self.kf.R = original_R  # and restore for the next update
        else:
            self.kf.update(measurement)

        # Update health and score metrics after a successful KF update

        # Update the smoothed anatomical score
        self.anatomical_integrity = self.config.ANATOMICAL_SCORE_ALPHA * skeleton.score + (
                    1 - self.config.ANATOMICAL_SCORE_ALPHA) * self.anatomical_integrity

        # Update the tracklet's health
        if inferred:
            # The update was based on an inferred point, so it's a bit less certain...
            # so restore health, but with a lil penalty
            self.health = 1.0 - self.config.INFERRED_HEALTH_PENALTY
        else:
            # The update was based on a direct measurement: this is a high-confidence event
            # so reset health to its maximum value
            self.health = 1.0

    def _infer_missing_central_kp(self, fragment: AssembledSkeleton) -> Optional[AssembledSkeleton]:
        """
        Infers the position of a missing central keypoint by aligning the last known
        full pose to the currently visible fragment using a rigid transform
        """

        prev_kps, curr_kps = self.skeleton.keypoints, fragment.keypoints
        common_names = list(prev_kps.keys() & curr_kps.keys())

        if len(common_names) < self.config.MIN_KPS_FOR_INFERENCE:
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

    @property
    def uncertainty(self) -> Dict[str, np.ndarray]:
        """
        Returns the uncertainty (variance) of the state variables from the
        Kalman Filter's covariance matrix P
        """
        diag_P = self.kf.P.diagonal()
        return {
            'position': diag_P[0:3],
            'velocity': diag_P[3:6],
            'scale': diag_P[6]
        }


class SkeletonAssembler:
    """ Assembles skeletons from a 'soup' of 3D reconstructed points for a single frame """

    def __init__(self,
            bones_list: List[Bone],
            bone_stats: Dict,
            assembler_config: AssemblerConfig,
            tracker_config: TrackerConfig
        ):

        self.config = assembler_config
        self.tracker_config = tracker_config

        self.bones_list = [frozenset(bone) for bone in bones_list]
        self.update_bone_stats(bone_stats)

        # Build the graph to determine topology
        skeleton_graph = nx.Graph([tuple(b) for b in self.bones_list])
        if not skeleton_graph.nodes:
            raise ValueError("Cannot assemble skeletons from an empty bone list.")

        self._skeleton_graph = skeleton_graph

        # Get keypoint degrees to determine topology
        degrees = dict(self._skeleton_graph.degree())
        self.leaf_nodes = {node for node, degree in degrees.items() if degree == 1}

        non_leaf_nodes = set(degrees.keys()) - self.leaf_nodes
        self.central_anchors = set()
        self.secondary_anchors = set()

        if non_leaf_nodes:
            sorted_anchors = sorted(list(non_leaf_nodes), key=lambda node: degrees[node], reverse=True)
            self.central_anchors = set(sorted_anchors[:self.config.MIN_CENTRAL_ANCHORS])
            self.secondary_anchors = non_leaf_nodes - self.central_anchors

        logger.debug(f"Central Anchors: {sorted(list(self.central_anchors))}")
        logger.debug(f"Secondary Anchors: {sorted(list(self.secondary_anchors))}")
        logger.debug(f"Leaf Nodes: {sorted(list(self.leaf_nodes))}")

        self.central_kp = max(degrees, key=degrees.get)
        logger.debug(f"Assembler determined central keypoint: '{self.central_kp}'")

    def update_bone_stats(self, new_stats: Dict):
        """ Allows the assembler's anatomical model to be updated on the fly """

        self.reference_bone: Bone = frozenset(new_stats['reference_bone'])
        self.median_ref_len = new_stats['median_reference_length']
        self.bones_ratios = {frozenset(k.split(';')): v for k, v in new_stats['bones_ratios'].items()}

    def assemble_frame(self, soup_points: List[SoupPoint]
        ) -> Tuple[List[CandidateSkeleton], Dict[Tuple[str, int], SoupPoint]]:
        """
        Main assembly entry point. Generates initial fragments and then creates all
        plausible merge hypotheses (without committing to them)
        """

        # Generate small, high-confidence initial fragments
        initial_fragments, points_map = self._generate_candidates(soup_points)
        if not initial_fragments:
            return [], {}

        # Generate all plausible merges between initial fragments (but keep initial fragments)
        merge_hypotheses = self._generate_merge_hypotheses(initial_fragments, points_map)

        # Return the complete set of hypotheses for the global solver
        all_hypotheses = initial_fragments + merge_hypotheses

        logger.debug(
            f"Assembler created {len(initial_fragments)} initial fragments and {len(merge_hypotheses)} merge hypotheses.")
        return all_hypotheses, points_map

    def _generate_merge_hypotheses(self,
                                   fragments: List[CandidateSkeleton],
                                   points_map: Dict[Tuple[str, int], SoupPoint]
                                   ) -> List[CandidateSkeleton]:
        """
        Generates all plausible merge hypotheses from a list of fragments without committing to any single merge.
        It is stateless (returns only the new, merged skeletons)
        """

        if len(fragments) < 2:
            return []

        # Each initial fragment is its own constituent part
        for i, frag in enumerate(fragments):
            # we use index i as a unique identifier for each initial fragment
            frag.constituent_indices = frozenset([i])

        merge_hypotheses = []

        # consider every possible pair of initial fragments for merging
        for i, j in combinations(range(len(fragments)), 2):
            skel_A, skel_B = fragments[i], fragments[j]

            # attempt to merge the two fragments
            merged_candidate = self._attempt_merge(skel_A, skel_B, points_map)

            if merged_candidate:
                # th emerged candidate's identity is the union of the parent fragments' IDs
                merged_candidate.constituent_indices = skel_A.constituent_indices.union(skel_B.constituent_indices)
                merge_hypotheses.append(merged_candidate)

        # TODO: could add a second layer of merging here by calling this function recursively on 'merge_hypotheses'...

        return merge_hypotheses

    def _generate_candidates(self,
            soup_points: List[SoupPoint]
        ) -> Tuple[List[CandidateSkeleton], Dict[Tuple[str, int], SoupPoint]]:
        """
        Generates all plausible skeleton candidates by exhaustively seeding from all anchor and leaf points.
        This creates the necessary redundancy for the downstream solver
        """

        if not soup_points:
            return [], {}

        points_map = {(p.keypoint_type, p.idx): p for p in soup_points}

        candidate_skeletons = []

        all_seed_nodes = list(points_map.keys())

        # used_as_seed is used to avoid starting a full growth from every single point in a large fragment
        # (which would be redundant). Only one point in a component should be used to seed
        used_as_seed = set()

        # Grow full skeletons from any anchor point
        all_anchor_types = self.central_anchors.union(self.secondary_anchors)
        anchor_seed_nodes = [node for node in all_seed_nodes if node[0] in all_anchor_types]

        for seed_node in anchor_seed_nodes:
            if seed_node in used_as_seed:
                continue

            candidate = self._grow_skeleton(
                seed_node,
                points_map,
                score_debt_tol=self.config.SCORE_DEBT_TOLERANCE
            )

            if candidate:
                candidate_skeletons.append(candidate)
                # Mark all nodes in the discovered fragment as 'used for seeding'
                for node in candidate.nodes:
                    used_as_seed.add(node)

        # Find any remaining isolated 2-point leaf fragments that weren't part of a larger growth
        # TODO: should it be more than 2?
        leaf_seed_nodes = [node for node in all_seed_nodes if node[0] in self.leaf_nodes]
        for seed_node in leaf_seed_nodes:
            if seed_node in used_as_seed:
                continue

            fragment = self._find_leaf_fragment(seed_node, points_map)
            if fragment:
                candidate_skeletons.append(fragment)
                # mark these two nodes as used as well
                for node in fragment.nodes:
                    used_as_seed.add(node)

        logger.debug(f"Generated {len(candidate_skeletons)} initial candidates for this frame.")
        return candidate_skeletons, points_map

    def _attempt_merge(self,
                       skel_A: CandidateSkeleton,
                       skel_B: CandidateSkeleton,
                       points_map:  Dict[Tuple[str, int], SoupPoint]
                       ) -> Optional[CandidateSkeleton]:
        """
        Attempts to merge two disjoint fragments into a larger skeleton

        1. Ensures fragments are fully disjoint (no shared points or keypoint types)
        2. Checks for "clone" fragments using Jaccard similarity on keypoint positions
        3. Verifies scale consistency
        4. Finds the best high-quality 'linking' bone between fragments
        5. Validates this link using a 'shared neighbour' graph heuristic
        """

        # Pre-check 1: must be fully disjoint
        if not skel_A.nodes.isdisjoint(skel_B.nodes):
            return None

        kps_A = {node[0] for node in skel_A.nodes}
        kps_B = {node[0] for node in skel_B.nodes}
        if not kps_A.isdisjoint(kps_B):
            return None

        # Pre-check 2: handle clones with Jaccard score
        common_kps = kps_A.intersection(kps_B)
        if common_kps:
            # this block should theoretically not be reached due to the check above but it's a good safeguard
            proximal_intersection = 0
            kps_map_A = {node[0]: points_map[node].position for node in skel_A.nodes}
            kps_map_B = {node[0]: points_map[node].position for node in skel_B.nodes}

            for name in common_kps:
                if np.linalg.norm(kps_map_A[name] - kps_map_B[name]) < self.tracker_config.CONFLICT_SOLVER_PROXIMITY_RADIUS:
                    proximal_intersection += 1

            union_size = len(kps_A.union(kps_B))
            jaccard_prox = proximal_intersection / union_size if union_size > 0 else 0

            if jaccard_prox > self.tracker_config.CONFLICT_SOLVER_JACCARD_THRESHOLD:
                # they are not merge candidates, they are conflicting clones
                return None

        # Scale consistency
        if abs(skel_A.scale - skel_B.scale) > self.config.MERGE_SCALE_TOLERANCE:
            return None
        combined_scale = (skel_A.scale + skel_B.scale) / 2.0

        # Find a high quailty linking bone
        best_link_score = -1.0
        best_linking_bone = None

        combined_nodes = skel_A.nodes.union(skel_B.nodes)
        combined_kps_map = {node[0]: points_map[node].position for node in combined_nodes}

        for kp_a in kps_A:
            for kp_b in kps_B:
                bone = frozenset((kp_a, kp_b))
                if bone in self.bones_ratios:
                    score = self._score_bone(bone, combined_kps_map, points_map, combined_nodes, combined_scale)
                    if score > best_link_score:
                        best_link_score = score
                        best_linking_bone = bone

        # Check if the best link we found is good enough to justify a merge
        if best_link_score < self.config.MERGE_LINKING_BONE_THRESHOLD:
            return None

        # Shared neighbour check
        kp_a_name, kp_b_name = tuple(best_linking_bone)
        neighbors_of_a = set(self._skeleton_graph.neighbors(kp_a_name))
        neighbors_of_b = set(self._skeleton_graph.neighbors(kp_b_name))

        shared_neighbors = neighbors_of_a.intersection(neighbors_of_b)
        shared_neighbors.discard(kp_a_name)
        shared_neighbors.discard(kp_b_name)

        if not any(n in kps_A or n in kps_B for n in shared_neighbors):
            # island-to-island link with no existing anchor point
            # It's the signature of a pathological merge. Reject it.
            return None

        # Merge is valid
        merged_nodes = skel_A.nodes.union(skel_B.nodes)

        # Estimate the new average score for the merged object
        avg_score_A = skel_A.anatomical_score / len(skel_A.nodes) if skel_A.nodes else 0
        avg_score_B = skel_B.anatomical_score / len(skel_B.nodes) if skel_B.nodes else 0
        num_bones_A = max(1, len(skel_A.nodes) - 1)
        num_bones_B = max(1, len(skel_B.nodes) - 1)

        new_total_score = (avg_score_A * num_bones_A) + (avg_score_B * num_bones_B) + best_link_score
        new_num_bones = num_bones_A + num_bones_B + 1
        new_avg_score = new_total_score / new_num_bones

        # Use the standard factory to create the final candidate
        return self._create_candidate(merged_nodes, new_avg_score, combined_scale)

    def _create_candidate(self,
            nodes:      FrozenSet[Tuple[str, int]],
            avg_score:  float,
            scale:      float
        ) -> CandidateSkeleton:
        """ Factory function to create a CandidateSkeleton object """

        if avg_score <= 0:
            # this can happen if the fragment is nonsensical
            # so we return a candidate with a score of 0
            return CandidateSkeleton(
                nodes=nodes,
                competition_score=0.0,
                anatomical_score=0.0,
                scale=scale
            )

        num_nodes = len(nodes)

        size_bonus_multiplier = num_nodes
        base_score = avg_score * size_bonus_multiplier

        # quality-over-quantity boost for exceptionally well-formed skeletons
        # (his helps high-quality fragments to punch above their weight in the conflict resolution stage)
        quality_bonus = 0.0

        high_quality_thresh = self.config.HIGH_QUALITY_THRESHOLD
        quality_bonus_factor = self.config.QUALITY_BONUS_FACTOR

        if avg_score > high_quality_thresh:
            # The bonus is a percentage of the base score
            quality_bonus = base_score * (quality_bonus_factor - 1.0)

        # The final 'competition score' (used by MWIS) is the sum of the base score and all bonuses
        competition_score = base_score + quality_bonus

        # The 'anatomical_score' reflects the pure anatomical fit multiplied by size
        # This is useful for the AnatomyLearner and for diagnostics
        anatomical_score = avg_score * num_nodes

        return CandidateSkeleton(
            nodes=nodes,
            competition_score=competition_score,
            anatomical_score=anatomical_score,
            scale=scale
        )

    def _find_leaf_fragment(self,
            leaf_node:      Tuple[str, int],
            points_map:     Dict[Tuple[str, int], SoupPoint]
        ) -> Optional[CandidateSkeleton]:
        """ lightweight method to find the single best 2-point fragment starting from a leaf node """

        leaf_kp_name = leaf_node[0]

        # a leaf has only one neighbor in the skeleton graph
        try:
            parent_kp_name = next(self._skeleton_graph.neighbors(leaf_kp_name))
        except StopIteration:
            # this shouldn't happen if the skeleton graph is correct, but as a safeguard:
            return None

        # Find all available points of the parent type
        parent_candidates = [node for node in points_map if node[0] == parent_kp_name]
        if not parent_candidates:
            return None

        best_parent_node = None
        best_bone_score = -1.0

        # Iterate through potential parents to find the one that forms the best single bone
        for parent_cand_node in parent_candidates:
            kps = {leaf_kp_name: points_map[leaf_node].position, parent_kp_name: points_map[parent_cand_node].position}
            nodes = frozenset([leaf_node, parent_cand_node])

            # We need to estimate scale even for a 2-point fragment
            # This is a bit of a chicken-and-egg problem but we can use a quick estimate
            scale = self._get_skeleton_scale(kps)
            if not (self.config.MIN_SANE_SCALE < scale < self.config.MAX_SANE_SCALE):
                continue

            bone = frozenset((leaf_kp_name, parent_kp_name))
            score = self._score_bone(bone, kps, points_map, nodes, scale)

            if score > best_bone_score:
                best_bone_score = score
                best_parent_node = parent_cand_node

        # Check if the best connection we found is good enough
        if best_parent_node and best_bone_score > self.config.MIN_BONE_SCORE_FOR_FRAGMENT:
            fragment_nodes = frozenset([leaf_node, best_parent_node])

            final_scale = self._get_skeleton_scale({n[0]: points_map[n].position for n in fragment_nodes})

            # Use the factory function to create the candidate with proper competition scoring
            return self._create_candidate(fragment_nodes, best_bone_score, final_scale)

        return None

    def _grow_skeleton(self,
           anchor_node:     Tuple[str, int],
           points_map:      Dict[Tuple[str, int], SoupPoint],
           score_debt_tol:  float
        ) -> Optional[CandidateSkeleton]:
        """ Grows a single skeleton candidate from an anchor point """

        # KDTree for fast spatial lookups
        all_nodes_list = list(points_map.keys())
        all_positions = np.array([points_map[node].position for node in all_nodes_list])
        if all_positions.shape[0] == 0: return None
        kdtree = cKDTree(all_positions)

        max_bone_len = self.median_ref_len * self.config.MAX_BONE_LEN

        # initialisation
        current_nodes = {anchor_node}
        current_kps = {anchor_node[0]: points_map[anchor_node].position}
        total_bone_score_sum = 0.0
        num_bones = 0

        while True:
            # Find all valid keypoints that could connect to the current skeleton
            # using topology (skeleton graph) and spatial proximity (KDTree)

            current_kp_names = {node[0] for node in current_nodes}
            nodes_to_evaluate = set()
            for node_in_skel in current_nodes:
                neighbor_kp_types = {n for n in self._skeleton_graph.neighbors(node_in_skel[0]) if
                                     n not in current_kp_names}
                if not neighbor_kp_types: continue

                center_pos = points_map[node_in_skel].position
                nearby_indices = kdtree.query_ball_point(center_pos, r=max_bone_len)
                for idx in nearby_indices:
                    cand_node = all_nodes_list[idx]
                    if cand_node[0] in neighbor_kp_types:
                        nodes_to_evaluate.add(cand_node)

            if not nodes_to_evaluate:
                break  # no more valid connections found

            # Calculate scale once per growth step
            current_step_scale = self._get_skeleton_scale(current_kps)
            if not (self.config.MIN_SANE_SCALE < current_step_scale < self.config.MAX_SANE_SCALE):
                # current skeleton has a bad scale so, abort
                break

            current_avg_score = (total_bone_score_sum / num_bones) if num_bones > 0 else 0
            current_base_score = current_avg_score * len(current_nodes)
            current_quality_bonus = 0.0

            if current_avg_score > self.config.HIGH_QUALITY_THRESHOLD:
                current_quality_bonus = current_base_score * (self.config.QUALITY_BONUS_FACTOR - 1.0)
            current_growth_score = current_base_score + current_quality_bonus

            # Evaluate each potential extension
            best_extension = None
            best_new_growth_score = -float('inf')

            for cand_node in nodes_to_evaluate:

                # Calculate the score contribution of only the new bones this candidate would form
                cand_kp_name = cand_node[0]
                temp_kps = current_kps.copy()
                temp_kps[cand_kp_name] = points_map[cand_node].position
                new_scale = current_step_scale

                new_bone_score_sum = 0
                new_bone_count = 0
                for existing_node in current_nodes:
                    bone = frozenset((cand_node[0], existing_node[0]))
                    if bone in self.bones_ratios:
                        temp_nodes = frozenset(current_nodes | {cand_node})
                        score = self._score_bone(bone, temp_kps, points_map, temp_nodes, new_scale)
                        if score < -500:  # An impossible bone was formed.
                            new_bone_score_sum = -float('inf')
                            break
                        new_bone_score_sum += score
                        new_bone_count += 1

                if new_bone_count == 0 or new_bone_score_sum == -float('inf'):
                    continue

                # Calculate the potential new holistic competition score
                new_total_bones = num_bones + new_bone_count
                new_total_score_sum = total_bone_score_sum + new_bone_score_sum
                new_avg_score = new_total_score_sum / new_total_bones

                new_num_nodes = len(current_nodes) + 1
                new_base_score = new_avg_score * new_num_nodes

                # smooth quality bonus
                bonus_factor = self.config.QUALITY_BONUS_FACTOR - 1.0

                # Map average score to 0-1 range, starting from a baseline (like 75) up to 100
                # Skeletons with avg score below 75 get no bonus
                baseline_quality = 75.0
                quality_range = 100.0 - baseline_quality

                # Normalized quality (0 to 1, clamped)
                normalized_quality = max(0, (new_avg_score - baseline_quality) / quality_range)

                # The bonus is the base score times the bonus factor, scaled by the normalized quality
                new_quality_bonus = new_base_score * bonus_factor * normalized_quality

                # final score to compare for the growth decision
                new_growth_score = new_base_score + new_quality_bonus

                # Keep track of the best extension found so far
                if new_growth_score > best_new_growth_score:
                    best_new_growth_score = new_growth_score
                    best_extension = {
                        "node": cand_node,
                        "score_contribution": new_bone_score_sum,
                        "bone_count_increase": new_bone_count
                    }

            # Make the growth decision (ompare the best potential new score with the current score)
            if best_extension and best_new_growth_score > (current_growth_score - score_debt_tol):
                # Yep, it's a good move. Accept growth.
                best_node = best_extension["node"]
                current_nodes.add(best_node)
                current_kps[best_node[0]] = points_map[best_node].position
                total_bone_score_sum += best_extension["score_contribution"]
                num_bones += best_extension["bone_count_increase"]
            else:
                # Nope, best potential growth is not good enough. Stop.
                break

        # Finalise and return the grown skeleton
        if len(current_nodes) < self.config.MIN_KPS_FOR_SKELETON:
            return None

        final_avg_score = (total_bone_score_sum / num_bones) if num_bones > 0 else 0.0
        if final_avg_score <= 0:
            return None

        final_scale = self._get_skeleton_scale(current_kps)

        return self._create_candidate(frozenset(current_nodes), final_avg_score, final_scale)

    def _get_skeleton_scale(self,
            keypoints:  Dict[str, np.ndarray]
        ) -> float:
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
        sane_scales = [s for s in scales if
                       self.config.MIN_SANE_SCALE <= s <= self.config.MAX_SANE_SCALE]

        return float(np.median(sane_scales)) if sane_scales else 1.0

    def _score_bone(self,
                    bone: Bone,
                    keypoints:  Dict[str, np.ndarray],
                    points_map: Dict[Tuple[str, int], SoupPoint],
                    nodes:      FrozenSet[Tuple[str, int]],
                    scale:      float
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
        conf1 = points_map[node1].confidence
        conf2 = points_map[node2].confidence

        expected_length = self.median_ref_len * stats['median_ratio'] * scale

        # Scale the expected deviation
        expected_mad = self.median_ref_len * stats['mad_ratio'] * scale + self.config.BONE_SCORE_MAD_EPSILON
        distance = np.linalg.norm(p1 - p2)

        # How many std dev (using MAD) the length is from the mean
        num_mads_away = abs(distance - expected_length) / (expected_mad + 1e-6)

        if num_mads_away > 2.0:  # Debug print for bones that are just a bit off
            logger.debug(f"      - Scoring bone {kp1_name}-{kp2_name}:\n"
                         f"          Scale: {scale:.2f}, Measured Dist: {distance:.2f}, Expected: {expected_length:.2f}"
                         f"          MADs away: {num_mads_away:.2f} (Thresh: {self.config.BONE_SCORE_MAD_THRESH})")

        if num_mads_away > self.config.BONE_SCORE_MAD_THRESH:
            # impossible bone: large negative penalty that will poison the average score
            return -1000.0

        # Gaussian-like fall off for the length score
        length_score = np.exp(-0.5 * num_mads_away ** 2)

        # Factor in the confidence of the 2D detections that created the 3D points
        confidence_score = (conf1 + conf2) / 2.0

        return length_score * confidence_score


class MultiObjectTracker:
    """ Main class for tracking multiple skeletons over time """

    def __init__(self, assembler: SkeletonAssembler, config: TrackerConfig = TrackerConfig()):

        self.assembler = assembler
        self.config = config
        self.frame_idx = -1
        self.tracklets: List[Tracklet] = []
        self.next_track_idx = 0

    def update(self,
               soup_points: List[SoupPoint],
               frame_idx: int
        ) -> List[Tracklet]:
        """ Processes a single frame using 'Hypothesize-and-Solve' workflow """

        self.frame_idx = frame_idx

        # Predict forward state of existing tracklets
        for tracklet in self.tracklets:
            tracklet.predict(self.frame_idx)

        # Generate all initial fragments from the 3D point soup and merge hypotheses from the assembler
        all_candidates, points_map = self.assembler.assemble_frame(soup_points)
        if not all_candidates:
            self.prune_tracklets()
            return self.get_active_tracklets()

        # Apply temporal guidance: boost scores of candidates that match tracklets
        bonuses = self._calculate_association_bonuses(all_candidates, points_map)
        for i, cand in enumerate(all_candidates):
            cand.competition_score += bonuses[i] * self.config.CONTINUITY_BONUS

        # Build a conflict graph that includes spatial conflicts and hierarchical conflicts (merge vs. its parts)
        conflict_graph = self._build_conflict_graph(all_candidates, points_map)

        # Solve for the globally optimal set of skeletons using MWIS
        # winner_indices = solve_mwis_SCIP(conflict_graph)
        winner_indices = solve_mwis_networkx(conflict_graph)

        winning_skeletons = [
            AssembledSkeleton(
                keypoints={node[0]: points_map[node].position for node in all_candidates[i].nodes},
                point_indices={node[0]: node[1] for node in all_candidates[i].nodes},
                score=all_candidates[i].anatomical_score,
                scale=all_candidates[i].scale
            )
            for i in winner_indices
        ]
        logger.debug(
            f"Conflict resolution selected {len(winner_indices)} skeletons from {len(all_candidates)} total candidates.")

        # Associate winning skeletons with tracklets and update state
        if self.tracklets and winning_skeletons:
            cost_matrix = self._build_final_assignment_cost_matrix(self.tracklets, winning_skeletons)
            tracklet_inds, winner_inds = linear_sum_assignment(cost_matrix)

            matched_winner_indices = set()
            for t_idx, w_idx in zip(tracklet_inds, winner_inds):
                if cost_matrix[t_idx, w_idx] < 1e9:
                    self.tracklets[t_idx].update(skeleton=winning_skeletons[w_idx], frame_idx=self.frame_idx)
                    matched_winner_indices.add(w_idx)
        else:
            matched_winner_indices = set()

        # Create new tracklets for unmatched skeletons
        for i, skel in enumerate(winning_skeletons):
            if i not in matched_winner_indices and self.assembler.central_kp in skel.keypoints:
                new_tracklet = Tracklet(self.next_track_idx, skel, self.frame_idx, self.assembler.central_kp, config=self.config)
                self.tracklets.append(new_tracklet)
                self.next_track_idx += 1

        # Prune old/lost tracklets
        self.prune_tracklets()

        return self.get_active_tracklets()

    def predict_only(self, frame_idx: int) -> List[Tracklet]:
        """ Handles frames with no detections by only running the prediction step """

        self.frame_idx = frame_idx

        for tracklet in self.tracklets:
            tracklet.predict(self.frame_idx)

        self.prune_tracklets()

        return self.get_active_tracklets()

    def get_active_tracklets(self) -> List[Tracklet]:
        """ Returns a list of tracklets that have been updated in the current frame """
        return [t for t in self.tracklets if t.time_since_update == 0]

    def prune_tracklets(self):
        """ Removes tracklets that have been lost for too long or that are very uncertain """

        self.tracklets = [
            t for t in self.tracklets
            if t.time_since_update <= self.config.MAX_TRACKLET_AGE and not np.sum(t.uncertainty['position']) > self.config.UNCERTAINTY_THRESHOLD
        ]

    def _build_conflict_graph(self,
            candidates: List[CandidateSkeleton],
            points_map: Dict[Tuple[str, int], SoupPoint]
        ) -> nx.Graph:
        """ Builds a conflict graph with both spatial and hierarchical conflicts """

        conflict_graph = nx.Graph()
        num_candidates = len(candidates)

        for i, cand in enumerate(candidates):
            weight = max(0, int(cand.competition_score * 100))
            conflict_graph.add_node(i, weight=weight)

        # Pre-calculate centroids for spatial checks
        centroids = np.array([
            np.mean([points_map[node].position for node in cand.nodes], axis=0) if cand.nodes else np.array([np.nan] * 3)
            for cand in candidates
        ])

        for i, j in combinations(range(num_candidates), 2):
            cand_i, cand_j = candidates[i], candidates[j]

            # Hierarchical conflict: if one candidate is composed of the other, they conflict
            if cand_i.constituent_indices and cand_j.constituent_indices:
                if cand_j.constituent_indices.issubset(cand_i.constituent_indices) or \
                        cand_i.constituent_indices.issubset(cand_j.constituent_indices):
                    conflict_graph.add_edge(i, j)
                    continue

            # Spatial Conflict (direct point sharing)
            shared_nodes = cand_i.nodes.intersection(cand_j.nodes)
            if len(shared_nodes) > self.config.CONFLICT_SOLVER_SHARED_POINTS_TOLERANCE:
                conflict_graph.add_edge(i, j)
                continue

            # Spatial Conflict (proximity)
            # Only check this if centroids are close
            dist_sq = np.sum((centroids[i] - centroids[j]) ** 2)
            if dist_sq < self.config.CONFLICT_SOLVER_BROAD_RADIUS ** 2:
                kps_i = {node[0]: points_map[node].position for node in cand_i.nodes}
                kps_j = {node[0]: points_map[node].position for node in cand_j.nodes}
                common = kps_i.keys() & kps_j.keys()
                union = kps_i.keys() | kps_j.keys()
                if not union:
                    continue

                proximal_intersection = sum(
                    1 for name in common if
                    np.linalg.norm(kps_i[name] - kps_j[name]) < self.config.CONFLICT_SOLVER_PROXIMITY_RADIUS
                )
                if proximal_intersection / len(union) > self.config.CONFLICT_SOLVER_JACCARD_THRESHOLD:
                    conflict_graph.add_edge(i, j)

        return conflict_graph

    def _calculate_association_bonuses(self,
                                       candidates: List[CandidateSkeleton],
                                       points_map: Dict[Tuple[str, int], SoupPoint]
                                       ) -> np.ndarray:
        """
        Calculates a score bonus (0 to 1) for each candidate skeleton based on its
        best possible alignment with any existing tracklet's prediction
        """

        if not self.tracklets or not candidates:
            return np.zeros(len(candidates))

        bonuses = np.zeros(len(candidates))

        for j, cand_skel in enumerate(candidates):
            skel_kps = {node[0]: points_map[node].position for node in cand_skel.nodes}

            if not skel_kps:
                continue

            max_bonus = 0.0
            for tracklet in self.tracklets:
                pred_pose = tracklet.predicted_pose

                if not pred_pose:
                    continue

                common_kps = pred_pose.keys() & skel_kps.keys()
                if len(common_kps) < self.config.ASSOCIATION_MIN_KPS:
                    continue

                # Calculate mean squared distance between common keypoints
                mean_dist_sq = sum(np.sum((pred_pose[kp] - skel_kps[kp]) ** 2) for kp in common_kps) / len(common_kps)

                # Gaussian bonus falls off as distance increases
                bonus = np.exp(-0.5 * mean_dist_sq / (self.config.ASSOCIATION_RADIUS ** 2))
                if bonus > max_bonus:
                    max_bonus = bonus

            bonuses[j] = max_bonus

        return bonuses

    def _build_final_assignment_cost_matrix(self,
                                            tracklets:  List[Tracklet],
                                            skeletons:  List[AssembledSkeleton]
                                            ) -> np.ndarray:
        """ Builds the cost matrix for the final assignment via Hungarian algorithm """

        cost_matrix = np.full((len(tracklets), len(skeletons)), 1e9)

        for i, tracklet in enumerate(tracklets):
            pred_pose = tracklet.predicted_pose

            if not pred_pose:
                continue

            for j, skel in enumerate(skeletons):
                common_kps = pred_pose.keys() & skel.keypoints.keys()

                if len(common_kps) < self.config.ASSOCIATION_MIN_KPS:
                    continue

                mean_dist_sq = sum(np.sum((pred_pose[kp] - skel.keypoints[kp]) ** 2)
                                   for kp in common_kps) / len(common_kps)

                if mean_dist_sq > self.config.ASSOCIATION_RADIUS ** 2:
                    continue

                # The cost is a weighted sum of pose distance and the skeleton's intrinsic score
                # A good skeleton (high score) should reduce the cost
                cost = (self.config.COST_POSE_DISTANCE_WEIGHT * mean_dist_sq +
                        self.config.COST_SKELETON_SCORE_WEIGHT * skel.score)
                cost_matrix[i, j] = cost

        return cost_matrix


if __name__ == '__main__':

    folder = Path().home() / 'Desktop' / '3d_ant_data'
    prefix = '240905-1616'
    session = 22

    anatomy_cfg = AnatomyConfig()
    assembler_cfg = AssemblerConfig()
    tracker_cfg = TrackerConfig()

    stats_output_file = folder / prefix / 'outputs' / f'bone_stats_session{session}.json'
    # prior_stats_file = Path().home() / 'Desktop' / 'bone_lengths.csv'
    prior_stats_file = None

    # Load data
    points_soup_file = folder / prefix / 'outputs' / f'points_soup_session{session}.pkl'
    skeleton_input_file = folder / prefix / 'inputs' / 'tracking'

    with open(points_soup_file, 'rb') as f:
        points_soup = pickle.load(f)

    if not points_soup:
        print("No points in the soup. Exiting.")
        exit()

    keypoints, bones, symmetry = fileio.load_skeleton_SLEAP(skeleton_input_file, symmetry=True)

    # Get Anatomical stats
    bootstrapper = StatsBootstrapper(
        output_path=stats_output_file,
        bones_list=bones,
        symmetry_map=symmetry,
        prior_stats_path=prior_stats_file,
        bootstrap_data=points_soup,
        config=anatomy_cfg
    )
    bone_stats = bootstrapper.get_initial_stats()

    # Initialise the tracking pipeline
    anatomy_learner = AnatomyLearner(initial_stats=bone_stats,
                                     config=anatomy_cfg)
    assembler = SkeletonAssembler(bones_list=bones,
                                  bone_stats=bone_stats,
                                  assembler_config=assembler_cfg,
                                  tracker_config=tracker_cfg)
    tracker = MultiObjectTracker(assembler=assembler,
                                 config=tracker_cfg)

    # Run tracking pipeline
    frames_indices = sorted(points_soup.keys())
    min_frame, max_frame = frames_indices[0], frames_indices[-1]

    tracklets_by_id = defaultdict(list)

    with alive_bar(title='Tracking Skeletons...', length=20, total=(max_frame - min_frame + 1), force_tty=True) as bar:
        for frame_idx in range(min_frame, max_frame + 1):

            # The assembler's anatomical model is updated with the latest learned stats
            current_stats = anatomy_learner.get_stats()
            assembler.update_bone_stats(current_stats)

            # Run the tracker for the frame
            if frame_idx in points_soup:
                active_tracklets = tracker.update(points_soup[frame_idx], frame_idx)
            else:
                active_tracklets = tracker.predict_only(frame_idx)

            # Feed the results back into the learner
            for tracklet in active_tracklets:
                # We only use skeletons from the current frame for learning
                if tracklet.last_update_frame == frame_idx:
                    anatomy_learner.add_sample(tracklet.skeleton)

            # Convert dataclasses back to dicts for serialization
            for tracklet in active_tracklets:
                skel_dict = tracklet.skeleton.to_dict()
                skel_dict['track_idx'] = tracklet.track_idx
                skel_dict['track_health'] = tracklet.health
                skel_dict['track_anatomical_integrity'] = tracklet.anatomical_integrity
                skel_dict['track_uncertainty_pos'] = tracklet.uncertainty['position'].tolist()
                skel_dict['track_velocity'] = tracklet.kf.x[3:6].flatten().tolist()
                skel_dict['track_predicted_pos'] = tracklet.predicted_position.tolist()
                skel_dict['time_since_update'] = tracklet.time_since_update

                # add the frame_idx to the skeleton dict
                skel_dict['frame_idx'] = frame_idx

                tracklets_by_id[tracklet.track_idx].append(skel_dict)

            bar()

    print("Tracking complete.")

    # Save final results (tracklet-centric format)
    output_file = folder / prefix / 'outputs' / f'tracklets_session{session}.pkl'
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'wb') as f:
        pickle.dump(dict(tracklets_by_id), f)

    print(f"Results saved to '{output_file}'")
