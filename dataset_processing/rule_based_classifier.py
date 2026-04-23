"""
Rule-based classifier for identifying asymmetric road scenes.

This module implements two main steps:
1. Lane width asymmetry detection
2. Curvature-based asymmetry detection
"""
import os
import json
from typing import List, Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion

from .config import *
from .geometry_utils import global_to_lidar, find_matched_gt_inst
from .map_utils import extract_lane_and_edges
from .curvature_analysis import (
    calculate_curvature, check_any_chunk_symmetrical_by_heading,
    identify_diverging_boundary_improved, calculate_region_curvature
)


class RuleBasedClassifier:
    """Rule-based classifier for asymmetric scene identification."""
    
    def __init__(self, nusc, nusc_maps, map_explorer, vector_map, car_img):
        """
        Initialize the classifier.
        
        Args:
            nusc: NuScenes instance
            nusc_maps: Dictionary of NuScenes maps
            map_explorer: Dictionary of map explorers
            vector_map: Vectorized map instance
            car_img: Car image for visualization
        """
        self.nusc = nusc
        self.nusc_maps = nusc_maps
        self.map_explorer = map_explorer
        self.vector_map = vector_map
        self.car_img = car_img
        
    def get_anns_results(self, input_dict):
        """Get annotation results for a given input dictionary."""
        
        lidar2ego = np.eye(4)
        lidar2ego[:3,:3] = Quaternion(input_dict['lidar2ego_rotation']).rotation_matrix
        lidar2ego[:3, 3] = input_dict['lidar2ego_translation']
        ego2global = np.eye(4)
        ego2global[:3,:3] = Quaternion(input_dict['ego2global_rotation']).rotation_matrix
        ego2global[:3, 3] = input_dict['ego2global_translation']

        lidar2global = ego2global @ lidar2ego

        lidar2global_translation = list(lidar2global[:3,3])
        lidar2global_rotation = list(Quaternion(matrix=lidar2global).q)

        location = input_dict['map_location']

        anns_results, vectors, polygon_geom = self.vector_map.gen_vectorized_samples(
            location, lidar2global_translation, lidar2global_rotation)
        
        return anns_results, vectors, polygon_geom
    
    def check_asymmetry_dist_change(self, left_sampled_points, right_sampled_points, 
                                   dist_offset_threshold=DIST_OFFSET_THRESHOLD):
        """
        Check asymmetry between two boundaries based on distance changes.
        
        Args:
            left_sampled_points: Points from the left boundary
            right_sampled_points: Points from the right boundary
            dist_offset_threshold: Threshold for determining asymmetry
            
        Returns:
            tuple: (is_asymmetric, left_points_aligned, right_points_aligned)
        """
        # Find points closest to ego vehicle (0,0)
        left_start_idx = np.argmin(np.linalg.norm(left_sampled_points - [0, 0], axis=1))
        right_start_idx = np.argmin(np.linalg.norm(right_sampled_points - [0, 0], axis=1))
        
        def check_direction(points, start_idx):
            """Check if points are ordered in ego vehicle's forward direction."""
            next_idx = min(start_idx + 1, len(points) - 1)
            prev_idx = max(start_idx - 1, 0)
            
            forward_vec = points[next_idx] - points[start_idx]
            backward_vec = points[prev_idx] - points[start_idx]
            
            ego_dir = np.array([0, 1])  # Forward direction
            
            forward_alignment = np.dot(forward_vec, ego_dir)
            backward_alignment = np.dot(backward_vec, ego_dir)
            
            return forward_alignment > backward_alignment

        # Check and fix boundary directions
        left_points = left_sampled_points
        if not check_direction(left_points, left_start_idx):
            left_points = left_points[::-1]
            left_start_idx = len(left_points) - 1 - left_start_idx
            
        right_points = right_sampled_points
        if not check_direction(right_points, right_start_idx):
            right_points = right_points[::-1]
            right_start_idx = len(right_points) - 1 - right_start_idx
        
        # Cut points from start indices
        left_points = left_points[left_start_idx:]
        right_points = right_points[right_start_idx:]

        # Calculate initial distance
        init_dist = np.linalg.norm(left_points[0] - right_points[0])
        
        # Check for asymmetry
        min_len = min(len(left_points), len(right_points))
        distances = np.linalg.norm(left_points[:min_len] - right_points[:min_len], axis=1)
        
        asymmetry = np.any(distances > init_dist + dist_offset_threshold)
        
        return asymmetry, left_points, right_points
    
    def check_point_distances(self, points, min_dist_closest=MIN_DIST_CLOSEST, 
                            max_dist_closest=MAX_DIST_CLOSEST, min_y=MIN_Y):
        """Check if points are within valid distance range from ego vehicle."""
        distances = np.linalg.norm(points, axis=1)
        closest_dist = np.min(distances)
        min_points_y = np.min(points[:, 1])

        return (min_dist_closest <= closest_dist <= max_dist_closest and 
                min_points_y >= min_y)
    
    def preprocess_scenes(self, nuscenes_infos: Dict[str, Any]) -> List[str]:
        """
        Process candidate scenes and extract valid road boundaries.
        
        Args:
            nuscenes_infos: NuScenes info dictionary
            
        Returns:
            List of valid sample tokens
        """
        
        # Setup output directory
        os.makedirs(SCENES_CANDIDATE_DIR, exist_ok=True)
        
        sample_token_candidates = []
        
        for i, info in enumerate(tqdm(nuscenes_infos['infos'], desc="Processing candidates")):
            sample_token = info['token']
            
            # Get map location
            sample = self.nusc.get('sample', sample_token)
            scene_token = sample['scene_token']
            scene = self.nusc.get('scene', scene_token)
            log_token = scene['log_token']
            log = self.nusc.get('log', log_token)
            location = log['location']

            # Get ego vehicle's pose
            lidar_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            lidar_pose_token = lidar_data['ego_pose_token']
            lidar_pose = self.nusc.get('ego_pose', lidar_pose_token)
            ego_trans_global = np.array(lidar_pose['translation'])
            ego_rot_global = np.array(lidar_pose['rotation'])

            # Get map layers
            layers = self.nusc_maps[location].explorer.layers_on_point(
                ego_trans_global[0], ego_trans_global[1])

            if layers['lane'] != '':
                lane_layer = self.nusc_maps[location].get('lane', layers['lane'])
            else:
                lane_layer = None
                
            if layers['road_segment'] != '':
                road_layer = self.nusc_maps[location].get('road_segment', layers['road_segment'])
            elif layers['road_block'] != '':
                road_layer = self.nusc_maps[location].get('road_block', layers['road_block'])
            else:
                road_layer = None

            # Validation checks
            if road_layer is None or lane_layer is None:
                continue
            if 'is_intersection' in road_layer and road_layer['is_intersection']:
                continue
            if len(road_layer['exterior_node_tokens']) > MAX_NODES_IN_MAP_NODE:
                continue

            # Extract lane boundaries
            lane_midline, left_pts, right_pts = extract_lane_and_edges(
                self.nusc_maps[location], road_layer, lane_layer)
            
            # Add z coordinate and transform to LiDAR frame
            lane_midline = np.hstack((lane_midline, np.zeros((lane_midline.shape[0], 1))))
            left_pts = np.hstack((left_pts, np.zeros((left_pts.shape[0], 1))))
            right_pts = np.hstack((right_pts, np.zeros((right_pts.shape[0], 1))))
            
            left_pts_lidar = global_to_lidar(left_pts, self.nusc, sample_token)[:, :2]
            right_pts_lidar = global_to_lidar(right_pts, self.nusc, sample_token)[:, :2]

            # Match GT instances
            anns_results, vectors, polygon_geom = self.get_anns_results(info)
            
            gt_insts_pts_loc = []
            gt_insts = anns_results['gt_vecs_pts_loc']
            gt_insts_label = anns_results['gt_vecs_label']
            for instance in gt_insts.instance_list:
                gt_insts_pts_loc.append(list(instance.coords))
                
            left_gt_inst_idx, min_dist_left = find_matched_gt_inst(
                left_pts_lidar, gt_insts_pts_loc, gt_insts_label, num_points=40)
            right_gt_inst_idx, min_dist_right = find_matched_gt_inst(
                right_pts_lidar, gt_insts_pts_loc, gt_insts_label, num_points=40)
            
            if left_gt_inst_idx is None or right_gt_inst_idx is None:
                continue
            if left_gt_inst_idx == right_gt_inst_idx:
                continue
            
            # Get sampled points and filter
            gt_insts_sampled_points = np.array(anns_results['gt_vecs_pts_loc'].fixed_interval_sampled_points)
            left_gt_sampled_points = gt_insts_sampled_points[left_gt_inst_idx]
            right_gt_sampled_points = gt_insts_sampled_points[right_gt_inst_idx]
            
            # Filter padding values
            left_gt_sampled_points = left_gt_sampled_points[left_gt_sampled_points[:, 0] != PADDING_VALUE]
            right_gt_sampled_points = right_gt_sampled_points[right_gt_sampled_points[:, 0] != PADDING_VALUE]
            
            # Additional validation
            if (len(left_gt_sampled_points[left_gt_sampled_points[:, 1] > 0]) < MIN_BOUNDARY_LENGTH or 
                len(right_gt_sampled_points[right_gt_sampled_points[:, 1] > 0]) < MIN_BOUNDARY_LENGTH):
                continue
            
            if (np.min(np.linalg.norm(left_gt_sampled_points - [0, 0], axis=1)) > MAX_DISTANCE_FROM_EGO and 
                np.min(np.linalg.norm(right_gt_sampled_points - [0, 0], axis=1)) > MAX_DISTANCE_FROM_EGO):
                continue
            
            if (np.linalg.norm(left_gt_sampled_points[0] - left_gt_sampled_points[-1]) < MIN_ENDPOINT_DISTANCE or 
                np.linalg.norm(right_gt_sampled_points[0] - right_gt_sampled_points[-1]) < MIN_ENDPOINT_DISTANCE):
                continue
            
            sample_token_candidates.append(sample_token)
            
            # Save visualization and data
            self._save_candidate_visualization(sample_token, left_gt_sampled_points, 
                                             right_gt_sampled_points)
            
        return sample_token_candidates
    
    def classify_by_lane_width(self, sample_tokens: List[str]) -> Tuple[List[str], List[str]]:
        """
        Classify samples based on lane width asymmetry.
        
        Args:
            sample_tokens: List of sample tokens to classify
            
        Returns:
            Tuple of (asymmetric_tokens, symmetric_tokens)
        """
        os.makedirs(SCENES_ASYMMETRIC_DIST_DIR, exist_ok=True)
        os.makedirs(SCENES_SYMMETRIC_DIST_DIR, exist_ok=True)
        
        sample_tokens_asymmetric_dist = []
        sample_tokens_symmetric_dist = []
        
        for sample_token in tqdm(sample_tokens, desc="Classifying by lane width"):
            # Load boundary data
            with open(f'{SCENES_CANDIDATE_DIR}/{sample_token}.json', 'r') as f:
                structured_data = json.load(f)
                
            left_gt_sampled_points = np.array(structured_data['map_elements'][0]['coordinates'])
            right_gt_sampled_points = np.array(structured_data['map_elements'][1]['coordinates'])
            
            # Check asymmetry
            is_asymmetric, left_aligned, right_aligned = self.check_asymmetry_dist_change(
                left_gt_sampled_points, right_gt_sampled_points)
            
            if is_asymmetric:
                sample_tokens_asymmetric_dist.append(sample_token)
                dst_dir = SCENES_ASYMMETRIC_DIST_DIR
            else:
                sample_tokens_symmetric_dist.append(sample_token)
                dst_dir = SCENES_SYMMETRIC_DIST_DIR
            
            # Filter points and save
            left_filtered = left_aligned[left_aligned[:, 1] >= MIN_Y_COORDINATE]
            right_filtered = right_aligned[right_aligned[:, 1] >= MIN_Y_COORDINATE]
            
            structured_data['map_elements'][0]['coordinates'] = left_filtered.tolist()
            structured_data['map_elements'][1]['coordinates'] = right_filtered.tolist()
            
            with open(f'{dst_dir}/{sample_token}.json', 'w') as f:
                json.dump(structured_data, f, indent=2)
            
            # Save visualization
            self._save_classification_visualization(sample_token, left_filtered, 
                                                  right_filtered, dst_dir)
        
        return sample_tokens_asymmetric_dist, sample_tokens_symmetric_dist
    
    def classify_by_curvature(self, sample_tokens_asymmetric_dist: List[str]) -> Tuple[List[str], List[str], List[str], Dict[str, List]]:
        """
        Classify asymmetric distance samples by curvature analysis.
        
        Args:
            sample_tokens_asymmetric_dist: List of asymmetric distance tokens
            
        Returns:
            Tuple of (asymmetric_curvature, symmetric_curvature, invalid_curvature, diverge_points_dict)
        """
        sample_tokens_asymmetric_curvature = []
        sample_tokens_asymmetric_curvature_invalid = []
        sample_tokens_symmetric_curvature = []
        diverge_points_dict = {}
        
        for sample_token in tqdm(sample_tokens_asymmetric_dist, desc="Analyzing curvature"):
            # Load boundary data
            with open(f'{SCENES_ASYMMETRIC_DIST_DIR}/{sample_token}.json', 'r') as f:
                structured_data = json.load(f)
            
            left_points = np.array(structured_data['map_elements'][0]['coordinates'])
            right_points = np.array(structured_data['map_elements'][1]['coordinates'])
            
            # Basic validation
            min_len = min(len(left_points), len(right_points))
            if min_len < MIN_POINTS:
                sample_tokens_asymmetric_curvature_invalid.append(sample_token)
                continue
            
            # Check symmetry using heading-based approach
            if check_any_chunk_symmetrical_by_heading(
                    left_points, right_points, 
                    chunk_size=CHUNK_SIZE, 
                    angle_thresh_rad=np.radians(ANGLE_THRESHOLD_DEG)):
                sample_tokens_symmetric_curvature.append(sample_token)
                continue
            
            # Identify diverging boundary
            left_points_trim = left_points[:min_len]
            right_points_trim = right_points[:min_len]
            diverge_boundary_tag, confidence, left_score, right_score = identify_diverging_boundary_improved(
                left_points_trim, right_points_trim)
            
            # Assign diverge and reference boundaries
            if diverge_boundary_tag == 'left':
                diverge_boundary_pts = left_points_trim
                reference_boundary_pts = right_points_trim
            else:
                diverge_boundary_pts = right_points_trim
                reference_boundary_pts = left_points_trim
            
            # Calculate curvatures
            diverge_curvatures = calculate_curvature(diverge_boundary_pts)
            reference_curvatures = calculate_curvature(reference_boundary_pts)
            curvature_diff = diverge_curvatures - reference_curvatures
            
            # Calculate regional curvatures
            diverge_region_curvature = calculate_region_curvature(diverge_boundary_pts, CHUNK_SIZE)
            reference_region_curvature = calculate_region_curvature(reference_boundary_pts, CHUNK_SIZE)
            
            # Check asymmetry conditions
            max_curvature_diff = np.max(curvature_diff)
            max_reference_curvature = np.max(reference_region_curvature)
            
            if (max_curvature_diff > CURVATURE_DIFF_THRESHOLD and 
                max_reference_curvature < REGION_CURVATURE_THRESHOLD):
                
                # Additional validations
                large_diff_indices = np.where(curvature_diff > CURVATURE_DIFF_THRESHOLD)[0]
                points_to_check = diverge_boundary_pts[large_diff_indices]
                
                if not self.check_point_distances(points_to_check):
                    sample_tokens_asymmetric_curvature_invalid.append(sample_token)
                    continue
                
                min_dist_to_diverge = np.min(np.linalg.norm(diverge_boundary_pts, axis=1))
                if min_dist_to_diverge > MIN_DIST_TO_DIVERGE_BOUNDARY:
                    sample_tokens_asymmetric_curvature_invalid.append(sample_token)
                    continue
                
                # Valid asymmetric sample
                sample_tokens_asymmetric_curvature.append(sample_token)
                diverge_points_dict[sample_token] = points_to_check.tolist()
            else:
                sample_tokens_symmetric_curvature.append(sample_token)
        
        # Remove problematic scenes
        tokens_to_remove = []
        for sample_token in sample_tokens_asymmetric_curvature:
            sample = self.nusc.get('sample', sample_token)
            scene_token = sample['scene_token']
            scene = self.nusc.get('scene', scene_token)
            if scene['name'] in SCENES_TO_REMOVE:
                tokens_to_remove.append(sample_token)
        
        for token in tokens_to_remove:
            sample_tokens_asymmetric_curvature.remove(token)
            sample_tokens_asymmetric_curvature_invalid.append(token)
        
        return (sample_tokens_asymmetric_curvature, sample_tokens_symmetric_curvature, 
                sample_tokens_asymmetric_curvature_invalid, diverge_points_dict)
    
    def _save_candidate_visualization(self, sample_token: str, left_points: np.ndarray, 
                                    right_points: np.ndarray):
        """Save visualization for candidate scene."""
        plt.figure(figsize=(5, 10))
        
        plt.plot(left_points[:, 0], left_points[:, 1], 'r-')
        mid_idx_left = len(left_points) // 2
        plt.text(left_points[mid_idx_left, 0] - 4, left_points[mid_idx_left, 1], 
                'Left', fontsize=18, color='red')
        
        plt.plot(right_points[:, 0], right_points[:, 1], 'b-')
        mid_idx_right = len(right_points) // 2
        plt.text(right_points[mid_idx_right, 0] + 1, right_points[mid_idx_right, 1], 
                'Right', fontsize=18, color='blue')
        
        plt.imshow(self.car_img, extent=[-1.2, 1.2, -1.5, 1.5])
        plt.arrow(0, 0, 0, 5, head_width=0.5, head_length=0.5, fc='red', ec='red')
        
        plt.axis('equal')
        plt.xlim(-15, 15)
        plt.ylim(-30, 30)
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.title('Input Map Ground Truth Visualization')
        
        plt.savefig(f'{SCENES_CANDIDATE_DIR}/{sample_token}.png')
        plt.close()
        
        # Save JSON data
        left_boundary = {"tag": "left", "coordinates": left_points.tolist()}
        right_boundary = {"tag": "right", "coordinates": right_points.tolist()}
        
        structured_data = {
            "map_elements": [left_boundary, right_boundary],
            "metadata": {
                "version": "1.0",
                "coordinate_system": "local_lidar_frame",
                "units": "meters"
            }
        }
        
        with open(f'{SCENES_CANDIDATE_DIR}/{sample_token}.json', 'w') as f:
            json.dump(structured_data, f, indent=2)
    
    def _save_classification_visualization(self, sample_token: str, left_points: np.ndarray, 
                                         right_points: np.ndarray, dst_dir: str):
        """Save visualization for classified scene."""
        plt.figure(figsize=(6, 7))
        
        plt.plot(left_points[:, 0], left_points[:, 1], 'r-')
        mid_idx_left = len(left_points) // 2 if len(left_points) > 0 else 0
        if len(left_points) > 0:
            plt.text(left_points[mid_idx_left, 0] - 4, left_points[mid_idx_left, 1], 
                    'Left', fontsize=18, color='red')
        
        plt.plot(right_points[:, 0], right_points[:, 1], 'b-')
        mid_idx_right = len(right_points) // 2 if len(right_points) > 0 else 0
        if len(right_points) > 0:
            plt.text(right_points[mid_idx_right, 0] + 1, right_points[mid_idx_right, 1], 
                    'Right', fontsize=18, color='blue')
        
        plt.imshow(self.car_img, extent=[-1.2, 1.2, -1.5, 1.5])
        plt.arrow(0, 0, 0, 5, head_width=0.5, head_length=0.5, fc='red', ec='red')
        
        plt.axis('equal')
        plt.xlim(-15, 15)
        plt.ylim(-5, 30)
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.title('Input Map Ground Truth Visualization')
        
        plt.savefig(f'{dst_dir}/{sample_token}.png')
        plt.close()

