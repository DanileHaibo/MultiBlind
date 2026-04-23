import os
import json
from pathlib import Path
import numpy as np
from typing import List, Tuple, Optional, Any
from shapely.geometry import LineString, Point, Polygon
import torch
import yaml

from nuscenes import NuScenes
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion


### File system utilities ###
def load_config():
    """Load configuration from YAML file"""
    config_path = Path(__file__).parent.parent.parent / "configs" / "HybridAStar.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
def setup_directories(directories):
    """
    Create or clean directories.
    
    Args:
        directories (list): List of directory paths to setup
    """
    for dir_path in directories:
        # if os.path.exists(dir_path):
        #     for file in os.listdir(dir_path):
        #         file_path = os.path.join(dir_path, file)
        #         if os.path.isfile(file_path):
        #             os.remove(file_path)
        # else:
        os.makedirs(dir_path, exist_ok=True)


### Geometry utilities ###
def global_to_lidar(points, nusc, sample_token):
    '''
    points: (num_points, 3) in global coordinate
    '''
    
    def get_matrix(calibrated_data, inverse=False):
        
        output = np.eye(4)
        output[:3, :3] = Quaternion(calibrated_data["rotation"]).rotation_matrix
        output[:3,  3] = calibrated_data["translation"]
        if inverse:
            output = np.linalg.inv(output)
        return output
    
    # global -> ego
    sample = nusc.get('sample', sample_token)
    pointsensor = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
    global_to_ego = get_matrix(poserecord, inverse=True)
    
    # ego -> lidar
    cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
    ego_to_lidar = get_matrix(cs_record, inverse=True)
    
    # global -> lidar
    global_to_lidar = ego_to_lidar @ global_to_ego
    
    hom_points = np.hstack((points, np.ones((points.shape[0], 1))))
    lidar_points = hom_points @ global_to_lidar.T
    
    return lidar_points[:, :-1]

def lidar_to_global(points, nusc, sample_token, ground=False):

    def get_matrix(calibrated_data, inverse=False):
        
        output = np.eye(4)
        output[:3, :3] = Quaternion(calibrated_data["rotation"]).rotation_matrix
        output[:3,  3] = calibrated_data["translation"]
        if inverse:
            output = np.linalg.inv(output)
        return output


    # lidar->ego 变换矩阵
    sample = nusc.get('sample', sample_token)
    pointsensor = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
    lidar_to_ego = get_matrix(cs_record)
    # print(lidar_to_ego)

    # 获取lidar数据时对应的车体位姿
    poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
    # ego->global 变换矩阵
    ego_to_global = get_matrix(poserecord)
    # print(ego_to_global)

    # lidar->global 变换矩阵
    lidar_to_global = ego_to_global @ lidar_to_ego
    # print(lidar_to_global)
    
    hom_points = np.hstack((points, np.ones((points.shape[0], 1))))
    global_points = hom_points @ lidar_to_global.T
    
    if ground:
        global_points[:, 2] = 0
    
    return global_points[:, :-1]

def is_in_range(points, x_range=(-15, 15), y_range=(0, 30), mode='any'):
    x_in_range = (x_range[0] <= points[:, 0]) & (points[:, 0] <= x_range[1])
    y_in_range = (y_range[0] <= points[:, 1]) & (points[:, 1] <= y_range[1])
    # A point is in range if both its x and y coordinates are in range
    points_in_range = x_in_range & y_in_range
    
    # Return True if any point is fully in range
    if mode == 'any':
        return points_in_range.any()
    elif mode == 'all':
        return points_in_range.all()


### Map utilities ###
def calculate_boundary_iou(boundary1: np.ndarray, boundary2: np.ndarray, distance_threshold: float = 1.0) -> float:
    """
    Calculate IoU between two boundary point sequences.
    
    Args:
        boundary1: First boundary points (N, 2)
        boundary2: Second boundary points (M, 2)
        distance_threshold: Distance threshold for point matching [m]
        
    Returns:
        float: IoU score between 0 and 1
    """
    matched1 = np.zeros(len(boundary1), dtype=bool)
    matched2 = np.zeros(len(boundary2), dtype=bool)
    
    overlap = 0
    for i, pt1 in enumerate(boundary1):
        for j, pt2 in enumerate(boundary2):
            if not matched1[i] and not matched2[j]:
                if np.linalg.norm(pt1 - pt2) < distance_threshold:
                    overlap += 1
                    matched1[i] = True
                    matched2[j] = True
                    break
    
    union = len(boundary1) + len(boundary2) - overlap
    if union == 0:
        return 1.0  # If both boundaries are identical
        
    return overlap / union

def calculate_boundary_iou_tensor(boundary1: torch.Tensor, boundary2: torch.Tensor, 
                                 distance_threshold: float = 1.0, 
                                 is_tensor: bool = True) -> torch.Tensor:
    """
    Calculate IoU between two boundary point sequences using PyTorch tensors.
    
    Args:
        boundary1: First boundary points tensor (N, 2)
        boundary2: Second boundary points tensor (M, 2)
        distance_threshold: Distance threshold for point matching [m]
        is_tensor: If True, returns a tensor, otherwise returns a float
        
    Returns:
        torch.Tensor or float: IoU score between 0 and 1
    """
    # Ensure inputs are on the same device
    device = boundary1.device
    
    # Compute pairwise distances between all points in both boundaries
    # Reshape to enable broadcasting
    b1 = boundary1.unsqueeze(1)  # Shape: (N, 1, 2)
    b2 = boundary2.unsqueeze(0)  # Shape: (1, M, 2)
    
    # Compute squared Euclidean distances
    distances = torch.sum((b1 - b2) ** 2, dim=2)  # Shape: (N, M)
    
    # Find minimum distances for each point in both boundaries
    min_dists_b1, _ = torch.min(distances, dim=1)  # Shape: (N,)
    min_dists_b2, _ = torch.min(distances, dim=0)  # Shape: (M,)
    
    # Count matches (points within threshold)
    matches_b1 = (min_dists_b1 < distance_threshold**2).sum()
    matches_b2 = (min_dists_b2 < distance_threshold**2).sum()
    
    # Take the average of matches from both directions for robustness
    overlap = (matches_b1 + matches_b2) / 2
    
    # Calculate union
    union = boundary1.size(0) + boundary2.size(0) - overlap
    
    # Handle edge case
    if union == 0:
        iou = torch.tensor(1.0, device=device)
    else:
        iou = overlap / union
    
    if is_tensor:
        return iou
    else:
        return iou.item()

def sample_boundaries_fixed_num(boundaries, num_points=50):
    """
    Sample fixed number of points from boundary lines
    Args:
        boundaries: numpy array of shape (n, 20, 2) containing boundary coordinates
        num_points: number of points to sample for each boundary (default: 50)
    Returns:
        numpy array of shape (n, num_points, 2) with sampled points
    """
    instance_points_list = []
    
    for boundary in boundaries:
        line = LineString(boundary)
        
        distances = np.linspace(0, line.length, num_points)
        sampled_points = np.array([list(line.interpolate(distance).coords[0]) 
                                 for distance in distances])
        
        instance_points_list.append(sampled_points)
    
    return np.array(instance_points_list)


### Planning Goals ###



### Planning evaluation utilities ###
def check_trajectory_collision(trajectory: List[np.ndarray], 
                            ground_truth_boundaries: np.ndarray,
                            collision_threshold: float = 0.5) -> bool:
    """
    Check if trajectory points collide with ground truth road boundaries
    
    Args:
        trajectory: List of states [x, y, heading] from planner
        ground_truth_boundaries: Ground truth road boundaries (n, 20, 2)
        vehicle_params: Vehicle parameters (unused in simplified version)
        collision_threshold: Minimum distance (meters) to consider as collision
    
    Returns:
        bool: True if collision detected, False otherwise
    """
    boundary_lines = []
    for boundary in ground_truth_boundaries:
        if np.allclose(boundary[0], boundary[-1]):
            continue
        boundary_lines.append(LineString(boundary))
    
    for state in trajectory:
        x, y, _ = state  # Ignore heading since we're just checking points
        point = Point(x, y)
        
        for boundary in boundary_lines:
            if point.distance(boundary) < collision_threshold:
                return True
                
    return False

def calculate_ade(traj1: np.ndarray, traj2: np.ndarray, handle_empty: str = 'zero_fill') -> float:
    """
    Calculate Average Displacement Error between two trajectories.
    
    Args:
        traj1: First trajectory, shape (n, 2) or (n, 3)
        traj2: Second trajectory, shape (m, 2) or (m, 3)
        handle_empty: How to handle empty trajectories
            - 'zero_fill': Fill empty trajectories with zeros
            - 'max_dist': Use maximum possible distance (worst case)
            - 'skip': Return None for empty trajectories
    
    Returns:
        float: ADE value, or None if handle_empty='skip' and trajectory is empty
    """
    traj1 = np.array(traj1)
    traj2 = np.array(traj2)
    
    if len(traj1) == 0 or len(traj2) == 0:
        if handle_empty == 'skip':
            return None
        elif handle_empty == 'max_dist':
            return 100.0  # Use a large value to represent maximum error
        elif handle_empty == 'zero_fill':
            if len(traj1) == 0:
                traj1 = np.zeros_like(traj2)
            else:
                traj2 = np.zeros_like(traj1)
    
    if len(traj1) != len(traj2):
        target_len = max(len(traj1), len(traj2))
        if len(traj1) < target_len:
            indices = np.linspace(0, len(traj1)-1, target_len)
            traj1 = np.array([np.interp(indices, range(len(traj1)), traj1[:, i]) for i in range(2)]).T
        else:
            indices = np.linspace(0, len(traj2)-1, target_len)
            traj2 = np.array([np.interp(indices, range(len(traj2)), traj2[:, i]) for i in range(2)]).T
    
    distances = np.linalg.norm(traj1[:, :2] - traj2[:, :2], axis=1)
    
    return float(np.mean(distances))

def calculate_fde(traj1: np.ndarray, traj2: np.ndarray, handle_empty: str = 'zero_fill') -> float:
    """
    Calculate Final Displacement Error between two trajectories.
    
    Args:
        traj1: First trajectory, shape (n, 2) or (n, 3)
        traj2: Second trajectory, shape (m, 2) or (m, 3)
        handle_empty: How to handle empty trajectories
            - 'zero_fill': Fill empty trajectories with zeros
            - 'max_dist': Use maximum possible distance (worst case)
            - 'skip': Return None for empty trajectories
    
    Returns:
        float: FDE value, or None if handle_empty='skip' and trajectory is empty
    """
    traj1 = np.array(traj1)
    traj2 = np.array(traj2)
    
    if len(traj1) == 0 or len(traj2) == 0:
        if handle_empty == 'skip':
            return None
        elif handle_empty == 'max_dist':
            return 100.0  # Use a large value to represent maximum error
        elif handle_empty == 'zero_fill':
            if len(traj1) == 0:
                traj1 = np.zeros_like(traj2)
            else:
                traj2 = np.zeros_like(traj1)
    
    final_distance = np.linalg.norm(traj1[-1, :2] - traj2[-1, :2])
    
    return float(final_distance)