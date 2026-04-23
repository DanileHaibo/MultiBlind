"""
Geometry utility functions for road boundary analysis and coordinate transformations.
"""
import numpy as np
import math
from typing import Optional, Tuple
from scipy.spatial.distance import cdist
from shapely.geometry import LineString
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion


def global_to_lidar(points, nusc, sample_token):
    """
    Transform points from global coordinate system to LiDAR coordinate system.
    
    Args:
        points: (num_points, 3) in global coordinate
        nusc: NuScenes instance
        sample_token: Sample token identifier
        
    Returns:
        np.ndarray: Points in LiDAR coordinate system
    """
    def get_matrix(calibrated_data, inverse=False):
        output = np.eye(4)
        output[:3, :3] = Quaternion(calibrated_data["rotation"]).rotation_matrix
        output[:3, 3] = calibrated_data["translation"]
        if inverse:
            output = np.linalg.inv(output)
        return output
    
    # Global -> ego
    sample = nusc.get('sample', sample_token)
    pointsensor = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
    global_to_ego = get_matrix(poserecord, inverse=True)
    
    # Ego -> LiDAR
    cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
    ego_to_lidar = get_matrix(cs_record, inverse=True)
    
    # Global -> LiDAR
    global_to_lidar_transform = ego_to_lidar @ global_to_ego
    
    hom_points = np.hstack((points, np.ones((points.shape[0], 1))))
    lidar_points = hom_points @ global_to_lidar_transform.T
    
    return lidar_points[:, :-1]


def lidar_to_img(points, nusc, sample_token, cam_name, img_shape, ground=False):
    """
    Transform points from LiDAR coordinate system to image coordinates.
    
    Args:
        points: (num_points, 3) in LiDAR coordinate
        nusc: NuScenes instance
        sample_token: Sample token identifier
        cam_name: Camera name
        img_shape: Image shape
        ground: Whether to project points to ground plane
        
    Returns:
        np.ndarray: Visible points in image coordinates
    """
    def translate(points, x):
        return points + x

    def rotate(points, rot_matrix):
        return points @ rot_matrix.T
    
    # Get camera and LiDAR data
    sample = nusc.get('sample', sample_token)
    cam = nusc.get('sample_data', sample['data'][cam_name])
    pointsensor = nusc.get('sample_data', sample['data']['LIDAR_TOP'])

    # Transform from LiDAR to ego
    cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
    points = rotate(points, Quaternion(cs_record['rotation']).rotation_matrix)
    points = translate(points, np.array(cs_record['translation']))

    # Transform from ego to global
    poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
    points = rotate(points, Quaternion(poserecord['rotation']).rotation_matrix)
    points = translate(points, np.array(poserecord['translation']))
    
    if ground:
        points[:, 2] = 0  # Project to ground plane
    
    # Transform from global to camera ego
    poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
    points = translate(points, -np.array(poserecord['translation']))
    points = rotate(points, Quaternion(poserecord['rotation']).rotation_matrix.T)
    
    # Transform from ego to camera
    cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    points = translate(points, -np.array(cs_record['translation']))
    points = rotate(points, Quaternion(cs_record['rotation']).rotation_matrix.T)
    
    # Project to image
    from nuscenes.utils.geometry_utils import view_points
    depths = points[:, 2]
    points = view_points(points.T, np.array(cs_record['camera_intrinsic']), normalize=True)
    points = points.T

    # Filter points
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > 1)
    mask = np.logical_and(mask, points[:, 0] > 1)
    mask = np.logical_and(mask, points[:, 0] < img_shape[1] - 1)
    mask = np.logical_and(mask, points[:, 1] > 1)
    mask = np.logical_and(mask, points[:, 1] < img_shape[0] - 1)
    
    return points[mask]


def find_matched_gt_inst(boundary_pts_loc, gt_insts_pts_loc, gt_insts_label, num_points=50):
    """
    Match polylines using interpolation and Chamfer distance.
    
    Args:
        boundary_pts_loc: Query polyline points (n, 2)
        gt_insts_pts_loc: List of GT polylines [(m1, 2), (m2, 2), ...]
        gt_insts_label: Labels for GT instances
        num_points: Number of points to interpolate
        
    Returns:
        tuple: (gt_inst_idx, min_dist)
    """
    min_dist = float('inf')
    gt_inst_idx = None
    
    # Interpolate query boundary to fixed number of points
    boundary_line = LineString(boundary_pts_loc)
    distances = np.linspace(0, boundary_line.length, num_points)
    boundary_interp = np.array([boundary_line.interpolate(dist).coords[0] for dist in distances])
    
    for idx, (gt_inst, label) in enumerate(zip(gt_insts_pts_loc, gt_insts_label)):
        if label != 2:  # Only consider boundary labels
            continue
            
        # Interpolate GT instance
        gt_line = LineString(gt_inst)
        gt_distances = np.linspace(0, gt_line.length, num_points)
        gt_interp = np.array([gt_line.interpolate(dist).coords[0] for dist in gt_distances])
        
        # Calculate Chamfer distance
        dist_matrix = cdist(boundary_interp, gt_interp)
        chamfer_dist = (np.mean(np.min(dist_matrix, axis=1)) + 
                       np.mean(np.min(dist_matrix, axis=0))) / 2
        
        if chamfer_dist < min_dist:
            min_dist = chamfer_dist
            gt_inst_idx = idx
    
    return gt_inst_idx, min_dist


def angle_wrap(radians):
    """Wrap angles to lie within [-pi, pi)."""
    return (radians + np.pi) % (2 * np.pi) - np.pi


def interpolate(pts, num_pts=None, max_dist=None):
    """
    Interpolate points either based on cumulative distances or maximum distance.
    
    Args:
        pts: XYZ(H) coordinates
        num_pts: Desired number of total points
        max_dist: Maximum distance between points
        
    Returns:
        np.ndarray: Interpolated coordinates
    """
    if num_pts is not None and max_dist is not None:
        raise ValueError("Only one of num_pts or max_dist can be used!")

    if pts.ndim != 2:
        raise ValueError("pts is expected to be 2 dimensional")

    pos_dim = min(pts.shape[-1], 3)
    has_heading = pts.shape[-1] == 4

    if num_pts is not None:
        assert num_pts > 1, f"num_pts must be at least 2, but got {num_pts}"

        if pts.shape[0] == num_pts:
            return pts

        cum_dist = np.cumsum(
            np.linalg.norm(np.diff(pts[..., :pos_dim], axis=0), axis=-1)
        )
        cum_dist = np.insert(cum_dist, 0, 0)

        steps = np.linspace(cum_dist[0], cum_dist[-1], num_pts)
        xyz_inter = np.empty((num_pts, pts.shape[-1]), dtype=pts.dtype)
        for i in range(pos_dim):
            xyz_inter[:, i] = np.interp(steps, xp=cum_dist, fp=pts[:, i])

        if has_heading:
            xyz_inter[:, 3] = angle_wrap(
                np.interp(steps, xp=cum_dist, fp=np.unwrap(pts[:, 3]))
            )

        return xyz_inter

    elif max_dist is not None:
        unwrapped_pts = pts
        if has_heading:
            unwrapped_pts[..., 3] = np.unwrap(unwrapped_pts[..., 3])

        segments = unwrapped_pts[..., 1:, :] - unwrapped_pts[..., :-1, :]
        seg_lens = np.linalg.norm(segments[..., :pos_dim], axis=-1)
        new_pts = [unwrapped_pts[..., 0:1, :]]
        
        for i in range(segments.shape[-2]):
            num_extra_points = seg_lens[..., i] // max_dist
            if num_extra_points > 0:
                step_vec = segments[..., i, :] / (num_extra_points + 1)
                new_pts.append(
                    unwrapped_pts[..., i, np.newaxis, :]
                    + step_vec[..., np.newaxis, :]
                    * np.arange(1, num_extra_points + 1)[:, np.newaxis]
                )

            new_pts.append(unwrapped_pts[..., i + 1 : i + 2, :])

        new_pts = np.concatenate(new_pts, axis=-2)
        if has_heading:
            new_pts[..., 3] = angle_wrap(new_pts[..., 3])

        return new_pts


def endpoints_intersect(left_edge, right_edge):
    """Check if endpoints of two edges intersect."""
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    A, B = left_edge[-1], right_edge[-1]
    C, D = right_edge[0], left_edge[0]
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def order_matches(pts, ref):
    """
    Evaluate whether pts is ordered the same as ref.
    
    Args:
        pts: The first array of points, shape (N, D)
        ref: The second array of points, shape (M, D)
        
    Returns:
        bool: True if pts's first point is closest to ref's first point
    """
    return np.linalg.norm(pts[0] - ref[0]) <= np.linalg.norm(pts[-1] - ref[0])

