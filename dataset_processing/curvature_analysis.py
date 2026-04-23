"""
Curvature analysis utilities for boundary asymmetry detection.
"""
import numpy as np
from typing import Tuple


def calculate_curvature(points):
    """
    Calculate the curvature of a polyline using the formula k = |x'y'' - y'x''|/(x'^2 + y'^2)^(3/2).
    
    Args:
        points: Nx2 array of (x, y) coordinates representing the polyline
        
    Returns:
        np.ndarray: Array of curvature values at each point
    """
    points = np.asarray(points)
    # Calculate first derivatives
    dx, dy = np.gradient(points[:, 0]), np.gradient(points[:, 1])
    # Calculate second derivatives
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    
    # Curvature formula
    numerator = np.abs(ddx * dy - ddy * dx)
    denominator = (dx**2 + dy**2)**1.5
    
    # Handle edge cases
    curvatures = np.zeros_like(numerator)
    valid_mask = denominator > 0
    curvatures[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
    
    return curvatures


def calculate_region_curvature(points, region_size=5):
    """
    Calculate the average curvature over sliding windows of points.
    
    Args:
        points: Nx2 array of (x, y) coordinates
        region_size: Size of sliding window
        
    Returns:
        np.ndarray: Array of regional curvature values
    """
    points = np.asarray(points)
    if len(points) < region_size:
        return np.zeros(len(points))
    
    num_regions = len(points) - region_size + 1
    regional_curvatures = np.zeros(num_regions)
    
    # Calculate curvature for each region
    for i in range(num_regions):
        region_points = points[i:i + region_size]
        regional_curvatures[i] = np.mean(calculate_curvature(region_points))
    
    return regional_curvatures


def curvature_difference(polyline1, polyline2):
    """
    Calculate the mean squared curvature difference between two polylines.
    
    Args:
        polyline1: Nx2 array of (x, y) coordinates for the first polyline
        polyline2: Nx2 array of (x, y) coordinates for the second polyline
    
    Returns:
        tuple: (max_curvature_diff, left_curvatures, right_curvatures)
    """
    min_len = min(len(polyline1), len(polyline2))
    
    # Calculate curvature for both boundaries
    left_curvatures = calculate_curvature(polyline1[:min_len])
    right_curvatures = calculate_curvature(polyline2[:min_len])
    
    # Return the maximum absolute curvature difference
    return np.max(np.abs(left_curvatures - right_curvatures)), left_curvatures, right_curvatures


def compute_heading(points):
    """
    Compute heading angles between consecutive points.
    
    Args:
        points: Nx2 array of (x, y) coordinates
        
    Returns:
        np.ndarray: Array of heading angles in radians [-π, π]
    """
    diffs = np.diff(points, axis=0)
    return np.arctan2(diffs[:, 1], diffs[:, 0])


def average_heading_of_chunk(points_chunk):
    """
    Compute average heading angle for a chunk of points using circular statistics.
    
    Args:
        points_chunk: Nx2 array of points
        
    Returns:
        float or None: Average heading angle in radians, or None if invalid
    """
    if len(points_chunk) < 2:
        return None
        
    headings = compute_heading(points_chunk)
    
    if len(headings) == 0:
        return None
        
    # Convert angles to unit vectors and average
    x_sum = np.sum(np.cos(headings))
    y_sum = np.sum(np.sin(headings))
    
    # Return the angle of the resultant vector
    if x_sum == 0 and y_sum == 0:
        return None  # No clear direction
        
    return np.arctan2(y_sum, x_sum)


def angle_diff(a, b):
    """
    Compute smallest absolute angular difference between two angles.
    
    Args:
        a, b: Angles in radians
        
    Returns:
        float: Absolute angular difference in [0, π]
    """
    diff = abs(a - b) % (2 * np.pi)
    return min(diff, 2 * np.pi - diff)


def check_any_chunk_symmetrical_by_heading(left_points, right_points, chunk_size=5, 
                                         angle_thresh_rad=np.radians(30), tail_length=3):
    """
    Check if any corresponding chunks of the two boundaries are symmetrical.
    
    Args:
        left_points, right_points: Nx2 arrays of boundary points
        chunk_size: Size of chunks to compare
        angle_thresh_rad: Angle threshold in radians
        tail_length: Length of tail to check
        
    Returns:
        bool: True if any chunk pair is symmetrical
    """
    # Check tail symmetry
    left_tail = left_points[-tail_length:]
    right_tail = right_points[-tail_length:]

    left_tail_heading = average_heading_of_chunk(left_tail)
    right_tail_heading = average_heading_of_chunk(right_tail)

    if left_tail_heading is not None and right_tail_heading is not None:
        if abs(angle_diff(left_tail_heading, right_tail_heading) - np.pi) < angle_thresh_rad:
            return True
            
    return False


def calculate_direction_change(points):
    """
    Calculate the overall direction change of a boundary.
    
    Args:
        points: Nx2 array of (x, y) coordinates
        
    Returns:
        float: Score representing the magnitude of direction change
    """
    if len(points) < 3:
        return 0.0
    
    # Calculate direction vectors between consecutive points
    vectors = points[1:] - points[:-1]
    
    # Normalize vectors safely
    normalized_vectors = np.zeros_like(vectors, dtype=float)
    for i in range(len(vectors)):
        norm = np.linalg.norm(vectors[i])
        if norm > 1e-6:
            normalized_vectors[i] = vectors[i] / norm
    
    # Calculate the angle between the first and last direction vectors
    start_dir = np.mean(normalized_vectors[:3], axis=0)
    end_dir = np.mean(normalized_vectors[-3:], axis=0)
    
    # Normalize again after averaging
    start_norm = np.linalg.norm(start_dir)
    end_norm = np.linalg.norm(end_dir)
    
    if start_norm > 1e-6:
        start_dir = start_dir / start_norm
    if end_norm > 1e-6:
        end_dir = end_dir / end_norm
    
    # Calculate angle
    cos_angle = np.clip(np.dot(start_dir, end_dir), -1.0, 1.0)
    angle = np.arccos(cos_angle)
    
    # Convert to degrees and normalize to a 0-1 score
    angle_deg = np.degrees(angle)
    score = angle_deg / 180.0
    
    return score


def calculate_continuous_turning(points, window_size=5):
    """
    Identify the largest continuous turning section in the boundary.
    
    Args:
        points: Nx2 array of (x, y) coordinates
        window_size: Size of sliding window for direction analysis
        
    Returns:
        float: Score representing the magnitude of continuous turning
    """
    if len(points) < window_size + 1:
        return 0.0
    
    # Calculate direction vectors between consecutive points
    vectors = points[1:] - points[:-1]
    
    # Use sliding windows to detect continuous turning
    turning_scores = []
    for i in range(len(vectors) - window_size + 1):
        window_vectors = vectors[i:i+window_size]
        # Calculate angle between first and last vector in window
        if np.linalg.norm(window_vectors[0]) > 1e-6 and np.linalg.norm(window_vectors[-1]) > 1e-6:
            v1_norm = window_vectors[0] / np.linalg.norm(window_vectors[0])
            v2_norm = window_vectors[-1] / np.linalg.norm(window_vectors[-1])
            cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
            angle = np.arccos(cos_angle)
            turning_scores.append(np.degrees(angle))
    
    # Return the maximum turning score, normalized to 0-1
    if turning_scores:
        max_turning = max(turning_scores)
        return min(max_turning / 90.0, 1.0)  # Cap at 1.0
    return 0.0


def identify_diverging_boundary_improved(left_boundary_pts, right_boundary_pts):
    """
    Identify which boundary is diverging using multiple geometric features.
    
    Args:
        left_boundary_pts: Nx2 array of left boundary points
        right_boundary_pts: Nx2 array of right boundary points
        
    Returns:
        tuple: (diverging_boundary, confidence, left_score, right_score)
    """
    # Convert to numpy if needed (for PyTorch compatibility)
    if hasattr(left_boundary_pts, 'detach'):
        left_boundary_pts = left_boundary_pts.detach().cpu().numpy()
    if hasattr(right_boundary_pts, 'detach'):
        right_boundary_pts = right_boundary_pts.detach().cpu().numpy()
    
    # Calculate direction change score
    left_dir_score = calculate_direction_change(left_boundary_pts)
    right_dir_score = calculate_direction_change(right_boundary_pts)
    
    # Calculate largest continuous turning section
    left_turning_score = calculate_continuous_turning(left_boundary_pts)
    right_turning_score = calculate_continuous_turning(right_boundary_pts)
    
    # Combine scores with appropriate weights
    w1, w3 = 0.5, 0.2
    left_total_score = w1 * left_dir_score + w3 * left_turning_score
    right_total_score = w1 * right_dir_score + w3 * right_turning_score
    
    # Calculate confidence as normalized difference
    confidence = abs(left_total_score - right_total_score) / max(left_total_score + right_total_score, 1e-6)
    
    return ('left' if left_total_score > right_total_score else 'right', 
            confidence, left_total_score, right_total_score)

