from PIL import Image
import torch
import numpy as np
import cv2

import math
import os
import copy
from itertools import product
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from shapely.geometry import LineString, Point
from mmdet3d.core import LiDARInstance3DBoxes
from torchvision.transforms.functional import perspective, InterpolationMode


''' common utils '''
def setup_dirs(dirs):
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)


''' loss related '''
def chamfer_distance(p1, p2, device=None):
    """Calculate chamfer distance between two point sets.
    
    Args:
        p1 (np.ndarray or torch.Tensor): First point set of shape (n1, d)
        p2 (np.ndarray or torch.Tensor): Second point set of shape (n2, d)
        device (torch.device, optional): Device to use if converting to torch.Tensor
    
    Returns:
        float: Chamfer distance between the two point sets
    """
    # Convert to torch tensors if they aren't already
    if not isinstance(p1, torch.Tensor):
        p1 = torch.from_numpy(p1).to(device)
    if not isinstance(p2, torch.Tensor):
        p2 = torch.from_numpy(p2).to(device)
    
    # Compute pairwise distances using broadcasting
    distances = torch.norm(p1.unsqueeze(1) - p2.unsqueeze(0), dim=2)
    
    # Compute minimum distances in both directions
    min_p1_to_p2 = distances.min(dim=1)[0].mean()  # Mean of min distances from p1 to p2
    min_p2_to_p1 = distances.min(dim=0)[0].mean()  # Mean of min distances from p2 to p1
    
    # Return the symmetric Chamfer distance
    return min_p1_to_p2 + min_p2_to_p1

def outward_inward_loss_interpolated(
    pred_boundary: torch.Tensor,   # shape (P, 2), e.g. (20, 2)
    gt_boundary: torch.Tensor,     # shape (B, 2), variable number of points
    centerline: torch.Tensor,      # shape (C, 2), variable number of points
    reference_boundary: torch.Tensor, # shape (R, 2), variable number of points
    alpha: float = 1.0,            # weight for outward objective
    beta: float = 1.0,             # weight for inward penalty
    visualize: bool = False
) -> torch.Tensor:
    """
    Computes a loss that:
      - Encourages the predicted boundary to move outward (off-road),
      - Penalizes inward deviations (invading the drivable area).
    
    This version uses Shapely to interpolate gt_boundary and centerline to match 
    the number of points in pred_boundary, avoiding the need for nearest-point matching.
    
    Args:
      pred_boundary: (P,2) predicted boundary points.
      gt_boundary:   (B,2) ground-truth boundary points.
      centerline:    (C,2) centerline points.
      alpha:         weight for outward-pushing term.
      beta:          weight for inward-penalty term.
      visualize:     whether to generate visualization for debugging.
      
    Returns:
      A scalar PyTorch tensor (loss).
    """
    
    # Get number of points in predicted boundary
    num_points = pred_boundary.shape[0]
    
    # ------------------------------
    # 1) Convert to numpy for Shapely
    # ------------------------------
    gt_np = gt_boundary.detach().cpu().numpy()
    centerline_np = centerline.detach().cpu().numpy()
    
    # ------------------------------
    # 2) Create Shapely LineStrings and interpolate
    # ------------------------------
    gt_line = LineString(gt_np)
    centerline_line = LineString(centerline_np)
    
    # Get total length
    gt_length = gt_line.length
    centerline_length = centerline_line.length
    
    # Create evenly spaced points
    gt_interp_np = np.array([
        gt_line.interpolate(i * gt_length / (num_points - 1)).coords[0]
        for i in range(num_points)
    ])
    
    centerline_interp_np = np.array([
        centerline_line.interpolate(i * centerline_length / (num_points - 1)).coords[0]
        for i in range(num_points)
    ])
    
    # Convert back to torch tensors
    device = pred_boundary.device
    gt_boundary_interp = torch.tensor(gt_interp_np, dtype=torch.float32, device=device)
    centerline_interp = torch.tensor(centerline_interp_np, dtype=torch.float32, device=device)
    
    # ------------------------------
    # 3) Define the outward direction d_i for each point
    #    d_i = (B_i - centerline_i) / ||...||
    # ------------------------------
    v = gt_boundary_interp - centerline_interp  # (P, 2)
    v_norm = v.norm(dim=-1, keepdim=True) + 1e-6
    d = v / v_norm  # (P, 2), the outward (normal) direction
    
    # ------------------------------
    # 4) Compute offset_i = (P_i - B_i) dot d_i
    #    If offset_i > 0 => predicted boundary is outward.
    #    If offset_i < 0 => predicted boundary is inward.
    # ------------------------------
    offset = ((pred_boundary - gt_boundary_interp) * d).sum(dim=-1)  # (P,)
    
    # ------------------------------
    # 5) Outward & Inward Terms
    # ------------------------------
    # outward_loss = - ReLU(offset)
    #   (We "minimize" negative offset so that we push offset to be larger positive.)
    outward_loss = -torch.nn.functional.relu(offset).mean()
    
    # inward_loss = ReLU(-offset)
    #   (Penalize negative offset, i.e., going inward.)
    inward_loss = torch.nn.functional.relu(-offset).mean()
    
    # Combine
    total_loss = alpha * outward_loss + beta * inward_loss
    
    return total_loss


''' map related '''
def find_best_matching_boundary(multiple_polylines, target_polyline, device='cuda'):
    """Find the polyline that best matches the target polyline using Chamfer distance.
    
    Args:
        multiple_polylines (torch.Tensor): Array of polylines with shape (n, m, 2) where
            n is number of lines, m is number of points per line (padded with -10000)
        target_polyline (torch.Tensor): Target polyline with shape (m, 2)
        device (str): Device to compute distances on. Defaults to 'cuda'.
    
    Returns:
        torch.Tensor or None: Best matching polyline with shape (m, 2), or None if no valid match
    """
    if len(multiple_polylines) == 0:
        return None
        
    min_dist = float('inf')
    best_matching_polyline = None
    
    for polyline in multiple_polylines:
        # Filter out padding values (-10000)
        valid_points = polyline[polyline[:, 0] != -10000]
        
        # Skip if no valid points
        if len(valid_points) == 0:
            continue
            
        # Calculate Chamfer distance between valid points and target
        dist = chamfer_distance(valid_points, target_polyline, device=device)
        
        # Update if this is the best match so far
        if dist < min_dist:
            min_dist = dist
            best_matching_polyline = valid_points
    
    return best_matching_polyline


''' attack location related '''
def sample_boundary_at_interval(boundary_pts, interval=0.5):
    """
    Sample points along a boundary at a specified interval.
    
    Args:
        boundary_pts (numpy.ndarray or torch.Tensor): Boundary points with shape (N, 2).
        interval (float): Sampling interval in meters.
        
    Returns:
        numpy.ndarray: Sampled points along the boundary with shape (M, 2).
    """
    
    # Convert to numpy if it's a torch tensor
    if isinstance(boundary_pts, torch.Tensor):
        boundary_pts = boundary_pts.detach().cpu().numpy()
    
    # Create a LineString from the boundary points
    line = LineString(boundary_pts)
    
    # Calculate the total length of the boundary
    total_length = line.length
    
    # Calculate the number of points to sample
    num_points = max(2, int(total_length / interval) + 1)
    
    # Sample points at equal distances along the line
    sampled_points = []
    for i in range(num_points):
        # Calculate the distance along the line
        distance = i * interval
        if distance > total_length:
            break
        
        # Get the point at this distance
        point = line.interpolate(distance)
        sampled_points.append((point.x, point.y))
    
    return np.array(sampled_points)

def generate_sampled_points(center, grid_size=1.0, num_points=4, mode='random'):
    """Generate random sampled points around a center point.
    
    Args:
        center (numpy.ndarray): Center point (x, y)
        grid_size (float): Size of search area in meters
        num_points (int): Number of points to sample
        mode (str): Sampling mode - 'left', 'right' or 'random'
    
    Returns:
        numpy.ndarray: Sampled points of shape (num_points, 2)
    """
    if mode == 'left':
        # Sample on the left side of the diverge point
        x = np.random.rand(num_points) * grid_size + (center[0] - grid_size)
        y = np.random.rand(num_points) * grid_size - grid_size/2 + center[1]
    elif mode == 'right':
        # Sample on the right side of the diverge point
        x = np.random.rand(num_points) * grid_size + center[0]
        y = np.random.rand(num_points) * grid_size - grid_size/2 + center[1]
    elif mode == 'random':
        # Sample randomly around the center
        x = np.random.rand(num_points) * grid_size - grid_size/2 + center[0]
        y = np.random.rand(num_points) * grid_size - grid_size/2 + center[1]
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    points = np.stack([x, y], axis=1)
    return points

def extend_boundary(
    reference_boundary,     # Nx2 array of (x, y) points defining the complete/reliable boundary
    incomplete_boundary,    # Mx2 array of (x, y) points defining the partial/incomplete boundary
    step=5
):
    """
    1) Compute average road width up to y_cutoff by comparing complete/incomplete boundaries
    2) Mirror the complete boundary with the computed width to create symmetric boundary
    3) Preserve the original ordering of reference_boundary points
    """

    #--------------------------------------------------------------------
    # Step 1: Build interpolation functions for both boundaries
    #         to measure road_width(y) = x_incomplete(y) - x_reference(y)
    #--------------------------------------------------------------------
    f_reference = interp1d(reference_boundary[:,1], reference_boundary[:,0],
                          kind='linear', fill_value="extrapolate")
    f_incomplete = interp1d(incomplete_boundary[:,1], incomplete_boundary[:,0],
                           kind='linear', fill_value="extrapolate")

    #--------------------------------------------------------------------
    # Step 2: Figure out the overlap in y
    #         We'll measure road width from y_overlap_min up to y_cutoff.
    #--------------------------------------------------------------------
    measure_points = min(step, len(incomplete_boundary))
    y_samples = incomplete_boundary[:measure_points, 1]

    #--------------------------------------------------------------------
    # Step 3: Sample road_width(y) = x_incomplete(y) - x_reference(y) 
    #         and compute average
    #--------------------------------------------------------------------
    road_width_samples = f_incomplete(y_samples) - f_reference(y_samples)
    avg_road_width = np.mean(road_width_samples)

    #--------------------------------------------------------------------
    # Step 4: Create mirrored boundary by shifting reference boundary 
    #         points by avg_road_width in x direction
    #--------------------------------------------------------------------
    mirrored_boundary = reference_boundary.copy()
    mirrored_boundary[:,0] += avg_road_width  # shift the x coordinate

    return mirrored_boundary, avg_road_width

def stitch_boundaries(
    incomplete_boundary,
    mirrored_boundary,
    step=5
):
    """
    Combine the incomplete boundary up to cutoff_y with the mirrored boundary above cutoff_y.
    Assumes each array is sorted by Y coordinate or follows a compatible ordering.
    
    Args:
        incomplete_boundary: Original partial boundary points
        mirrored_boundary: Boundary points created by mirroring reference boundary
        step: Number of points to keep from incomplete boundary
    
    Returns:
        np.ndarray: Combined boundary points maintaining continuous ordering
    """
    
    reliable_part = incomplete_boundary[:step].tolist()
    
    # Get the mirrored points after the y-coordinate of the step-th point
    y_cutoff = incomplete_boundary[step-1][1] if step > 0 else incomplete_boundary[0][1]
    mirrored_part = []
    for point in mirrored_boundary:
        if point[1] >= y_cutoff:
            mirrored_part.append(point)
    
    # Combine parts to create complete boundary
    complete_boundary = np.array(reliable_part + mirrored_part)
    return complete_boundary

def calculate_curvature(points):
    """
    Calculate the curvature of a polyline represented by points.
    
    Parameters:
        points (ndarray): Nx2 array of (x, y) coordinates representing the polyline.

    Returns:
        curvatures (ndarray): Array of curvature values at each point (excluding endpoints).
    """
    points = np.array(points)
    dx = np.gradient(points[:, 0])  # First derivative of x
    dy = np.gradient(points[:, 1])  # First derivative of y
    ddx = np.gradient(dx)  # Second derivative of x
    ddy = np.gradient(dy)  # Second derivative of y
    
    # Curvature formula
    curvatures = np.abs(ddx * dy - ddy * dx) / (dx**2 + dy**2)**1.5
    curvatures[np.isinf(curvatures)] = 0  # Handle division by zero if needed
    curvatures = np.nan_to_num(curvatures)  # Replace NaNs with 0
    
    return curvatures

def calculate_region_curvature(points, region_size=5):
    """
    Calculate the average curvature over sliding windows of points.
    
    Args:
        points (ndarray): Nx2 array of (x, y) coordinates.
        region_size (int): Size of sliding window. Default is 5.

    Returns:
        ndarray: Array of regional curvature values.
    """
    points = np.asarray(points)
    if len(points) < region_size:
        return np.zeros(len(points))
    
    num_regions = len(points) - region_size + 1
    regional_curvatures = np.zeros(num_regions)
    
    # Calculate curvature for each region using vectorized operations
    for i in range(num_regions):
        region_points = points[i:i + region_size]
        regional_curvatures[i] = np.mean(calculate_curvature(region_points))
    
    return regional_curvatures

def get_asymmetry_anchors(diverge_boundary_pts, reference_boundary_pts, CURVATURE_DIFF_THRESHOLD=0.1, top_k=5):
    """
    Find points with peak curvature differences that are well separated.
    
    Args:
        diverge_boundary_pts: Nx2 array of boundary points
        reference_boundary_pts: Nx2 array of boundary points
        CURVATURE_DIFF_THRESHOLD: Minimum curvature difference threshold
        top_k: Number of peak points to return
    
    Returns:
        np.ndarray: Selected boundary points with peak curvature differences
    """
    
    if isinstance(diverge_boundary_pts, torch.Tensor):
        diverge_boundary_pts = diverge_boundary_pts.detach().cpu().numpy()
    if isinstance(reference_boundary_pts, torch.Tensor):
        reference_boundary_pts = reference_boundary_pts.detach().cpu().numpy()
    
    # calculate curvature
    min_length = min(diverge_boundary_pts.shape[0], reference_boundary_pts.shape[0])
    diverge_curvatures = calculate_curvature(diverge_boundary_pts[:min_length])
    reference_curvatures = calculate_curvature(reference_boundary_pts[:min_length])
    curvature_diff = np.abs(diverge_curvatures - reference_curvatures)
    
    large_diff_indices = np.where(curvature_diff > CURVATURE_DIFF_THRESHOLD)[0]
    
    # If no points exceed threshold, return top k points with largest curvature difference
    if len(large_diff_indices) == 0:
        # Sort indices by curvature difference (descending)
        sorted_indices = np.argsort(curvature_diff)[::-1]
        # Take top k indices
        top_indices = sorted_indices[:top_k]
        return diverge_boundary_pts[top_indices]
    else:
        # Return points that exceed threshold
        return diverge_boundary_pts[large_diff_indices]

def find_diverge_point(boundary_pts, reference_pts=None, method='curvature'):
    """
    Find the point where the boundary starts to diverge/turn.
    
    Args:
        boundary_pts: Points of the boundary to analyze
        reference_pts: Optional reference boundary for comparison
        method: Method to detect divergence - 'curvature' or 'difference'
    
    Returns:
        int: Index of the diverge point
    """
    if method == 'curvature':
        # Use curvature to find where the road starts to turn
        curvatures = calculate_curvature(boundary_pts)
        
        # Smooth the curvature values
        window_size = 5
        smoothed_curvatures = np.convolve(curvatures, np.ones(window_size)/window_size, mode='valid')
        
        # Find the first significant increase in curvature
        # Skip the first few points as they might have artifacts
        start_idx = 5
        CURVATURE_THRESHOLD = 0.05  # Adjust based on your data
        
        for i in range(start_idx, len(smoothed_curvatures)):
            if smoothed_curvatures[i] > CURVATURE_THRESHOLD:
                # We found a point where curvature starts increasing
                return i
        
        # If no clear diverge point, use a default value
        return min(10, len(boundary_pts) // 3)
    
    elif method == 'difference' and reference_pts is not None:
        # Use difference from reference boundary
        min_len = min(len(boundary_pts), len(reference_pts))
        diffs = np.linalg.norm(boundary_pts[:min_len] - reference_pts[:min_len], axis=1)
        
        # Smooth the differences
        window_size = 5
        smoothed_diffs = np.convolve(diffs, np.ones(window_size)/window_size, mode='valid')
        
        # Find where the difference starts increasing significantly
        start_idx = 5
        DIFF_THRESHOLD = 0.2  # Adjust based on your data
        
        for i in range(start_idx, len(smoothed_diffs) - 1):
            if (smoothed_diffs[i+1] - smoothed_diffs[i]) > DIFF_THRESHOLD:
                return i
        
        # If no clear diverge point, use a default value
        return min(10, len(boundary_pts) // 3)
    
    else:
        # Default method
        return min(10, len(boundary_pts) // 3)


def create_natural_early_turn(diverge_boundary_pts, reference_boundary_pts, diverge_boundary_tag, step=5, backtrack_distance=5):
    """
    Create a realistic early turn by identifying where the original turn begins
    and shifting that same turn pattern to start earlier.
    
    Args:
        diverge_boundary_pts: Points of the diverging boundary
        reference_boundary_pts: Points of the reference boundary
        diverge_boundary_tag: 'left' or 'right' indicating which boundary is diverging
        step: Number of reliable initial points to keep
        backtrack_distance: How many points to backtrack from diverge point
    
    Returns:
        np.ndarray: Modified boundary points with an earlier natural turn
    """
    # 1. Find the diverge point - where the boundary starts to turn
    diverge_idx = find_diverge_point(diverge_boundary_pts, reference_boundary_pts, method='curvature')
    
    # 2. Determine where to start the early turn
    # We'll backtrack from the diverge point by backtrack_distance
    early_turn_start_idx = max(diverge_idx - backtrack_distance, 1)
    
    # 3. Keep the points up to early_turn_start_idx
    early_part = diverge_boundary_pts[:early_turn_start_idx].copy()
    
    # 4. Extract the turn pattern from the original boundary
    turn_pattern = diverge_boundary_pts[diverge_idx:].copy()
    
    # 5. Calculate the y-offset to shift the pattern earlier
    y_offset = diverge_boundary_pts[diverge_idx, 1] - diverge_boundary_pts[early_turn_start_idx, 1]
    
    # 6. Create the shifted turn pattern
    shifted_turn = turn_pattern.copy()
    shifted_turn[:, 1] -= y_offset  # Shift y coordinates earlier
    
    # 7. Connect the early part with the shifted turn
    # Ensure the connection is smooth by adjusting the x-coordinate of the first shifted point
    x_adjust = early_part[-1, 0] - shifted_turn[0, 0]
    shifted_turn[:, 0] += x_adjust
    
    # 8. Combine to create the final boundary
    target_boundary = np.vstack([early_part, shifted_turn])
    
    return target_boundary

def get_target_boundary_pts(diverge_boundary_pts, reference_boundary_pts, diverge_boundary_tag, dataset, step=5):
    
    # check inputs
    if isinstance(diverge_boundary_pts, torch.Tensor):
        diverge_boundary_pts = diverge_boundary_pts.detach().cpu().numpy()
    if isinstance(reference_boundary_pts, torch.Tensor):
        reference_boundary_pts = reference_boundary_pts.detach().cpu().numpy()
    
    if dataset == 'symmetric':
        # Handle symmetric case - create gradual curve
        target_boundary = diverge_boundary_pts.copy()
        
        # Starting index where deviation begins
        start_idx = 8
        if start_idx >= len(target_boundary):
            return target_boundary
        
        # Get reference y coordinate at start point
        y_start = target_boundary[start_idx, 1]
        
        # Determine curve direction based on boundary tag
        # Left boundary curves right, right boundary curves left
        curve_direction = -1 if diverge_boundary_tag == 'left' else 1
        
        # Apply gradual lateral shift to subsequent points
        for i in range(start_idx, len(target_boundary)):
            # Calculate distance from starting point in y-axis
            dy = target_boundary[i, 1] - y_start
            # Quadratic scaling factor for natural curve
            delta_x = curve_direction * 0.05 * (dy ** 2)
            target_boundary[i, 0] += delta_x
        
        return target_boundary
    
    elif dataset == 'asymmetric':
        
        # extend the diverge boundary
        extended_diverge_boundary, avg_road_width = extend_boundary(
            reference_boundary_pts, diverge_boundary_pts, step=step)
        # stitch the boundaries
        target_boundary = stitch_boundaries(
            diverge_boundary_pts, extended_diverge_boundary, step=step)
        
        return target_boundary
    
    else:
        raise ValueError(f'Invalid dataset: {dataset}')


''' visualization related '''
def normalize_img(img: torch.Tensor, img_norm_cfg):
    
    img_mean = torch.tensor(img_norm_cfg['mean'], dtype=img.dtype, device=img.device).reshape(3, 1, 1)
    img_std = torch.tensor(img_norm_cfg['std'], dtype=img.dtype, device=img.device).reshape(3, 1, 1)

    return (img - img_mean) / img_std


def denormalize_img(img: torch.Tensor, img_norm_cfg):
    # img shape (3, 480, 800)
    # mean, std represent the mean and std of the 3 channels
    img_mean = torch.tensor(img_norm_cfg['mean']).reshape(3, 1, 1).to(img.device)
    img_std = torch.tensor(img_norm_cfg['std']).reshape(3, 1, 1).to(img.device)

    return img * img_std + img_mean


def plot_bboxes(ax, pts_3d, labels_3d, pc_range, car_img, colors_plt):
    """ Plot bounding boxes on the given axes. """
    ax.set_xlim(pc_range[0], pc_range[3])
    ax.set_ylim(pc_range[1], pc_range[4])
    ax.axis('off')
    
    for label_3d, pts in zip(labels_3d, pts_3d):
        x, y = pts[:, 0], pts[:, 1]
        ax.plot(x, y, color=colors_plt[label_3d], linewidth=1, alpha=0.8, zorder=-1)
        ax.scatter(x, y, color=colors_plt[label_3d], s=2, alpha=0.8, zorder=-1)
    
    ax.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])


def plot_attack_locs_bev(ax, attack_locs_bev, zorder=12):
    """Overlay dual blind attack 3D points on BEV (uses x,y in lidar).

    attack_locs_bev: optional length-2 sequence; each entry is None or array-like (x,y,z).
    P1 = first flare (RSA stage-1), P2 = second flare (curve search).
    """
    if attack_locs_bev is None:
        return
    styles = [
        {'c': '#e6007a', 'marker': '*', 's': 140, 'label': 'P1'},
        {'c': '#00c8ff', 'marker': 'X', 's': 130, 'label': 'P2'},
    ]
    for i, pt in enumerate(attack_locs_bev[:2]):
        if pt is None:
            continue
        arr = np.asarray(pt, dtype=np.float64).reshape(-1)
        if arr.size < 2:
            continue
        x, y = float(arr[0]), float(arr[1])
        kw = {k: v for k, v in styles[i].items() if k != 'label'}
        ax.scatter(
            [x], [y],
            edgecolors='white',
            linewidths=1.0,
            zorder=zorder,
            **kw,
        )
        ax.text(
            x, y,
            styles[i]['label'],
            fontsize=7,
            color='white',
            ha='center',
            va='center',
            zorder=zorder + 1,
            fontweight='bold',
        )


def visualize_results(result, vis_seg_dir, sample_idx, pc_range, car_img, colors_plt, show_score_thr=0.3):
    
    plt.figure(figsize=(2, 4))
    plt.xlim(pc_range[0], pc_range[3])
    plt.ylim(pc_range[1], pc_range[4])
    plt.axis('off')

    # visualize pred
    # import pdb;pdb.set_trace()
    result_dic = result[0]['pts_bbox']
    boxes_3d = result_dic['boxes_3d'] # bbox: xmin, ymin, xmax, ymax
    scores_3d = result_dic['scores_3d']
    labels_3d = result_dic['labels_3d']
    pts_3d = result_dic['pts_3d']
    keep = scores_3d > show_score_thr

    for pred_score_3d, pred_bbox_3d, pred_label_3d, pred_pts_3d in zip(scores_3d[keep], boxes_3d[keep],labels_3d[keep], pts_3d[keep]):

        pred_pts_3d = pred_pts_3d.numpy()
        pts_x = pred_pts_3d[:,0]
        pts_y = pred_pts_3d[:,1]
        plt.plot(pts_x, pts_y, color=colors_plt[pred_label_3d],linewidth=1,alpha=0.8,zorder=-1)
        plt.scatter(pts_x, pts_y, color=colors_plt[pred_label_3d],s=1,alpha=0.8,zorder=-1)


        pred_bbox_3d = pred_bbox_3d.numpy()
        xy = (pred_bbox_3d[0],pred_bbox_3d[1])
        width = pred_bbox_3d[2] - pred_bbox_3d[0]
        height = pred_bbox_3d[3] - pred_bbox_3d[1]
        pred_score_3d = float(pred_score_3d)
        pred_score_3d = round(pred_score_3d, 2)
        s = str(pred_score_3d)

    plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])

    # map_path = osp.join(sample_dir, 'PRED_MAP_plot.png')
    map_path = os.path.join(f'{sample_idx}.png')
    plt.savefig(map_path, bbox_inches='tight', format='png',dpi=1200)
    print(f'save map to {map_path}')
    plt.close()
    
    
def visualize_gt(gt_res, vis_gt_dir, sample_idx, pc_range, car_img, colors_plt):
    
    plt.figure(figsize=(2, 4))
    plt.xlim(pc_range[0], pc_range[3])
    plt.ylim(pc_range[1], pc_range[4])
    plt.axis('off')
    
    # Plot ground truth
    gt_bboxes_3d, gt_labels_3d = gt_res
    gt_bboxes = np.array([bbox.numpy() for bbox in gt_bboxes_3d.fixed_num_sampled_points])
    for label_3d, pts in zip(gt_labels_3d, gt_bboxes):
        x, y = pts[:, 0], pts[:, 1]
        plt.plot(x, y, color=colors_plt[label_3d], linewidth=1, alpha=0.8, zorder=-1)
        plt.scatter(x, y, color=colors_plt[label_3d], s=2, alpha=0.8, zorder=-1)
    
    plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])
    
    # Save the figure
    vis_seg_fn = os.path.join(vis_gt_dir, f'{sample_idx}.png')
    plt.savefig(vis_seg_fn, bbox_inches='tight', format='png', dpi=1200)
    print(f'GT results saved to {vis_seg_fn}')
    plt.close()
    
def visualize_attack_results(
    gt_res,
    orig_res,
    attack_res,
    vis_seg_dir,
    sample_idx,
    pc_range,
    car_img,
    colors_plt,
    segment=None,
    show_score_thr=0.4,
    attack_locs_bev=None,
):
    
    _, axs = plt.subplots(1, 4, figsize=(8, 4))  # Adjusted to 4 subplots and increased width
    plt.tight_layout()
    
    # Plot ground truth
    gt_bboxes_3d, gt_labels_3d = gt_res
    gt_bboxes = np.array([bbox.numpy() for bbox in gt_bboxes_3d.fixed_num_sampled_points])
    plot_bboxes(axs[1], gt_bboxes, gt_labels_3d, pc_range, car_img, colors_plt)
    axs[1].set_title('Ground Truth', fontsize=10)
    
    # Plot prediction without attack
    result_dic = orig_res['pts_bbox']
    keep = result_dic['scores_3d'] > show_score_thr
    plot_bboxes(axs[2], np.array(result_dic['pts_3d'])[keep], np.array(result_dic['labels_3d'])[keep], pc_range, car_img, colors_plt)
    axs[2].set_title('Prediction (wo attack)', fontsize=10)
    plot_attack_locs_bev(axs[2], attack_locs_bev)
    
    # Plot prediction with attack
    result_dic = attack_res['pts_bbox']
    keep = result_dic['scores_3d'] > show_score_thr
    plot_bboxes(axs[3], np.array(result_dic['pts_3d'])[keep], np.array(result_dic['labels_3d'])[keep], pc_range, car_img, colors_plt)
    axs[3].set_title('Prediction (w attack)', fontsize=10)
    plot_attack_locs_bev(axs[3], attack_locs_bev)
    
    if segment is not None:
        # Plot segment (num_points, 2)
        plot_bboxes(axs[0], segment[np.newaxis], np.array([2]), pc_range, car_img, colors_plt)
        axs[0].set_title('Asymmetric Segment', fontsize=10)
    
    # Save the figure
    vis_seg_fn = os.path.join(vis_seg_dir, f'{sample_idx}.png')
    print(f'Save prediction results to: {vis_seg_fn}')
    plt.savefig(vis_seg_fn, bbox_inches='tight', format='png', dpi=1200)
    plt.close()

