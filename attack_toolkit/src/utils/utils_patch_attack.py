import torch
import math
import copy
import numpy as np
from mmdet3d.core import LiDARInstance3DBoxes
from torchvision.transforms.functional import perspective, InterpolationMode

from attack_toolkit.src.utils.utils_attack import normalize_img, denormalize_img

def get_proj_scale(lat_dist, long_dist, ori_img_width, camera_height=1.51):
    # nuScenes camera intrinsic parameters
    # 5mm 1/1.8" 1/1.8'' CMOS sensor of 1600x1200 resolution.
    # 
    focal_length_mm = 5.5  # mm
    sensor_width_mm = 7.2
    resolution_width_px = ori_img_width  # e.g., 800, 1600 (width usually not padded)

    # Compute distance d
    d = math.sqrt(camera_height**2 + long_dist**2 + lat_dist**2)

    # Convert focal length from mm to pixels
    # Using an approximate conversion based on the sensor width and image resolution width
    pixel_size_width = sensor_width_mm / resolution_width_px
    focal_length_px = focal_length_mm / pixel_size_width

    # Calculate scale factor s
    scale_factor = focal_length_px / d

    return scale_factor

def create_pseudo_area(patch_cfg, ori_img_shape, proj_scale=50):
    
    H_phy, W_phy = patch_cfg['height'], patch_cfg['width']  # e.g, 2, 2
    _, ori_H_img, ori_W_img = ori_img_shape  # 450, 800
    
    # Check if patch dimensions are valid for the given projection scale
    if H_phy > ori_H_img // proj_scale or W_phy > ori_W_img // proj_scale:
        return None
    
    pseudo_area = (ori_H_img - H_phy * proj_scale - 1, (ori_W_img - W_phy * proj_scale) // 2, H_phy * proj_scale, W_phy * proj_scale)
    pseudo_area = tuple([int(v) for v in pseudo_area])
    
    return pseudo_area

def get_patch_heading(diverge_point, diverge_boundary_pts):
    '''
    get patch heading based on local direction around diverge point
    
    Args:
        diverge_point: numpy array (2,)
        diverge_boundary_pts: numpy array (N, 2)
    
    Returns:
        heading: float, heading angle in radians
    '''
    # Calculate heading based on local direction around diverge point
    # Find closest points to diverge_point
    distances = np.linalg.norm(diverge_boundary_pts - diverge_point, axis=1)
    center_idx = np.argmin(distances)
    
    # Take n points before and after the center point
    n_points = 1  # number of points to use on each side
    start_idx = max(0, center_idx - n_points)
    end_idx = min(len(diverge_boundary_pts), center_idx + n_points + 1)
    
    # Get local segment points
    local_segment = diverge_boundary_pts[start_idx:end_idx]
    
    # Calculate direction vector using first and last point of local segment
    direction_vector = local_segment[-1] - local_segment[0]
    
    # Calculate heading angle in radians (arctangent of y/x)
    bev_heading = np.arctan2(direction_vector[1], direction_vector[0])
    
    # Convert BEV heading to LiDAR heading
    # BEV coordinate: x to the right, y to the front, heading 0 to the right, counterclockwise
    # LiDAR coordinate: x to the front, y to the left, heading 0 to the front, clockwise
    lidar_heading = -bev_heading
    
    return lidar_heading

def get_patch_heading_facing_ego(diverge_point):
    '''
    Calculate patch heading to make the patch perpendicular to the direction from ego to diverge point
    
    Args:
        diverge_point: numpy array (2,) or (3,) containing x,y coordinates
        
    Returns:
        heading: float, heading angle in radians that makes patch perpendicular to ego direction
    '''
    # Extract x,y coordinates of the diverge point
    x, y = diverge_point[0], diverge_point[1]
    
    # Calculate the direction from ego vehicle (0,0) to diverge point
    direction_from_ego = np.array([x, y])
    
    # Calculate heading angle in radians (arctangent of y/x)
    # This gives the direction pointing from ego toward the diverge point
    bev_heading = np.arctan2(direction_from_ego[1], direction_from_ego[0])
    
    # Add 90 degrees (π/2 radians) to get perpendicular direction
    bev_heading_perpendicular = bev_heading - np.pi/2
    
    # Convert BEV heading to LiDAR heading
    # bev coordinate: x to the right, y to the front, heading 0 to the right, counterclockwise
    # lidar coordinate: x to the front, y to the left, heading 0 to the right, clockwise
    lidar_heading = -bev_heading_perpendicular
    
    # Normalize to [-π, π]
    lidar_heading = (lidar_heading + np.pi) % (2 * np.pi) - np.pi
    
    return lidar_heading

def init_patch_mask(ori_img_shape, img_norm_cfg, device, pseudo_area, mode='random'):
    
    # init patch
    if mode == 'random':
        patch = torch.randn(ori_img_shape).to(device) * 255
    elif mode == 'zero':
        patch = torch.zeros(ori_img_shape).to(device)
    elif mode == 'mean':
        patch = torch.randn(ori_img_shape).to(device)
        patch = denormalize_img(patch, img_norm_cfg)
    else:
        raise ValueError('mode not supported')
    patch.requires_grad_(True)
    
    # init mask
    _, ori_h, ori_w = ori_img_shape
    mask = torch.zeros((1, ori_h, ori_w)).to(device)
    t, l, h, w = pseudo_area
    mask[:, t:t+h, l:l+w] = 1
    
    return patch, mask

def get_phy_patch_mask(patch, mask, patch_cfg, pseudo_area, lidar2global, global2img, img_shape, ground=True):
    
    # calc patch corners start
    t, l, h, w = pseudo_area
    patch_corners_img_startpoints = [
        [l, t], [l+w, t], [l+w, t+h] , [l, t+h]
    ]  # top-left, top-right, bottom-right, bottom-left

    # calc patch corners end
    # LiDARInstance3DBoxes: 
    # - x (+right), y(+front), z(ground=-1.84/2)
    # - dx (left-right), dy (front-back), dz (up-down)
    # - yaw (0: right, -pi/2: front)
    if patch_cfg['type'] == 'vertical':
        # patch_bbox3d_lidar = LiDARInstance3DBoxes(tensor=[[patch_cfg['lat'], patch_cfg['long'], -patch_cfg['lidar2ground']/2+patch_cfg['height']/2, 
        #                                                     patch_cfg['width'], 0, patch_cfg['height'], patch_cfg['heading']]],
        #                                                     origin=(0.5, 0.5, 0.5))
        patch_bbox3d_lidar = LiDARInstance3DBoxes(tensor=[[patch_cfg['lat'], patch_cfg['long'], -patch_cfg['lidar2ground'], 
                                                    patch_cfg['width'], 0, patch_cfg['height'], patch_cfg['heading']]],
                                                    origin=(0.5, 0.5, 0.5))
    elif patch_cfg['type'] == 'ground':
        patch_bbox3d_lidar = LiDARInstance3DBoxes(tensor=[[patch_cfg['lat'], patch_cfg['long'], -patch_cfg['lidar2ground'], 
                                                            patch_cfg['width'], patch_cfg['height'], 0, patch_cfg['heading']]],
                                                            origin=(0.5, 0.5, 0.5))
    else:
        raise ValueError('Invalid patch type')
    
    visible_projections = get_patch_corners_on_img(patch_bbox3d_lidar, lidar2global, global2img, img_shape, ground=True)
    if visible_projections is None:
        return None, None, None, None
    
    # Initialize lists for transformed patches and masks
    patch_trans_list = []
    mask_trans_list = []
    visible_cam_indices = []
    
    for cam_idx, patch_corners_img_endpoints in visible_projections:

        # transform patch and mask
        patch_trans = perspective(patch, patch_corners_img_startpoints, 
                            patch_corners_img_endpoints, 
                            interpolation=InterpolationMode.BILINEAR, fill=0)
        mask_trans = perspective(mask, patch_corners_img_startpoints, 
                                    patch_corners_img_endpoints, 
                                    interpolation=InterpolationMode.BILINEAR, fill=0)
        
        patch_trans = patch_trans.unsqueeze(0).unsqueeze(0)
        mask_trans = mask_trans.unsqueeze(0).unsqueeze(0)  # to (1, 1, 1, img_h, img_w)
        
        patch_trans_list.append(patch_trans)
        mask_trans_list.append(mask_trans)
        visible_cam_indices.append(cam_idx)
    
    return patch_trans_list, mask_trans_list, visible_cam_indices

def get_patch_corners_on_img(bboxes3d, lidar2global, global2img, img_shape, ground=False):
    """Project 3D box corners to 2D image planes for multiple cameras.
    
    Args:
        bboxes3d: LiDAR 3D boxes
        lidar2global (torch.Tensor): Transform from LiDAR to global frame (4, 4)
        global2img (torch.Tensor): Transform from global to image frames (6, 4, 4)
        img_shape (tuple): Image shape in (H, W) format
        ground (bool): If True, set z coordinate to 0 in global frame
    
    Returns:
        tuple: (best_coords, visible_cam_idx) where best_coords are the 2D coordinates 
        in the best camera view and visible_cam_idx is that camera's index
    """
    def translate(points, x):
        """Apply translation to points"""
        return points + x

    def rotate(points, rot_matrix):
        """Apply rotation to points"""
        return points @ rot_matrix.T

    img_h, img_w = img_shape
    corners_3d = bboxes3d.corners  # (num_boxes, 8, 3)
    num_bbox = corners_3d.shape[0]
    
    # First transform to global frame
    corners_3d_reshaped = corners_3d.reshape(-1, 3)  # (num_boxes*8, 3)
    
    # Convert inputs to numpy if they're torch tensors
    if isinstance(lidar2global, torch.Tensor):
        lidar2global = lidar2global.detach().cpu().numpy()
    
    # Transform to global frame using rotate and translate
    corners_global = rotate(corners_3d_reshaped, lidar2global[:3, :3])
    corners_global = translate(corners_global, lidar2global[:3, 3])
    
    # Set z to 0 if ground is True
    if ground:
        corners_global[[0, 3, 4, 7], 2] = 0
        corners_global[[1, 2, 5, 6], 2] = bboxes3d.height.item()
        
    # Convert to homogeneous coordinates
    pts_4d = np.concatenate([corners_global, np.ones((num_bbox * 8, 1))], axis=-1)  # (num_boxes*8, 4)

    # Handle multiple cameras
    num_cams = global2img.shape[0]
    visible_projections = []

    for cam_idx in range(num_cams):
        # Get transformation matrix for this camera
        cam_matrix = copy.deepcopy(global2img[cam_idx])
        if isinstance(cam_matrix, torch.Tensor):
            cam_matrix = cam_matrix.cpu().numpy()
            
        # Project points from global to image
        pts_2d = rotate(pts_4d, cam_matrix)  # Using rotate for matrix multiplication
        
        # Check points in front of the camera
        depth = pts_2d[:, 2]
        valid_depth_mask = depth > 1e-5
        
        if valid_depth_mask.any():
            pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=1e5)
            pts_2d[:, 0] /= pts_2d[:, 2]
            pts_2d[:, 1] /= pts_2d[:, 2]
            
            # Check points within image bounds
            valid_img_mask = (
                (pts_2d[:, 0] >= 0) & 
                (pts_2d[:, 0] < img_w) &
                (pts_2d[:, 1] >= 0) & 
                (pts_2d[:, 1] < img_h)
            )
            
            # Combine masks for fully visible points
            valid_mask = valid_depth_mask & valid_img_mask
            visible_pts = valid_mask.sum()
            
            if visible_pts > 0:
            # if visible_pts >= 4:
                visible_imgfov_pts_2d = pts_2d[..., :2].reshape(8, 2)
                _, unique_indices = np.unique(visible_imgfov_pts_2d, axis=0, return_index=True)
                visible_imgfov_pts_2d = visible_imgfov_pts_2d[unique_indices]  # (4, 2)
                
                # sort points to find corners in clockwise order starting from top-left
                center = visible_imgfov_pts_2d.mean(axis=0)
                angles = np.arctan2(visible_imgfov_pts_2d[:, 1] - center[1], 
                                visible_imgfov_pts_2d[:, 0] - center[0])
                sorted_indices = np.argsort(angles)
                ordered_pts = visible_imgfov_pts_2d[sorted_indices]
                
                visible_projections.append((cam_idx, ordered_pts))

    return visible_projections


def apply_patch(imgs_adv, patch_trans_list, mask_trans_list, img_norm_cfg, visible_cam_indices):
    """ Apply patch to the image.
    
    Args:
        imgs_adv (torch.Tensor): (1, 6, 3, 480, 800)
        patch_trans (torch.Tensor): (1, 1, 3, 450, 800)
        mask_trans (torch.Tensor): (1, 1, 1, 450, 800)
        img_norm_cfg (dict): image normalization config
        visible_cam_idx (int): index of the visible camera
    
    Returns:
        imgs_adv (torch.Tensor): (1, 6, 3, 480, 800)
    """
    for cam_idx, patch_trans, mask_trans in zip(visible_cam_indices, patch_trans_list, mask_trans_list):
        # Handle padding if needed
        if patch_trans.shape[-2] < imgs_adv.shape[-2]:
            patch_trans = torch.nn.functional.pad(
                patch_trans, 
                (0, 0, 0, imgs_adv.shape[-2]-patch_trans.shape[-2], 0, 0), 
                'constant', 
                0
            )
        if mask_trans.shape[-2] < imgs_adv.shape[-2]:
            mask_trans = torch.nn.functional.pad(
                mask_trans,
                (0, 0, 0, imgs_adv.shape[-2]-mask_trans.shape[-2], 0, 0),
                'constant',
                0
            )
        
        # Normalize patch
        patch_trans_norm = normalize_img(patch_trans, img_norm_cfg)
        
        # Apply patch to this camera view
        imgs_adv[:, [cam_idx], ...] = torch.where(
            mask_trans>0, 
            patch_trans_norm, 
            imgs_adv[:, [cam_idx], ...]
        )
    
    return imgs_adv