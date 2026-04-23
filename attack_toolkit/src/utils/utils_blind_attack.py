import cv2
import torch
import numpy as np

from attack_toolkit.src.utils.utils_attack import normalize_img, denormalize_img


''' attack position ranking utils '''
def calculate_combined_score(point, img_metas, divergent_points, max_beam_angle=np.radians(40)):
    """
    Position scoring function for attack position ranking.
    
    Args:
        point: 3D attack position [x, y, z]
        img_metas: Image metadata containing camera transforms
        divergent_points: Array of divergent boundary points
        max_beam_angle: Maximum angle from camera-to-divergent line
    
    Returns:
        float: Combined visibility score (higher = better attack position)
    """

    MIN_DISTANCE = 1.0  # Avoid division by zero
    FALLOFF_POWER = 2  # Gentler than inverse square law
    
    
    # Get camera positions in lidar coordinates
    camera_positions = []
    ego2lidar = np.linalg.inv(img_metas['lidar2ego'])
    for camera2ego in img_metas['camera2ego']:
        camera_pos_ego = camera2ego[:3, 3]
        camera_pos_lidar = (ego2lidar @ np.append(camera_pos_ego, 1))[:3]
        camera_positions.append(camera_pos_lidar)
    
    total_score = 0
    
    for cam_pos in camera_positions:
       
        for div_point in divergent_points:
            # Vector from camera to attack point
            cam_to_attack = point - cam_pos
            cam_to_attack_dist = np.linalg.norm(cam_to_attack)
            cam_to_attack_dir = cam_to_attack / cam_to_attack_dist
            
            # Vector from camera to divergent point
            cam_to_div = div_point - cam_pos
            cam_to_div_dist = np.linalg.norm(cam_to_div)
            cam_to_div_dir = cam_to_div / cam_to_div_dist
            
            # Calculate angle between these vectors
            angle = np.arccos(np.clip(np.dot(cam_to_attack_dir, cam_to_div_dir), -1.0, 1.0))
            
            # Only consider points within the maximum angle
            if angle < max_beam_angle:

                angle_score = 1.0 - (angle / max_beam_angle)  # 1.0 = perfect alignment, 0.0 = at edge
                distance_score = 1.0 / (cam_to_attack_dist ** 2)  # Closer = better
                
                total_score += angle_score * distance_score
    
    return total_score


''' lens flare utils '''
def get_camera_params(img_metas, cam_idx):
    """
    Get camera parameters from NuScenes metadata and convert to LiDAR coordinates
    
    Args:
        img_metas: image metadata from dataset
        cam_idx: camera index (0-5 for nuScenes)
    Returns:
        dict: camera parameters including position and direction in LiDAR coordinates
    """
    # Get camera to ego vehicle transform
    camera2ego = img_metas['camera2ego'][cam_idx]  # (4, 4) matrix
    
    # Get ego to LiDAR transform
    ego2lidar = np.linalg.inv(img_metas['lidar2ego'])  # (4, 4) matrix
    
    # Convert camera position from ego to LiDAR coordinates
    camera_pos_ego = camera2ego[:3, 3]
    camera_pos_homo = np.append(camera_pos_ego, 1)  # homogeneous coordinates
    camera_pos_lidar = (ego2lidar @ camera_pos_homo)[:3]
    
    # Convert camera direction from ego to LiDAR coordinates
    # First get direction in ego coordinates (negative z-axis of camera)
    camera_direction_ego = -camera2ego[:3, 2]
    camera_direction_homo = np.append(camera_direction_ego, 0)  # use 0 for direction vector
    # Convert to LiDAR coordinates (only apply rotation)
    camera_direction_lidar = (ego2lidar @ camera_direction_homo)[:3]
    camera_direction_lidar = camera_direction_lidar / np.linalg.norm(camera_direction_lidar)
    
    return {
        'position': camera_pos_lidar,
        'direction': camera_direction_lidar
    }
    
    
def calculate_light_direction(attacker_position, ego_position):
    """
    Calculate light direction from attacker to ego vehicle
    
    Args:
        attacker_position: np.array([x, y, z]) in LiDAR coordinates
        ego_position: np.array([x, y, z]) typically [0, 0, 0] in LiDAR coordinates
    Returns:
        np.array: normalized direction vector
    """
    direction = ego_position - attacker_position
    return direction / np.linalg.norm(direction)



def project_3d_to_2d(points_3d, lidar2global, global2img, ground=False):
    """Convert points from LiDAR coordinate to camera image coordinates.
    Args:
        points_3d (torch.Tensor): Points in LiDAR coordinate with shape (3,) or (n, 3)
        lidar2global (torch.Tensor): Transformation matrix from LiDAR to global with shape (4, 4)
        global2img (torch.Tensor): Transformation matrix from global to image with shape (4, 4)
        ground (bool): Whether to project points to ground plane. Default: False
    Returns:
        points_2d (torch.Tensor): Points in image coordinate with shape (2,) or (n, 2)
    """
    def translate(points, x):
        """Apply translation to points"""
        return points + x

    def rotate(points, rot_matrix):
        """Apply rotation to points"""
        return torch.matmul(points, rot_matrix.transpose(-2, -1))

    # Handle single point case
    single_point = False
    if points_3d.dim() == 1:
        points_3d = points_3d.unsqueeze(0)
        single_point = True

    # Transform points from lidar to global coordinates
    points_global = rotate(points_3d, lidar2global[:3, :3])
    points_global = translate(points_global, lidar2global[:3, 3])
    
    if ground:
        points_global[:, 2] = 0
    
    # Convert to homogeneous coordinates
    points_global_homo = torch.cat([points_global, torch.ones_like(points_global[:, :1])], dim=-1)  # (n, 4)
    
    # Transform from global to image coordinates directly
    points_img = torch.matmul(global2img, points_global_homo.T).T  # (n, 4)
    
    # # Project to 2D
    # points_2d = points_img[:, :2] / points_img[:, 2:3]  # (n, 2)
    
    # Filter out points with depth <= 1e-5
    valid_depth_mask = points_img[:, 2] > 1e-5
    # Project to 2D (only for valid depth points)
    points_2d = torch.zeros_like(points_img[:, :2])
    points_2d[valid_depth_mask] = points_img[valid_depth_mask, :2] / points_img[valid_depth_mask, 2:3]
    
    # Return single point if input was single point
    if single_point:
        points_2d = points_2d.squeeze(0)
        
    return points_2d


def is_camera_affected(camera_params, light_source_params):
    """
    Determine if a camera should be affected by the light source
    
    Args:
        camera_params: dict with camera position and direction
        light_source_params: dict with light source position
    Returns:
        bool: Whether camera should be affected
    """
    # Constants
    CAMERA_ANGLE_THRESHOLD = np.pi / 3  # 60 degrees
    
    # Calculate vectors
    light_direction = calculate_light_direction(
        light_source_params['position'],
        camera_params['position']
    )
    
    # Calculate angle between camera direction and light direction
    angle = np.arccos(np.dot(camera_params['direction'], light_direction))
    
    # Stricter angle threshold (60 degrees from camera direction)
    if angle > CAMERA_ANGLE_THRESHOLD:
        return False
        
    # Check if light source is on the same side as camera
    # Get camera position relative to vehicle center (assumed to be [0,0,0])
    cam_to_center = camera_params['position']
    light_to_center = light_source_params['position']
    
    # If camera and light are on opposite sides of the vehicle, light should be blocked
    # Check x-y plane quadrants
    if np.sign(cam_to_center[0]) != np.sign(light_to_center[0]) and \
       np.sign(cam_to_center[1]) != np.sign(light_to_center[1]):
        return False
        
    return True


def generate_lens_flare(imgs, img_metas, light_source_params, img_norm_cfg, cam_idx, img_shape):
    """
    Generate a physically-based lens flare effect with improved long-distance handling
    """
    
    # Setup
    device = imgs.device
    ori_h, ori_w = img_shape
    
    # Constants for lens flare effect
    BASE_RADIUS_DIVISOR = 2  # Initial size of the flare (ori_size // 2)
    MIN_RADIUS_DIVISOR = 8   # Minimum radius of the flare (ori_size // 8)
    REFERENCE_DISTANCE = 1.0  # Reference distance for size calculation
    MAX_DISTANCE_FACTOR = 30  # At what multiple of reference_distance should we reach min_radius
    INTENSITY_SCALE = 0.02    # Scaling factor for intensity
    MIN_INTENSITY = 0.6       # Minimum intensity for close distances
    FALLOFF_POWER = 1.5       # Power for inverse distance law (slower than inverse square)
    BLUE_TINT_FACTOR = 1.1    # Factor to boost blue channel
    CAMERA_ANGLE_THRESHOLD = np.pi / 3  # 60 degrees
    
    # Calculate dynamic parameters
    base_radius = min(ori_h, ori_w) // BASE_RADIUS_DIVISOR
    min_radius = min(ori_h, ori_w) // MIN_RADIUS_DIVISOR
    
    # Get camera parameters
    camera_params = get_camera_params(img_metas, cam_idx)
    
    # Check if this camera should be affected
    if not is_camera_affected(camera_params, light_source_params):
        return imgs
    
    # Calculate distance between light source and camera
    distance = np.linalg.norm(
        light_source_params['position'][:2] - camera_params['position'][:2]
    )
    
    # Project light source position to image plane
    light_pos_3d = torch.tensor(light_source_params['position']).to(device)
    light_pos_2d = project_3d_to_2d(
        light_pos_3d,
        torch.tensor(img_metas['lidar2global']).to(device),
        torch.tensor(img_metas['global2img'][cam_idx]).to(device),
        ground=False
    )
    
    # Convert to numpy for OpenCV processing
    light_pos_2d = light_pos_2d.detach().cpu().numpy().astype(int)
    
    # Check if light source is potentially visible
    is_visible = (0 <= light_pos_2d[0] < ori_w) and (0 <= light_pos_2d[1] < ori_h)
    
    # Even if not directly visible, allow flare effect from nearby
    if not is_visible:
        # Find closest point on image
        light_pos_2d[0] = np.clip(light_pos_2d[0], 0, ori_w-1)
        light_pos_2d[1] = np.clip(light_pos_2d[1], 0, ori_h-1)
    
    # 1. Calculate blur size with logarithmic falloff
    normalized_distance = min(distance / REFERENCE_DISTANCE, MAX_DISTANCE_FACTOR)
    log_factor = 1 - (np.log(normalized_distance) / np.log(MAX_DISTANCE_FACTOR))
    blur_radius = int(min_radius + (base_radius - min_radius) * max(0, log_factor))
    
    # 2. Calculate intensity with slower falloff
    max_intensity = light_source_params.get('power', 2000.0)
    effective_distance = max(distance, REFERENCE_DISTANCE)
    intensity = max_intensity / (effective_distance ** FALLOFF_POWER)
    intensity = min(1.0, intensity * INTENSITY_SCALE)
    
    # Ensure minimum effect intensity within reasonable range
    intensity = max(intensity, MIN_INTENSITY)
    
    # 3. Generate the lens flare
    img_adv = imgs[0, cam_idx].clone()  # min: -1.5828, max: 2.64
    img_denorm = denormalize_img(img_adv, img_norm_cfg).permute(1, 2, 0).cpu().numpy()  # (480, 800, 3)
    img_denorm = img_denorm.astype(np.float32)
    
    # Create blank flare image
    flare = np.zeros_like(img_denorm, dtype=np.float32)  # (480, 800, 3)
    flare = np.ascontiguousarray(flare)
    
    # Create the flare circle
    cv2.circle(
        flare,
        (int(light_pos_2d[0]), int(light_pos_2d[1])),
        blur_radius,
        (255.0, 255.0, 255.0),
        -1
    )
    
    # Apply Gaussian blur
    kernel_size = min(2 * blur_radius + 1, min(ori_h, ori_w) - 1)
    kernel_size = max(3, kernel_size)
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    flare = cv2.GaussianBlur(flare, (kernel_size, kernel_size), blur_radius / 2)
    
    # Add slight blue tint common in lens flares
    flare[:, :, 0] *= BLUE_TINT_FACTOR
    flare = np.clip(flare, 0, 255)
    
    # Add the flare to the original image
    result = cv2.addWeighted(img_denorm, 1.0, flare, intensity, 0)
    result = np.clip(result, 0, 255)
    
    # Convert back to normalized tensor
    result = torch.from_numpy(result).permute(2, 0, 1).to(device)
    result = normalize_img(result, img_norm_cfg)
    
    # Update the image
    imgs[0, cam_idx] = result
    
    return imgs
