import cv2
import torch
import numpy as np

from shapely import affinity
from shapely.geometry import LineString, box

from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points

from mmdet.datasets.builder import PIPELINES


''' map related '''
def overlap_filter(mask, filter_mask):
    C, _, _ = mask.shape
    for c in range(C-1, -1, -1):
        filter = np.repeat((filter_mask[c] != 0)[None, :], c, axis=0)
        mask[:c][filter] = 0

    return mask


def get_patch_coord(patch_box, patch_angle=0.0):
    patch_x, patch_y, patch_h, patch_w = patch_box

    x_min = patch_x - patch_w / 2.0
    y_min = patch_y - patch_h / 2.0
    x_max = patch_x + patch_w / 2.0
    y_max = patch_y + patch_h / 2.0

    patch = box(x_min, y_min, x_max, y_max)
    patch = affinity.rotate(patch, patch_angle, origin=(
        patch_x, patch_y), use_radians=False)

    return patch


def get_discrete_degree(vec, angle_class=36):
    deg = np.mod(np.degrees(np.arctan2(vec[1], vec[0])), 360)
    deg = (int(deg / (360 / angle_class) + 0.5) % angle_class) + 1
    return deg


def mask_for_lines(lines, mask, thickness, idx, type='index', angle_class=36):
    coords = np.asarray(list(lines.coords), np.int32)
    coords = coords.reshape((-1, 2))
    if len(coords) < 2:
        return mask, idx
    if type == 'backward':
        coords = np.flip(coords, 0)

    if type == 'index':
        cv2.polylines(mask, [coords], False, color=idx, thickness=thickness)
        idx += 1
    else:
        for i in range(len(coords) - 1):
            cv2.polylines(mask, [coords[i:]], False, color=get_discrete_degree(
                coords[i + 1] - coords[i], angle_class=angle_class), thickness=thickness)
    return mask, idx


def line_geom_to_mask(layer_geom, confidence_levels, local_box, canvas_size, thickness, idx, type='index', angle_class=36):
    patch_x, patch_y, patch_h, patch_w = local_box

    patch = get_patch_coord(local_box)

    canvas_h = canvas_size[0]
    canvas_w = canvas_size[1]
    scale_height = canvas_h / patch_h
    scale_width = canvas_w / patch_w

    trans_x = -patch_x + patch_w / 2.0
    trans_y = -patch_y + patch_h / 2.0

    map_mask = np.zeros(canvas_size, np.uint8)

    for line in layer_geom:
        if isinstance(line, tuple):
            line, confidence = line
        else:
            confidence = None
        new_line = line.intersection(patch)
        if not new_line.is_empty:
            new_line = affinity.affine_transform(
                new_line, [1.0, 0.0, 0.0, 1.0, trans_x, trans_y])
            new_line = affinity.scale(
                new_line, xfact=scale_width, yfact=scale_height, origin=(0, 0))
            confidence_levels.append(confidence)
            if new_line.geom_type == 'MultiLineString':
                for new_single_line in new_line.geoms:
                    map_mask, idx = mask_for_lines(
                        new_single_line, map_mask, thickness, idx, type, angle_class)
            else:
                map_mask, idx = mask_for_lines(
                    new_line, map_mask, thickness, idx, type, angle_class)
    return map_mask, idx


def preprocess_map(vectors, patch_size, canvas_size, max_channel, thickness, angle_class):
    confidence_levels = [-1]
    vector_num_list = {}
    for i in range(max_channel):
        vector_num_list[i] = []

    for vector in vectors:
        if vector['pts_num'] >= 2:
            vector_num_list[vector['type']].append(
                LineString(vector['pts'][:vector['pts_num']]))

    local_box = (0.0, 0.0, patch_size[0], patch_size[1])

    idx = 1
    filter_masks = []
    instance_masks = []
    forward_masks = []
    backward_masks = []
    for i in range(max_channel):
        map_mask, idx = line_geom_to_mask(
            vector_num_list[i], confidence_levels, local_box, canvas_size, thickness, idx)
        instance_masks.append(map_mask)
        filter_mask, _ = line_geom_to_mask(
            vector_num_list[i], confidence_levels, local_box, canvas_size, thickness + 4, 1)
        filter_masks.append(filter_mask)
        forward_mask, _ = line_geom_to_mask(
            vector_num_list[i], confidence_levels, local_box, canvas_size, thickness, 1, type='forward', angle_class=angle_class)
        forward_masks.append(forward_mask)
        backward_mask, _ = line_geom_to_mask(
            vector_num_list[i], confidence_levels, local_box, canvas_size, thickness, 1, type='backward', angle_class=angle_class)
        backward_masks.append(backward_mask)

    filter_masks = np.stack(filter_masks)
    instance_masks = np.stack(instance_masks)
    forward_masks = np.stack(forward_masks)
    backward_masks = np.stack(backward_masks)

    instance_masks = overlap_filter(instance_masks, filter_masks)
    forward_masks = overlap_filter(
        forward_masks, filter_masks).sum(0).astype('int32')
    backward_masks = overlap_filter(
        backward_masks, filter_masks).sum(0).astype('int32')

    semantic_masks = instance_masks != 0
    semantic_masks = torch.from_numpy(semantic_masks)
    #semantic_masks = torch.cat([(~torch.any(semantic_masks, axis=0)).unsqueeze(0), semantic_masks])

    return semantic_masks, instance_masks, forward_masks, backward_masks


class RasterizeMapVectors(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self,
                 map_grid_conf,
                 bev_w,
                 bev_h,
                 map_max_channel=3,
                 map_thickness=1,
                 map_angle_class=36
                 ):

        self.map_max_channel = map_max_channel
        self.map_thickness = map_thickness
        self.map_angle_class = map_angle_class

        map_xbound, map_ybound = map_grid_conf['xbound'], map_grid_conf['ybound']

        # patch_size: 在 y, x 方向上的坐标 range
        patch_h = map_ybound[1] - map_ybound[0]
        patch_w = map_xbound[1] - map_xbound[0]

        self.map_patch_size = (patch_h, patch_w)  # (30, 60)
        self.bevfeat_canvas_size = (bev_w, bev_h)  # (100, 200)


    def __call__(self, results):
        vectors = results['vectors']
        for vector in vectors:
            vector['pts'] = vector['pts'][:, [1, 0]]  # permute the coordinate to (x, y)

        bev_semantic_masks, bev_instance_masks, _, _ = preprocess_map(
            vectors, self.map_patch_size, self.bevfeat_canvas_size, self.map_max_channel, self.map_thickness, self.map_angle_class)
        
        results.update({
            'bev_semantic_map': bev_semantic_masks.unsqueeze(0),  # (1, 3, 100, 200)
            'instance_map_bev': torch.from_numpy(bev_instance_masks).unsqueeze(0),  # (1, 3, 100, 200)
        })

        return results



''' common '''
def lidar_to_camera_points(points_3d, lidar2global, global2img, img_shape, min_dist=1.0, ground=False):
    """Convert points from LiDAR coordinate to camera image coordinates.
    Args:
        points_3d (torch.Tensor): Points in LiDAR coordinate with shape (n, 3)
        lidar2global (torch.Tensor): Transformation matrix from LiDAR to global with shape (4, 4)
        global2img (torch.Tensor): Transformation matrix from global to image with shape (num_cam, 4, 4)
        camera_intrinsics (torch.Tensor): Camera intrinsics with shape (num_cam, 4, 4)
        img_shape (tuple): Image shape in (H, W) format
        min_dist (float): Minimum distance of points from camera. Default: 1.0
    Returns:
        tuple: visible_cam_indices, visible_points_2d
    """
    def translate(points, x):
        """Apply translation to points"""
        return points + x

    def rotate(points, rot_matrix):
        """Apply rotation to points"""
        return torch.matmul(points, rot_matrix.transpose(-2, -1))

    device = points_3d.device
    num_points = points_3d.shape[0]
    num_cam = global2img.shape[0]
    img_h, img_w = img_shape

    # Transform points from lidar to global coordinates
    points_global = rotate(points_3d, lidar2global[:3, :3])
    points_global = translate(points_global, lidar2global[:3, 3])
    
    if ground:
        points_global[:, 2] = 0
    
    # Initialize lists for visible points
    visible_cam_indices = []
    visible_points_2d = []
    
    # Convert to homogeneous coordinates
    points_global_homo = torch.cat([points_global, torch.ones_like(points_global[:, :1])], dim=-1)  # (n, 4)
    
    # Process each camera
    for cam_idx in range(num_cam):
        # Transform from global to image coordinates directly
        points_img = torch.matmul(global2img[cam_idx], points_global_homo.T).T  # (n, 4)
        
        # Get depths (z coordinate before projection)
        depths = points_img[:, 2]
        
        # Project to 2D
        points_2d = points_img[:, :2] / points_img[:, 2:3]  # (n, 2)
            
        # Create visibility mask
        visible_mask = (depths > min_dist) & \
                      (points_2d[:, 1] > 1) & \
                      (points_2d[:, 1] < img_h - 1) & \
                      (points_2d[:, 0] > 1) & \
                      (points_2d[:, 0] < img_w - 1)
        
        # Add visible points to results
        if visible_mask.any():
            visible_points = points_2d[visible_mask]
            visible_cam_indices.extend([cam_idx] * len(visible_points))
            visible_points_2d.extend(visible_points)

    return visible_cam_indices, visible_points_2d



def lidar_to_img(points, nusc, sample_token, cam_name, img_shape, ground=False):
    '''
    points: (num_points, 3) in lidar coordinate
    '''
    
    def translate(points, x: np.ndarray) -> None:
        """
        Applies a translation to the point cloud.
        :param points: <np.float: n, 3>. Points to transform.
        :param x: <np.float: 3>. Translation in x, y, z.
        """
        return points + x

    def rotate(points, rot_matrix: np.ndarray) -> None:
        """
        Applies a rotation.
        :param points: <np.float: n, 3>. Points to transform.
        :param rot_matrix: <np.float: 3, 3>. Rotation matrix.
        """
        return points @ rot_matrix.T
    
    
    # init
    sample = nusc.get('sample', sample_token)
    cam = nusc.get('sample_data', sample['data'][cam_name])
    pointsensor = nusc.get('sample_data', sample['data']['LIDAR_TOP'])

    # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
    # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
    cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
    points = rotate(points, Quaternion(cs_record['rotation']).rotation_matrix)
    points = translate(points, np.array(cs_record['translation']))

    # Second step: transform from ego to the global frame.
    poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
    points = rotate(points, Quaternion(poserecord['rotation']).rotation_matrix)
    points = translate(points, np.array(poserecord['translation']))
    if ground:
        points[:, 2] = 0  # set z value in global frame to 0
    
    # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
    poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
    points = translate(points, -np.array(poserecord['translation']))
    points = rotate(points, Quaternion(poserecord['rotation']).rotation_matrix.T)
    
    # Fourth step: transform from ego into the camera.
    cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    points = translate(points, -np.array(cs_record['translation']))
    points = rotate(points, Quaternion(cs_record['rotation']).rotation_matrix.T)
    
    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = points[:, 2]
    
    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    points = view_points(points.T, np.array(cs_record['camera_intrinsic']), normalize=True)
    points = points.T  # Convert back to (n,3))

    # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
    # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
    # casing for non-keyframes which are slightly out of sync.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > 1)
    mask = np.logical_and(mask, points[:, 0] > 1)
    mask = np.logical_and(mask, points[:, 0] < img_shape[0] - 1)
    mask = np.logical_and(mask, points[:, 1] > 1)
    mask = np.logical_and(mask, points[:, 1] < img_shape[1] - 1)
    points_visible = points[mask]

    return points_visible