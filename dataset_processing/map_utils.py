"""
Map processing utilities for extracting lane information from NuScenes maps.
"""
import numpy as np
from typing import Tuple
from scipy.spatial.distance import cdist
from shapely.geometry import LineString
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.map_api import NuScenesMap
from .geometry_utils import interpolate, endpoints_intersect, order_matches


def extract_lane_center(nusc_map: NuScenesMap, lane_record) -> np.ndarray:
    """
    Extract lane center points from NuScenes map.
    
    Args:
        nusc_map: NuScenes map instance
        lane_record: Lane record from NuScenes
        
    Returns:
        np.ndarray: Lane midline points
    """
    # Get lane center's points
    curr_lane = nusc_map.arcline_path_3.get(lane_record["token"], [])
    lane_midline = np.array(
        arcline_path_utils.discretize_lane(curr_lane, resolution_meters=0.5)
    )[:, :2]

    # Remove duplicate entries
    duplicate_check = np.where(
        np.linalg.norm(np.diff(lane_midline, axis=0, prepend=0), axis=1) < 1e-10
    )[0]
    if duplicate_check.size > 0:
        lane_midline = np.delete(lane_midline, duplicate_check, axis=0)

    return lane_midline


def extract_lane_and_edges(nusc_map: NuScenesMap, road_record, lane_record) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract lane center and edge boundaries from NuScenes map.
    
    Args:
        nusc_map: NuScenes map instance
        road_record: Road segment record
        lane_record: Lane record
        
    Returns:
        tuple: (lane_midline, left_pts, right_pts)
    """
    # Get polygon vertices
    road_polygon_obj = nusc_map.get("polygon", road_record["polygon_token"])
    polygon_nodes = [
        nusc_map.get("node", node_token)
        for node_token in road_polygon_obj["exterior_node_tokens"]
    ]
    polygon_pts = np.array([(node["x"], node["y"]) for node in polygon_nodes])

    # Get lane center points
    lane_midline = extract_lane_center(nusc_map, lane_record)

    # Compute closest lane center point to each polygon vertex
    closest_midlane_pt = np.argmin(cdist(polygon_pts, lane_midline), axis=1)
    
    # Compute local direction of the lane at each point
    direction_vectors = np.diff(
        lane_midline,
        axis=0,
        prepend=lane_midline[[0]] - (lane_midline[[1]] - lane_midline[[0]]),
    )

    # Select direction vectors at closest lane center points
    local_dir_vecs = direction_vectors[closest_midlane_pt]
    origin_to_polygon_vecs = polygon_pts - lane_midline[closest_midlane_pt]

    # Compute perpendicular dot product
    perp_dot_product = (
        local_dir_vecs[:, 0] * origin_to_polygon_vecs[:, 1]
        - local_dir_vecs[:, 1] * origin_to_polygon_vecs[:, 0]
    )

    # Determine which indices are on the right of the lane center
    on_right = perp_dot_product < 0

    # Find boundary between left/right polygon vertices
    transitions = np.where(np.roll(on_right, 1) < on_right)[0]

    if len(transitions) == 1:
        idx_changes = transitions.item()
    else:
        idx_changes = transitions[0].item()      

    if idx_changes > 0:
        # Roll array to put boundary at index 0
        polygon_pts = np.roll(polygon_pts, shift=-idx_changes, axis=0)
        on_right = np.roll(on_right, shift=-idx_changes)

    left_pts = polygon_pts[~on_right]
    right_pts = polygon_pts[on_right]

    # Final ordering check
    if endpoints_intersect(left_pts, right_pts):
        if not order_matches(left_pts, lane_midline):
            left_pts = left_pts[::-1]
        else:
            right_pts = right_pts[::-1]

    # Ensure left and right have same number of points
    if left_pts.shape[0] < right_pts.shape[0]:
        left_pts = interpolate(left_pts, num_pts=right_pts.shape[0])
    elif right_pts.shape[0] < left_pts.shape[0]:
        right_pts = interpolate(right_pts, num_pts=left_pts.shape[0])

    return lane_midline, left_pts, right_pts

