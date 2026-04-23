import os
import json
import argparse
from typing import List

import numpy as np
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Polygon, Point
import networkx as nx

from nuscenes.nuscenes import NuScenes

import lanelet2
from lanelet2.projection import UtmProjector
from lanelet2.io import Origin

from attack_toolkit.src.utils.utils_plan import lidar_to_global, global_to_lidar, is_in_range

np.set_printoptions(precision=3, suppress=True)

data_root = 'data/nuscenes'
# Initialize nuScenes
nusc = NuScenes(
    version='v1.0-trainval',  # or 'v1.0-trainval' for full dataset
    dataroot=data_root,  # Change this to your nuScenes data path
    verbose=True
)

def find_lanelets_in_region(lanelet_map, sample_token, x_range=30, y_range=60):
    """
    Find all lanelets within a rectangular region centered at point_xy
    
    Args:
        lanelet_map: Loaded lanelet map
        point_xy: Tuple of (x,y) in NuScenes global coordinates
        heading: Heading angle in radians
        x_range: Range in x direction (meters)
        y_range: Range in y direction (meters)
        projector: UTM projector used to load the map
    
    Returns:
        List of lanelets in the region
    """
    corners_lidar = np.array([
        [-x_range/2, y_range/2, 0],  # left front
        [x_range/2, y_range/2, 0],  # right front
        [x_range/2, -y_range/2, 0],  # right rear
        [-x_range/2, -y_range/2, 0],  # left rear
    ])
    corners_global = lidar_to_global(corners_lidar, nusc, sample_token, ground=True)
    
    # get the min and max x and y values
    min_x = np.min(corners_global[:, 0])
    max_x = np.max(corners_global[:, 0])
    min_y = np.min(corners_global[:, 1])
    max_y = np.max(corners_global[:, 1])
    
    # Create BasicPoint3d objects for the bounding box corners
    p_min = lanelet2.core.BasicPoint2d(min_x, min_y)
    p_max = lanelet2.core.BasicPoint2d(max_x, max_y)
    bounding_box_2d = lanelet2.core.BoundingBox2d(p_min, p_max)
    
    # Get all lanelets within the bounding box
    # The search2d method returns all primitives that intersect with the bounding box
    lanelets_in_region = lanelet_map.laneletLayer.search(bounding_box_2d)
    
    return lanelets_in_region

def get_adjacent_lanelets(lanes, sample_token):
    adjacent_lanes = []
    
    for lane in lanes:
        center_points = np.array([[p.x, p.y, p.z] for p in lane.centerline])
        center_points_lidar = global_to_lidar(center_points, nusc, sample_token)
        # Check if any line segment of the centerline crosses y=0
        for i in range(len(center_points_lidar)-1):
            y1 = center_points_lidar[i][1]
            y2 = center_points_lidar[i+1][1]
            
            # If y1 and y2 have different signs, the line segment crosses y=0
            if y1 * y2 <= 0:
                # Determine direction based on y values
                if y1 <= 0 and y2 > 0:
                    direction = 'next'  # Moving towards positive y
                else:
                    direction = 'prev'  # Moving towards negative y
                adjacent_lanes.append((lane, direction))
                break
        
    return adjacent_lanes

def create_lane_graph(lanes: List[lanelet2.core.Lanelet], graph: lanelet2.routing.RoutingGraph) -> nx.DiGraph:
    """
    Creates a directed graph representation of lane connections.
    
    Args:
        lanes: List of all available lanes
        nusc: NuScenes instance
        sample_token: Current sample token
    
    Returns:
        NetworkX DiGraph where nodes are lane IDs and edges represent connections
    """
    G = nx.DiGraph()
    
    # Create mapping of lane_id to lane object for quick lookup
    lane_map = {lane.id: lane for lane in lanes}
    
    for lane in lanes:
        # Add node with lane object as attribute
        G.add_node(lane.id, lane_obj=lane)
        
        # Add edges for next connections
        for next_lane in graph.following(lane):
            if next_lane.id in lane_map:
                G.add_edge(lane.id, next_lane.id, direction='next')
        
        # Add edges for previous connections
        for prev_lane in graph.previous(lane):
            if prev_lane.id in lane_map:
                G.add_edge(lane.id, prev_lane.id, direction='prev')
    
    return G

def find_all_routes(
    start_lanes: List[lanelet2.core.Lanelet],
    lane_graph: nx.DiGraph,
    gt_boundaries: np.ndarray,
    sample_token: str,
    max_depth: int = 50
) -> List[List[lanelet2.core.Lanelet]]:
    """
    Finds all valid routes from given start points using graph search.
    
    Args:
        start_lanes: List of valid start points (lane, direction)
        lane_graph: DirectedGraph of lane connections
        nusc: NuScenes instance
        sample_token: Current sample token
        max_depth: Maximum number of lane segments to traverse
    
    Returns:
        List of valid route sequences
    """
    
    def is_valid_route(route: List[lanelet2.core.Lanelet], gt_boundaries: np.ndarray,
                       x_range=(-15, 15), y_range=(-30, 30),
                       resolution=0.5) -> bool:
        """
        Checks if a route stays within visualization bounds and is reachable from ego position.
        
        Args:
            route: List of RoadLanes to check
            gt_boundaries: Ground truth boundary points
        """
        
        # First check if route is in range
        points = []
        for lane in route:
            lane_points = np.array([[p.x, p.y, p.z] for p in lane.centerline])
            points_lidar = global_to_lidar(lane_points, nusc, sample_token)
            points.extend(points_lidar)
        points = np.array(points)
        
        if not is_in_range(points, x_range=x_range, y_range=(0, 30), mode='any'):
            return False
        
        # Create grid map with configurable dimensions
        x_size = int((x_range[1] - x_range[0]) / resolution)
        y_size = int((y_range[1] - y_range[0]) / resolution)
        grid = np.zeros((y_size, x_size), dtype=np.uint8)  # 0: free, 1: occupied
        
        # Convert boundaries to geometries
        boundaries = []
        for boundary_points in gt_boundaries:
            if np.allclose(boundary_points[0], boundary_points[-1]):
                boundaries.append(Polygon(boundary_points))
            else:
                boundaries.append(LineString(boundary_points))
        # add a line at y=-5 to prevent flooding backward
        boundaries.append(LineString([(-15, -5), (15, -5)]))
        
        # Convert coordinates to grid indices
        def world_to_grid(x, y):
            grid_x = int((x - x_range[0]) / resolution)
            grid_y = int((y - y_range[0]) / resolution)
            return np.clip(grid_x, 0, x_size-1), np.clip(grid_y, 0, y_size-1)
    
        def is_in_boundary(x: float, y: float) -> bool:
            """Check if point is within or very close to any boundary"""
            point = Point(x, y)
            for boundary in boundaries:
                if isinstance(boundary, Polygon):
                    if point.within(boundary):
                        return True
                else:  # LineString
                    if point.distance(boundary) < resolution:
                        return True
            return False
        
        # Mark grid cells as occupied if they intersect with boundaries
        for i in range(y_size):
            for j in range(x_size):
                x = -15 + j * resolution
                y = -30 + i * resolution
                if is_in_boundary(x, y):
                    grid[i, j] = 1
        
        # Mark range boundaries as occupied
        grid[0, :] = 1  # y = -30
        grid[-1, :] = 1  # y = 30
        grid[:, 0] = 1  # x = -15
        grid[:, -1] = 1  # x = 15
        
        # Flood fill from ego position
        start_x, start_y = world_to_grid(0, 0)
        visited = np.zeros_like(grid)
        queue = [(start_x, start_y)]
        visited[start_y, start_x] = 1
        
        # 8-connected neighborhood
        directions = [(0,1), (1,0), (0,-1), (-1,0), 
                    (1,1), (-1,1), (1,-1), (-1,-1)]
        
        while queue:
            x, y = queue.pop(0)
            for dx, dy in directions:
                new_x, new_y = x + dx, y + dy
                if (0 <= new_x < x_size and 0 <= new_y < y_size and 
                    not visited[new_y, new_x] and not grid[new_y, new_x]):
                    queue.append((new_x, new_y))
                    visited[new_y, new_x] = 1
                    
        # Check if route points are reachable
        for point in points:
            grid_x, grid_y = world_to_grid(point[0], point[1])
            if visited[grid_y, grid_x]:
                return True
                
        return False
    
    def is_subset_of_existing_routes(new_route: List[lanelet2.core.Lanelet], existing_routes: List[List[lanelet2.core.Lanelet]]) -> bool:
        """
        Checks if the new route is a subset of any existing route.
        
        Args:
            new_route: List of RoadLanes representing the new route
            existing_routes: List of existing valid routes
            
        Returns:
            bool: True if new_route is a subset of any existing route
        """
        new_route_ids = [lane.id for lane in new_route]
        for existing_route in existing_routes:
            existing_route_ids = [lane.id for lane in existing_route]
            
            # Check if new_route is a contiguous subset of existing_route
            # print(new_route_ids)
            # print(existing_route_ids)
            str_new = ','.join([str(id) for id in new_route_ids])
            str_existing = ','.join([str(id) for id in existing_route_ids])
            if str_new in str_existing:
                return True
        return False
    
    valid_routes = []
    temp_routes = []
    
    # single direction search
    for start_lane, direction in start_lanes:
        # Use DFS to find all possible routes
        stack = [(start_lane.id, [start_lane])]
        while stack:
            current_id, current_route = stack.pop()
            
            # Check forward connections
            for next_id in lane_graph.neighbors(current_id):
                if lane_graph.adj[current_id][next_id]['direction'] == direction:
                    next_lane = lane_graph.nodes[next_id]['lane_obj']
                    new_route = current_route + [next_lane]
                    
                    if len(new_route) > max_depth:
                        continue
                        
                    if is_valid_route(new_route, gt_boundaries):
                        temp_routes.append(new_route)
                        stack.append((next_id, new_route))
    
    # Filter out subset routes
    # Sort routes by length (descending) to check longer routes first
    temp_routes.sort(key=lambda x: len(x), reverse=True)
    
    for route in temp_routes:
        # Only add route if it's not a subset of any existing route
        if not is_subset_of_existing_routes(route, valid_routes):
            valid_routes.append(route)
    
    return valid_routes

def convert_routes_to_points(
    routes: List[List[lanelet2.core.Lanelet]],
    gt_boundaries: np.ndarray,
    sample_token: str,
    interval=1.0,
    x_range=(-15, 15),
    y_range=(0, 30),
    dist2boundary=0.5
) -> List[np.ndarray]:
    """
    Converts route sequences to point sequences in lidar coordinates.
    
    Args:
        routes: List of route sequences
        nusc: NuScenes instance
        sample_token: Current sample token
    
    Returns:
        List of point sequences in lidar coordinates
    """
    point_sequences = []
    
    # Convert boundaries to shapely LineStrings for intersection checking
    boundary_lines = []
    for boundary in gt_boundaries:
        line = LineString(boundary)
        boundary_lines.append(line)
    
    for route in routes:
        points = []
        
        # check the direction of the route
        start_lane = route[0]
        start_lane_centers = np.array([[p.x, p.y, p.z] for p in start_lane.centerline])
        start_lane_centers_lidar = global_to_lidar(start_lane_centers, nusc, sample_token)
        direction = 'next' if start_lane_centers_lidar[0, 1] < 0 else 'prev'
        
        # connect the lanes in a route
        for lane in route:
            lane_centers = np.array([[p.x, p.y, p.z] for p in lane.centerline])
            lane_centers_lidar = global_to_lidar(lane_centers, nusc, sample_token)
            if direction == 'prev':
                lane_centers_lidar = lane_centers_lidar[::-1]
            points.extend(lane_centers_lidar)
            
        # interpolate the points (convert to polyline and interpolate)
        line = LineString([(p[0], p[1]) for p in points])
        
        line_length = line.length
        distances = np.arange(0, line_length, interval)
        points_interpolated = []
        for dist in distances:
            point = line.interpolate(dist)
            points_interpolated.append([point.x, point.y, 0]) # Set z=0 since we only interpolate x,y

        if line_length > distances[-1]:
            point = line.interpolate(line_length)
            points_interpolated.append([point.x, point.y, 0])
            
        points_interpolated = np.array(points_interpolated)
        
        # clip the points to the range
        mask = (points_interpolated[:, 0] >= x_range[0]) & (points_interpolated[:, 0] <= x_range[1]) & \
               (points_interpolated[:, 1] >= y_range[0]) & (points_interpolated[:, 1] <= y_range[1])
        points_interpolated = points_interpolated[mask]
        
        # clip the points to the boundary
        points_wo_collision = []
        for point in points_interpolated:
            x, y, z = point  # Ignore heading since we're just checking points
            point_shapely = Point(x, y)
            collision = False
            
            for boundary in boundary_lines:
                if point_shapely.distance(boundary) < dist2boundary:
                    collision = True
                    break
            if not collision:
                points_wo_collision.append([x, y, z])
            else:
                break
        
        point_sequences.append(np.array(points_wo_collision))
    
    return point_sequences

def select_goal_states(route_center_points, interval=1.0, n=3, dist2range=2.0):
    """
    Selects n diverse goal points from route centerlines with margin from range boundaries.
    
    Args:
        route_center_points: List of route point sequences
        n: Number of goal points to select
        dist2range: Distance to track back along route (in meters)
        
    Returns:
        List of selected points with their headings (point, heading)
    """
    # Extract candidate points - use last point of each route
    candidate_points = []
    for points in route_center_points:
        if len(points) < 2:  # Need at least 2 points for heading
            continue
            
        # Get last point and calculate heading using previous point
        last_point = points[-1]
        prev_point = points[-2]
        heading = np.arctan2(last_point[1] - prev_point[1],
                           last_point[0] - prev_point[0])
            
        candidate_points.append((last_point, heading, points, len(points)-1))
    
    if not candidate_points:
        return []
        
    selected = []
    remaining = candidate_points.copy()
    
    # First select farthest from ego
    farthest = max(remaining, key=lambda x: np.linalg.norm(x[0]))
    selected.append(farthest)
    remaining = [r for r in remaining if not np.array_equal(r[0], farthest[0])]
    
    # Subsequent selections based on max-min distance
    while len(selected) < n and remaining:
        max_min_dist = -1
        best_candidate = None
        
        for candidate in remaining:
            # Calculate minimum distance to already selected points
            min_dist = min(np.linalg.norm(candidate[0] - s[0]) for s in selected)
            
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_candidate = candidate
                
        if best_candidate:
            selected.append(best_candidate)
            remaining = [r for r in remaining if not np.array_equal(r[0], best_candidate[0])]
    
    # Post-process: track back along routes
    final_selected = []
    steps_back = int(dist2range / interval)  # Convert distance to number of points
    
    for point, heading, route_points, idx in selected[:n]:
        # Calculate new index after tracking back
        new_idx = max(0, idx - steps_back)
        new_point = route_points[new_idx]
        
        # Recalculate heading at new point
        if new_idx + 1 < len(route_points):
            next_point = route_points[new_idx + 1]
            new_heading = np.arctan2(next_point[1] - new_point[1],
                                   next_point[0] - new_point[0])
        else:
            prev_point = route_points[new_idx - 1]
            new_heading = np.arctan2(new_point[1] - prev_point[1],
                                   new_point[0] - prev_point[0])
                                   
        final_selected.append((new_point, new_heading))
    
    return final_selected


def main(args):

    # Search Region Parameters
    SEARCH_X_RANGE = 30  # meters, total width of search region
    SEARCH_Y_RANGE = 60  # meters, total height of search region

    # Route Finding Parameters
    MAX_ROUTE_DEPTH = 100  # maximum number of lane segments in a route

    ROUTE_X_RANGE = (-15, 15)  # meters, visualization window x-range
    ROUTE_Y_RANGE = (0, 30)    # meters, visualization window y-range

    ROUTE_INTERVAL = 0.5     # meters, interval between interpolated points
    DIST2BOUNDARY = 0.5      # meters, minimum distance to keep from boundaries

    # Goal Selection Parameters
    NUM_GOALS = 3            # number of goal states to select
    DIST2RANGE = 2.0        # meters, distance to track back from end of route

    dataset_tag = args.dataset_tag
    model = args.model

    gt_dir = f'dataset/{model}/train_blind_targeted_{dataset_tag}/results/map/gt'
    if not os.path.exists(gt_dir):
        raise ValueError(f'{gt_dir} does not exist')

    goal_points_dir = f'dataset/goal_states_{dataset_tag}_lanelet2'
    if not os.path.exists(goal_points_dir):
        os.makedirs(goal_points_dir)

    with open(f'dataset/sample_tokens_{dataset_tag}.txt', 'r') as f:
        sample_tokens = f.readlines()
    sample_tokens = [token.strip() for token in sample_tokens]

    # Main processing loop
    valid_routes_num = []
    for sample_idx, sample_token in enumerate(sample_tokens):
        ''' Initialize scene and map data '''
        # get the location of sample token
        sample = nusc.get('sample', sample_token)
        scene = nusc.get('scene', sample['scene_token'])
        log = nusc.get('log', scene['log_token'])
        location = log['location']
        
        # read gt road boundaries
        gt_path = os.path.join(gt_dir, f'{sample_token}.json')
        with open(gt_path, 'r') as f:
            gt_data = json.load(f)
        gt_boundaries = [
            bbox for bbox, label in zip(gt_data['bboxes'], gt_data['labels'])
            if label == 2
        ]
        gt_vectorized_map = np.array(gt_boundaries)
        
        # load osm map
        MAP_ORIGINS = {
            'boston-seaport': [42.336849169438615, -71.05785369873047],
            'singapore-onenorth': [1.2882100868743724, 103.78475189208984],
            'singapore-hollandvillage': [1.2993652317780957, 103.78217697143555],
            'singapore-queenstown': [1.2782562240223188, 103.76741409301758]
        }
        map_path = f'data/lanelet2_for_nuScenes/{location}.osm'
        origin = Origin(MAP_ORIGINS[location][0], MAP_ORIGINS[location][1])
        projector = UtmProjector(origin)
        
        lanelet_map = lanelet2.io.load(map_path, projector)
        traffic_rules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany, 
                                            lanelet2.traffic_rules.Participants.Vehicle)
        lanelet_graph = lanelet2.routing.RoutingGraph(lanelet_map, traffic_rules)
        
        ''' Step 1: get lanelets '''
        # get lanelets in the region
        lanelets = find_lanelets_in_region(lanelet_map, sample_token, x_range=SEARCH_X_RANGE, y_range=SEARCH_Y_RANGE)
        # get ego adjacent lanelets
        ego_adjacent_lanelets = get_adjacent_lanelets(lanelets, sample_token)
        
        ''' Step 2: find valid routes '''
        graph = create_lane_graph(lanelets, lanelet_graph)
        
        start_lanelets = ego_adjacent_lanelets
        routes = find_all_routes(start_lanelets, graph, gt_vectorized_map, sample_token, max_depth=MAX_ROUTE_DEPTH)
        print(f'{sample_token} valid routes: {len(routes)}')
        valid_routes_num.append(len(routes))
        
        ''' Step 4: Convert routes to point sequences '''
        route_center_points = convert_routes_to_points(routes, gt_vectorized_map, sample_token, interval=ROUTE_INTERVAL, x_range=ROUTE_X_RANGE, y_range=ROUTE_Y_RANGE, dist2boundary=DIST2BOUNDARY)
        
        ''' Step 5: Select goal points '''
        selected_goals = select_goal_states(route_center_points, interval=ROUTE_INTERVAL, n=NUM_GOALS, dist2range=DIST2RANGE)
        
        # Plot routes
        # plot the route center points
        plt.figure(figsize=(4,8))
        for gt in gt_vectorized_map:
            plt.plot(gt[:, 0], gt[:, 1], 'k-', alpha=0.7)
        for route_idx, center_points in enumerate(route_center_points):
            plt.plot(center_points[:, 0], center_points[:, 1], '-', alpha=0.7)
            for goal_point, goal_heading in selected_goals:
                plt.scatter(center_points[0, 0], center_points[0, 1], color='green', s=40, label='Start')
                plt.scatter(goal_point[0], goal_point[1], color='red', s=40, label='Goal')
        plt.xlim(-15, 15)
        plt.ylim(-30, 30)
        plt.savefig(f'{goal_points_dir}/{sample_token}.png')
        plt.close()
        
        ''' Step 6: Save the selected goals '''
        # combine goal_point and goal_heading into a list
        output_goals = []
        for goal_point, goal_heading in selected_goals:
            goal_point[-1] = goal_heading  # replace z to heading
            output_goals.append(goal_point.tolist())
        selected_goals_path = os.path.join(goal_points_dir, f'{sample_token}.json')
        with open(selected_goals_path, 'w') as f:
            json.dump(output_goals, f)
    
    print('average valid routes num: ', np.mean(valid_routes_num))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate planning goals for nuScenes dataset")
    parser.add_argument("--dataset_tag", type=str, default="asymmetric", help="Dataset tag")
    parser.add_argument("--model", type=str, default="maptr-bevpool", help="Model name")
    args = parser.parse_args()
    
    main(args)