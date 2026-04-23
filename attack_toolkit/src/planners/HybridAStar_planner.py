#!/usr/bin/env python
"""
HybridAStar Trajectory Planner for Adversarial Map Attack Evaluation

This module implements a Hybrid A* path planner used to evaluate the impact of 
adversarial attacks on HD map perception. The planner takes predicted road boundaries
from attacked/clean models and generates safe trajectories for autonomous vehicles.

Usage:
    python HybridAStar_planner.py --dataset asymmetric --root_dir results/ 
                                   --gt_traj_dir gt/ --clean_traj_dir clean/
"""

import os
import sys
import json
import multiprocessing
from pathlib import Path
from functools import partial
from typing import List, Tuple
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from shapely.geometry import Point, Polygon
from mmcv import Config, DictAction

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from attack_toolkit.src.utils.utils_plan import *
from attack_toolkit.src.planners.HybridAStar.hybrid_a_star import hybrid_a_star_planning


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default=None)
parser.add_argument('--root_dir', type=str, default=None)
parser.add_argument('--gt_traj_dir', type=str, default=None)
parser.add_argument('--clean_traj_dir', type=str, default=None)
parser.add_argument('--attack_traj_dir', type=str, default=None)
parser.add_argument(
    '--attack-options',
    nargs='+',
    action=DictAction,
    help='override some settings in the attack config, the key-value pair '
    'in xxx=yyy format will be merged into config file. If the value to '
    'be overridden is a list, it should be like key="[a,b]" or key=a,b '
    'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
    'Note that the quotation marks are necessary and that no white space '
    'is allowed.')
parser.add_argument('--collision_threshold', type=float, default=0.5)
parser.add_argument('--vis', action='store_true', default=False)
args = parser.parse_args()


@dataclass
class VehicleState:
    """
    Represents vehicle state in Bird's Eye View (BEV) coordinates.
    
    This class encapsulates the kinematic state of an autonomous vehicle for
    trajectory planning purposes. Coordinates follow NuScenes convention:
    - x: lateral position (positive = right)
    - y: longitudinal position (positive = forward)
    - heading: yaw angle (0 = forward, positive = counterclockwise)
    
    Attributes:
        x: Lateral position in meters
        y: Longitudinal position in meters  
        heading: Vehicle heading angle in radians [0, 2π]
        velocity: Forward velocity in m/s
        steering_angle: Steering angle in radians [-max_steer, +max_steer]
    """
    x: float  # [m] lateral position
    y: float  # [m] longitudinal position
    heading: float  # [rad] vehicle yaw angle
    velocity: float  # [m/s] forward velocity
    steering_angle: float = 0.0  # [rad] steering angle


class HybridAStarPlanner:
    """Hybrid A* planner for autonomous vehicle trajectory planning."""
    
    def __init__(self, CONFIG: dict):
        self.CONFIG = CONFIG
        self.road_network = None
        self.timeout = CONFIG.get('planning', {}).get('timeout', 120)  # Default 2 minutes
        
    def initialize_map(self, vectorized_map: np.ndarray) -> None:
        """Initialize road network from vectorized map."""
        self.road_network = vectorized_map
        
    def _get_obstacle_coordinates(self) -> Tuple[List[float], List[float]]:
        """
        Convert occupancy grid to obstacle coordinates for HybridAStar.
        
        Returns:
            ox: List of x coordinates of obstacles
            oy: List of y coordinates of obstacles
        """
        if self.road_network is None:
            raise ValueError("Road network not initialized")
            
        ox, oy = [], []
        
        # Add road boundaries from map
        boundaries = np.array(self.road_network).reshape(-1, 2)
        ox = boundaries[:, 0].tolist()
        oy = boundaries[:, 1].tolist()
        
        # Add vertical and horizontal edges
        x_min, x_max = -16, 16
        y_min, y_max = -31, 31
        resolution = 1  # [m] obstacle sampling interval
        
        for y in np.arange(y_min, y_max + resolution, resolution):
            ox.append(x_min)
            oy.append(y)
            ox.append(x_max)
            oy.append(y)
            
        for x in np.arange(x_min, x_max + resolution, resolution):
            ox.append(x)
            oy.append(y_min)
            ox.append(x)
            oy.append(y_max)
            
        # Remove obstacles inside vehicle region
        vehicle_width = self.CONFIG['vehicle']['width']
        vehicle_front = self.CONFIG['vehicle']['front_length'] 
        vehicle_back = self.CONFIG['vehicle']['back_length']
        
        # Calculate vehicle corners
        vehicle_corners = [
            [-vehicle_width/2, -vehicle_back],  # rear left
            [vehicle_width/2, -vehicle_back],   # rear right
            [vehicle_width/2, vehicle_front],   # front right
            [-vehicle_width/2, vehicle_front]   # front left
        ]
        
        # Create vehicle polygon and filter obstacles
        vehicle_polygon = Polygon(vehicle_corners)
        filtered_points = []
        for x, y in zip(ox, oy):
            point = Point(x, y)
            if not vehicle_polygon.contains(point):
                filtered_points.append((x, y))
                
        # Update obstacle lists
        if filtered_points:
            ox, oy = zip(*filtered_points)
            ox = list(ox)
            oy = list(oy)
            
        # Add artificial barrier to prevent planning in negative y
        barrier_y = -1  # Slightly below y=0 to allow the vehicle to start at y=0
        for x in range(int(self.CONFIG['grid']['x_range'][0]), int(self.CONFIG['grid']['x_range'][1]) + 1):
            ox.append(x)
            oy.append(barrier_y)

        return ox, oy
    
    def _smooth_trajectory(self, trajectory, weight_data=0.5, weight_smooth=0.3, tolerance=0.00001):
        """
        Apply path smoothing to the planned trajectory
        
        Args:
            trajectory: Original trajectory points
            weight_data: Weight for keeping trajectory close to original data
            weight_smooth: Weight for smoothing
            tolerance: Convergence tolerance
            
        Returns:
            Smoothed trajectory
        """
        if len(trajectory) <= 2:
            return trajectory
            
        smoothed = np.copy(trajectory)
        change = tolerance + 1
        
        while change > tolerance:
            change = 0
            for i in range(1, len(trajectory) - 1):
                for j in range(len(trajectory[0])):
                    aux = smoothed[i][j]
                    
                    # Apply smoothing formula
                    smoothed[i][j] = (weight_data * trajectory[i][j] + 
                                    weight_smooth * (smoothed[i-1][j] + smoothed[i+1][j])) / \
                                    (weight_data + 2 * weight_smooth)
                    
                    change += abs(aux - smoothed[i][j])
                    
        return smoothed
        
    def _plan_trajectory_worker(self, current_state, goal_state) -> np.ndarray:
        """
        Plan trajectory using HybridAStar.
        
        Args:
            current_state: Current vehicle state
            goal_state: Goal state [x, y, heading]
            
        Returns:
            Array of shape (n_steps, 3) containing x, y, heading
        """
        try:
            # Get obstacle coordinates
            ox, oy = self._get_obstacle_coordinates()
            
            # Get start and goal states
            start = [current_state.x, current_state.y, current_state.heading]
            goal = [goal_state[0], goal_state[1], goal_state[2]]
            
            # Run HybridAStar planning
            path = hybrid_a_star_planning(
                start=start,
                goal=goal,
                ox=ox,
                oy=oy,
                xy_resolution=self.CONFIG['grid']['xy_resolution'],
                yaw_resolution=self.CONFIG['grid']['yaw_resolution'],
                grid_params=self.CONFIG['grid'],
                cost_params=self.CONFIG['costs'],
                goal_params=self.CONFIG['goal']
            )

            if path is None:
                return np.array([])
            else:
                # Convert path to trajectory format
                path_length = len(path.x_list)
                trajectory = np.zeros((path_length, 3))
                
                # Interpolate path to match desired timesteps
                for i in range(path_length):
                    trajectory[i] = [path.x_list[i], path.y_list[i], path.yaw_list[i]]
                
                trajectory = self._smooth_trajectory(trajectory)
            
            return trajectory
                
        except Exception as e:
            print(f'Error during planning: {e}')
            return np.array([])
        
    def plan_trajectory(self, current_state, goal_state) -> np.ndarray:
        """
        Plan trajectory using HybridAStar with timeout.
        
        Args:
            current_state: Current vehicle state
            goal_state: Goal state [x, y, heading]
            
        Returns:
            Array of shape (n_steps, 3) containing x, y, heading
            Returns empty array if planning fails or times out
        """
        
        # Use spawn context to avoid issues with CUDA/GPU resources
        ctx = multiprocessing.get_context('spawn')
        with ctx.Pool(1) as pool:
            try:
                # Create the worker function with arguments
                worker = partial(self._plan_trajectory_worker, current_state, goal_state)
                
                # Start the planning process with timeout
                result = pool.apply_async(worker)
                
                # Wait for result with timeout
                trajectory = result.get(timeout=self.timeout)
                
                return trajectory
                
            except multiprocessing.TimeoutError:
                print(f'Planning timed out after {self.timeout} seconds')
                pool.terminate()  # Terminate the process
                return np.array([])
            except Exception as e:
                print(f'Error during planning: {e}')
                return np.array([])
            finally:
                pool.terminate()  # Ensure process is terminated


def process_boundaries(result_vectors, confidence_threshold=0.3, num_points=50):
    """
    Process and filter boundaries from prediction results.
    
    Args:
        result_vectors: List of vector predictions
        confidence_threshold: Minimum confidence for boundary filtering
        num_points: Number of points to sample for each boundary
        
    Returns:
        Processed vectorized map as numpy array
    """
    boundaries = [
        {
            'pts': np.array(vector['pts']),
            'confidence': vector['confidence_level']
        }
        for vector in result_vectors
        if vector['cls_name'] == 'boundary' and vector['confidence_level'] > confidence_threshold
    ]
    
    # Apply non-maximum suppression
    kept_boundaries = []
    boundaries.sort(key=lambda x: x['confidence'], reverse=True)
    
    while boundaries:
        best = boundaries.pop(0)
        kept_boundaries.append(best['pts'])
        
        # Filter out overlapping boundaries
        if len(boundaries) > 1:
            boundaries = [
                boundary for boundary in boundaries
                if calculate_boundary_iou(best['pts'], boundary['pts']) < 0.5
            ]
    
    vectorized_map = np.array(kept_boundaries)
    return sample_boundaries_fixed_num(vectorized_map, num_points=num_points)


def main():
    """Main function to run the planner."""
    
    # Load configuration
    cfg = Config(load_config())
    
    cfg.attack = {}
    if args.attack_options is not None:
        for key, value in args.attack_options.items():
            keys = key.split('.')
            cfg.attack[keys[-1]] = value

    # Initialize planner
    planner = HybridAStarPlanner(CONFIG=cfg)

    # Initialize current state
    current_state = VehicleState(
        x=0.0,              # Initial x position [m]
        y=0.0,              # Initial y position [m] 
        heading=1.57,       # Initial heading [rad]
        velocity=5.0,       # Initial velocity [m/s]
        steering_angle=0.0  # Initial steering angle [rad]
    )

    # Initialize output directories
    root_dir = args.root_dir
    map_results_dir = os.path.join(root_dir, 'results', 'map')

    gt_dir = os.path.join(map_results_dir, 'gt')
    clean_dir = os.path.join(map_results_dir, 'clean')
    attack_dir = os.path.join(map_results_dir, 'attack')

    output_dir = os.path.join(root_dir, 'results', 'planning')
    gt_output_dir = os.path.join(output_dir, 'gt')
    clean_output_dir = os.path.join(output_dir, 'clean')
    attack_output_dir = os.path.join(output_dir, 'attack')
    vis_dir = os.path.join(output_dir, 'vis')

    setup_directories([output_dir, gt_output_dir, clean_output_dir, attack_output_dir, vis_dir])

    # Load prediction results
    with open(os.path.join(clean_dir, 'pts_bbox/nuscmap_results.json'), 'r') as f:
        clean_results = json.load(f)['results']
        
    with open(os.path.join(attack_dir, 'pts_bbox/nuscmap_results.json'), 'r') as f:
        attack_results = json.load(f)['results']

    # Initialize lists to store sample tokens
    gt_safe_tokens = []
    gt_failure_tokens = []
    gt_collision_tokens = []

    clean_safe_tokens = []  # Successfully planned and collision-free
    clean_failure_tokens = []  # No valid trajectory found
    clean_collision_tokens = []  # Valid trajectory but with collision

    attack_safe_tokens = []
    attack_failure_tokens = []
    attack_collision_tokens = []

    ades_clean_attack = []
    fdes_clean_attack = []
    ades_gt_clean = []
    fdes_gt_clean = []
    ades_gt_attack = []
    fdes_gt_attack = []

    for idx, (clean_result, attack_result) in tqdm(enumerate(zip(clean_results, attack_results))):
        # Get scene location
        sample_token = clean_result['sample_token']

        # Process predicted boundaries
        clean_vectorized_map = process_boundaries(clean_result['vectors'])
        attack_vectorized_map = process_boundaries(attack_result['vectors'])

        # Load ground truth
        gt_path = os.path.join(gt_dir, f'{sample_token}.json')
        with open(gt_path, 'r') as f:
            gt_data = json.load(f)
        gt_boundaries = [
            bbox for bbox, label in zip(gt_data['bboxes'], gt_data['labels'])
            if label == 2
        ]
        gt_vectorized_map = np.array(gt_boundaries)
        gt_vectorized_map = sample_boundaries_fixed_num(gt_vectorized_map, num_points=50)
        
        # Load goal point
        goal_path = Path(root_dir).parent.parent / f'goal_states_{args.dataset}_lanelet2' / f'{sample_token}.json'
        with open(goal_path, 'r') as f:
            goal_states = json.load(f)


        ''' Plan and evaluate trajectories '''
        ### GT case ###
        gt_traj_path = os.path.join(args.gt_traj_dir, f'{sample_token}.json') if args.gt_traj_dir else None
        if gt_traj_path and os.path.exists(gt_traj_path):
            with open(gt_traj_path, 'r') as f:
                gt_trajectory_list = json.load(f)
            gt_trajectory_list = [np.array(traj) for traj in gt_trajectory_list]
        else:
            gt_trajectory_list = []
            for goal_idx, goal_state in enumerate(goal_states):
                planner.initialize_map(gt_vectorized_map)
                gt_trajectory = planner.plan_trajectory(
                    current_state=current_state,
                    goal_state=goal_state,
                )
                gt_trajectory_list.append(gt_trajectory)

        # Evaluate gt trajectory
        for goal_idx, gt_trajectory in enumerate(gt_trajectory_list):
            if len(gt_trajectory) == 0:
                gt_failure_tokens.append((sample_token, goal_idx))
            else:
                gt_safe_tokens.append((sample_token, goal_idx))
                
        
        ## Clean case ###
        clean_traj_path = os.path.join(args.clean_traj_dir, f'{sample_token}.json') if args.clean_traj_dir else None
        if clean_traj_path and os.path.exists(clean_traj_path):
            with open(clean_traj_path, 'r') as f:
                clean_trajectory_list = json.load(f)
            clean_trajectory_list = [np.array(traj) for traj in clean_trajectory_list]
        else:
            clean_trajectory_list = []
            for goal_idx, goal_state in enumerate(goal_states):
                planner.initialize_map(clean_vectorized_map)
                clean_trajectory = planner.plan_trajectory(
                    current_state=current_state,
                    goal_state=goal_state,
                )
                clean_trajectory_list.append(clean_trajectory)
        
        # Evaluate clean trajectory
        for goal_idx, clean_trajectory in enumerate(clean_trajectory_list):
            if len(clean_trajectory) == 0:
                clean_failure_tokens.append((sample_token, goal_idx))
            else:
                if check_trajectory_collision(trajectory=clean_trajectory, 
                                            ground_truth_boundaries=gt_vectorized_map, 
                                            collision_threshold=args.collision_threshold):
                    clean_collision_tokens.append((sample_token, goal_idx))
                else:
                    clean_safe_tokens.append((sample_token, goal_idx))
                
        
        ### Attack case ###
        attack_traj_path = os.path.join(args.attack_traj_dir, f'{sample_token}.json') if args.attack_traj_dir else None
        if attack_traj_path and os.path.exists(attack_traj_path):
            with open(attack_traj_path, 'r') as f:
                attack_trajectory_list = json.load(f)
            attack_trajectory_list = [np.array(traj) for traj in attack_trajectory_list]
        else:
            attack_trajectory_list = []
            for goal_idx, goal_state in enumerate(goal_states):
                planner.initialize_map(attack_vectorized_map)
                attack_trajectory = planner.plan_trajectory(
                    current_state=current_state,
                    goal_state=goal_state,
                )
                attack_trajectory_list.append(attack_trajectory)

        # Evaluate attack trajectory
        for goal_idx, attack_trajectory in enumerate(attack_trajectory_list):
            if len(attack_trajectory) == 0:
                attack_failure_tokens.append((sample_token, goal_idx))
            else:
                if check_trajectory_collision(trajectory=attack_trajectory, 
                                            ground_truth_boundaries=gt_vectorized_map, 
                                            collision_threshold=args.collision_threshold):
                    attack_collision_tokens.append((sample_token, goal_idx))
                else:
                    attack_safe_tokens.append((sample_token, goal_idx))
        
        
        ''' Save planned trajectory '''
        gt_trajectory_list = [traj.tolist() for traj in gt_trajectory_list if isinstance(traj, np.ndarray)]
        gt_traj_path = os.path.join(gt_output_dir, f'{sample_token}.json')
        with open(gt_traj_path, 'w') as f:
            json.dump(gt_trajectory_list, f)
        
        clean_trajectory_list = [traj.tolist() for traj in clean_trajectory_list if isinstance(traj, np.ndarray)]
        clean_traj_path = os.path.join(clean_output_dir, f'{sample_token}.json')
        with open(clean_traj_path, 'w') as f:
            json.dump(clean_trajectory_list, f)
        
        attack_trajectory_list = [traj.tolist() for traj in attack_trajectory_list if isinstance(traj, np.ndarray)]
        attack_traj_path = os.path.join(attack_output_dir, f'{sample_token}.json')
        with open(attack_traj_path, 'w') as f:
            json.dump(attack_trajectory_list, f)
        
            
        ''' Calculate planning metrics '''
        for gt_trajectory, clean_trajectory, attack_trajectory in zip(gt_trajectory_list, clean_trajectory_list, attack_trajectory_list):
            if len(gt_trajectory) > 0 and len(clean_trajectory) > 0 and len(attack_trajectory) > 0:
                handle_empty = 'zero_fill'

                ade_gt_clean = calculate_ade(gt_trajectory, clean_trajectory, handle_empty=handle_empty)
                fde_gt_clean = calculate_fde(gt_trajectory, clean_trajectory, handle_empty=handle_empty)
                ade_gt_attack = calculate_ade(gt_trajectory, attack_trajectory, handle_empty=handle_empty)
                fde_gt_attack = calculate_fde(gt_trajectory, attack_trajectory, handle_empty=handle_empty)
                
                ades_gt_attack.append(ade_gt_attack)
                fdes_gt_attack.append(fde_gt_attack)
                ades_gt_clean.append(ade_gt_clean)
                fdes_gt_clean.append(fde_gt_clean)
                
        for clean_trajectory, attack_trajectory in zip(clean_trajectory_list, attack_trajectory_list):
            if len(clean_trajectory) > 0 and len(attack_trajectory) > 0:
                handle_empty = 'zero_fill'

                ade_clean_attack = calculate_ade(clean_trajectory, attack_trajectory, handle_empty=handle_empty)
                fde_clean_attack = calculate_fde(clean_trajectory, attack_trajectory, handle_empty=handle_empty)
                
                ades_clean_attack.append(ade_clean_attack)
                fdes_clean_attack.append(fde_clean_attack)
        
        ''' Visualize results '''
        if args.vis:
            fig, axs = plt.subplots(1, 3, figsize=(12, 6))

            # Color & Style Definitions
            start_color = 'green'
            goal_color = 'purple'
            gt_boundary_color = 'blue'
            clean_boundary_color = 'orange'
            attack_boundary_color = 'orange'
            trajectory_line_style = '--'
            gt_boundary_line_style = '-'
            clean_boundary_line_style = '-'
            attack_boundary_line_style = '-'

            # Colormap for Trajectories
            trajectory_colors = plt.cm.Reds(np.linspace(0.4, 0.8, len(goal_states)))

            # Plot start and goal points
            for ax in axs.flat:
                # Start point as circle
                ax.scatter(current_state.x, current_state.y, color=start_color, s=80, label='Start', zorder=5)

                # Goal points as stars
                for goal_idx, goal_state in enumerate(goal_states):
                    ax.scatter(goal_state[0], goal_state[1], color=goal_color, marker='*', s=150, label='Goal' if goal_idx == 0 else None, zorder=5)

                # GT boundaries (solid blue)
                for i, boundary in enumerate(gt_vectorized_map):
                    label = 'GT Boundaries' if i == 0 else None
                    ax.plot(boundary[:, 0], boundary[:, 1], color=gt_boundary_color, linestyle=gt_boundary_line_style, linewidth=2.5, label=label, alpha=0.7)

            # Plot GT Trajectories
            for i, gt_trajectory in enumerate(gt_trajectory_list):
                traj = [state for state in gt_trajectory if state[0] != 0 and state[1] != 0]
                if traj:
                    traj_x = [state[0] for state in traj]
                    traj_y = [state[1] for state in traj]
                    axs[0].plot(traj_x, traj_y, color=trajectory_colors[i], linestyle=trajectory_line_style, linewidth=2, label=f'GT Trajectory {i+1}')
            axs[0].set_title('GT Scenario')

            # Plot Clean Boundaries & Trajectories
            for i, boundary in enumerate(clean_vectorized_map):
                label = 'Clean Boundaries' if i == 0 else None
                axs[1].plot(boundary[:, 0], boundary[:, 1], color=clean_boundary_color, linestyle=clean_boundary_line_style, linewidth=2, label=label)

            for i, clean_trajectory in enumerate(clean_trajectory_list):
                traj = [state for state in clean_trajectory if state[0] != 0 and state[1] != 0]
                if traj:
                    traj_x = [state[0] for state in traj]
                    traj_y = [state[1] for state in traj]
                    axs[1].plot(traj_x, traj_y, color=trajectory_colors[i], linestyle=trajectory_line_style, linewidth=2, label=f'Clean Trajectory {i+1}')
            axs[1].set_title('Clean Scenario')

            # Plot Attack Boundaries & Trajectories
            for i, boundary in enumerate(attack_vectorized_map):
                label = 'Attack Boundaries' if i == 0 else None
                axs[2].plot(boundary[:, 0], boundary[:, 1], color=attack_boundary_color, linestyle=attack_boundary_line_style, linewidth=2, label=label)

            for i, attack_trajectory in enumerate(attack_trajectory_list):
                traj = [state for state in attack_trajectory if state[0] != 0 and state[1] != 0]
                if traj:
                    traj_x = [state[0] for state in traj]
                    traj_y = [state[1] for state in traj]
                    axs[2].plot(traj_x, traj_y, color=trajectory_colors[i], linestyle=trajectory_line_style, linewidth=2, label=f'Attack Trajectory {i+1}')
            axs[2].set_title('Attack Scenario')

            # Common Plot Adjustments
            for ax in axs:
                ax.set_xlim(-15, 15)
                ax.set_ylim(-30, 30)
                ax.set_xlabel('X [m]')
                ax.set_ylabel('Y [m]')
                ax.legend(loc='lower left', fontsize='medium')
                ax.grid(True)

            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f'{sample_token}.png'))
            print(f'saved to {os.path.join(vis_dir, f"{sample_token}.png")}')
            plt.close()
    

    ''' Calculate and display summary statistics '''
    # Get unique sample tokens for each category
    clean_failure_samples = set(token for token, _ in clean_failure_tokens)
    clean_collision_samples = set(token for token, _ in clean_collision_tokens)
    attack_failure_samples = set(token for token, _ in attack_failure_tokens)
    attack_collision_samples = set(token for token, _ in attack_collision_tokens)
    
    # Calculate metrics
    total_samples = len(set(token for token, _ in clean_safe_tokens + clean_failure_tokens + clean_collision_tokens))
    clean_unreachable_goal_rate = len(clean_failure_samples) / total_samples
    attack_unreachable_goal_rate = len(attack_failure_samples) / total_samples
    clean_unsafe_trajectory_rate = len(clean_collision_samples) / total_samples
    attack_unsafe_trajectory_rate = len(attack_collision_samples) / total_samples
    
    print(f'\n{"="*60}')
    print(f'Evaluate {cfg.attack.loss} attack using {cfg.attack.type} in {args.dataset} dataset.')
    print(f'{"="*60}')
    
    # Main results based on attack type
    print(f'\n--- MAIN ATTACK EVALUATION METRICS ---')
    if cfg.attack.loss == 'rsa':
        print(f'Unreachable Goal Rate (UGR):')
        print(f'  Clean:  {clean_unreachable_goal_rate*100:.1f}%')
        print(f'  Attack: {attack_unreachable_goal_rate*100:.1f}%')
        print(f'  Increase: +{(attack_unreachable_goal_rate - clean_unreachable_goal_rate)*100:.1f}%')
    elif cfg.attack.loss == 'eta':
        print(f'Unsafe Planned Trajectory Rate (UPTR):')
        print(f'  Clean:  {clean_unsafe_trajectory_rate*100:.1f}%')
        print(f'  Attack: {attack_unsafe_trajectory_rate*100:.1f}%')
        print(f'  Increase: +{(attack_unsafe_trajectory_rate - clean_unsafe_trajectory_rate)*100:.1f}%')
    else:
        raise ValueError(f'Invalid attack loss: {cfg.attack.loss}')
    
    # Add detailed metrics
    print(f'\n--- DETAILED METRICS ---')
    print(f'Unreachable Goal Rate - Clean: {clean_unreachable_goal_rate*100:.1f}%, Attack: {attack_unreachable_goal_rate*100:.1f}%')
    print(f'Unsafe Planned Trajectory Rate - Clean: {clean_unsafe_trajectory_rate*100:.1f}%, Attack: {attack_unsafe_trajectory_rate*100:.1f}%')
    
    if ades_clean_attack and ades_gt_clean and ades_gt_attack:
        print(f'\nTrajectory Deviation Metrics:')
        print(f'  ADE (Clean vs Attack): {np.mean(ades_clean_attack):.2f} ± {np.std(ades_clean_attack):.2f}')
        print(f'  ADE (GT vs Attack): {np.mean(ades_gt_attack):.2f} ± {np.std(ades_gt_attack):.2f}')
        print(f'  ADE (GT vs Clean): {np.mean(ades_gt_clean):.2f} ± {np.std(ades_gt_clean):.2f}')
        
        print(f'  FDE (Clean vs Attack): {np.mean(fdes_clean_attack):.2f} ± {np.std(fdes_clean_attack):.2f}')
        print(f'  FDE (GT vs Attack): {np.mean(fdes_gt_attack):.2f} ± {np.std(fdes_gt_attack):.2f}')
        print(f'  FDE (GT vs Clean): {np.mean(fdes_gt_clean):.2f} ± {np.std(fdes_gt_clean):.2f}')
    
    print(f'{"="*60}\n')
    
    # Save detailed summary to file
    with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
        f.write('PLANNING EVALUATION RESULTS \n')
        f.write('='*60 + '\n')
        f.write(f'Attack Type: {cfg.attack.type}\n')
        f.write(f'Attack Loss: {cfg.attack.loss}\n')
        f.write(f'Dataset: {args.dataset}\n')
        
        f.write('Clean Setting Statistics:\n')
        f.write(f'Unreachable Goal Rate: {clean_unreachable_goal_rate*100:.1f}%\n')
        f.write(f'Unsafe Planned Trajectory Rate: {clean_unsafe_trajectory_rate*100:.1f}%\n\n')

        f.write('Attack Setting Statistics:\n')
        f.write(f'Unreachable Goal Rate: {attack_unreachable_goal_rate*100:.1f}%\n')
        f.write(f'Unsafe Planned Trajectory Rate: {attack_unsafe_trajectory_rate*100:.1f}%\n\n')

        if ades_clean_attack and ades_gt_clean and ades_gt_attack:
            f.write('Trajectory Deviation Metrics (m):\n')
            f.write(f'ADE Clean vs Attack: {np.mean(ades_clean_attack):.2f} ± {np.std(ades_clean_attack):.2f}\n')
            f.write(f'ADE GT vs Attack: {np.mean(ades_gt_attack):.2f} ± {np.std(ades_gt_attack):.2f}\n')
            f.write(f'ADE GT vs Clean: {np.mean(ades_gt_clean):.2f} ± {np.std(ades_gt_clean):.2f}\n')
            
            f.write(f'FDE Clean vs Attack: {np.mean(fdes_clean_attack):.2f} ± {np.std(fdes_clean_attack):.2f}\n')
            f.write(f'FDE GT vs Attack: {np.mean(fdes_gt_attack):.2f} ± {np.std(fdes_gt_attack):.2f}\n')
            f.write(f'FDE GT vs Clean: {np.mean(fdes_gt_clean):.2f} ± {np.std(fdes_gt_clean):.2f}\n')
        
    # Save summary statistics to JSON file with updated terminology
    results_dict = {
        "unreachable_goal_rate": {
            "clean": float(clean_unreachable_goal_rate),
            "attack": float(attack_unreachable_goal_rate),
            "increase": float(attack_unreachable_goal_rate - clean_unreachable_goal_rate)
        },
        "unsafe_planned_trajectory_rate": {
            "clean": float(clean_unsafe_trajectory_rate),
            "attack": float(attack_unsafe_trajectory_rate),
            "increase": float(attack_unsafe_trajectory_rate - clean_unsafe_trajectory_rate)
        }
    }
    
    # Add trajectory deviation metrics if available
    if ades_clean_attack and ades_gt_clean and ades_gt_attack:
        results_dict["trajectory_deviation"] = {
            "ade": {
                "clean_vs_attack_mean": float(np.mean(ades_clean_attack)),
                "clean_vs_attack_std": float(np.std(ades_clean_attack))
            },
            "fde": {
                "clean_vs_attack_mean": float(np.mean(fdes_clean_attack)),
                "clean_vs_attack_std": float(np.std(fdes_clean_attack))
            }
        }
        results_dict["trajectory_deviation"]["ade"].update({
            "gt_vs_clean_mean": float(np.mean(ades_gt_clean)),
            "gt_vs_clean_std": float(np.std(ades_gt_clean)),
            "gt_vs_attack_mean": float(np.mean(ades_gt_attack)),
            "gt_vs_attack_std": float(np.std(ades_gt_attack))
        })
        results_dict["trajectory_deviation"]["fde"].update({
            "gt_vs_clean_mean": float(np.mean(fdes_gt_clean)),
            "gt_vs_clean_std": float(np.std(fdes_gt_clean)),
            "gt_vs_attack_mean": float(np.mean(fdes_gt_attack)),
            "gt_vs_attack_std": float(np.std(fdes_gt_attack))
        })
    
    with open(os.path.join(output_dir, 'summary.json'), 'w') as json_file:
        json.dump(results_dict, json_file, indent=4)
    
    
if __name__ == '__main__':
    main()
