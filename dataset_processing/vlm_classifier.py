"""
VLM-based classifier for refining asymmetric scene classification.

This module processes GPT-4V responses to further filter asymmetric scenes.
"""
import os
import json
import random
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from matplotlib.patches import Rectangle
from PIL import Image

from .config import *
from .geometry_utils import lidar_to_img
from .curvature_analysis import identify_diverging_boundary_improved
from .vlm_client import OpenAIVLMClient, VLMSceneClassifier


class VLMClassifier:
    """VLM-based classifier for final scene refinement."""
    
    def __init__(self, nusc, car_img, vlm_client=None):
        """
        Initialize the VLM classifier.
        
        Args:
            nusc: NuScenes instance
            car_img: Car image for visualization
            vlm_client: Optional VLM client for scene classification
        """
        self.nusc = nusc
        self.car_img = car_img
        self.vlm_scene_classifier = None
        if vlm_client is not None:
            self.vlm_scene_classifier = VLMSceneClassifier(vlm_client)
        
    def get_unique_scene_tokens(self, sample_tokens: List[str]) -> Dict[str, List[str]]:
        """Get mapping of scene tokens to their sample tokens."""
        scene_to_samples = {}
        for sample_token in sample_tokens:
            scene_token = self.nusc.get('sample', sample_token)['scene_token']
            if scene_token not in scene_to_samples:
                scene_to_samples[scene_token] = []
            scene_to_samples[scene_token].append(sample_token)
        return scene_to_samples
    
    def sample_from_scenes(self, scene_to_samples: Dict[str, List[str]], target_count: int) -> set:
        """Sample tokens trying to maximize scene diversity."""
        unique_scenes = list(scene_to_samples.keys())
        num_scenes = len(unique_scenes)
        selected_samples = set()

        if num_scenes >= target_count:
            # Take one sample from each scene
            selected_scenes = random.sample(unique_scenes, target_count)
            for scene in selected_scenes:
                selected_samples.add(random.choice(scene_to_samples[scene]))
        else:
            # Take one sample from each available scene
            for scene in unique_scenes:
                selected_samples.add(random.choice(scene_to_samples[scene]))
            
            # Fill remaining slots with random samples
            remaining = target_count - len(selected_samples)
            all_remaining_samples = [
                sample for scene in unique_scenes 
                for sample in scene_to_samples[scene] 
                if sample not in selected_samples
            ]
            if remaining > 0 and all_remaining_samples:
                additional_samples = random.sample(all_remaining_samples, 
                                                min(remaining, len(all_remaining_samples)))
                selected_samples.update(additional_samples)
        
        return selected_samples
    
    def generate_selected_samples(self, sample_tokens_asymmetric: List[str], 
                                sample_tokens_symmetric: List[str]) -> Tuple[set, set]:
        """
        Generate selected samples for VLM processing.
        
        Args:
            sample_tokens_asymmetric: List of asymmetric sample tokens
            sample_tokens_symmetric: List of symmetric sample tokens
            
        Returns:
            Tuple of (selected_asymmetric, selected_symmetric)
        """
        target_count = len(sample_tokens_asymmetric)
        
        # Get scene mappings
        scene_samples_asymmetric = self.get_unique_scene_tokens(sample_tokens_asymmetric)
        scene_samples_symmetric = self.get_unique_scene_tokens(sample_tokens_symmetric)
        
        print(f'Total {len(scene_samples_asymmetric)} asymmetric scenes, '
              f'{len(scene_samples_symmetric)} symmetric scenes')
        
        # Sample tokens
        random_samples_asymmetric = self.sample_from_scenes(scene_samples_asymmetric, target_count)
        random_samples_symmetric = self.sample_from_scenes(scene_samples_symmetric, target_count)
        
        print(f'Selected {len(random_samples_asymmetric)} asymmetric samples and '
              f'{len(random_samples_symmetric)} symmetric samples')
        
        return random_samples_asymmetric, random_samples_symmetric
    
    def process_selected_samples(self, random_samples_asymmetric: set, 
                               random_samples_symmetric: set,
                               diverge_points_dict: Dict[str, List]) -> None:
        """
        Process selected samples and generate visualizations.
        
        Args:
            random_samples_asymmetric: Set of selected asymmetric samples
            random_samples_symmetric: Set of selected symmetric samples
            diverge_points_dict: Dictionary of diverge points
        """
        # Setup directories
        os.makedirs(SCENES_ASYMMETRIC_CURVATURE_SELECTED_DIR, exist_ok=True)
        os.makedirs(SCENES_SYMMETRIC_CURVATURE_SELECTED_DIR, exist_ok=True)
        
        all_samples = list(random_samples_asymmetric) + list(random_samples_symmetric)
        
        for sample_token in all_samples:
            # Determine source and destination directories
            if sample_token in random_samples_asymmetric:
                dst_dir = SCENES_ASYMMETRIC_CURVATURE_SELECTED_DIR
            else:
                dst_dir = SCENES_SYMMETRIC_CURVATURE_SELECTED_DIR
            
            # Find source directory
            src_dir = None
            if os.path.exists(f'{SCENES_ASYMMETRIC_DIST_DIR}/{sample_token}.json'):
                src_dir = SCENES_ASYMMETRIC_DIST_DIR
            elif os.path.exists(f'{SCENES_SYMMETRIC_DIST_DIR}/{sample_token}.json'):
                src_dir = SCENES_SYMMETRIC_DIST_DIR
            else:
                raise ValueError(f'Source file not found for {sample_token}')
            
            # Load and process data
            with open(f'{src_dir}/{sample_token}.json', 'r') as f:
                structured_data = json.load(f)
            
            # Update data with diverge points and boundary identification
            structured_data['diverge_points'] = diverge_points_dict.get(sample_token, [])
            structured_data['diverge_boundary_tag'] = identify_diverging_boundary_improved(
                np.array(structured_data['map_elements'][0]['coordinates']),
                np.array(structured_data['map_elements'][1]['coordinates'])
            )
            
            # Save updated data
            with open(f'{dst_dir}/{sample_token}.json', 'w') as f:
                json.dump(structured_data, f, indent=2)
            
            # Generate visualizations
            self._plot_map_visualization(structured_data, sample_token, dst_dir, diverge_points_dict)
            self._plot_camera_visualization(sample_token, dst_dir, diverge_points_dict)
    
    def run_vlm_classification(self, scene_dir: str, output_dir: str, 
                              scene_tokens: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Run VLM classification on scenes.
        
        Args:
            scene_dir: Directory containing scene visualizations
            output_dir: Directory to save VLM responses
            scene_tokens: Optional list of specific scene tokens to process
            
        Returns:
            Dictionary mapping scene tokens to classifications
        """
        if self.vlm_scene_classifier is None:
            raise ValueError("VLM client not initialized. Set vlm_client in constructor.")
        
        return self.vlm_scene_classifier.classify_scenes(scene_dir, output_dir, scene_tokens)
    
    def process_vlm_responses(self, vlm_response_dir: str) -> Tuple[List[str], List[str]]:
        """
        Process VLM responses to get final classification.
        
        Args:
            vlm_response_dir: Directory containing VLM responses
            
        Returns:
            Tuple of (final_asymmetric_tokens, final_symmetric_tokens)
        """
        # Get initial classifications from rule-based approach
        sample_tokens_asymmetric_vlm = [
            token.split('.')[0] for token in os.listdir(vlm_response_dir)
            if not token.startswith('._') and token.endswith('.json')
        ]
        
        sample_tokens_symmetric_curvature_selected = [
            token.split('.')[0] for token in os.listdir(SCENES_SYMMETRIC_CURVATURE_SELECTED_DIR)
            if token.endswith('.json') and not token.startswith('._')
        ]
        
        sample_tokens_asymmetric = []
        sample_tokens_symmetric = sample_tokens_symmetric_curvature_selected.copy()
        
        # Process asymmetric samples based on VLM responses
        for sample_token in sample_tokens_asymmetric_vlm:
            response_file = f"{sample_token}.json"
            response_path = os.path.join(vlm_response_dir, response_file)
            
            with open(response_path, 'r') as f:
                response = json.load(f)
                
            if response['classification'] == 'symmetric':
                sample_tokens_symmetric.append(sample_token)
            else:
                sample_tokens_asymmetric.append(sample_token)
        
        print(f'Total symmetric scenes: {len(sample_tokens_symmetric)}')
        print(f'Total asymmetric scenes: {len(sample_tokens_asymmetric)}')
        
        return sample_tokens_asymmetric, sample_tokens_symmetric
    
    def process_gpt_responses(self, gpt_response_dir: str) -> Tuple[List[str], List[str]]:
        """
        Process GPT-4V responses to get final classification.
        
        Args:
            gpt_response_dir: Directory containing GPT responses
            
        Returns:
            Tuple of (final_asymmetric_tokens, final_symmetric_tokens)
        """
        # Get initial classifications from rule-based approach
        sample_tokens_asymmetric_gpt = [
            token.split('.')[0] for token in os.listdir(gpt_response_dir)
            if not token.startswith('._') and token.endswith('.json')
        ]
        
        sample_tokens_symmetric_curvature_selected = [
            token.split('.')[0] for token in os.listdir(SCENES_SYMMETRIC_CURVATURE_SELECTED_DIR)
            if token.endswith('.json') and not token.startswith('._')
        ]
        
        sample_tokens_asymmetric = []
        sample_tokens_symmetric = sample_tokens_symmetric_curvature_selected.copy()
        
        # Process asymmetric samples based on GPT responses
        for sample_token in sample_tokens_asymmetric_gpt:
            response_file = f"{sample_token}.json"
            response_path = os.path.join(gpt_response_dir, response_file)
            
            with open(response_path, 'r') as f:
                response = json.load(f)
                
            if response['classification'] == 'symmetric':
                sample_tokens_symmetric.append(sample_token)
            else:
                sample_tokens_asymmetric.append(sample_token)
        
        print(f'Total symmetric scenes: {len(sample_tokens_symmetric)}')
        print(f'Total asymmetric scenes: {len(sample_tokens_asymmetric)}')
        
        return sample_tokens_asymmetric, sample_tokens_symmetric
    
    def create_final_dataset(self, sample_tokens_asymmetric: List[str], 
                           sample_tokens_symmetric: List[str],
                           final_sample_num: int = FINAL_SAMPLE_NUM) -> Tuple[List[str], List[str]]:
        """
        Create final dataset by randomly sampling from classified tokens.
        
        Args:
            sample_tokens_asymmetric: List of asymmetric tokens
            sample_tokens_symmetric: List of symmetric tokens
            final_sample_num: Number of samples to select for each class
            
        Returns:
            Tuple of (final_asymmetric_tokens, final_symmetric_tokens)
        """
        # Randomly select final samples
        final_sample_tokens_asymmetric = random.sample(sample_tokens_asymmetric, final_sample_num)
        final_sample_tokens_symmetric = random.sample(sample_tokens_symmetric, final_sample_num)
        
        # Save sample tokens to files
        with open(SAMPLE_TOKENS_ASYMMETRIC_PATH, 'w') as f:
            for token in final_sample_tokens_asymmetric:
                f.write(f'{token}\n')
                
        with open(SAMPLE_TOKENS_SYMMETRIC_PATH, 'w') as f:
            for token in final_sample_tokens_symmetric:
                f.write(f'{token}\n')
        
        # Copy files to final directories
        os.makedirs(SCENES_ASYMMETRIC_DIR, exist_ok=True)
        os.makedirs(SCENES_SYMMETRIC_DIR, exist_ok=True)
        
        for sample_token in final_sample_tokens_asymmetric:
            self._copy_files(sample_token, SCENES_ASYMMETRIC_DIR)
        for sample_token in final_sample_tokens_symmetric:
            self._copy_files(sample_token, SCENES_SYMMETRIC_DIR)
        
        return final_sample_tokens_asymmetric, final_sample_tokens_symmetric
    
    def _copy_files(self, sample_token: str, dst_dir: str):
        """Copy files for a sample token to destination directory."""
        if os.path.exists(f'{SCENES_ASYMMETRIC_CURVATURE_SELECTED_DIR}/{sample_token}.json'):
            src_dir = SCENES_ASYMMETRIC_CURVATURE_SELECTED_DIR
        elif os.path.exists(f'{SCENES_SYMMETRIC_CURVATURE_SELECTED_DIR}/{sample_token}.json'):
            src_dir = SCENES_SYMMETRIC_CURVATURE_SELECTED_DIR
        else:
            raise ValueError(f'No json file found for {sample_token}')
        
        shutil.copy(f'{src_dir}/{sample_token}.json', f'{dst_dir}/{sample_token}.json')
        shutil.copy(f'{src_dir}/{sample_token}_map.png', f'{dst_dir}/{sample_token}_map.png')
    
    def _plot_map_visualization(self, structured_data: Dict, sample_token: str, 
                              dst_dir: str, diverge_points_dict: Dict[str, List]):
        """Generate and save map visualization."""
        sampled_points_left = np.array(structured_data['map_elements'][0]['coordinates'])
        sampled_points_right = np.array(structured_data['map_elements'][1]['coordinates'])
        
        plt.figure(figsize=(6, 7))
        
        # Plot boundaries
        plt.plot(sampled_points_left[:, 0], sampled_points_left[:, 1], 'r-', label='Left Boundary')
        plt.plot(sampled_points_right[:, 0], sampled_points_right[:, 1], 'b-', label='Right Boundary')
        
        # Add labels
        if len(sampled_points_left) > 0:
            mid_idx_left = len(sampled_points_left) // 2
            plt.text(sampled_points_left[mid_idx_left, 0] - 4, 
                    sampled_points_left[mid_idx_left, 1], 
                    'Left', fontsize=18, color='red')
        
        if len(sampled_points_right) > 0:
            mid_idx_right = len(sampled_points_right) // 2
            plt.text(sampled_points_right[mid_idx_right, 0] + 1, 
                    sampled_points_right[mid_idx_right, 1], 
                    'Right', fontsize=18, color='blue')
        
        # Plot diverge points if they exist
        if sample_token in diverge_points_dict:
            diverge_points = diverge_points_dict[sample_token]
            plt.scatter([p[0] for p in diverge_points], [p[1] for p in diverge_points], 
                       color='green', marker='o', label='Diverge Point Candidates')
        
        # Add car and arrow
        plt.imshow(self.car_img, extent=[-1.2, 1.2, -1.5, 1.5])
        plt.arrow(0, 0, 0, 5, head_width=0.5, head_length=0.5, fc='red', ec='red')
        
        # Set plot properties
        plt.legend()
        plt.axis('equal')
        plt.xlim(-15, 15)
        plt.ylim(-5, 30)
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.title('Input Map Ground Truth Visualization')
        
        plt.savefig(f'{dst_dir}/{sample_token}_map.png', bbox_inches='tight', pad_inches=0)
        plt.close()
    
    def _plot_camera_visualization(self, sample_token: str, dst_dir: str, 
                                 diverge_points_dict: Dict[str, List]):
        """Generate and save camera visualization."""
        sample = self.nusc.get('sample', sample_token)
        cam_tokens = {
            'CAM_FRONT_LEFT': sample['data']['CAM_FRONT_LEFT'],
            'CAM_FRONT': sample['data']['CAM_FRONT'], 
            'CAM_FRONT_RIGHT': sample['data']['CAM_FRONT_RIGHT']
        }
        
        # Load camera images
        images = {}
        for cam_name, token in cam_tokens.items():
            cam_data = self.nusc.get('sample_data', token)
            img_path = os.path.join(self.nusc.dataroot, cam_data['filename'])
            images[cam_name] = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        
        # Create camera visualization
        fig, axes = plt.subplots(1, len(images), figsize=(12, 3))
        plt.subplots_adjust(wspace=0, hspace=0)
        
        # Ensure axes is a list even for single camera
        if len(images) == 1:
            axes = [axes]
        
        # Process diverge points if they exist
        if sample_token in diverge_points_dict:
            diverge_points_3d = np.array(diverge_points_dict[sample_token])
            diverge_points_3d = np.concatenate([
                diverge_points_3d,
                np.ones((len(diverge_points_3d), 1)) * -1.84
            ], axis=1)
            
            # Plot camera views with diverge points
            for cam_idx, (cam_name, ax) in enumerate(zip(images.keys(), axes)):
                ax.imshow(images[cam_name])
                visible_points_2d = lidar_to_img(diverge_points_3d.copy(), self.nusc, 
                                               sample_token, cam_name, 
                                               images[cam_name].shape, ground=True)
                
                if len(visible_points_2d) > 0:
                    ax.scatter(visible_points_2d[:, 0], visible_points_2d[:, 1], c='green', s=10)
                    
                    # Add bounding box
                    min_x = max(np.min(visible_points_2d[:, 0]) - 100, 4)
                    max_x = min(np.max(visible_points_2d[:, 0]) + 100, images[cam_name].shape[1] - 4)
                    min_y = max(np.min(visible_points_2d[:, 1]) - 100, 4)
                    max_y = min(np.max(visible_points_2d[:, 1]) + 100, images[cam_name].shape[0] - 4)
                    
                    rect = Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, 
                                   linewidth=2, edgecolor='red', facecolor='none')
                    ax.add_patch(rect)
                
                ax.axis('off')
        else:
            # Plot camera views without diverge points
            for cam_idx, (cam_name, ax) in enumerate(zip(images.keys(), axes)):
                ax.imshow(images[cam_name])
                ax.axis('off')
        
        # Save camera visualization
        plt.savefig(f'{dst_dir}/{sample_token}_cameras.png', bbox_inches='tight', pad_inches=0)
        plt.close()

