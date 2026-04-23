"""
Data loading and preprocessing utilities for NuScenes dataset processing.
"""
import os
import json
import pickle
import shutil
from typing import Dict, List, Any

from .config import *


class DataUtils:
    """Utility class for data loading and file management."""
    
    @staticmethod
    def setup_directories(directories: List[str]):
        """
        Create or clean directories.
        
        Args:
            directories: List of directory paths to setup
        """
        for dir_path in directories:
            if os.path.exists(dir_path):
                # Clean existing directory
                for file in os.listdir(dir_path):
                    file_path = os.path.join(dir_path, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
            else:
                os.makedirs(dir_path)
    
    @staticmethod
    def load_nuscenes_infos(info_path: str) -> Dict[str, Any]:
        """
        Load NuScenes info file.
        
        Args:
            info_path: Path to the info file
            
        Returns:
            Dictionary containing NuScenes info data
        """
        with open(info_path, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def save_nuscenes_infos(infos: Dict[str, Any], output_path: str):
        """
        Save NuScenes info file.
        
        Args:
            infos: Info data to save
            output_path: Output file path
        """
        with open(output_path, 'wb') as f:
            pickle.dump(infos, f)
    
    @staticmethod
    def load_sample_tokens(token_file: str) -> List[str]:
        """
        Load sample tokens from text file.
        
        Args:
            token_file: Path to token file
            
        Returns:
            List of sample tokens
        """
        with open(token_file, 'r') as f:
            tokens = f.readlines()
        return [token.strip() for token in tokens]
    
    @staticmethod
    def save_sample_tokens(tokens: List[str], output_path: str):
        """
        Save sample tokens to text file.
        
        Args:
            tokens: List of sample tokens
            output_path: Output file path
        """
        with open(output_path, 'w') as f:
            for token in tokens:
                f.write(f"{token}\n")
    
    @staticmethod
    def create_filtered_info_file(original_infos: Dict[str, Any], 
                                 sample_tokens: List[str], 
                                 output_path: str):
        """
        Create filtered NuScenes info file based on sample tokens.
        
        Args:
            original_infos: Original NuScenes info data
            sample_tokens: List of sample tokens to include
            output_path: Output file path
        """
        filtered_infos = {
            'infos': [],
            'metadata': original_infos['metadata']
        }
        
        for scene_info in original_infos['infos']:
            if scene_info['token'] in sample_tokens:
                filtered_infos['infos'].append(scene_info)
        
        DataUtils.save_nuscenes_infos(filtered_infos, output_path)
        print(f'Saved filtered info file to {output_path}')
    
    @staticmethod
    def create_filtered_map_anns(original_anns_path: str, 
                                sample_tokens: List[str], 
                                output_path: str):
        """
        Create filtered map annotations file based on sample tokens.
        
        Args:
            original_anns_path: Path to original annotations file
            sample_tokens: List of sample tokens to include
            output_path: Output file path
        """
        with open(original_anns_path, 'r') as f:
            original_anns = json.load(f)
        
        filtered_anns = {'GTs': []}
        for ann in original_anns['GTs']:
            if ann['sample_token'] in sample_tokens:
                filtered_anns['GTs'].append(ann)
        
        with open(output_path, 'w') as f:
            json.dump(filtered_anns, f, indent=2)
        
        print(f'Saved filtered map annotations to {output_path}')


class DatasetProcessor:
    """Main dataset processor class."""
    
    def __init__(self, nusc):
        """
        Initialize the dataset processor.
        
        Args:
            nusc: NuScenes instance
        """
        self.nusc = nusc
    
    def create_candidate_dataset(self, sample_tokens_candidates: List[str]):
        """
        Create candidate dataset files.
        
        Args:
            sample_tokens_candidates: List of candidate sample tokens
        """
        # Save sample tokens
        DataUtils.save_sample_tokens(sample_tokens_candidates, SAMPLE_TOKEN_CANDIDATES_PATH)
        
        # Load original data
        original_infos = DataUtils.load_nuscenes_infos(NUSCENES_INFOS_PATH)
        
        # Create filtered info file
        DataUtils.create_filtered_info_file(
            original_infos, sample_tokens_candidates, NUSCENES_INFOS_CANDIDATES_PATH)
        
        # Create filtered map annotations
        DataUtils.create_filtered_map_anns(
            NUSCENES_MAP_ANNS_PATH, sample_tokens_candidates, NUSCENES_MAP_ANNS_CANDIDATES_PATH)
    
    def create_final_datasets(self, final_asymmetric_tokens: List[str], 
                            final_symmetric_tokens: List[str]):
        """
        Create final asymmetric and symmetric datasets.
        
        Args:
            final_asymmetric_tokens: List of final asymmetric sample tokens
            final_symmetric_tokens: List of final symmetric sample tokens
        """
        # Load original data
        original_infos = DataUtils.load_nuscenes_infos(NUSCENES_INFOS_PATH)
        
        # Create asymmetric dataset
        DataUtils.create_filtered_info_file(
            original_infos, final_asymmetric_tokens, NUSCENES_INFOS_ASYMMETRIC_PATH)
        DataUtils.create_filtered_map_anns(
            NUSCENES_MAP_ANNS_PATH, final_asymmetric_tokens, NUSCENES_MAP_ANNS_ASYMMETRIC_PATH)
        
        # Create symmetric dataset
        DataUtils.create_filtered_info_file(
            original_infos, final_symmetric_tokens, NUSCENES_INFOS_SYMMETRIC_PATH)
        DataUtils.create_filtered_map_anns(
            NUSCENES_MAP_ANNS_PATH, final_symmetric_tokens, NUSCENES_MAP_ANNS_SYMMETRIC_PATH)
    
    def get_scene_statistics(self, sample_tokens: List[str]) -> Dict[str, int]:
        """
        Get statistics about scenes in the sample tokens.
        
        Args:
            sample_tokens: List of sample tokens
            
        Returns:
            Dictionary with scene statistics
        """
        scene_to_samples = {}
        
        for sample_token in sample_tokens:
            sample = self.nusc.get('sample', sample_token)
            scene_token = sample['scene_token']
            
            if scene_token not in scene_to_samples:
                scene_to_samples[scene_token] = []
            scene_to_samples[scene_token].append(sample_token)
        
        return {
            'total_samples': len(sample_tokens),
            'unique_scenes': len(scene_to_samples),
            'scene_to_samples': scene_to_samples
        }

