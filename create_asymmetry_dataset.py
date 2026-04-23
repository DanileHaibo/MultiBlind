#!/usr/bin/env python3
"""
Main script for creating asymmetry dataset from nuScenes.

This script implements the two-stage asymmetry scene identification method:
1. Rule-based classifier: Filters scenes based on lane width and curvature differences
2. VLM-based classifier: Uses Vision Language Model responses to further refine classification
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
from nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer

# Add utils path
from attack_toolkit.src.utils.utils_prompt import VectorizedLocalMap

# Import our dataset processing modules
from dataset_processing import (
    RuleBasedClassifier, 
    VLMClassifier,
    OpenAIVLMClient,
    DataUtils,
    DatasetProcessor,
    MAPS,
    PATCH_SIZE,
    FIXED_PTSNUM_PER_LINE,
    PADDING_VALUE,
    MAP_CLASSES,
    NUSCENES_INFOS_PATH,
    RESPONSE_GPT4O_ASYMMETRIC_DIR,
    SCENES_ASYMMETRIC_CURVATURE_SELECTED_DIR,
    VLM_MODEL,
    VLM_API_KEY_FILE
)



def setup_nuscenes_environment(data_root: str):
    """
    Setup NuScenes environment including maps and vectorized map.
    
    Args:
        data_root: Path to NuScenes data root
        
    Returns:
        Tuple of (nusc, nusc_maps, map_explorer, vector_map, car_img)
    """
    print("Initializing NuScenes...")
    
    # Initialize NuScenes
    nusc = NuScenes(version='v1.0-trainval', dataroot=data_root, verbose=True)
    
    # Initialize NuScenes Maps
    nusc_maps = {}
    map_explorer = {}
    for loc in MAPS:
        nusc_maps[loc] = NuScenesMap(dataroot=data_root, map_name=loc)
        map_explorer[loc] = NuScenesMapExplorer(nusc_maps[loc])
    
    # Initialize vectorized map
    vector_map = VectorizedLocalMap(
        data_root,
        patch_size=PATCH_SIZE, 
        map_classes=MAP_CLASSES,
        fixed_ptsnum_per_line=FIXED_PTSNUM_PER_LINE,
        padding_value=PADDING_VALUE
    )
    
    # Load car image for visualization
    car_img_path = './figs/lidar_car.png'
    if not os.path.exists(car_img_path):
        # Create a simple placeholder if image doesn't exist
        car_img = Image.new('RGBA', (24, 30), (255, 0, 0, 128))
    else:
        car_img = Image.open(car_img_path)
    
    return nusc, nusc_maps, map_explorer, vector_map, car_img


def main():
    """Main function for dataset creation."""
    parser = argparse.ArgumentParser(description='Create asymmetry dataset from nuScenes')
    parser.add_argument('--data_root', type=str, default='./data/nuscenes',
                       help='Path to NuScenes data root')
    parser.add_argument('--skip_preprocess', action='store_true',
                       help='Skip candidate scene processing')
    parser.add_argument('--skip_rule_based', action='store_true',
                       help='Skip rule-based classification')
    parser.add_argument('--vlm_response_dir', type=str, default=RESPONSE_GPT4O_ASYMMETRIC_DIR,
                       help='Directory containing VLM responses')
    parser.add_argument('--vlm_model', type=str, default=VLM_MODEL,
                       help='OpenAI model name (default: gpt-4o)')
    parser.add_argument('--api_key_file', type=str, default=VLM_API_KEY_FILE,
                       help='File containing OpenAI API key')
    parser.add_argument('--skip_vlm_call', action='store_true',
                       help='Skip VLM API calls (use existing responses)')
    parser.add_argument('--final_sample_num', type=int, default=100,
                       help='Number of final samples per class')
    
    args = parser.parse_args()
    
    # Setup environment
    nusc, nusc_maps, map_explorer, vector_map, car_img = setup_nuscenes_environment(args.data_root)
    
    # Initialize VLM client if needed
    vlm_client = None
    if not args.skip_vlm_call:
        if os.path.exists(args.api_key_file):
            try:
                with open(args.api_key_file, 'r') as f:
                    api_key = f.read().strip()
                vlm_client = OpenAIVLMClient(api_key, args.vlm_model)
                print(f"Initialized OpenAI VLM client: {args.vlm_model}")
            except Exception as e:
                print(f"Warning: Failed to initialize VLM client: {e}")
                print("VLM classification will be skipped.")
        else:
            raise ValueError(f"VLM API key file {args.api_key_file} not found")
    
    # Initialize processors
    rule_classifier = RuleBasedClassifier(nusc, nusc_maps, map_explorer, vector_map, car_img)
    vlm_classifier = VLMClassifier(nusc, car_img, vlm_client)
    dataset_processor = DatasetProcessor(nusc)
    
    # Step 1: Process candidate scenes (if not skipped)
    if not args.skip_preprocess:
        print("\n=== Step 1: Processing candidate scenes ===")
        
        # Load NuScenes info data
        nuscenes_infos = DataUtils.load_nuscenes_infos(NUSCENES_INFOS_PATH)
        print(f"Loaded {len(nuscenes_infos['infos'])} scenes from NuScenes")
        
        # Process candidates
        sample_token_candidates = rule_classifier.preprocess_scenes(nuscenes_infos)
        print(f"Found {len(sample_token_candidates)} candidate scenes")
        
        # Create candidate dataset files
        dataset_processor.create_candidate_dataset(sample_token_candidates)
        
        # Get scene statistics
        stats = dataset_processor.get_scene_statistics(sample_token_candidates)
        print(f"Candidate dataset: {stats['total_samples']} samples from {stats['unique_scenes']} unique scenes")
    else:
        # Load existing candidates
        sample_token_candidates = DataUtils.load_sample_tokens(
            'dataset/sample_token_candidates.txt')
        print(f"Loaded {len(sample_token_candidates)} existing candidate scenes")
    
    
    # Step 2: Rule-based classification (if not skipped)
    if not args.skip_rule_based:
        print("\n=== Step 2: Rule-based classification ===")
        
        # Step 2.1: Lane width asymmetry classification
        print("Step 2.1: Classifying by lane width asymmetry...")
        sample_tokens_asymmetric_dist, sample_tokens_symmetric_dist = rule_classifier.classify_by_lane_width(
            sample_token_candidates)
        print(f"Lane width classification: {len(sample_tokens_asymmetric_dist)} asymmetric, "
              f"{len(sample_tokens_symmetric_dist)} symmetric")
        
        # Step 2.2: Curvature-based classification
        print("\nStep 2.2: Classifying by curvature asymmetry...")
        (sample_tokens_asymmetric_curvature, 
         sample_tokens_symmetric_curvature, 
         sample_tokens_invalid,
         diverge_points_dict) = rule_classifier.classify_by_curvature(sample_tokens_asymmetric_dist)
        
        # Add symmetric samples from distance classification
        sample_tokens_symmetric_curvature.extend(sample_tokens_symmetric_dist)
        
        print(f"Curvature classification: {len(sample_tokens_asymmetric_curvature)} asymmetric, "
              f"{len(sample_tokens_symmetric_curvature)} symmetric, "
              f"{len(sample_tokens_invalid)} invalid")
        
        # Step 2.3: Generate selected samples for VLM processing
        print("\nStep 2.3: Generating samples for VLM processing...")
        random_samples_asymmetric, random_samples_symmetric = vlm_classifier.generate_selected_samples(
            sample_tokens_asymmetric_curvature, sample_tokens_symmetric_curvature)
        
        # Process selected samples (generate visualizations)
        vlm_classifier.process_selected_samples(
            random_samples_asymmetric, random_samples_symmetric, diverge_points_dict)
        
        print(f"Generated {len(random_samples_asymmetric)} asymmetric and "
              f"{len(random_samples_symmetric)} symmetric samples for VLM processing")
    else:
        print("Skipping rule-based classification")
    
    
    # Step 3: VLM-based classification
    print("\n=== Step 3: VLM-based classification ===")
    
    # Step 3.1: Run VLM classification (if not skipped)
    if not args.skip_vlm_call and vlm_client is not None:
        print("Step 3.1: Running VLM classification...")
        
        # Get asymmetric scenes to classify
        asymmetric_scene_tokens = [
            token.split('.')[0] for token in os.listdir(SCENES_ASYMMETRIC_CURVATURE_SELECTED_DIR)
            if token.endswith('.json') and not token.startswith('._')
        ]
        
        print(f"Running VLM classification on {len(asymmetric_scene_tokens)} asymmetric scenes")
        
        # Run VLM classification
        vlm_classifications = vlm_classifier.run_vlm_classification(
            SCENES_ASYMMETRIC_CURVATURE_SELECTED_DIR, 
            args.vlm_response_dir,
            asymmetric_scene_tokens
        )
        
        print(f"VLM classification completed. Results saved to {args.vlm_response_dir}")
        
        # Print classification summary
        symmetric_count = sum(1 for c in vlm_classifications.values() if c == 'symmetric')
        asymmetric_count = sum(1 for c in vlm_classifications.values() if c == 'asymmetric')
        print(f"VLM classification results: {asymmetric_count} asymmetric, {symmetric_count} symmetric")
    
    # Step 3.2: Process VLM responses
    if os.path.exists(args.vlm_response_dir):
        print(f"Step 3.2: Processing VLM responses from {args.vlm_response_dir}")
        
        # Process VLM responses
        sample_tokens_asymmetric, sample_tokens_symmetric = vlm_classifier.process_vlm_responses(
            args.vlm_response_dir)
        
        print(f"VLM classification: {len(sample_tokens_asymmetric)} asymmetric, "
              f"{len(sample_tokens_symmetric)} symmetric")
        
        # Step 4: Create final dataset
        print("\n=== Step 4: Creating final dataset ===")
        
        final_asymmetric_tokens, final_symmetric_tokens = vlm_classifier.create_final_dataset(
            sample_tokens_asymmetric, sample_tokens_symmetric, args.final_sample_num)
        
        # Create final dataset files
        dataset_processor.create_final_datasets(final_asymmetric_tokens, final_symmetric_tokens)
        
        print(f"Final dataset created:")
        print(f"  - Asymmetric: {len(final_asymmetric_tokens)} samples")
        print(f"  - Symmetric: {len(final_symmetric_tokens)} samples")
        print(f"  - Files saved to dataset/ directory")
        
        # Get final statistics
        async_stats = dataset_processor.get_scene_statistics(final_asymmetric_tokens)
        sync_stats = dataset_processor.get_scene_statistics(final_symmetric_tokens)
        
        print(f"\nFinal dataset statistics:")
        print(f"  - Asymmetric: {async_stats['total_samples']} samples from {async_stats['unique_scenes']} scenes")
        print(f"  - Symmetric: {sync_stats['total_samples']} samples from {sync_stats['unique_scenes']} scenes")
        
    else:
        print(f"Warning: VLM response directory {args.vlm_response_dir} not found")
        print("VLM classification step skipped. Run VLM classification separately and provide response directory.")
    
    # print("\n=== Dataset creation completed ===")
    # print("Generated files:")
    # print("  - dataset/sample_tokens_asymmetric.txt")
    # print("  - dataset/sample_tokens_symmetric.txt") 
    # print("  - dataset/scenes_asymmetric/ (visualizations)")
    # print("  - dataset/scenes_symmetric/ (visualizations)")
    # print("  - dataset/nuscenes_infos_temporal_val_maptr_asymmetric.pkl")
    # print("  - dataset/nuscenes_infos_temporal_val_maptr_symmetric.pkl")
    # print("  - dataset/nuscenes_map_anns_val_asymmetric.json")
    # print("  - dataset/nuscenes_map_anns_val_symmetric.json")


if __name__ == "__main__":
    main()

