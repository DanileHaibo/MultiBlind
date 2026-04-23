"""
Configuration file for asymmetry scene identification from nuScenes dataset.
"""
import os

# Data paths
DATA_ROOT = './data/nuscenes'
RESULT_ROOT = 'dataset'

# NuScenes maps
MAPS = ['boston-seaport', 'singapore-hollandvillage',
        'singapore-onenorth', 'singapore-queenstown']

# File paths for sample tokens
SAMPLE_TOKEN_CANDIDATES_PATH = os.path.join(RESULT_ROOT, 'sample_token_candidates.txt')
SAMPLE_TOKENS_SELECTED_PATH = os.path.join(RESULT_ROOT, 'sample_token_candidates_selected.txt')
SAMPLE_TOKENS_ASYMMETRIC_PATH = os.path.join(RESULT_ROOT, 'sample_tokens_asymmetric.txt')
SAMPLE_TOKENS_SYMMETRIC_PATH = os.path.join(RESULT_ROOT, 'sample_tokens_symmetric.txt')

# Output directories
SCENES_CANDIDATE_DIR = f'{RESULT_ROOT}/scenes_candidate'
SCENES_CANDIDATE_SELECTED_DIR = f'{RESULT_ROOT}/scenes_candidate_selected'

SCENES_ASYMMETRIC_DIST_DIR = f'{RESULT_ROOT}/scenes_asymmetric_dist'
SCENES_SYMMETRIC_DIST_DIR = f'{RESULT_ROOT}/scenes_symmetric_dist'

SCENES_ASYMMETRIC_CURVATURE_DIR = f'{RESULT_ROOT}/scenes_asymmetric_curvature'
SCENES_ASYMMETRIC_CURVATURE_INVALID_DIR = f'{RESULT_ROOT}/scenes_asymmetric_curvature_invalid'
SCENES_ASYMMETRIC_CURVATURE_SELECTED_DIR = f'{RESULT_ROOT}/scenes_asymmetric_curvature_selected'

SCENES_SYMMETRIC_CURVATURE_DIR = f'{RESULT_ROOT}/scenes_symmetric_curvature'
SCENES_SYMMETRIC_CURVATURE_SELECTED_DIR = f'{RESULT_ROOT}/scenes_symmetric_curvature_selected'

RESPONSE_GPT4O_ASYMMETRIC_DIR = f'{RESULT_ROOT}/response_gpt4o_asymmetric'
RESPONSE_GPT4O_SYMMETRIC_DIR = f'{RESULT_ROOT}/response_gpt4o_symmetric'

SCENES_ASYMMETRIC_DIR = f'{RESULT_ROOT}/scenes_asymmetric'
SCENES_SYMMETRIC_DIR = f'{RESULT_ROOT}/scenes_symmetric'

# NuScenes info files
NUSCENES_INFOS_PATH = 'data/nuscenes/nuscenes_infos_temporal_val.pkl'
NUSCENES_INFOS_CANDIDATES_PATH = os.path.join(RESULT_ROOT, 'nuscenes_infos_temporal_val_maptr_candidates.pkl')
NUSCENES_INFOS_SYMMETRIC_PATH = os.path.join(RESULT_ROOT, 'nuscenes_infos_temporal_val_maptr_symmetric.pkl')
NUSCENES_INFOS_ASYMMETRIC_PATH = os.path.join(RESULT_ROOT, 'nuscenes_infos_temporal_val_maptr_asymmetric.pkl')

# NuScenes map files (for evaluation)
NUSCENES_MAP_ANNS_PATH = 'data/nuscenes/nuscenes_map_anns_val.json'
NUSCENES_MAP_ANNS_CANDIDATES_PATH = os.path.join(RESULT_ROOT, 'nuscenes_map_anns_val_candidates.json')
NUSCENES_MAP_ANNS_SYMMETRIC_PATH = os.path.join(RESULT_ROOT, 'nuscenes_map_anns_val_symmetric.json')
NUSCENES_MAP_ANNS_ASYMMETRIC_PATH = os.path.join(RESULT_ROOT, 'nuscenes_map_anns_val_asymmetric.json')

# Boundary filtering parameters
MAX_NODES_IN_MAP_NODE = 1000
MIN_BOUNDARY_LENGTH = 10  # Minimum length of road boundary instances
MIN_ENDPOINT_DISTANCE = 5  # Minimum distance between first and last point of boundary
MAX_DISTANCE_FROM_EGO = 10  # Maximum distance from ego vehicle to consider boundaries

# Vectorized map parameters
PATCH_SIZE = (60, 30)
FIXED_PTSNUM_PER_LINE = 20
PADDING_VALUE = -10000
MAP_CLASSES = ['divider', 'ped_crossing', 'boundary']

# Rule-based classifier parameters
DIST_OFFSET_THRESHOLD = 5  # Threshold for determining asymmetry in lane width
MIN_Y_COORDINATE = 0  # Minimum y-coordinate to filter points behind the vehicle

# Curvature analysis parameters
CURVATURE_DIFF_THRESHOLD = 0.1
REGION_CURVATURE_THRESHOLD = 0.3
MIN_POINTS = 5
CHUNK_SIZE = 5
ANGLE_THRESHOLD_DEG = 30  # degrees

# Distance validation parameters
MIN_DIST_CLOSEST = 3
MAX_DIST_CLOSEST = 15
MIN_Y = 3
MIN_DIST_TO_DIVERGE_BOUNDARY = 10

# Scene blacklist (problematic scenes)
SCENES_TO_REMOVE = [
    'scene-0329', 'scene-907', 'scene-0908', 'scene-0557', 'scene-0560', 
    'scene-0561', 'scene-0632', 'scene-0109', 'scene-0784'
]

# Final sample parameters
FINAL_SAMPLE_NUM = 100

# VLM configuration
VLM_MODEL = 'gpt-4o'  # OpenAI model name
VLM_API_KEY_FILE = './dataset_processing/api.txt'  # File containing OpenAI API key

