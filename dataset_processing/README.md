# Asymmetry Scene Identification 

## Overview

The dataset processing implements a two-stage method to identify asymmetric scenes from the nuScenes validation set:

1. **Rule-based Classifier**: Filters scenes based on lane width and curvature differences.
2. **VLM-based Classifier**: Uses VLM (GPT-4o) to further refine classification.

⚠️ The VLM-based classifier requires an OpenAI API key. You can obtain one from [OpenAI's API page](https://openai.com/api/). Or you can directly use our generated asymmetric dataset in [dataset/](../dataset/) (highly recommended👍 to ensure consistent results).

## File Structure

```
dataset_processing/
├── __init__.py                 # Package initialization
├── config.py                   # Configuration parameters
├── geometry_utils.py           # Coordinate transformations and geometric utilities
├── curvature_analysis.py       # Curvature calculation and asymmetry detection
├── map_utils.py               # NuScenes map processing utilities
├── rule_based_classifier.py   # Rule-based classification implementation
├── vlm_classifier.py          # VLM-based classification implementation
├── vlm_client.py              # OpenAI VLM client implementation
├── data_utils.py              # Data loading and file management utilities
├── prompt_template.txt        # VLM system prompt
└── examples/                  # Example images for CoT in system prompt
    ├── example1_map.png
    ├── example2_map.png
    └── example3_map.png

create_asymmetry_dataset.py    # Main script for asymmetry scene identification
README_dataset_processing.md   # This file
```


## Usage

### Set Up OpenAI API Key

Put your OpenAI API key at [api.txt](api.txt).

### Basic Usage

To run the complete dataset processing pipeline:

```bash
python create_asymmetry_dataset.py --data_root ./data/nuscenes
```

### Advanced Usage

The script supports several command-line options for flexibility:

```bash
python create_asymmetry_dataset.py \
    --data_root ./data/nuscenes \  # Path to NuScenes data root
    --skip_preprocess \  # Skip candidate scene processing (useful if already done)
    --skip_rule_based \  # Skip rule-based classification (useful if already done)
    --skip_vlm_call \    # Skip VLM API calls and only process existing responses
    --vlm_response_dir dataset/response_gpt4o_asymmetric \  # Directory containing VLM responses
    --vlm_model gpt-4o \  # VLM model name
    --api_key_file ./dataset_processing/api.txt \  # File containing OpenAI API key
    --final_sample_num 100  # Number of final samples per class (default: 100)
```

## Pipeline Stages

### Stage 1: Candidate Scene Processing

This stage processes the full nuScenes validation set to identify candidate scenes that contain valid road boundaries:

**Output**: 
- `dataset/sample_token_candidates.txt`
- `dataset/scenes_candidate/` (visualizations)

### Stage 2: Rule-based Classification

#### Stage 2.1: Lane Width Asymmetry Detection

Classifies scenes based on lane width changes.

**Output**:
- `dataset/scenes_asymmetric_dist/`
- `dataset/scenes_symmetric_dist/`

#### Stage 2.2: Curvature-based Asymmetry Detection

Further refines asymmetric scenes using curvature differences.

**Output**:
- `dataset/scenes_asymmetric_curvature/`
- `dataset/scenes_symmetric_curvature/`
- `dataset/scenes_asymmetric_curvature_selected/`
- `dataset/scenes_symmetric_curvature_selected/`

### Stage 3: VLM-based Classification

#### Stage 3.1: VLM API Calls

If VLM client is configured and not skipped, calls OpenAI GPT-4o API to classify scenes.

**Output**:
- `dataset/response_gpt4o_asymmetric/`

#### Stage 3.2: VLM Response Processing

Processes VLM responses to get final classification.

**Output**:
- `dataset/sample_tokens_asymmetric.txt`
- `dataset/sample_tokens_symmetric.txt`
- `dataset/scenes_asymmetric/`
- `dataset/scenes_symmetric/`

## Configuration

Key parameters can be modified in `dataset_processing/config.py`:

### Scene Filtering Parameters
- `MAX_NODES_IN_MAP_NODE`: Maximum nodes in map polygon (default: 1000)
- `MIN_BOUNDARY_LENGTH`: Minimum boundary length in meters (default: 10)
- `MIN_ENDPOINT_DISTANCE`: Minimum endpoint distance (default: 5)
- `MAX_DISTANCE_FROM_EGO`: Maximum distance from ego vehicle (default: 10)

### Classification Parameters
- `DIST_OFFSET_THRESHOLD`: Lane width asymmetry threshold (default: 5)
- `CURVATURE_DIFF_THRESHOLD`: Curvature difference threshold (default: 0.1)
- `REGION_CURVATURE_THRESHOLD`: Regional curvature threshold (default: 0.3)

### VLM Parameters
- `VLM_MODEL`: OpenAI model name (default: 'gpt-4o')
- `VLM_API_KEY_FILE`: File containing OpenAI API key (default: 'api.txt')

## Output Files

The processing generates several output files:

### Sample Token Files
- `dataset/sample_tokens_asymmetric.txt`: Final asymmetric sample tokens
- `dataset/sample_tokens_symmetric.txt`: Final symmetric sample tokens

### NuScenes Info Files
- `dataset/nuscenes_infos_temporal_val_maptr_asymmetric.pkl`: Asymmetric dataset info
- `dataset/nuscenes_infos_temporal_val_maptr_symmetric.pkl`: Symmetric dataset info

### Map Annotation Files
- `dataset/nuscenes_map_anns_val_asymmetric.json`: Asymmetric map annotations
- `dataset/nuscenes_map_anns_val_symmetric.json`: Symmetric map annotations

### Visualization Directories
- `dataset/scenes_asymmetric/`: Final asymmetric scene visualizations
- `dataset/scenes_symmetric/`: Final symmetric scene visualizations

## Notes

1. **Data Requirements**: The scripts expect the nuScenes dataset to be available at the specified data root.

2. **Reproducibility**: Random sampling may produce different sets of 100 asymmetric and symmetric scenes, leading to variations in attack results. For consistent results, use our provided asymmetric dataset.