"""
VLM client utilities for scene classification using OpenAI GPT-4V.
"""
import base64
import json
import os
from typing import Dict, List, Optional, Any
from tqdm import tqdm


def encode_image(image_path: str) -> str:
    """
    Encode an image file as base64 string.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        str: Base64 encoded image
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


class OpenAIVLMClient:
    """OpenAI GPT-4V client for scene classification."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        """
        Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key
            model: Model name (default: gpt-4o)
        """
        self.api_key = api_key
        self.model = model
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("OpenAI package not found. Install with: pip install openai")
    
    def classify_scene(self, messages: List[Dict[str, Any]]) -> str:
        """
        Classify scene using OpenAI GPT-4V.
        
        Args:
            messages: Messages in OpenAI format
            
        Returns:
            str: JSON response from GPT-4V
        """
        completion = self.client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=messages
        )
        return completion.choices[0].message.content


class VLMSceneClassifier:
    """Scene classifier using VLM."""
    
    def __init__(self, vlm_client: OpenAIVLMClient, prompt_template_path: Optional[str] = None,
                 example_images_dir: Optional[str] = None):
        """
        Initialize scene classifier.
        
        Args:
            vlm_client: OpenAI VLM client instance
            prompt_template_path: Path to prompt template file
            example_images_dir: Directory containing example images
        """
        self.vlm_client = vlm_client
        self.system_prompt = self._load_system_prompt(prompt_template_path)
        self.example_images_dir = example_images_dir or "dataset_processing/examples"
    
    def _load_system_prompt(self, prompt_template_path: Optional[str]) -> str:
        """Load system prompt from file or use default."""
        if prompt_template_path and os.path.exists(prompt_template_path):
            with open(prompt_template_path, "r") as file:
                return file.read()
        else:
            # Use default prompt from dataset_processing directory
            default_prompt_path = os.path.join("dataset_processing", "prompt_template.txt")
            if os.path.exists(default_prompt_path):
                with open(default_prompt_path, "r") as file:
                    return file.read()
            else:
                # Fallback minimal prompt
                return """You are an expert driving scene analyst. Classify each scene as either 'symmetric' or 'asymmetric' 
                based on road boundary geometry. Return JSON format: {"classification": "symmetric/asymmetric", 
                "details": {"asymmetry_type": "type or null", "reasoning": "explanation"}}"""
    
    def _prepare_example_images(self) -> List[Dict[str, Any]]:
        """Prepare example images for few-shot learning."""
        examples = []
        v
        # Define example mappings
        example_configs = [
            {
                "image": "example1_map.png",
                "description": "This is Example 1 showing an asymmetric case where the left boundary diverges sharply to the left while the right boundary goes straight, creating a fork."
            },
            {
                "image": "example2_map.png", 
                "description": "This is Example 2 showing a symmetric case where both boundaries follow consistent geometry with similar spacing."
            },
            {
                "image": "example3_map.png",
                "description": "This is Example 3 showing another symmetric case where the left boundary turns to the left while the right boundary turns to the right, forming mirrored geometry. This is a typical pattern at crossroads."
            }
        ]
        
        for config in example_configs:
            image_path = os.path.join(self.example_images_dir, config["image"])
            if os.path.exists(image_path):
                examples.extend([
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encode_image(image_path)}"
                        }
                    },
                    {
                        "type": "text",
                        "text": config["description"]
                    }
                ])
        
        return examples
    
    def _create_messages(self, scene_dir: str, sample_token: str) -> List[Dict[str, Any]]:
        """
        Create messages for VLM classification.
        
        Args:
            scene_dir: Directory containing scene data
            sample_token: Sample token identifier
            
        Returns:
            List of messages for VLM
        """
        # Load map data
        with open(os.path.join(scene_dir, f"{sample_token}.json"), "r") as file:
            map_data = json.load(file)
        
        # Prepare image paths
        map_image_path = os.path.join(scene_dir, f"{sample_token}_map.png")
        camera_image_path = os.path.join(scene_dir, f"{sample_token}_cameras.png")
        
        # Encode images
        base64_map_image = encode_image(map_image_path)
        base64_camera_image = encode_image(camera_image_path)
        
        # Prepare user content with examples
        user_content = [
            {
                "type": "text",
                "text": "First, let me show you some examples to demonstrate the classification:"
            }
        ]
        
        # Add example images
        user_content.extend(self._prepare_example_images())
        
        # Add current scene to classify
        user_content.extend([
            {
                "type": "text",
                "text": f"\n\nNow, please analyze the following scene:\n\n{json.dumps(map_data)}"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_map_image}"
                }
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_camera_image}"
                }
            }
        ])
        
        # Create messages
        messages = [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": user_content
            }
        ]
        
        return messages
    
    def classify_scene(self, scene_dir: str, sample_token: str) -> Dict[str, Any]:
        """
        Classify a single scene.
        
        Args:
            scene_dir: Directory containing scene data
            sample_token: Sample token identifier
            
        Returns:
            Dict containing classification results
        """
        messages = self._create_messages(scene_dir, sample_token)
        response = self.vlm_client.classify_scene(messages)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Fallback if response is not valid JSON
            return {
                "classification": "symmetric",  # Default fallback
                "details": {
                    "asymmetry_type": None,
                    "reasoning": f"Failed to parse VLM response: {response}"
                }
            }
    
    def classify_scenes(self, scene_dir: str, output_dir: str, 
                       scene_tokens: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Classify multiple scenes.
        
        Args:
            scene_dir: Directory containing scene data
            output_dir: Directory to save VLM responses
            scene_tokens: Optional list of specific scene tokens to process
            
        Returns:
            Dict mapping scene tokens to classifications
        """
        # Get scene list
        if scene_tokens is None:
            scene_list = os.listdir(scene_dir)
            scene_tokens = [item.split('.')[0] for item in scene_list 
                           if item.endswith('.json') and not item.startswith('._')]
            scene_tokens = list(set(scene_tokens))
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Check for existing responses
        existing_responses = []
        if os.path.exists(output_dir):
            existing_files = os.listdir(output_dir)
            existing_responses = [item.split('.')[0] for item in existing_files 
                                if item.endswith('.json')]
        
        print(f"Total {len(scene_tokens)} scenes to process")
        print(f"Total {len(existing_responses)} scenes already processed")
        
        # Process scenes
        classifications = {}
        
        for sample_token in tqdm(scene_tokens, desc="Classifying scenes"):
            if sample_token in existing_responses:
                # Load existing response
                response_path = os.path.join(output_dir, f"{sample_token}.json")
                with open(response_path, 'r') as f:
                    result = json.load(f)
                classifications[sample_token] = result.get('classification', 'symmetric')
                continue
            
            try:
                # Classify scene
                result = self.classify_scene(scene_dir, sample_token)
                
                # Save response
                response_path = os.path.join(output_dir, f"{sample_token}.json")
                with open(response_path, 'w') as f:
                    json.dump(result, f, indent=2)
                
                classifications[sample_token] = result.get('classification', 'symmetric')
                
            except Exception as e:
                print(f"Error processing {sample_token}: {e}")
                # Save error response
                error_result = {
                    "classification": "symmetric",  # Default fallback
                    "details": {
                        "asymmetry_type": None,
                        "reasoning": f"Error during classification: {str(e)}"
                    }
                }
                response_path = os.path.join(output_dir, f"{sample_token}.json")
                with open(response_path, 'w') as f:
                    json.dump(error_result, f, indent=2)
                
                classifications[sample_token] = "symmetric"
        
        return classifications
