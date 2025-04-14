import requests
from PIL import Image
import io
import argparse
from datetime import datetime
import os

class StableDiffusionClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url

    def generate_image(
        self,
        prompt: str,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        negative_prompt: str = "low quality, bad anatomy, worst quality, low resolution"
    ) -> Image.Image:
        """
        Generate an image using Stable Diffusion v3.5
        
        Args:
            prompt: Text description of the desired image
            num_inference_steps: Number of denoising steps (higher = better quality but slower)
            guidance_scale: How closely to follow the prompt (higher = more faithful but less creative)
            negative_prompt: Text description of what to avoid in the image
            
        Returns:
            PIL Image object
        """
        payload = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "negative_prompt": negative_prompt
        }

        response = requests.post(f"{self.base_url}/predict", json=payload)
        
        if response.status_code != 200:
            raise Exception(f"Error generating image: {response.text}")
            
        # Convert raw bytes to PIL Image
        return Image.open(io.BytesIO(response.content))

def save_image_with_timestamp(image: Image.Image, prompt: str, output_dir: str = "generated_images"):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename with timestamp and sanitized prompt
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Sanitize prompt for filename (keep only alphanumeric chars and replace spaces with underscores)
    sanitized_prompt = "".join(c for c in prompt if c.isalnum() or c.isspace()).replace(" ", "_")
    # Truncate prompt if it's too long
    sanitized_prompt = sanitized_prompt[:50]  # Limit to 50 characters
    
    filename = f"{timestamp}_{sanitized_prompt}.png"
    filepath = os.path.join(output_dir, filename)
    
    # Save the image
    image.save(filepath)
    print(f"Image saved as: {filepath}")
    return filepath

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate image using Stable Diffusion')
    parser.add_argument('prompt', type=str, help='Text prompt for image generation')
    parser.add_argument('--output-dir', type=str, default='generated_images',
                      help='Directory to save generated images (default: generated_images)')
    
    args = parser.parse_args()
    
    # Generate and save image
    client = StableDiffusionClient()
    try:
        image = client.generate_image(prompt=args.prompt)
        save_image_with_timestamp(image, args.prompt, args.output_dir)
    except Exception as e:
        print(f"Error: {e}")
