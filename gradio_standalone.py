import gradio as gr
import torch
from diffusers import StableDiffusion3Pipeline
from PIL import Image
import io
from datetime import datetime
import os
from huggingface_hub import login

class StableDiffusionStandalone:
    def __init__(self):
        # Login to Hugging Face if token is available
        if "HF_TOKEN" in os.environ:
            login(token=os.environ.get('HF_TOKEN'))
        
        # Determine device
        if torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"  # Apple Silicon GPU
        else:
            self.device = "cpu"
        print(f"Using device: {self.device}")
        
        # Initialize the pipeline
        self.pipe = None  # We'll load it on demand
        self.model_loaded = False
        
        # CLIP token limit
        self.max_token_limit = 77
    
    def load_model(self):
        if self.model_loaded:
            return
            
        print("Loading Stable Diffusion 3.5 Medium model...")
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3.5-medium", 
            torch_dtype=torch.float16,
            variant="fp16"
        ).to(self.device)

        # Enable memory efficient attention
        self.pipe.enable_attention_slicing()
        
        # Enable torch compile for faster inference if available
        if hasattr(torch, 'compile') and self.device != "mps":  # MPS doesn't support torch.compile yet
            self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead", fullgraph=True)
            
        self.model_loaded = True
        print("Model loaded successfully!")
        
        # Get tokenizer from text encoder for proper token counting
        self.tokenizer = self.pipe.tokenizer
    
    def truncate_prompt(self, prompt):
        """Properly truncate prompt to stay within CLIP token limit using actual tokenizer"""
        if not prompt or not self.model_loaded:
            return prompt
            
        # Tokenize the prompt
        tokens = self.tokenizer.encode(prompt)
        
        # Check if truncation is needed
        if len(tokens) <= self.max_token_limit:
            return prompt
            
        # Truncate tokens and decode back to text
        truncated_tokens = tokens[:self.max_token_limit]
        truncated_prompt = self.tokenizer.decode(truncated_tokens)
        
        return truncated_prompt

    def generate(self, prompt, num_steps, guidance_scale, negative_prompt):
        try:
            # Load model if not already loaded
            if not self.model_loaded:
                self.load_model()
                
            # Truncate prompts to fit within CLIP token limits
            truncated_prompt = self.truncate_prompt(prompt)
            truncated_negative_prompt = self.truncate_prompt(negative_prompt)
            
            if truncated_prompt != prompt:
                print(f"Prompt was truncated due to token limit. Original length: {len(prompt)}")
            
            # Generate image using the pipeline
            output = self.pipe(
                prompt=truncated_prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                negative_prompt=truncated_negative_prompt
            )
            
            image = output.images[0]
            
            # Save the image
            filepath = self.save_image_with_timestamp(image, prompt)
            
            return image, f"Image saved to {filepath}"
        except Exception as e:
            # Create error image
            error_img = Image.new('RGB', (512, 512), color=(255, 200, 200))
            return error_img, f"Error: {str(e)}"
    
    def save_image_with_timestamp(self, image, prompt, output_dir="generated_images"):
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

def launch_interface():
    # Create interface object
    interface = StableDiffusionStandalone()
    
    # Create Gradio interface
    with gr.Blocks(title="Stable Diffusion 3.5 Medium (Standalone)") as demo:
        gr.Markdown("# Stable Diffusion 3.5 Medium - Standalone")
        gr.Markdown("Generate high-quality images directly using Stable Diffusion 3.5 Medium")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input components
                prompt = gr.Textbox(label="Prompt", placeholder="A photo of a cat wearing a hat")
                negative_prompt = gr.Textbox(
                    label="Negative Prompt", 
                    placeholder="low quality, bad anatomy, worst quality, low resolution",
                    value="low quality, bad anatomy, worst quality, low resolution"
                )
                
                with gr.Row():
                    num_steps = gr.Slider(minimum=1, maximum=50, value=28, step=1, label="Inference Steps")
                    guidance_scale = gr.Slider(minimum=1, maximum=15, value=3.5, step=0.1, label="Guidance Scale")
                
                generate_btn = gr.Button("Generate Image", variant="primary")
                output_message = gr.Textbox(label="Status")
                
                # Info accordion
                with gr.Accordion("ℹ️ Model Information", open=False):
                    gr.Markdown("""
                    - **Model**: Stable Diffusion 3.5 Medium by Stability AI
                    - **First run** will download the model (several GB) which may take time
                    - **Default settings** (28 steps, 3.5 guidance) are optimized for SD 3.5
                    - **Generation time** depends on your hardware
                    """)
            
            with gr.Column(scale=1):
                # Output image
                output_image = gr.Image(label="Generated Image", type="pil")
        
        # Load model on startup button
        load_model_btn = gr.Button("Load Model (Do this first)")
        load_model_btn.click(
            fn=interface.load_model,
            inputs=[],
            outputs=[]
        )
        
        # Set up the button click event
        generate_btn.click(
            fn=interface.generate,
            inputs=[prompt, num_steps, guidance_scale, negative_prompt],
            outputs=[output_image, output_message]
        )
    
    # Launch the app
    demo.launch(share=False)

if __name__ == "__main__":
    launch_interface() 