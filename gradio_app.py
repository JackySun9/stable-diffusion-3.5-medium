import gradio as gr
import os
import sys
import requests
from PIL import Image
import io
from datetime import datetime

# Import directly from the local directory
from client import StableDiffusionClient, save_image_with_timestamp

class GradioInterface:
    def __init__(self, base_url="http://localhost:8000"):
        self.client = StableDiffusionClient(base_url=base_url)
        
    def generate(self, prompt, num_steps, guidance_scale, negative_prompt):
        try:
            # Call the existing client to generate the image
            image = self.client.generate_image(
                prompt=prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt
            )
            
            # Save the image
            filepath = save_image_with_timestamp(image, prompt)
            
            return image, f"Image saved to {filepath}"
        except Exception as e:
            # Create error image
            error_img = Image.new('RGB', (512, 512), color=(255, 200, 200))
            return error_img, f"Error: {str(e)}"

def launch_interface():
    # Create interface object
    interface = GradioInterface()
    
    # Create Gradio interface
    with gr.Blocks(title="Stable Diffusion 3.5 Medium") as demo:
        gr.Markdown("# Stable Diffusion 3.5 Medium")
        gr.Markdown("Generate high-quality images with Stable Diffusion 3.5 Medium")
        
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
            
            with gr.Column(scale=1):
                # Output image
                output_image = gr.Image(label="Generated Image", type="pil")
        
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