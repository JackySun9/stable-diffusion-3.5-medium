# Stable Diffusion 3.5 Medium Interface

This directory contains code to run and interact with the Stable Diffusion 3.5 Medium model using various interfaces.

## Requirements

- Python 3.9+
- Hugging Face account and token (for model download)
- Approximately 10GB disk space for models

## Installation

1. Clone this repository or navigate to the project directory

2. Create a virtual environment and activate it:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install the required dependencies:
```bash
pip install diffusers gradio litserve pillow torch huggingface_hub
```

4. Set your Hugging Face token as an environment variable (required to download the model):
```bash
export HF_TOKEN=your_huggingface_token
```

## Usage Options

You have multiple ways to use the Stable Diffusion 3.5 Medium model:

### Option 1: Using the Server and Client (Recommended for multi-user setups)

#### Running the Server

The server provides a REST API for generating images:

```bash
python server.py
```

This will start a server at http://localhost:8000

#### Using the Gradio Web Interface with Server

For a user-friendly web interface that connects to the running server:

```bash
python gradio_app.py
```

This will start a Gradio web interface at http://localhost:7860

**Important**: Make sure the server is running before starting the Gradio interface.

#### Using the Python Client

There's a Python client provided that can be used to generate images programmatically:

```python
from client import StableDiffusionClient, save_image_with_timestamp

# Initialize client
client = StableDiffusionClient(base_url="http://localhost:8000")

# Generate an image
image = client.generate_image(
    prompt="A beautiful sunset over the ocean",
    num_inference_steps=28,
    guidance_scale=3.5,
    negative_prompt="low quality, bad anatomy, worst quality, low resolution"
)

# Save the image
save_image_with_timestamp(image, "sunset", "my_images")
```

#### Using the Command Line Interface

You can also generate images directly from the command line:

```bash
python client.py "A beautiful sunset over the ocean" --output-dir my_images
```

### Option 2: Standalone Gradio App (Simplest to use)

For a single-user setup without running a separate server:

```bash
python gradio_standalone.py
```

This will start a Gradio web interface at http://localhost:7860 that directly loads and runs the model.

## Notes

- The default parameters (28 inference steps, 3.5 guidance scale) are optimized for Stable Diffusion 3.5.
- The first run will download the model weights from Hugging Face, which may take some time.
- The model requires significant GPU memory; if you're running on a CPU, the generation will be much slower.
- The standalone version is simpler to use but loads the model in the same process as the web server, while the server-client approach can be more efficient for serving multiple users. 