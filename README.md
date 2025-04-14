# Stable Diffusion Image Generator

A local API server and client for generating images using Stable Diffusion v3.5.

## Features

- REST API server for image generation
- Simple Python client
- Support for macOS with MPS acceleration (Apple Silicon)
- Automatic device detection (CUDA, MPS, CPU)
- Configurable image generation parameters

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

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your Hugging Face token (optional but recommended):
   ```bash
   export HF_TOKEN=your_token_here
   ```
   Without a token, the model download might be rate-limited or fail.

## Usage

### Starting the Server

Run the server with:

```bash
python server.py
```

The server will:
1. Automatically detect the best available device (CUDA GPU, Apple Silicon MPS, or CPU)
2. Download the Stable Diffusion v3.5 model if not already cached
3. Start a REST API server on port 8000
4. Accept image generation requests via the `/predict` endpoint

### Generating Images

You can generate images in two ways:

#### Using the included Python client

```bash
python client.py "a photograph of a mountain lake at sunset"
```

The generated image will be saved in the `generated_images` directory.

#### Using the API directly

Send a POST request to the `/predict` endpoint:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a photograph of a mountain lake at sunset", "num_inference_steps": 20, "guidance_scale": 7.5}'
```

The response will be a PNG image.

## Configuration Options

When generating images, you can configure:

- `prompt`: Text description of the desired image
- `num_inference_steps`: Number of denoising steps (higher = better quality but slower)
- `guidance_scale`: How closely to follow the prompt (higher = more faithful but less creative)
- `negative_prompt`: Text description of what to avoid in the image

## Performance Notes

- On Apple Silicon Macs (M1/M2/M3), the model uses MPS acceleration
- First generation is slower due to model loading and compilation
- Subsequent generations are faster
- Memory is automatically cleared between generations on Apple Silicon
- Generation time depends on device performance and inference steps

## Customization

The default model is Stable Diffusion v3.5, but the codebase can be modified to use other diffusion models available in the diffusers library.

## Troubleshooting

- If you encounter "Connection refused" errors, make sure the server is running
- If you see CUDA or MPS related errors, try falling back to CPU by modifying the device detection in server.py
- For "out of memory" errors, reduce the batch size or use fewer inference steps
- If model download fails, check your internet connection and Hugging Face token 