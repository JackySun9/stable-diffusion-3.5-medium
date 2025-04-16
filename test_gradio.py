import gradio as gr
from client import StableDiffusionClient

# Test that we can import the client
print("Successfully imported StableDiffusionClient")

# Test that we can create a simple Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Test Interface")
    input_text = gr.Textbox(label="Prompt")
    output_text = gr.Textbox(label="Output")
    
    def echo_input(text):
        return f"You entered: {text}"
    
    submit_btn = gr.Button("Submit")
    submit_btn.click(fn=echo_input, inputs=input_text, outputs=output_text)

if __name__ == "__main__":
    # Don't actually launch, just verify the code compiles/imports correctly
    print("All imports successful, Gradio interface created")
    print("You can now run 'python gradio_app.py' to start the full interface") 