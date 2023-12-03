import torch
from diffusers import AutoPipelineForImage2Image
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
import gradio as gr

# log gpu availabilitu
print(f"Is CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")


def image_to_template(generated_image, logo, button_text, punchline, theme_color):
    # Resize logo if needed
    logo = logo.resize((100, 100))  # Example size, adjust as needed

    # Create a blank canvas with extra space for logo, punchline, and button
    canvas_width = max(generated_image.width, logo.width) * 2
    canvas_height = generated_image.height + logo.height + 100
    canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
    
    # Paste the logo and the generated image onto the canvas
    canvas.paste(logo, (10, 10))  # Adjust position as needed
    canvas.paste(generated_image, (0, logo.height + 20))

    # Add punchline and button
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()  # Or use a custom font
    text_color = theme_color

    # Draw punchline
    draw.text((10, logo.height + generated_image.height + 30), punchline, fill=text_color, font=font)

    # Draw button
    button_position = (10, logo.height + generated_image.height + 60)  # Adjust as needed
    draw.rectangle([button_position, (canvas_width - 10, canvas_height - 10)], outline=theme_color, fill=text_color)
    draw.text(button_position, button_text, font=font)

    return canvas

def generate_template(initial_image, logo, prompt, button_text, punchline, image_color, theme_color):
    pipeline = AutoPipelineForImage2Image.from_pretrained(
        "./models/kandinsky-2-2-decoder", 
        torch_dtype=torch.float16, 
        variant="fp16", 
        use_safetensors=True
    )

    # pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=True)
    pipeline.enable_model_cpu_offload()

    prompt = f"{prompt}, include the color {image_color}"
    negative_prompt = "low quality, bad quality, blurry, unprofessional"

    generated_image = pipeline(
        prompt=prompt, 
        negative_prompt=negative_prompt,
        image=initial_image,  
        height=256, 
        width=256).images[0]

    template_image = image_to_template(generated_image, logo, button_text, punchline, theme_color)

    return template_image

# Set up Gradio interface
iface = gr.Interface(
    fn=generate_template,
    inputs=[gr.inputs.Image(type="pil", label="Initial Image"), 
            gr.inputs.Image(type="pil", label="Logo"), 
            gr.inputs.Textbox(label="Prompt"), 
            gr.inputs.Textbox(label="Button Text"),
            gr.inputs.Textbox(label="Punchline"),
            gr.inputs.ColorPicker(label="Image Color"),
            gr.inputs.ColorPicker(label="Theme Color")],
    outputs=[gr.outputs.Image(type="pil")],
    title="Ad Template Generation Using Diffusion Models Demo",
    description="Generate ad template based on your inputs using a trained model.",
    concurrency_limit=2,
    # examples=[
    #     []
    # ]
)

# Run the interface
iface.launch()
