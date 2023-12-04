import torch
from diffusers import AutoPipelineForImage2Image
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
import gradio as gr
import gc
import textwrap

# log gpu availability
print(f"Is CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")


def image_to_template(generated_image, logo, button_text, punchline, theme_color):
    template_width = 540
    button_font_size = 10
    punchline_font_size = 30
    decoration_height = 10
    margin = 20
    # wrap punchline text
    punchline = textwrap.wrap(punchline, width=35)
    n_of_lines_punchline = len(punchline)

    generated_image = generated_image.convert("RGBA")
    logo = logo.convert("RGBA")

    # image shape
    image_width = template_width // 2
    image_height = image_width * generated_image.height // generated_image.width
    image_shape = (image_width, image_height)

    # logo shape
    logo_width = image_width // 3
    logo_height = logo_width * logo.height // logo.width
    logo_shape = (logo_width, logo_height)

    # Define fonts
    button_font = ImageFont.truetype("./assets/Montserrat-Bold.ttf", button_font_size)
    punchline_font = ImageFont.truetype("./assets/Montserrat-Bold.ttf", punchline_font_size)

    # button shape
    button_width = template_width // 3
    button_height = button_font_size * 3

    # template height calculation
    template_height = (
        image_height
        + logo_height
        + button_height
        + n_of_lines_punchline * punchline_font_size
        + (5 * margin)
        + (2 * decoration_height)
    )

    # Calculate positions for the centered layout
    logo_pos = ((template_width - logo_width) // 2, margin + decoration_height)
    image_pos = (
        (template_width - image_width) // 2,
        logo_pos[1] + logo_height + margin,
    )

    # Decoration positions
    top_decoration_pos = [
        margin,
        -decoration_height // 2,
        template_width - margin,
        decoration_height // 2,
    ]
    bottom_decoration_pos = [
        margin,
        template_height - decoration_height // 2,
        template_width - margin,
        template_height + decoration_height // 2,
    ]

    # Generate Components
    generated_image.thumbnail(image_shape, Image.LANCZOS)
    logo.thumbnail(logo_shape, Image.LANCZOS)
    background = Image.new("RGBA", (template_width, template_height), "WHITE")
    # round the corners of generated image
    mask = Image.new("L", generated_image.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0) + generated_image.size, 20, fill=255)
    generated_image.putalpha(mask)
    # Paste the logo and the generated image onto the background
    background.paste(logo, logo_pos, logo)
    background.paste(generated_image, image_pos, generated_image)
    # Draw the decorations, punchline, and button
    draw = ImageDraw.Draw(background)
    # Decorations on top and bottom
    draw.rounded_rectangle(bottom_decoration_pos, radius=20, fill=theme_color)
    draw.rounded_rectangle(top_decoration_pos, radius=20, fill=theme_color)
    # Punchline text
    text_heights = []
    for line in punchline:
        text_width, text_height = draw.textsize(line, font=punchline_font)
        punchline_pos = (
            (template_width - text_width) // 2,
            image_pos[1] + generated_image.height + margin + sum(text_heights),
        )
        draw.text(punchline_pos, line, fill=theme_color, font=punchline_font)
        text_heights.append(text_height)

    # Button with rounded corners
    button_text_width, button_text_height = draw.textsize(button_text, font=button_font)
    button_shape = [
        ((template_width - button_width) // 2, punchline_pos[1] + text_height + margin),
        (
            (template_width + button_width) // 2,
            punchline_pos[1] + text_height + margin + button_height,
        ),
    ]
    draw.rounded_rectangle(button_shape, radius=20, fill=theme_color)
    # Button text
    button_text_pos = (
        (template_width - button_text_width) // 2,
        button_shape[0][1] + (button_height - button_text_height) // 2,
    )
    draw.text(button_text_pos, button_text, fill="white", font=button_font)

    return background


def generate_template(
    initial_image, logo, prompt, button_text, punchline, image_color, theme_color
):
    pipeline = AutoPipelineForImage2Image.from_pretrained(
        "./models/kandinsky-2-2-decoder",
        torch_dtype=torch.float16,
        use_safetensors=True,
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
        width=256,
    ).images[0]

    template_image = image_to_template(
        generated_image, logo, button_text, punchline, theme_color
    )

    # free cpu and gpu memory
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()

    return template_image


# Set up Gradio interface
iface = gr.Interface(
    fn=generate_template,
    inputs=[
        gr.Image(type="pil", label="Initial Image"),
        gr.Image(type="pil", label="Logo"),
        gr.Textbox(label="Prompt"),
        gr.Textbox(label="Button Text"),
        gr.Textbox(label="Punchline"),
        gr.ColorPicker(label="Image Color"),
        gr.ColorPicker(label="Theme Color"),
    ],
    outputs=[gr.Image(type="pil")],
    title="Ad Template Generation Using Diffusion Models Demo",
    description="Generate ad template based on your inputs using a trained model.",
    concurrency_limit=2,
    examples=[
        [
            "./assets/city_image.jpg",  # Initial Image
            "./assets/logo.png",  # Logo
            "Big bank building finance",  # Prompt
            "Discover More!",  # Button Text
            "We Maximize Risk-Adusted Returns for Our Customers",  # Punchline
            "#00FF00",  # Image Color
            "#0000FF",  # Theme Color
        ]
    ],
)

# Run the interface
iface.launch(debug=True)
