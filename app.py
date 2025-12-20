import gradio as gr
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
import spaces
from molmo_utils import process_vision_info
from typing import Iterable

from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes

colors.orange_red = colors.Color(
    name="orange_red",
    c50="#FFF0E5",
    c100="#FFE0CC",
    c200="#FFC299",
    c300="#FFA366",
    c400="#FF8533",
    c500="#FF4500",
    c600="#E63E00",
    c700="#CC3700",
    c800="#B33000",
    c900="#992900",
    c950="#802200",
)

class OrangeRedTheme(Soft):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.gray,
        secondary_hue: colors.Color | str = colors.orange_red, # Use the new color
        neutral_hue: colors.Color | str = colors.slate,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Outfit"), "Arial", "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            background_fill_primary="*primary_50",
            background_fill_primary_dark="*primary_900",
            body_background_fill="linear-gradient(135deg, *primary_200, *primary_100)",
            body_background_fill_dark="linear-gradient(135deg, *primary_900, *primary_800)",
            button_primary_text_color="white",
            button_primary_text_color_hover="white",
            button_primary_background_fill="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_primary_background_fill_hover="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_dark="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_hover_dark="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_secondary_text_color="black",
            button_secondary_text_color_hover="white",
            button_secondary_background_fill="linear-gradient(90deg, *primary_300, *primary_300)",
            button_secondary_background_fill_hover="linear-gradient(90deg, *primary_400, *primary_400)",
            button_secondary_background_fill_dark="linear-gradient(90deg, *primary_500, *primary_600)",
            button_secondary_background_fill_hover_dark="linear-gradient(90deg, *primary_500, *primary_500)",
            slider_color="*secondary_500",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_primary_shadow="*shadow_drop_lg",
            button_large_padding="11px",
            color_accent_soft="*primary_100",
            block_label_background_fill="*primary_200",
        )

orange_red_theme = OrangeRedTheme()

MODEL_ID = "allenai/SAGE-MM-Qwen3-VL-4B-SFT_RL"

print(f"Loading {MODEL_ID}...")
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    dtype="auto",
    device_map="auto"
)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    dtype="auto",
    device_map="auto"
)
print("Model loaded successfully.")

@spaces.GPU
def process_video(user_text, video_path, max_new_tokens):
    if not video_path:
        return "Please upload a video."

    # Use default prompt if user input is empty
    if not user_text.strip():
        user_text = "Describe this video in detail."

    # Construct messages for Molmo/Qwen
    messages = [
        {
            "role": "user",
            "content": [
                dict(type="text", text=user_text),
                dict(type="video", video=video_path),
            ],
        }
    ]

    # Process Vision Info using molmo_utils
    # This samples frames and handles resizing logic automatically
    try:
        _, videos, video_kwargs = process_vision_info(messages)
        videos, video_metadatas = zip(*videos)
        videos, video_metadatas = list(videos), list(video_metadatas)
    except Exception as e:
        return f"Error processing video frames: {e}"

    # Apply chat template
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Prepare inputs
    inputs = processor(
        videos=videos,
        video_metadata=video_metadatas,
        text=text,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens
        )

    generated_tokens = generated_ids[0, inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return generated_text

css = """
#main-title h1 {font-size: 2.4em !important;}
"""

with gr.Blocks() as demo:
    gr.Markdown("# **SAGE-MM-Video-Reasoning**", elem_id="main-title")
    gr.Markdown("Upload a video to get a detailed explanation or ask specific questions using [SAGE-MM-Qwen3-VL](https://huggingface.co/allenai/SAGE-MM-Qwen3-VL-4B-SFT_RL).")
    
    with gr.Row():
        with gr.Column():
            vid_input = gr.Video(label="Input Video", format="mp4", height=350)
            
            vid_prompt = gr.Textbox(
                label="Prompt", 
                value="Describe this video in detail.", 
                placeholder="Type your question here..."
            )
            
            with gr.Accordion("Advanced Settings", open=False):
                max_tokens_slider = gr.Slider(
                    minimum=128,
                    maximum=4096,
                    value=1024,
                    step=128,
                    label="Max New Tokens",
                    info="Controls the length of the generated text."
                )

            vid_btn = gr.Button("Analyze Video", variant="primary")
        
        with gr.Column():
            vid_text_out = gr.Textbox(label="Model Response", interactive=True, lines=23)
            
    gr.Examples(
        examples=[
            ["example-videos/1.mp4"],
            ["example-videos/2.mp4"],
            ["example-videos/3.mp4"],
            ["example-videos/4.mp4"],
            ["example-videos/5.mp4"],
        ],
        inputs=[vid_input],
        label="Video Examples"
    )

    vid_btn.click(
        fn=process_video,
        inputs=[vid_prompt, vid_input, max_tokens_slider],
        outputs=[vid_text_out]
    )

if __name__ == "__main__":
    demo.launch(theme=orange_red_theme, css=css, mcp_server=True, ssr_mode=False)