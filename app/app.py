import gradio as gr
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "google/medgemma-4b-it"

processor = AutoProcessor.from_pretrained(MODEL_ID)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True
)

model.eval()


def generate_report_medgemma(image):
    if image is None:
        return "Please upload a chest X-ray image."

    image = image.convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": (
                        "Generate a concise chest X-ray radiology report.\n"
                        "Use exactly this format:\n\n"
                        "Findings: ...\n"
                        "Impression: ..."
                    )
                }
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=180,
            do_sample=False
        )

    result = processor.decode(
        outputs[0][input_len:],
        skip_special_tokens=True
    )

    torch.cuda.empty_cache()
    return result.strip()


with gr.Blocks(title="Multi-Modal Chest X-Ray Intelligence System") as demo:
    gr.Markdown("# Multi-Modal Chest X-Ray Intelligence System")
    gr.Markdown("## Mode 1: Report Generation")

    image_input = gr.Image(type="pil", label="Upload Chest X-ray Image")
    generate_btn = gr.Button("Generate Report")
    report_output = gr.Textbox(label="Generated Report", lines=8)

    generate_btn.click(
        fn=generate_report_medgemma,
        inputs=image_input,
        outputs=report_output
    )

demo.launch(share=True)
