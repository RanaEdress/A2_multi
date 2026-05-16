
import torch
from PIL import Image


def generate_report_medgemma(
    image_path,
    processor,
    model,
    max_new_tokens=180
):

    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": (
                        "Generate a concise chest X-ray radiology report. "
                        "Use exactly this format:\n"
                        "Findings: ...\n"
                        "Impression: ..."
                    )
                },
            ],
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
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

    generated_report = processor.decode(
        outputs[0][input_len:],
        skip_special_tokens=True
    )

    return generated_report.strip()
