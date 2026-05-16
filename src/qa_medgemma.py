
import torch


def answer_question_medgemma(
    question,
    context,
    processor,
    model,
    max_new_tokens=80
):

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are a medical QA assistant.\n"
                        "Use only the provided retrieved context.\n"
                        "Answer concisely.\n\n"
                        f"Retrieved context:\n{context}\n\n"
                        f"Question: {question}"
                    )
                }
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

    decoded = processor.decode(
        outputs[0][input_len:],
        skip_special_tokens=True
    )

    return decoded.strip()
