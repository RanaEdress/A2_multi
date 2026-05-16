
import math
import torch
from PIL import Image
from tqdm import tqdm


def move_batch_to_device_and_half(batch, device):
    new_batch = {}

    for k, v in batch.items():
        v = v.to(device)

        if torch.is_floating_point(v):
            v = v.half()

        new_batch[k] = v

    return new_batch


def load_colpali_model(model_id="vidore/colpali-v1.3"):
    from colpali_engine.models import ColPali, ColPaliProcessor

    model = ColPali.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cuda:0"
    ).eval()

    processor = ColPaliProcessor.from_pretrained(model_id)

    return processor, model


def build_image_embeddings(df, processor, model, max_images=50):
    embeddings = []
    metadata = []

    subset = df.head(max_images).reset_index(drop=True)

    for idx, row in tqdm(subset.iterrows(), total=len(subset)):
        try:
            image = Image.open(row["image_path"]).convert("RGB")

            batch_images = processor.process_images([image])
            batch_images = move_batch_to_device_and_half(
                batch_images,
                model.device
            )

            with torch.no_grad():
                image_embedding = model(**batch_images)

            image_embedding = image_embedding.detach().cpu().float()

            embeddings.append(image_embedding)

            metadata.append({
                "sample_id": idx,
                "subject_id": row["subject_id"],
                "image_path": row["image_path"],
                "report": row["first_report"]
            })

        except Exception as e:
            print(f"Error at index {idx}: {e}")

    return embeddings, metadata


def retrieve_with_colpali(
    question,
    image_embeddings,
    metadata,
    processor,
    model,
    top_k=3
):
    batch_queries = processor.process_queries([question])
    batch_queries = move_batch_to_device_and_half(
        batch_queries,
        model.device
    )

    with torch.no_grad():
        query_embedding = model(**batch_queries)

    query_embedding = query_embedding.detach().cpu().float()

    scores = []

    for idx, image_embedding in enumerate(image_embeddings):
        try:
            score = processor.score_multi_vector(
                query_embedding,
                image_embedding
            )[0].item()
        except Exception:
            score = -1e9

        if math.isnan(score) or math.isinf(score):
            score = -1e9

        scores.append((idx, score))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    retrieved = []

    for idx, score in scores[:top_k]:
        item = metadata[idx].copy()
        item["score"] = score
        retrieved.append(item)

    return retrieved
