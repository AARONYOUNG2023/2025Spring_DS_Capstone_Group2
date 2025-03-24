import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import os

def evaluate_clip_alignment(image_paths, generated_texts, clip_model_name="openai/clip-vit-base-patch32"):
    """
    Computes average image-text alignment scores (cosine similarity) using CLIP.
    :param image_paths: list of image file paths aligned with generated_texts
    :param generated_texts: list of textual reports
    :param clip_model_name: Hugging Face CLIP model name
    :return: average similarity score (0 to 1) and list of individual similarities
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(clip_model_name).to(device)
    processor = CLIPProcessor.from_pretrained(clip_model_name)

    similarities = []

    for img_path, text in zip(image_paths, generated_texts):
        if not os.path.exists(img_path):
            continue
        image = Image.open(img_path).convert("RGB")

        inputs = processor(
            text=[text],
            images=image,
            return_tensors="pt",
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds

        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        sim = (image_embeds * text_embeds).sum(dim=-1).item()  # single float
        similarities.append(sim)


    avg_sim = np.mean(similarities) if similarities else 0.0
    return avg_sim, similarities

if __name__ == "__main__":
    example_image_paths = [
        r"/path/to/image1.png",
        r"/path/to/image2.png"
    ]
    example_generated_texts = [
        "A normal chest radiograph with no signs of consolidation.",
        "Findings consistent with pneumonia in the right lower lobe."
    ]

    avg_sim, sim_list = evaluate_clip_alignment(example_image_paths, example_generated_texts)
    print(f"Average CLIP Similarity: {avg_sim:.4f}")
    print("Individual Similarities:", sim_list)
