import os
import re
import pandas as pd
import torch
from torch.utils.data import DataLoader
from dataset import SimpleChestXRayDataset
from combined_model import ChestXRayReportGenerator

def clean_generated_text(text: str) -> str:
    #
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)
    text = re.sub(r'([:.,])\1+', r'\1', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = ' '.join(text.split())
    return text

def main():
    csv_path = r"D:\S4Cap\pythonProject\master_dataset.csv"
    df = pd.read_csv(csv_path)

    df["image_path"] = df["image_path"].apply(
        lambda p: p.replace(
            "/home/ubuntu/DS_capstone/raw_dataset/test_data/NLMCXR_png",
            r"D:\S4Cap\pythonProject\2025Spring_DS_Capstone_Group2\data\raw\NLMCXR_png"
        ) if isinstance(p, str) else p
    )

    dataset = SimpleChestXRayDataset(df)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChestXRayReportGenerator().to(device)

    #
    model_path = r"D:\S4Cap\pythonProject\finetuned_model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded fine-tuned model from {model_path}")
    else:
        print("Using randomly initialized weights (no finetuned model found)")

    model.eval()
    tokenizer = model.text_decoder.tokenizer

    # Forward Pass（Loss）
    for batch in data_loader:
        images = batch["image"].to(device)
        texts = batch["text"]

        encoding = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        labels = input_ids.clone()

        outputs = model(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs["loss"]
        print(f"Forward pass done. Batch size: {images.size(0)} | Loss: {loss.item():.4f}")
        break


    try:
        single_sample = dataset[0]
        single_image = single_sample["image"].unsqueeze(0).to(device)
        prompt_text = "Findings:"
        num_mask_tokens = 8

        generated_report = model.generate_text(
            images=single_image,
            prompt_text=prompt_text,
            num_mask_tokens=num_mask_tokens
        )
        cleaned_report = clean_generated_text(generated_report)

        print("\nGenerated Report:")
        print(cleaned_report)

    except Exception as e:
        print(f"Error during generation: {e}")

if __name__ == "__main__":
    main()
