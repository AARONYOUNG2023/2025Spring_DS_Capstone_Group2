"""
main.py

Example script that:
1. Loads the processed CSV dataset (master_dataset.csv).
2. Builds a DataLoader using SimpleChestXRayDataset.
3. Initializes the ChestXRayReportGenerator model.
4. Demonstrates a single training batch (forward pass) and a simple generation example.
"""

import pandas as pd
import torch
from torch.utils.data import DataLoader
from dataset import SimpleChestXRayDataset
from combined_model import ChestXRayReportGenerator

def main():
    # -------------------------------------------------------------------------
    # 1) Load the dataset
    # -------------------------------------------------------------------------
    csv_path = r"C:\Users\yangy\PycharmProjects\2025Spring_DS_Capstone_Group2\data\processed\master_dataset.csv"
    df = pd.read_csv(csv_path)

    dataset = SimpleChestXRayDataset(df)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=2)

    # -------------------------------------------------------------------------
    # 2) Initialize the combined model
    # -------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChestXRayReportGenerator().to(device)

    # -------------------------------------------------------------------------
    # 3) Tokenizer for the text side (from the text_decoder inside our model)
    # -------------------------------------------------------------------------
    tokenizer = model.text_decoder.tokenizer

    # -------------------------------------------------------------------------
    # 4) Demonstrate one batch forward pass
    # -------------------------------------------------------------------------
    for batch in data_loader:
        images = batch["image"].to(device)
        texts = batch["text"]

        encoding = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
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
        print(f"Batch size: {images.size(0)} | Loss: {loss.item():.4f}")
        break

    # -------------------------------------------------------------------------
    # 5) Demonstrate generation (batch_size=1 usage)
    # -------------------------------------------------------------------------
    model.eval()
    single_sample = dataset[0]
    single_image = single_sample["image"].unsqueeze(0).to(device)
    prompt_text = "The chest X-ray reveals"

    generated_report = model.generate_text(
        images=single_image,
        prompt_text=prompt_text,
        num_mask_tokens=15
    )
    print("Generated (BERT-MLM style) text:\n", generated_report)

if __name__ == "__main__":
    main()
