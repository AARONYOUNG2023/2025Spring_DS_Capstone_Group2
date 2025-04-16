import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import pandas as pd
from dataset import SimpleChestXRayDataset
from combined_model import ChestXRayReportGenerator

EPOCHS = 5
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSV_PATH = "D:/S4Cap/pythonProject/master_dataset.csv"
SAVE_PATH = "finetuned_model.pth"

def main():
    df = pd.read_csv(CSV_PATH)
    df["image_path"] = df["image_path"].apply(
        lambda p: p.replace(
            "/home/ubuntu/DS_capstone/raw_dataset/test_data/NLMCXR_png",
            r"D:/S4Cap/pythonProject/2025Spring_DS_Capstone_Group2/data/raw/NLMCXR_png"
        ) if isinstance(p, str) else p
    )

    dataset = SimpleChestXRayDataset(df)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    model = ChestXRayReportGenerator().to(DEVICE)
    tokenizer = model.text_decoder.tokenizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images = batch["image"].to(DEVICE)
            texts = batch["text"]

            encoding = tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            )
            input_ids = encoding["input_ids"].to(DEVICE)
            attention_mask = encoding["attention_mask"].to(DEVICE)
            labels = input_ids.clone()

            outputs = model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Average Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Model saved to {SAVE_PATH}")

if __name__ == '__main__':
    main()
