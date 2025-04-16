import os, torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import pandas as pd
from dataset import SimpleChestXRayDataset
from combined_model import ChestXRayReportGenerator

EPOCHS      = 10
BATCH_SIZE  = 16
LR          = 2e-5
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSV_PATH    = r"C:\Users\yangy\Desktop\2025Spring\DSCapstone\data\processed\master_dataset.csv"
SAVE_PATH   = "finetuned_model.pth"

def main():
    df = pd.read_csv(CSV_PATH)
    df["image_path"] = df["image_path"].str.replace(
        "/home/ubuntu/DS_capstone/raw_dataset/test_data/NLMCXR_png",
        r"D:/S4Cap/pythonProject/2025Spring_DS_Capstone_Group2/data/raw/NLMCXR_png")

    model     = ChestXRayReportGenerator().to(DEVICE)
    tokenizer = model.text_decoder.tokenizer

    dataset = SimpleChestXRayDataset(df, tokenizer)
    dataloader = DataLoader(dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            num_workers=os.cpu_count(),
                            pin_memory=True,
                            persistent_workers=True)

    optimizer = AdamW(model.parameters(), lr=LR)
    scaler    = torch.cuda.amp.GradScaler()        # mixed precision
    model     = torch.compile(model)               # PyTorch ≥ 2.0
    torch.backends.cudnn.benchmark = True

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images         = batch["image"].to(DEVICE, non_blocking=True)
            input_ids      = batch["input_ids"].to(DEVICE, non_blocking=True)
            attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
            labels         = batch["labels"].to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                loss = model(images=images,
                             input_ids=input_ids,
                             attention_mask=attention_mask,
                             labels=labels)["loss"]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} | Avg Loss: {total_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Model saved to {SAVE_PATH}")

if __name__ == "__main__":
    main()
