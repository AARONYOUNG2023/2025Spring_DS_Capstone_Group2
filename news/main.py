import os, re, pandas as pd, torch
from torch.utils.data import DataLoader
from dataset import SimpleChestXRayDataset
from combined_model import ChestXRayReportGenerator

# ----------------------------------------------------------------------
def clean_generated_text(text: str) -> str:
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)        # rm repeated words
    text = re.sub(r'([:.,])\1+', r'\1', text)             # collapse punct
    text = re.sub(r'[^\x00-\x7F]+', '', text)             # strip non‑ASCII
    return ' '.join(text.split())

# ----------------------------------------------------------------------
def main():
    csv_path = "/home/ubuntu/PycharmProjects/2025Spring_DS_Capstone_Group2/data/processed/master_dataset.csv"
    df = pd.read_csv(csv_path)

    df["image_path"] = df["image_path"].str.replace(
        "/home/ubuntu/DS_capstone/raw_dataset/test_data/NLMCXR_png",
        "/home/ubuntu/PycharmProjects/2025Spring_DS_Capstone_Group2/data/raw/NLMCXR_png"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- build model first so we can grab its tokenizer ----
    model = ChestXRayReportGenerator().to(device)
    tokenizer = model.text_decoder.tokenizer           # << NEW >>

    # ---- dataset / loader ----
    dataset = SimpleChestXRayDataset(df, tokenizer)    # << pass tokenizer >>
    loader  = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)

    # ---- load finetuned weights (if any) ----
    model_path = "/home/ubuntu/PycharmProjects/2025Spring_DS_Capstone_Group2/finetuned_model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded fine‑tuned model from {model_path}")
    else:
        print("Using randomly initialised weights")

    # ------------------------------------------------------------------
    # Forward pass on one batch (note: dataset already gives token tensors)
    # ------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images         = batch["image"].to(device)
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            loss = model(images=images,
                         input_ids=input_ids,
                         attention_mask=attention_mask,
                         labels=labels)["loss"]
            print(f"Forward pass done. Batch size: {images.size(0)} | Loss: {loss.item():.4f}")
            break                                           # only first batch

    # ------------------------------------------------------------------
    # Single‑image report generation
    # ------------------------------------------------------------------
    try:
        sample       = dataset[0]
        single_img   = sample["image"].unsqueeze(0).to(device)
        generated    = model.generate_text(single_img, prompt_text="Findings:", num_mask_tokens=8)
        print("\nGenerated report:\n", clean_generated_text(generated))
    except Exception as e:
        print("Error during generation:", e)

# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
