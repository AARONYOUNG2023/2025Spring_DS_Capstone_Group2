"""
train_ar.py
Auto‑regressive training loop for EfficientNet‑B4 + BioGPT.

Run:
    python train_ar.py --epochs 5 --bs 8
"""

import os, math, time, warnings, nltk
import torch, pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
from models.combined_model import ChestXRayReportGenerator


warnings.filterwarnings("ignore")


# ------------------------------------------------------------------ #
# Dataset: loads image + tokenised report
# ------------------------------------------------------------------ #
class ARDataset(Dataset):
    def __init__(self, df, tokenizer, transform):
        self.df   = df.reset_index(drop=True)
        self.tok  = tokenizer
        self.tfms = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # -------- image --------
        img_path = row["image_path"]
        img = Image.open(img_path).convert("RGB") if (
            isinstance(img_path, str) and os.path.exists(img_path)
        ) else Image.new("RGB", (224, 224), "black")
        img = self.tfms(img)

        # -------- text → ids --------
        text_val = row["report"]
        text = str(text_val) if pd.notnull(text_val) else ""
        ids = self.tok(
            text,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        return {"image": img, "input_ids": ids, "labels": ids.clone(), "report": text}

# ------------------------------------------------------------------ #
def build_loaders(csv_path, tokenizer, bs, num_workers=4):
    df = pd.read_csv(csv_path)
    tfms = Compose([Resize((224,224)),
                    ToTensor(),
                    Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

    train_ds = ARDataset(df[df.split=="train"], tokenizer, tfms)
    val_ds   = ARDataset(df[df.split=="val"],   tokenizer, tfms)


    def collate(batch):
        images = torch.stack([b["image"] for b in batch])
        ids = torch.stack([b["input_ids"] for b in batch])
        labels = torch.stack([b["labels"] for b in batch])
        reports = [b["report"] for b in batch]
        return {"images": images, "input_ids": ids, "labels": labels, "reports": reports}

    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True,
                          num_workers=num_workers, collate_fn=collate, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False,
                        num_workers=num_workers, collate_fn=collate, pin_memory=True)

    return train_dl, val_dl

# ------------------------------------------------------------------ #
def bleu_score(preds, refs):
    smoothie = SmoothingFunction().method4
    refs_tok = [[r.lower().split()] for r in refs]
    preds_tok = [p.lower().split()  for p in preds]
    return corpus_bleu(refs_tok, preds_tok, smoothing_function=smoothie)

# ------------------------------------------------------------------ #
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = ChestXRayReportGenerator().to(device)
    model.text_decoder.lm.gradient_checkpointing_enable()
    tok    = model.text_decoder.tok

    train_dl, val_dl = build_loaders(args.csv, tok, args.bs)

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sch = CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler()

    best_bleu, patience = 0.0, 0
    for ep in range(1, args.epochs+1):
        model.train()
        tot, n = 0.0, 0
        t0 = time.time()
        for batch in train_dl:
            imgs = batch["images"].to(device, non_blocking=True)
            ids  = batch["input_ids"].to(device, non_blocking=True)
            labels= batch["labels"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                loss = model(images=imgs, input_ids=ids, labels=labels)["loss"]
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()

            tot += loss.item(); n += 1
        sch.step()
        print(f"Epoch {ep}/{args.epochs}  train loss {tot/n:.4f}  "
              f"time {(time.time()-t0):.1f}s", flush=True)

        # -------- validation BLEU ------------------------------------
        model.eval();
        preds, refs = [], []
        with torch.no_grad():
            for batch in val_dl:
                imgs = batch["images"]
                for img, ref in zip(imgs, batch["reports"]):
                    gen = model.generate_text(
                        img.unsqueeze(0).to(device),  # (1,3,224,224)
                        prompt="",
                        max_new_tokens=64,
                    )
                    preds.append(gen)
                    refs.append(str(ref))
        bleu = bleu_score(preds, refs)
        print(f"          val BLEU‑4 {bleu:.4f}")

        # early‑stop
        if bleu > best_bleu:
            best_bleu, patience = bleu, 0
            torch.save(model.state_dict(), args.out)
            print("          ✓ best BLEU → model saved")
        else:
            patience += 1
            if patience >= args.patience:
                print("          no improvement → early stop")
                break

# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import argparse, pathlib
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",
        default="/home/ubuntu/PycharmProjects/2025Spring_DS_Capstone_Group2/data/processed/master_dataset.csv")
    parser.add_argument("--out",
        default="/home/ubuntu/PycharmProjects/2025Spring_DS_Capstone_Group2/finetuned_biogpt.pth")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--bs",     type=int, default=4)
    parser.add_argument("--lr",     type=float, default=2e-5)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    train(args)
