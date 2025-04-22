"""
dataset.py

SimpleChestXRayDataset: loads an image + tokenised text.
This version keeps everything the same **except** the tokenizer call,
so we reserve one slot for the image/CLS token and avoid the 513‑vs‑512
broadcast error.
"""

from PIL import Image
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset


class SimpleChestXRayDataset(Dataset):
    """
    Returns dict:
        {
            "image"        : 3×224×224 tensor,
            "input_ids"    : LongTensor[L],
            "attention_mask": LongTensor[L],
            "labels"       : LongTensor[L]   # copy of input_ids
        }
    """

    def __init__(self, df, tokenizer, max_len: int = 512):
        self.df        = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len   = max_len

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ---------- 1. image ----------
        img_path = row["image_path"]
        if isinstance(img_path, str):
            image = Image.open(img_path).convert("RGB")
        else:
            image = Image.new("RGB", (224, 224), color="black")
        image = self.transform(image)

        # ---------- 2. text → tokens ----------
        txt = f"{row['findings']} {row['impression']}".strip()

        enc = self.tokenizer(
            txt,
            add_special_tokens=False,     # model adds CLS/SEP later
            padding="max_length",
            truncation=True,
            max_length=self.max_len - 1,  # reserve 1 slot for extra token
            return_tensors="pt",
        )

        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["image"]  = image
        item["labels"] = item["input_ids"].clone()
        return item
