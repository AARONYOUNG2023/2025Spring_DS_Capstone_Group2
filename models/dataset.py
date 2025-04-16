"""
dataset.py

Contains the SimpleChestXRayDataset for loading and transforming
Chest‑X‑ray images *and already‑tokenised* text.
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class SimpleChestXRayDataset(Dataset):
    """
    One item returns:
        {
            "image"        : 3×224×224 tensor,
            "input_ids"    : Tensor[L]  (int64),
            "attention_mask": Tensor[L] (int64),
            "labels"       : Tensor[L]  (int64, copy of input_ids)
        }
    """

    def __init__(self, df, tokenizer, max_len: int = 512):
        """
        Parameters
        ----------
        df : pandas.DataFrame
            Must contain columns  [image_path, findings, impression].
        tokenizer : transformers.PreTrainedTokenizer
            The tokenizer that belongs to your text decoder
            (e.g.  model.text_decoder.tokenizer).
        max_len : int
            Maximum token length (padded / truncated to this).
        """
        self.df        = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len   = max_len

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ---------- 1. image ----------
        img_path = row["image_path"]
        if isinstance(img_path, str):
            image = Image.open(img_path).convert("RGB")
        else:                                  # missing image fallback
            image = Image.new("RGB", (224, 224), color="black")
        image = self.transform(image)

        # ---------- 2. text → tokens (happens ONCE here) ----------
        txt = f"{row['findings']} {row['impression']}".strip()
        enc = self.tokenizer(
            txt,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}  # drop batch dim
        item["image"]  = image
        item["labels"] = item["input_ids"].clone()        # for causal LM loss
        return item
