"""
dataset.py

Contains the SimpleChestXRayDataset for loading and transforming
Chest X-ray images and their associated text fields.
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class SimpleChestXRayDataset(Dataset):
    """
    A PyTorch Dataset for chest X-ray images and associated text (findings/impression).
    """

    def __init__(self, df):
        """
        :param df: A pandas DataFrame with columns like [image_path, findings, impression].
        """
        self.df = df.reset_index(drop=True)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row["image_path"]
        if image_path and isinstance(image_path, str):
            image = Image.open(image_path).convert("RGB")
        else:
            image = Image.new("RGB", (224, 224), color="black")


        image = self.transform(image)
        text_input = (str(row["findings"]) + " " + str(row["impression"])).strip()

        return {
            "image": image,
            "text": text_input
        }
