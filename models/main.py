"""
combined_model.py

Demonstrates how to combine the DensenetImageEncoder (from image_encoder.py)
and the ClinicalBertDecoder (from clinical_bert_decoder.py) to form a pipeline
for generating or refining medical text from Chest X-ray images.

We assume you have a preprocessed DataFrame (e.g. master_dataset.csv)
with columns: [image_path, findings, impression], plus any other data needed.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd

# Local imports from the two modules we created:
from image_encoder import DensenetImageEncoder
from clinical_bert_decoder import ClinicalBertDecoder

# Optionally, you can reuse your transforms from your data preprocessing script
# from data_preprocessing import NLMChestXRayDataset, get_transforms   # Example


class ChestXRayReportGenerator(nn.Module):
    """
    End-to-end pipeline: CNN image encoder + ClinicalBERT "decoder".
    """

    def __init__(self,
                 image_feature_dim: int = 768,
                 bert_model_name: str = "medicalai/ClinicalBERT"):
        """
        :param image_feature_dim: Dimensionality of the image encoder's projection.
        :param bert_model_name: Hugging Face model ID for the ClinicalBERT weights.
        """
        super().__init__()
        self.image_encoder = DensenetImageEncoder(feature_dim=image_feature_dim)
        self.text_decoder = ClinicalBertDecoder(
            bert_model_name=bert_model_name,
            hidden_dim=image_feature_dim
        )

    def forward(self,
                images: torch.Tensor,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                labels: torch.Tensor = None):
        """
        Forward pass: image -> feature -> appended to BERT for masked language modeling.
        :param images: (batch_size, 3, 224, 224) chest X-ray images
        :param input_ids: (batch_size, seq_len) token IDs for textual input
        :param attention_mask: (batch_size, seq_len)
        :param labels: (batch_size, seq_len), optional for MLM loss
        :return: dictionary with "loss" (if labels) and "logits"
        """
        # 1) Encode images
        image_embeds = self.image_encoder(images)  # (batch_size, image_feature_dim)

        # 2) Decode text
        output_dict = self.text_decoder(
            image_embeds=image_embeds,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return output_dict

    @torch.no_grad()
    def generate_text(self, images: torch.Tensor, prompt_text: str = "", num_mask_tokens: int = 20):
        """
        Single-sample generation method that:
        1) Encodes a single image.
        2) Calls the decoder's generate() with a prompt and
           a set of mask tokens to fill.

        :param images: (1, 3, 224, 224) single chest X-ray image
        :param prompt_text: partial text prompt
        :param num_mask_tokens: how many tokens to attempt to fill
        :return: generated text as a string
        """
        # Assume a batch of size 1 for simplicity
        image_embeds = self.image_encoder(images)  # (1, image_feature_dim)
        generated = self.text_decoder.generate(
            image_embeds=image_embeds,
            prompt_text=prompt_text,
            num_mask_tokens=num_mask_tokens
        )
        return generated


def main():
    """
    Example of loading the processed CSV dataset, building a DataLoader,
    and running one batch through the combined model for demonstration.
    """

    # -------------------------------------------------------------------------
    # 1) Load the dataset
    # -------------------------------------------------------------------------
    csv_path = r"C:\Users\yangy\PycharmProjects\2025Spring_DS_Capstone_Group2\data\processed\master_dataset.csv"
    df = pd.read_csv(csv_path)

    # Suppose you have a custom dataset & transforms from your data preprocessing script
    # from your_data_script import NLMChestXRayDataset, get_transforms
    # transform = get_transforms()
    # dataset = NLMChestXRayDataset(df, transform=transform)

    # For demonstration, we won't import your entire data preprocessing again;
    # We'll just show a pseudo-code approach:
    from torch.utils.data import Dataset
    from PIL import Image
    import torchvision.transforms as transforms

    class SimpleChestXRayDataset(Dataset):
        def __init__(self, df):
            self.df = df.reset_index(drop=True)
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
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

            # We'll combine findings + impression as a single string for demonstration:
            text_input = (str(row["findings"]) + " " + str(row["impression"])).strip()

            return {
                "image": image,
                "text": text_input
            }

    dataset = SimpleChestXRayDataset(df)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=2)

    # -------------------------------------------------------------------------
    # 2) Initialize the combined model
    # -------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChestXRayReportGenerator().to(device)

    # -------------------------------------------------------------------------
    # 3) Tokenizer for the text side
    #    We can reuse the ClinicalBertDecoder’s tokenizer
    # -------------------------------------------------------------------------
    tokenizer = model.text_decoder.tokenizer

    # -------------------------------------------------------------------------
    # 4) Demonstrate one batch
    # -------------------------------------------------------------------------
    for batch in data_loader:
        images = batch["image"].to(device)  # (batch_size, 3, 224, 224)
        texts = batch["text"]

        # Tokenize texts
        encoding = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        # Optionally, build MLM labels (you would normally do random masking)
        # For a simple demonstration, let's keep the labels the same as input_ids:
        labels = input_ids.clone()

        # Forward pass
        outputs = model(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs["loss"]
        logits = outputs["logits"]

        print(f"Batch size: {images.size(0)} | Loss: {loss.item():.4f}")
        break

    # -------------------------------------------------------------------------
    # 5) Demonstrate generation (batch_size=1 usage)
    # -------------------------------------------------------------------------
    # Just take the first sample from the DataLoader for generation
    single_sample = dataset[0]
    single_image = single_sample["image"].unsqueeze(0).to(device)
    prompt_text = "The chest X-ray reveals"

    model.eval()
    generated_report = model.generate_text(
        images=single_image,
        prompt_text=prompt_text,
        num_mask_tokens=15
    )
    print("Generated (BERT-MLM style) text:\n", generated_report)


if __name__ == "__main__":
    main()
