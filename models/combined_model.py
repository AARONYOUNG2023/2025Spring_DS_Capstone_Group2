"""
combined_model.py

Demonstrates how to combine the DensenetImageEncoder (from image_encoder.py)
and the ClinicalBertDecoder (from transformers_text_decoder.py) to form
a pipeline for generating or refining medical text from Chest X-ray images.

Classes:
    ChestXRayReportGenerator
"""

import torch
import torch.nn as nn
from image_encoder import DensenetImageEncoder
from transformers_text_decoder import ClinicalBertDecoder


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
        image_embeds = self.image_encoder(images)

        output_dict = self.text_decoder(
            image_embeds=image_embeds,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return output_dict

    @torch.no_grad()
    def generate_text(self,
                      images: torch.Tensor,
                      prompt_text: str = "",
                      num_mask_tokens: int = 20) -> str:
        """
        Single-sample generation method that:
        1) Encodes a single image.
        2) Calls the decoder's generate() with a prompt and a set of mask tokens to fill.

        :param images: (1, 3, 224, 224) single chest X-ray image
        :param prompt_text: partial text prompt
        :param num_mask_tokens: how many tokens to attempt to fill
        :return: generated text as a string
        """
        image_embeds = self.image_encoder(images)  # (1, image_feature_dim)
        generated = self.text_decoder.generate(
            image_embeds=image_embeds,
            prompt_text=prompt_text,
            num_mask_tokens=num_mask_tokens
        )
        return generated
