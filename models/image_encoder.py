"""
image_encoder.py

Defines a Densenet-based image encoder using a model hosted on Hugging Face:
https://huggingface.co/timm/densenet121.ra_in1k

It extracts a feature vector from a 224x224 chest X-ray image, suitable for
downstream text decoding models.
"""

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
import timm


class DensenetImageEncoder(nn.Module):
    """
    CNN-based Image Encoder wrapping a DenseNet-121 architecture
    (weights from timm/huggingface).
    Produces an image feature vector of dimension `feature_dim`.
    """

    def __init__(self, feature_dim: int = 768, hub_repo_id: str = "timm/densenet121.ra_in1k"):
        """
        :param feature_dim: Dimension of the final projected image embedding.
        :param hub_repo_id: Hugging Face repo ID where the DenseNet weights are stored.
        """
        super().__init__()
        checkpoint_filepath = hf_hub_download(repo_id=hub_repo_id, filename="pytorch_model.bin")
        self.base_model = timm.create_model("densenet121", pretrained=False)
        state_dict = torch.load(checkpoint_filepath, map_location="cpu")
        self.base_model.load_state_dict(state_dict)
        self.base_model.classifier = nn.Identity()
        in_features = 1024
        self.projector = nn.Linear(in_features, feature_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for image encoder.

        :param images: A batch of images, shape (batch_size, 3, 224, 224).
        :return: A batch of image embeddings, shape (batch_size, feature_dim).
        """
        features = self.base_model(images)
        projected = self.projector(features)
        return projected
