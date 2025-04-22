# models/combined_model.py
import torch
import torch.nn as nn
from .image_encoder import ImageEncoder      # ← relative import
from .biogpt_decoder import BioGPTDecoder    # ← relative import


class ChestXRayReportGenerator(nn.Module):
    """EfficientNet‑B4 image encoder + BioGPT‑large text decoder."""

    def __init__(self, img_dim: int = 768):
        super().__init__()
        self.image_encoder = ImageEncoder(out_dim=img_dim)
        self.text_decoder  = BioGPTDecoder(img_dim=img_dim)

    # ---------------- training / inference forward ------------------
    def forward(self, images: torch.Tensor,
                input_ids: torch.Tensor,
                labels: torch.Tensor | None = None):
        img_emb = self.image_encoder(images)          # (B, 768)
        return self.text_decoder(img_emb, input_ids, labels)

    # ---------------- convenience generate wrapper ------------------
    @torch.no_grad()
    def generate_text(self,
                      images: torch.Tensor,
                      prompt: str = "",
                      max_new_tokens: int = 128) -> str:
        img_emb = self.image_encoder(images)
        return self.text_decoder.generate(img_emb, prompt, max_new_tokens)
