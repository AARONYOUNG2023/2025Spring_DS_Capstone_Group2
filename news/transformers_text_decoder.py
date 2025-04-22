"""
clinical_bert_decoder.py

A "decoder" that adapts ClinicalBERT (a masked‑LM) for vision‑language
tasks: a DenseNet image feature is projected to a BERT‑sized vector and
prepended as a *visual token* so the model can attend to the image.

Only functional changes:

1.  **Device‑safety** – every tensor created in `forward()` or `generate()`
    is moved to the same device as the incoming `image_embeds`.
2.  A tiny safeguard in `forward()` to cast `attention_mask` to `long`
    (some callers pass `bool` masks).

No other logic was touched.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM, AutoTokenizer


class ClinicalBertDecoder(nn.Module):
    """
    BERT‑MLM + one learnable visual token to inject image information.
    """

    def __init__(
        self,
        bert_model_name: str = "medicalai/ClinicalBERT",
        hidden_dim: int = 768,
        max_length: int = 50,
    ):
        super().__init__()
        self.max_length = max_length

        # --- load tokenizer & MLM head ---
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.bert_mlm  = AutoModelForMaskedLM.from_pretrained(bert_model_name)

        # --- image→token projection ---
        self.visual_token_proj = nn.Linear(hidden_dim, hidden_dim)

    # ------------------------------------------------------------------ #
    def forward(
        self,
        image_embeds: torch.Tensor,      # (B, hidden_dim)
        input_ids: torch.Tensor,         # (B, L)
        attention_mask: torch.Tensor,    # (B, L)
        labels: torch.Tensor | None = None,
    ):
        """
        Prepends the projected image embedding to token embeddings and
        feeds everything to ClinicalBERT's MLM head.
        """
        device = image_embeds.device

        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device, dtype=torch.long)

        # 1.  make the visual token
        projected_img = self.visual_token_proj(image_embeds)    # (B, H)
        visual_token  = projected_img.unsqueeze(1)              # (B, 1, H)

        # 2.  look up token embeddings
        token_embeds  = self.bert_mlm.get_input_embeddings()(input_ids)
        combined_embs = torch.cat([visual_token, token_embeds], dim=1)  # (B, 1+L, H)

        # 3.  expand mask to cover visual token
        B = input_ids.size(0)
        expanded_mask = torch.cat(
            [
                torch.ones(B, 1, device=device, dtype=torch.long),  # for visual token
                attention_mask,
            ],
            dim=1,
        )

        outputs = self.bert_mlm(
            inputs_embeds=combined_embs,
            attention_mask=expanded_mask,
            return_dict=True,
        )
        logits = outputs.logits                                     # (B, 1+L, vocab)

        # 4.  compute MLM loss if labels provided
        loss = None
        if labels is not None:
            dummy = torch.full((B, 1), -100, device=device, dtype=torch.long)
            shifted_labels = torch.cat([dummy, labels.to(device)], dim=1)

            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), shifted_labels.view(-1))

        return {"loss": loss, "logits": logits}

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate(
        self,
        image_embeds: torch.Tensor,       # (1, hidden_dim)
        prompt_text: str = "",
        num_mask_tokens: int = 20,
    ) -> str:
        """
        Fills in `num_mask_tokens` after the prompt using MLM logits.
        Note: supports batch_size = 1 for simplicity.
        """
        device = image_embeds.device
        if image_embeds.size(0) != 1:
            raise ValueError("generate() supports batch_size=1 only.")

        # 1. tokenise prompt on the *same device*
        enc = self.tokenizer.encode_plus(
            prompt_text, add_special_tokens=True, return_tensors="pt"
        )
        input_ids      = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        # 2. append [MASK] tokens (also on device)
        mask_id = self.tokenizer.mask_token_id
        mask_tokens = torch.full(
            (1, num_mask_tokens), mask_id, dtype=torch.long, device=device
        )
        input_ids      = torch.cat([input_ids,      mask_tokens], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(mask_tokens)], dim=1)

        # 3. forward pass (image + tokens)
        logits = self.forward(
            image_embeds=image_embeds,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )["logits"]

        # 4. greedy fill the mask positions
        generated_ids = input_ids.clone()
        start = enc["input_ids"].shape[1]           # first mask index
        for i in range(num_mask_tokens):
            pos = start + i
            predicted = logits[0, pos].argmax(dim=-1)
            generated_ids[0, pos] = predicted

        # 5. decode & return
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
