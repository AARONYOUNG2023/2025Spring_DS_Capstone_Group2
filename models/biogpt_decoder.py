# biogpt_decoder.py
import torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class BioGPTDecoder(nn.Module):
    """
    Projects an image embedding to BioGPT hidden size, prepends it,
    then generates text auto‑regressively.
    """
    def __init__(self,
                 lm_id: str = "microsoft/biogpt",
                 img_dim: int = 768):
        super().__init__()
        self.tok  = AutoTokenizer.from_pretrained(lm_id)
        self.lm   = AutoModelForCausalLM.from_pretrained(lm_id)

        self.proj = nn.Linear(img_dim, self.lm.config.hidden_size)

    # -------- training forward ---------------------------------------
    def forward(self, img_emb, input_ids, labels=None):
        """
        img_emb : (B, img_dim)
        input_ids / labels : (B, L)
        """
        vis = self.proj(img_emb).unsqueeze(1)  # (B,1,H)
        tok_emb = self.lm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([vis, tok_emb], 1)  # (B,1+L,H)

        if labels is not None:
            # prepend dummy ignore_index so len(labels) == len(logits)
            pad = torch.full(
                (labels.size(0), 1),
                self.lm.config.pad_token_id,  # usually 1
                dtype=torch.long,
                device=labels.device,
            )
            labels = torch.cat([pad, labels], 1)  # (B,1+L)

        out = self.lm(inputs_embeds=inputs_embeds, labels=labels)
        return {"loss": out.loss, "logits": out.logits}

    # -------- greedy generation --------------------------------------
    @torch.no_grad()
    def generate(self, img_emb, prompt: str, max_new: int = 128):
        device = img_emb.device
        vis = self.proj(img_emb).unsqueeze(1)

        ids = self.tok(prompt, return_tensors="pt").input_ids.to(device)
        emb = self.lm.get_input_embeddings()(ids)
        gen = self.lm.generate(
            inputs_embeds=torch.cat([vis, emb], 1),
            max_new_tokens=max_new,
            eos_token_id=self.tok.eos_token_id
        )
        return self.tok.decode(gen[0], skip_special_tokens=True).strip()
