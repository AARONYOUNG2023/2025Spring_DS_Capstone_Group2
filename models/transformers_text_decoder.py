"""
clinical_bert_decoder.py

Defines a "decoder" module that leverages the ClinicalBERT model:
https://huggingface.co/medicalai/ClinicalBERT

BERT is traditionally an encoder architecture. Here, we illustrate a workaround
to incorporate image features (from a CNN) by prepending a learnable "visual token"
to the word embeddings, letting the model attend to image features in a
masked language modeling scenario.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM, AutoTokenizer


class ClinicalBertDecoder(nn.Module):
    """
    A pseudo-decoder that uses BERT for masked language modeling.
    We inject image embeddings by projecting them into BERT's embedding space
    and prepending them as a "visual token" to the textual input. This is not a
    classic seq2seq approach, but demonstrates how one might incorporate
    domain-specific BERT (ClinicalBERT) for generation-like tasks.
    """

    def __init__(self,
                 bert_model_name: str = "medicalai/ClinicalBERT",
                 hidden_dim: int = 768,
                 max_length: int = 50):
        """
        :param bert_model_name: Name or path of the ClinicalBERT checkpoint.
        :param hidden_dim: The hidden size for projection from image feature to a "token" embedding.
        :param max_length: Maximum sequence length for generation.
        """
        super().__init__()
        self.max_length = max_length

        # Load BERT (MLM) model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.bert_mlm = AutoModelForMaskedLM.from_pretrained(bert_model_name)

        # We will project the image embedding into BERT's embedding space
        # and treat it as a learned token appended to the sequence.
        self.visual_token_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self,
                image_embeds: torch.Tensor,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                labels: torch.Tensor = None):
        """
        Forward pass for training or fine-tuning with masked language modeling.

        :param image_embeds: (batch_size, hidden_dim) from the CNN encoder.
        :param input_ids: (batch_size, seq_len) tokenized input text IDs.
        :param attention_mask: (batch_size, seq_len) attention mask.
        :param labels: (batch_size, seq_len) for MLM training if doing supervised learning.
        :return: BERT MLM outputs with loss (if labels are provided).
        """
        batch_size = image_embeds.size(0)

        # Project image embeddings to match BERT embedding dimension
        projected_img = self.visual_token_proj(image_embeds)  # (batch_size, hidden_dim)

        # Expand dims to (batch_size, 1, hidden_dim)
        visual_token = projected_img.unsqueeze(1)

        # Get the BERT word embeddings
        # We'll do an embedding lookup manually to combine with the visual token
        # The BERT model's embeddings are at:
        bert_embeddings = self.bert_mlm.bert.embeddings.word_embeddings

        # Convert input_ids to their embeddings
        token_embeds = bert_embeddings(input_ids)  # (batch_size, seq_len, hidden_dim)

        # Prepend the visual token embeddings
        combined_embeds = torch.cat([visual_token, token_embeds], dim=1)  # (batch_size, seq_len+1, hidden_dim)

        # Adjust the attention mask to account for new prepended token
        # The new token is always "visible" (1), so:
        expanded_mask = torch.cat([
            torch.ones(batch_size, 1, device=attention_mask.device, dtype=attention_mask.dtype),
            attention_mask
        ], dim=1)  # (batch_size, seq_len+1)

        # We pass these embeddings to the BERT model's forward, but we need to bypass
        # the standard input_ids route. We'll use `inputs_embeds`:
        outputs = self.bert_mlm.bert(
            inputs_embeds=combined_embeds,
            attention_mask=expanded_mask,
            return_dict=True
        )

        # For MLM, we use the final hidden states from BERT's encoder output:
        sequence_output = outputs.last_hidden_state  # (batch_size, seq_len+1, hidden_dim)

        # BERT's masked LM head is self.bert_mlm.cls
        prediction_scores = self.bert_mlm.cls(sequence_output)  # (batch_size, seq_len+1, vocab_size)

        # If we have labels, we should shift them to match the shape of the prediction scores
        # The first token in the output is the visual token, which might not have a label
        # Typically we might shift the labels by 1 or ignore the first token label.
        # For simplicity, let's assume we ignore the first token label:
        if labels is not None:
            # Expand labels to the same size with a dummy label for the new visual token
            dummy_label = torch.full((batch_size, 1), -100, device=labels.device)
            shifted_labels = torch.cat([dummy_label, labels], dim=1)  # (batch_size, seq_len+1)
        else:
            shifted_labels = None

        # Compute MLM loss if labels are provided
        loss = None
        if shifted_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # Flatten the prediction scores and labels
            prediction_scores_2d = prediction_scores.view(-1, prediction_scores.size(-1))
            shifted_labels_2d = shifted_labels.view(-1)
            loss = loss_fct(prediction_scores_2d, shifted_labels_2d)

        return {
            "loss": loss,
            "logits": prediction_scores
        }

    @torch.no_grad()
    def generate(self, image_embeds: torch.Tensor, prompt_text: str = "", num_mask_tokens: int = 20):
        """
        A simplistic "generate" method that demonstrates filling in masked tokens.
        We create a single input with a series of [MASK] tokens after the prompt, and
        rely on the masked language modeling head to fill them.
        This is NOT a full auto-regressive generation approach (which BERT does not natively support).
        Instead, it shows one technique to produce tokens from a BERT-MLM model.

        :param image_embeds: (batch_size=1, hidden_dim) single image embedding.
        :param prompt_text: Optional partial text to start with.
        :param num_mask_tokens: Number of masked tokens to attempt to fill.
        :return: Generated text string (best-effort with MLM).
        """
        # Currently, this method is set up for batch_size=1 usage
        batch_size = image_embeds.size(0)
        if batch_size != 1:
            raise ValueError("generate() currently supports batch_size=1 only.")

        # Tokenize the prompt
        encoding = self.tokenizer.encode_plus(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=True
        )
        input_ids = encoding["input_ids"]  # shape (1, seq_len)
        attention_mask = encoding["attention_mask"]

        # Create masked tokens after the prompt
        mask_token_id = self.tokenizer.mask_token_id
        mask_tokens = torch.full((1, num_mask_tokens), mask_token_id, dtype=torch.long)
        input_ids = torch.cat([input_ids, mask_tokens], dim=1)  # shape (1, seq_len + num_mask_tokens)
        attention_mask = torch.cat([attention_mask, torch.ones_like(mask_tokens)], dim=1)

        # Forward pass
        outputs = self.forward(
            image_embeds=image_embeds,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # We only take the logits from the final step
        logits = outputs["logits"]  # (1, seq_len+num_mask_tokens+1, vocab_size)

        # The simplest approach: for each mask position, pick the most likely token (greedy fill).
        # We ignore the appended "visual token" offset here for demonstration.
        generated_ids = input_ids.clone()
        vocab_size = logits.size(-1)

        # Indices that correspond to the newly masked tokens
        # Those are positions from [original seq_len, original seq_len+num_mask_tokens)
        original_seq_len = encoding["input_ids"].shape[1]
        for i in range(num_mask_tokens):
            current_position = original_seq_len + i
            # The corresponding logits: (1, vocab_size)
            token_logits = logits[0, current_position, :]
            predicted_id = torch.argmax(token_logits, dim=-1).unsqueeze(0)
            generated_ids[0, current_position] = predicted_id

        # Decode the resulting tokens
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return generated_text
