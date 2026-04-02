"""
inference.py
------------
Classify a single commit message string using the trained dual-head model.
Mirrors classify_review() from your notebook, extended for both output heads.
"""

import torch
from dataset import INTENT_LABELS, SEVERITY_LABELS


def classify_commit(text: str, model, tokenizer, device,
                    max_length: int = None, pad_token_id: int = 50256) -> dict:
    """
    Given a raw commit message string, return predicted intent and severity.

    Returns
    -------
    {
        "intent":           "Memory Leak",
        "intent_confidence": 0.92,
        "severity":          "High",
        "severity_confidence": 0.78,
    }
    """
    model.eval()

    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]

    # Truncate  (same logic as classify_review in your notebook)
    if max_length is not None:
        input_ids = input_ids[:min(max_length, supported_context_length)]
    else:
        input_ids = input_ids[:supported_context_length]

    # Pad to max_length
    if max_length is not None:
        input_ids += [pad_token_id] * (max_length - len(input_ids))

    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)

    with torch.no_grad():
        intent_logits, severity_logits = model(input_tensor)

    intent_probs   = torch.softmax(intent_logits,   dim=-1).squeeze()
    severity_probs = torch.softmax(severity_logits, dim=-1).squeeze()

    intent_idx   = torch.argmax(intent_probs).item()
    severity_idx = torch.argmax(severity_probs).item()

    return {
        "intent":             INTENT_LABELS[intent_idx],
        "intent_confidence":  round(intent_probs[intent_idx].item(), 4),
        "severity":           SEVERITY_LABELS[severity_idx],
        "severity_confidence": round(severity_probs[severity_idx].item(), 4),
    }
