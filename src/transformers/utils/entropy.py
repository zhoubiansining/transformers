import torch
import torch.nn.functional as F
from typing import Optional


def calculate_cosine_similarity(h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarity between two hidden state tensors.

    Args:
        h1: First hidden states, shape [batch_size, hidden_size]
        h2: Second hidden states, shape [batch_size, hidden_size]

    Returns:
        similarity: Tensor of shape [batch_size] with cosine similarity per sample
    """
    h1_norm = F.normalize(h1, p=2, dim=-1)
    h2_norm = F.normalize(h2, p=2, dim=-1)
    return torch.sum(h1_norm * h2_norm, dim=-1)


def calculate_information_entropy(logits: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """
    Compute normalized information entropy of the softmax distribution over logits.

    Args:
        logits: Logits tensor, shape [batch_size, vocab_size]
        normalize: Whether to normalize entropy to [0, 1] range

    Returns:
        entropy: Tensor of shape [batch_size]
    """
    probs = F.softmax(logits.float(), dim=-1)
    entropy = -torch.sum(probs * torch.log2(probs + 1e-12), dim=-1)
    if normalize:
        vocab_size = logits.shape[-1]
        max_entropy = torch.log2(torch.tensor(vocab_size, dtype=torch.float32, device=logits.device))
        entropy = entropy / max_entropy
    return entropy


def _select_layers_from_entropies(entropies_per_layer: list[torch.Tensor]) -> torch.Tensor:
    """
    Select the layer with minimum entropy (entropy valley) per sample.

    Starting from the last layer, selects the first layer where entropy stops
    decreasing (i.e., the valley). This implements the "trough" strategy.

    Args:
        entropies_per_layer: List of entropy tensors, each shape [B], length = num_layers

    Returns:
        selected_idx: Long tensor of shape [B] with the selected layer index per sample
    """
    assert len(entropies_per_layer) >= 1

    last_entropy = entropies_per_layer[-1]
    B = last_entropy.shape[0]
    device = last_entropy.device

    selected_idx = torch.full((B,), len(entropies_per_layer) - 1, dtype=torch.long, device=device)
    selected_entropy = last_entropy
    active = torch.ones(B, dtype=torch.bool, device=device)

    for i in range(len(entropies_per_layer) - 2, -1, -1):
        curr = entropies_per_layer[i]
        better = (curr < selected_entropy) & active
        selected_idx = torch.where(better, torch.full_like(selected_idx, i), selected_idx)
        selected_entropy = torch.where(better, curr, selected_entropy)
        stop_mask = (~(curr < selected_entropy)) & active
        if stop_mask.any():
            active = active & (~stop_mask)
        if not active.any():
            break

    return selected_idx
