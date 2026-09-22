import torch

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask = None,
) -> torch.Tensor:
    """
    Returns a float32 attention tensor with shape (batch, query length, value width).
    """
    d_k = key.shape[-1]
    attn_scores = query @ key.transpose(1, 2)

    if mask is not None:
        attn_scores.masked_fill_(mask, -torch.inf)

    attn_weights = torch.softmax(attn_scores / d_k ** 0.5, dim=-1)
    return attn_weights @ value