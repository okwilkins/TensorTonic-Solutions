import torch

def multi_head_attention(
    hidden_states: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    num_heads: int,
    causal: bool = False,
) -> torch.Tensor:
    """
    Returns a float32 tensor with the same shape as hidden_states.
    """
    batch, num_tokens, d_in = hidden_states.shape
    head_dim = d_in // num_heads
    mask = torch.triu(torch.ones(num_tokens, num_tokens, dtype=torch.bool), diagonal=1)

    # (B, T, D) -> (B, T, H, D) -> (B, H, T, D)
    queries = torch.matmul(hidden_states, w_q).view(batch, num_tokens, num_heads, head_dim).transpose(1, 2,)
    keys = torch.matmul(hidden_states, w_k).view(batch, num_tokens, num_heads, head_dim).transpose(1, 2)
    values = torch.matmul(hidden_states, w_v).view(batch, num_tokens, num_heads, head_dim).transpose(1, 2)

    attn_scores = queries @ keys.transpose(2, 3)
    if causal:
        attn_scores.masked_fill_(mask[:num_tokens, :num_tokens], -torch.inf)

    attn_weights = torch.softmax(attn_scores / head_dim ** 0.5, dim=-1)
    # (B, H, T, D) -> (B, T, H, D)
    # Used to all the head's dimensions can be collapsed per head
    context_vectors = (attn_weights @ values).transpose(1, 2)
    # Collapse the context vectors of each head into one dimension
    context_vectors = context_vectors.contiguous().view(batch, num_tokens, d_in)

    return context_vectors @ w_o