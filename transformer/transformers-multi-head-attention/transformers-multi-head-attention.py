import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    # Your code here
    batch_size, seq_len_q, d_model = Q.shape
    seq_len_k = K.shape[1]
    d_k = d_model // num_heads

    # Linear projections
    Q_proj = Q @ W_q   # (batch, seq_len_q, d_model)
    K_proj = K @ W_k   # (batch, seq_len_k, d_model)
    V_proj = V @ W_v   # (batch, seq_len_k, d_model)

    # Split into heads: (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)
    def split_heads(x, seq_len):
        x = x.reshape(batch_size, seq_len, num_heads, d_k)
        return x.transpose(0, 2, 1, 3)

    Qh = split_heads(Q_proj, seq_len_q)
    Kh = split_heads(K_proj, seq_len_k)
    Vh = split_heads(V_proj, seq_len_k)

    # Scaled dot-product attention per head
    scores = Qh @ Kh.transpose(0, 1, 3, 2) / np.sqrt(d_k)   # (batch, num_heads, seq_len_q, seq_len_k)
    attn_weights = softmax(scores)
    head_outputs = attn_weights @ Vh   # (batch, num_heads, seq_len_q, d_k)

    # Concatenate heads: (batch, num_heads, seq_len_q, d_k) -> (batch, seq_len_q, d_model)
    concat = head_outputs.transpose(0, 2, 1, 3).reshape(batch_size, seq_len_q, d_model)

    # Output projection
    output = concat @ W_o

    return output
    pass