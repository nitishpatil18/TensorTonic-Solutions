import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    # Your code here
    mu = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mu) / np.sqrt(var + eps)
    return gamma * x_norm + beta
    pass

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    # Your code here
    batch_size, seq_len_q, d_model = Q.shape
    seq_len_k = K.shape[1]
    d_k = d_model // num_heads

    Q_proj = Q @ W_q
    K_proj = K @ W_k
    V_proj = V @ W_v

    def split_heads(x, seq_len):
        x = x.reshape(batch_size, seq_len, num_heads, d_k)
        return x.transpose(0, 2, 1, 3)

    Qh = split_heads(Q_proj, seq_len_q)
    Kh = split_heads(K_proj, seq_len_k)
    Vh = split_heads(V_proj, seq_len_k)

    scores = Qh @ Kh.transpose(0, 1, 3, 2) / np.sqrt(d_k)
    attn_weights = softmax(scores, axis=-1)
    head_outputs = attn_weights @ Vh

    concat = head_outputs.transpose(0, 2, 1, 3).reshape(batch_size, seq_len_q, d_model)

    return concat @ W_o
    pass

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    # Your code here
    hidden = np.dot(x, W1) + b1
    relu_out = np.maximum(0, hidden)
    return np.dot(relu_out, W2) + b2
    pass

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    # Your code here
    attn_out = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)
    x_prime = layer_norm(x + attn_out, gamma1, beta1)

    ffn_out = feed_forward(x_prime, W1, b1, W2, b2)
    output = layer_norm(x_prime + ffn_out, gamma2, beta2)
    return output
    pass