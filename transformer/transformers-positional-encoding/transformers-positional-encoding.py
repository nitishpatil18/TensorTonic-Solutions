import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # Your code here
    position = np.arange(seq_length)[:, np.newaxis]          # (seq_length, 1)
    i = np.arange(d_model // 2)[np.newaxis, :]                # (1, d_model/2)
    
    angle_rates = 1.0 / np.power(10000, (2 * i) / d_model)     # (1, d_model/2)
    angles = position * angle_rates                            # (seq_length, d_model/2)
    
    pe = np.zeros((seq_length, d_model), dtype=np.float64)
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles)

    return pe
    pass