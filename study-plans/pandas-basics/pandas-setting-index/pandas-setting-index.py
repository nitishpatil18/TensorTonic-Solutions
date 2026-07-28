import pandas as pd

def set_index_column(data, index_col):
    df = pd.DataFrame(data)
    df = df.set_index(index_col)
    return {
        "index_values": df.index.tolist(),
        "columns": list(df.columns),
        "data": df.to_dict(orient="list")
    }
    pass