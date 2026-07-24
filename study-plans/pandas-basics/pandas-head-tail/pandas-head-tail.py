import pandas as pd

def head_tail(data, n):
    df = pd.DataFrame(data)
    head = df.head(n).to_dict(orient="list")
    tail = df.tail(n).to_dict(orient="list")
    return {
        "head": head,
        "tail": tail
    }
    pass