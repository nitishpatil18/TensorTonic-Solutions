import pandas as pd

def select_columns(data, columns):
    df = pd.DataFrame(data)
    selected = df[columns]
    return selected.to_dict(orient="list")
    pass