import pandas as pd

def select_column(data, column):
    df = pd.DataFrame(data)
    values = df[column].tolist()
    return {
        "values": values,
        "length": len(values)
    }
    pass