import pandas as pd
from collections import Counter

def data_types_overview(data):
    df = pd.DataFrame(data)
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    type_counts = dict(Counter(dtypes.values()))
    return {
        "dtypes": dtypes,
        "type_counts": type_counts,
        "num_columns": len(df.columns)
    }
    pass