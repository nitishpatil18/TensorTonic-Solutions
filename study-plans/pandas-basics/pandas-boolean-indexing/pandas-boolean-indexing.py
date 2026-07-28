import pandas as pd

def boolean_filter(data, column, threshold):
    df = pd.DataFrame(data)
    mask = df[column] > threshold
    filtered = df[mask]
    filtered_data = filtered.to_dict(orient="list")
    count = len(filtered)
    return {
        "filtered_data": filtered_data,
        "count": count
    }
    pass