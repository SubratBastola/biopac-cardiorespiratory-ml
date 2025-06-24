import pandas as pd

def load_data(filepath: str):
    df = pd.read_csv(filepath)
    X = df.drop(columns=["Activity_label"])
    y = df["Activity_label"]
    return X, y

