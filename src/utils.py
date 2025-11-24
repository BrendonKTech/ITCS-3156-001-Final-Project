import pandas as pd

def load_data(path: str):
    df = pd.read_csv(path, sep=';')
    df = df.dropna(axis=1, how="all")
    return df
