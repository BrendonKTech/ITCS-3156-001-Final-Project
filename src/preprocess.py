import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def clean_data(df):
    # Replace -200 with NaN
    df = df.replace(-200, np.nan)

    # Convert comma decimal formatting (e.g. '3,2' -> '3.2')
    df = df.applymap(
        lambda x: str(x).replace(",", ".") if isinstance(x, str) else x
    )

    # Convert all columns that can be converted to numeric
    df = df.apply(pd.to_numeric, errors="ignore")

    # Drop remaining NaN rows
    df = df.dropna()

    # Drop non-numeric columns (Date and Time)
    df = df.drop(columns=[c for c in ["Date", "Time"] if c in df.columns])

    return df


def get_features_targets(df, target):
    X = df.drop(columns=[target])
    y = df[target]

    # Select only numeric columns for X
    X = X.select_dtypes(include=["float64", "int64"])
    return X, y

def split_and_scale(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test
