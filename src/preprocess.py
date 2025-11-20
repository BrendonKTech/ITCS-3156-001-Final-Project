import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def replace_invalid_values(df: pd.DataFrame, invalid_value: float = -200) -> pd.DataFrame:
    """Replace invalid sensor values with NaN."""
    return df.replace(invalid_value, np.nan)

def impute_missing(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    """Impute missing values using the specified strategy (mean/median)."""
    imputer = SimpleImputer(strategy=strategy)
    imputed_array = imputer.fit_transform(df)
    imputed_df = pd.DataFrame(imputed_array, columns=df.columns)
    return imputed_df

def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize features to zero mean and unit variance."""
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_array, columns=df.columns)
    return scaled_df
