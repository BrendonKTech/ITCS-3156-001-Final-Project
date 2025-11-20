import pandas as pd

def add_time_features(df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
    """
    Extract hour, day, month from datetime column.
    
    Args:
        df (pd.DataFrame)
        datetime_col (str): name of datetime column
    
    Returns:
        pd.DataFrame: dataframe with added features
    """
    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df['hour'] = df[datetime_col].dt.hour
    df['day'] = df[datetime_col].dt.day
    df['month'] = df[datetime_col].dt.month
    return df
