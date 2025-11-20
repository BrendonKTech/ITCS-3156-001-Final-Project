import pandas as pd

def load_raw_data(filepath:str) -> pd.DataFrame:
    """
    Load raw CSV dataset into a pandas DataFrame.
    
    Args:
        filepath (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    df = pd.read_csv(filepath)
    return df

def load_processed_data(filepath: str) -> pd.DataFrame:
    """
    Load processed CSV dataset.
    
    Args:
        filepath (str): Path to processed CSV.
        
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    df = pd.read_csv(filepath)
    return df