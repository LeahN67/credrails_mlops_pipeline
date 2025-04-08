import pytest
import pandas as pd


def test_class_balance():
    """
    Test that classes exist
    
    """
    df = pd.read_parquet('data_processing/modified_dataset.parquet')
    assert len(df['Classification_Tag'].unique()) > 1, "Need multiple classes "