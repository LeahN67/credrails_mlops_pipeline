import pytest
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# Test if training artifacts are created
def test_training_artifacts_exist():
    """Test that all expected output files are created"""
    required_files = [
        'model_outputs/classification_model.pkl',
        'model_outputs/status_encoder.pkl',
        'model_outputs/dayofweek_encoder.pkl',
        'model_outputs/classification_tag_encoder.pkl',
        'model_outputs/confusion_matrix.png',
        'model_outputs/feature_importance.png',
        'model_outputs/evaluation_metrics.txt'
    ]
    
    for file_path in required_files:
        assert os.path.exists(file_path), f"Missing expected file: {file_path}"

# Test data loading and preprocessing
def test_data_loading():
    """Test that data loads and preprocesses correctly"""
    df = pd.read_parquet('data_processing/modified_dataset.parquet')
    
    # Check basic dataframe structure
    assert not df.empty, "Data should not be empty"
    assert 'Classification_Tag' in df.columns, "Target column missing"
    assert len(df.columns) > 1, "Should have multiple features"
    
    # Check categorical columns were encoded
    assert df['Status'].dtype == 'int64', "Status should be encoded as integer"
    assert df['DayOfWeek'].dtype == 'int64', "DayOfWeek should be encoded as integer"

# Test model training
def test_model_training():
    """Test that the model can be loaded and has expected structure"""
    model = joblib.load('model_outputs/classification_model.pkl')
    
    # Check it's an XGBoost model
    assert str(type(model)).endswith("XGBClassifier'>"), "Model should be XGBClassifier"
    
    # Check it has been fitted
    assert hasattr(model, 'feature_importances_'), "Model should be fitted"
    assert len(model.feature_importances_) > 0, "Model should have features"

# Test evaluation metrics
def test_evaluation_metrics():
    """Test that evaluation metrics file has content"""
    with open('model_outputs/evaluation_metrics.txt') as f:
        content = f.read()
    
    assert len(content) > 0, "Metrics file should not be empty"
    assert 'precision' in content, "Metrics should include precision"
    assert 'recall' in content, "Metrics should include recall"
    assert 'f1-score' in content, "Metrics should include f1-score"

# Test class imbalance handling
def test_class_balance():
    """Test that classes are balanced after resampling"""
    df = pd.read_parquet('data_processing/modified_dataset.parquet')
    class_counts = df['Classification_Tag'].value_counts()
    
    # Check original data has imbalance (at least one class with <30% of data)
    assert any(count/len(df) < 0.3 for count in class_counts), "Original data should have class imbalance"
    
    # For the upsampled data (you might need to modify this based on your actual resampling)
    X_resampled = pd.read_parquet('data_processing/modified_dataset.parquet')  # Replace with actual resampled data path if different
    y_resampled = X_resampled['Classification_Tag']
    resampled_counts = y_resampled.value_counts()
    
    # Check all classes have similar counts after resampling
    max_count = max(resampled_counts)
    for count in resampled_counts:
        assert pytest.approx(count, rel=0.1) == max_count, "Classes should be balanced after resampling"