import joblib
import os

def test_model_saved():
    """Test that model artifacts exist"""
    assert os.path.exists('model_outputs/classification_model.pkl')
    assert os.path.exists('model_outputs/status_encoder.pkl')
    
def test_model_performance():
    """Test model meets minimum performance"""
    with open('model_outputs/evaluation_metrics.txt') as f:
        report = f.read()
    assert 'weighted avg' in report
    # Add specific performance threshold checks