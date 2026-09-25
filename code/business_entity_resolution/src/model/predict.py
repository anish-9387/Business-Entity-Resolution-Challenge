import numpy as np

def predict_scores(model, X):
    """
    Get probability scores from model.
    model has .predict_proba(X) method.
    Returns np.ndarray of probabilities.
    """
    return np.array(model.predict_proba(X))

def predict_calibrated(model, X, calibrator=None):
    """
    Predict and optionally calibrate.
    Returns np.ndarray of (calibrated) probabilities.
    """
    from src.model.calibrate import calibrate
    raw = predict_scores(model, X)
    return calibrate(calibrator, raw)
