from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
import numpy as np

def fit_calibrator(y_true, raw_scores):
    """
    Fit isotonic regression calibrator.
    y_true: array of 0/1 labels
    raw_scores: array of model predicted probabilities
    Returns fitted IsotonicRegression object.
    """
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(raw_scores, y_true)
    return calibrator

def calibrate(calibrator, scores):
    """
    Apply calibrator to raw scores.
    Returns calibrated probabilities as np.ndarray.
    If calibrator is None, return scores unchanged.
    """
    if calibrator is None:
        return np.array(scores)
    return calibrator.predict(scores)
