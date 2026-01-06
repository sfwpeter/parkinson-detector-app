# utils/preprocessing.py
import numpy as np

def build_feature_vector(input_values: dict, feature_order: list):
    """
    Build a feature vector (list/np.array) in the order requested by the model.
    """
    return [float(input_values[f]) for f in feature_order]

def validate_inputs(input_values: dict, feature_info: dict):
    """
    Basic validation: checks numeric types and optional min/max constraints from feature_info.
    Returns (bool, message)
    """
    for feat, val in input_values.items():
        try:
            v = float(val)
        except Exception:
            return False, f"Feature {feat} must be a number."
        info = feature_info.get(feat, {})
        mn = info.get("min", None)
        mx = info.get("max", None)
        if mn is not None and v < mn:
            return False, f"{feat} below minimum ({v} < {mn})"
        if mx is not None and v > mx:
            return False, f"{feat} above maximum ({v} > {mx})"
    return True, ""