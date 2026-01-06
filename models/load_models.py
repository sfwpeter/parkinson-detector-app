# models/load_models.py
"""Utility to load the trained model and scaler used by the Streamlit app.

This module keeps the model-loading logic in one place and uses
Streamlit's cache to avoid reloading on every interaction.
"""
from pathlib import Path
import joblib
import streamlit as st


@st.cache_resource
def load_models() -> tuple:
    """Load model and scaler from the `models/` directory.

    Returns (model, scaler) where scaler may be None if not found.
    """
    try:
        # models/load_models.py lives in the `models` package directory
        models_dir = Path(__file__).parent

        # filenames in this repo
        model_path = models_dir / "parkinsons_best_model.pkl"
        scaler_path = models_dir / "parkinsons_scaler.pkl"

        if not model_path.exists():
            st.error(f"Model file not found at {model_path}")
            return None, None

        model = joblib.load(model_path)

        if not scaler_path.exists():
            st.warning(f"Scaler file not found at {scaler_path}")
            scaler = None
        else:
            scaler = joblib.load(scaler_path)

        return model, scaler

    except Exception as exc:
        st.error(f"Error loading models: {exc}")
        return None, None


def get_model_info(model: object) -> dict:
    """Return small dictionary describing the loaded model."""
    try:
        model_type = type(model).__name__
        info = {
            "model_type": model_type,
            "has_feature_importance": hasattr(model, 'feature_importances_'),
            "has_predict_proba": hasattr(model, 'predict_proba'),
        }

        if hasattr(model, 'feature_importances_'):
            info["feature_importances"] = list(model.feature_importances_)

        return info
    except Exception as exc:
        st.error(f"Error extracting model info: {exc}")
        return None
