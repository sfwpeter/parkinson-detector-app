# models/load_models.py
import os
import pickle
import streamlit as st
from pathlib import Path

@st.cache_resource
def load_models():
    """
    Load pre-trained models and scaler from pickle files.
    Using Streamlit's cache to load models only once.
    """
    try:
        # Get the directory where this file is located
        current_dir = Path(__file__).parent.parent
        models_dir = current_dir / "models"
        
        # Load the main model
        model_path = models_dir / "parkinsons_model.pkl"
        if not model_path.exists():
            st.error(f"Model file not found at {model_path}")
            return None, None
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load the scaler
        scaler_path = models_dir / "scaler.pkl"
        if not scaler_path.exists():
            st.warning(f"Scaler file not found at {scaler_path}")
            scaler = None
        else: 
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        
        return model, scaler
    
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None


def get_model_info(model):
    """
    Extract information about the loaded model.
    """
    try:
        model_type = type(model).__name__
        info = {
            "model_type": model_type,
            "has_feature_importance": hasattr(model, 'feature_importances_'),
            "has_predict_proba": hasattr(model, 'predict_proba'),
        }
        
        if hasattr(model, 'feature_importances_'):
            info["feature_importances"] = model.feature_importances_
        
        return info
    except Exception as e:
        st. error(f"Error extracting model info: {str(e)}")
        return None
