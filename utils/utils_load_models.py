# utils/load_models.py
import joblib
from pathlib import Path

def load_model_and_scaler(models_dir: Path):
    """
    Loads the model and scaler from the models directory.

    Expected files:
      - parkinsons_model.pkl
      - scaler.pkl

    Raises informative errors if not found or invalid.
    """
    models_dir = Path(models_dir)
    model_path = models_dir / "parkinsons_model.pkl"
    scaler_path = models_dir / "scaler.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}\n"
            "Place your trained model pickle in models/parkinsons_model.pkl"
        )
    if not scaler_path.exists():
        # scaler is optional but recommended
        raise FileNotFoundError(
            f"Scaler not found: {scaler_path}\n"
            "Place your fitted scaler (e.g. StandardScaler) in models/scaler.pkl"
        )

    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

    try:
        scaler = joblib.load(scaler_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load scaler: {e}")

    return model, scaler