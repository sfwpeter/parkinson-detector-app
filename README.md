# Parkinson Detector App

This repository contains a Streamlit application that loads a trained machine learning model to detect Parkinson's disease characteristics from voice measurements.

Quickstart (local)

1. Create and activate a Python environment (recommended Python 3.10+).

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the Streamlit app:

```powershell
streamlit run app.py
```

What you need

- A trained model file at `models/parkinsons_best_model.pkl` (joblib/pickle format).
- (Optional but recommended) a fitted scaler at `models/parkinsons_scaler.pkl`.

If you trained the model in the included notebook, export it to the `models/` folder with the names above.

Deploying to Streamlit Cloud

- Push this repository to GitHub.
- On Streamlit Cloud, select the repo and set the main file to `app.py`.
- Ensure `requirements.txt` is present (already included).

Notes and next steps

- The app uses `models/load_models.py` to load model and scaler.
- Input feature order is defined in `models/preprocessing.py` (the model expects that order).
- If you edit feature names or model files, update `models/preprocessing.py` accordingly.

License: MIT
