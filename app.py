# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from utils.load_models import load_model_and_scaler
from utils.preprocessing import build_feature_vector, validate_inputs
from pathlib import Path

st.set_page_config(page_title="Parkinson's Detector", page_icon="🧠", layout="wide")

# --- paths ---
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
ASSETS_DIR = ROOT / "assets"

FEATURE_INFO_PATH = DATA_DIR / "feature_info.json"

# --- load feature info (for defaults and validation) ---
@st.cache_data
def load_feature_info(path=FEATURE_INFO_PATH):
    with open(path, "r") as f:
        return json.load(f)

feature_info = load_feature_info()

# --- load the model & scaler (if available) ---
@st.cache_resource
def load_models():
    model, scaler = load_model_and_scaler(MODELS_DIR)
    return model, scaler

# Attempt to load models and show a helpful message on error
try:
    model, scaler = load_models()
    model_loaded = True
except Exception as e:
    model = None
    scaler = None
    model_loaded = False
    model_load_error = str(e)

# --- Styling ---
def local_css(file_path=ASSETS_DIR / "styles.css"):
    try:
        with open(file_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

local_css()

# --- Sidebar ---
st.sidebar.image(str(ASSETS_DIR / "logo.png"), use_column_width=False, width=120)  # placeholder
st.sidebar.title("Parkinson's Detector")
st.sidebar.markdown("A compact Streamlit app to predict Parkinson's disease from voice features.")
nav = st.sidebar.radio("Navigation", ["Home", "Predict", "Feature Importance", "About"])

# --- Helpers for UI ---
def render_header():
    st.title("Parkinson's Disease Detection")
    st.markdown(
        """
        This app uses a machine learning model trained on voice measurements to predict the likelihood of Parkinson's disease.
        Enter the measurements in the form or load example values, then press Predict.
        """
    )

def render_result_card(pred_label, pred_prob, threshold=0.5):
    # pred_label: 0/1 or negative/positive depending on model
    status = "Parkinson's Detected" if pred_label else "No Parkinson's Detected"
    color = "danger" if pred_label else "success"
    st.markdown(f"### Prediction: **{status}**")
    st.write(f"Confidence: **{pred_prob*100:.2f}%**")
    if pred_label:
        st.info(
            """
            The model predicts presence of Parkinson’s disease with the probability shown above.
            This is a screening aid only — always consult clinical specialists and confirm with diagnostic tests.
            """
        )
    else:
        st.success(
            """
            The model predicts low likelihood of Parkinson’s disease based on given voice features.
            This is not a diagnosis — consult a clinician for confirmation.
            """
        )

# --- Pages ---
if nav == "Home":
    render_header()
    st.markdown("## Quick Start")
    st.markdown(
        """
        1. Go to the *Predict* page in the sidebar.
        2. Enter the voice features (defaults are dataset medians).
        3. Click *Predict* to get a probability and explanation.
        """
    )
    st.markdown("## Model status")
    if model_loaded:
        st.success("Model and scaler loaded successfully.")
        st.write(f"Model type: `{type(model).__name__}`")
    else:
        st.error("Model not loaded.")
        st.write("Place your pickle files in the `models/` folder with the names:\n- parkinsons_model.pkl\n- scaler.pkl")
        st.write("Error: " + model_load_error)

    st.markdown("---")
    st.markdown("## Notes")
    st.markdown(
        """
        - Input values are validated against the training dataset ranges if available.
        - If the saved model supports predict_proba, probability will be shown; otherwise an approximate score is computed.
        - A feature importance chart is available under 'Feature Importance'.
        """
    )

elif nav == "Predict":
    render_header()
    if not model_loaded:
        st.error("Model/scaler not loaded. See Home page for instructions.")
        st.stop()

    with st.form("prediction_form"):
        st.subheader("Patient voice feature inputs")
        # layout: three columns per row for compactness
        cols = st.columns(3)
        input_values = {}
        features = list(feature_info.keys())
        # iterate and create fields with defaults
        for i, feat in enumerate(features):
            col = cols[i % 3]
            info = feature_info[feat]
            default = info.get("median", 0.0)
            minv = info.get("min", None)
            maxv = info.get("max", None)
            help_text = info.get("description", "")
            # Use number_input for numeric features
            input_values[feat] = col.number_input(
                label=f"{feat}",
                min_value=minv if minv is not None else -1e6,
                max_value=maxv if maxv is not None else 1e6,
                value=float(default),
                format="%.6f",
                help=help_text,
                step=0.0001
            )
        submitted = st.form_submit_button("Predict")

    if submitted:
        # Validation
        valid, message = validate_inputs(input_values, feature_info)
        if not valid:
            st.error("Input validation error: " + message)
        else:
            X = build_feature_vector(input_values, features)
            # scale
            if scaler is not None:
                X_scaled = scaler.transform([X])
            else:
                X_scaled = np.array([X])
            # Predict
            try:
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_scaled)[0]
                    # assume classes [0,1] where 1 -> Parkinson's
                    if len(proba) == 2:
                        pred_prob = float(proba[1])
                    else:
                        # fallback to max
                        pred_prob = float(np.max(proba))
                elif hasattr(model, "decision_function"):
                    # map to probability-like via sigmoid
                    df = model.decision_function(X_scaled)[0]
                    pred_prob = 1 / (1 + np.exp(-df))
                else:
                    # fallback: predict then give 0.99 for predicted class
                    pred = model.predict(X_scaled)[0]
                    pred_prob = 0.99 if pred == 1 else 0.01
                pred_label = int(pred_prob >= 0.5)
            except Exception as e:
                st.error("Model prediction failed: " + str(e))
                st.stop()

            # Display results
            st.subheader("Result")
            render_result_card(pred_label, pred_prob)
            st.markdown("### Raw outputs")
            st.write({
                "predicted_label": int(pred_label),
                "predicted_probability": float(pred_prob)
            })

            # show input summary
            st.subheader("Input summary")
            st.table(pd.DataFrame([input_values]).T.rename(columns={0: "value"}))

            st.markdown("---")
            st.subheader("Explainability & Next Steps")
            st.markdown(
                """
                - Feature importance chart is available on the *Feature Importance* page.
                - This model is a screening aid — follow up with clinical tests and specialist consultation.
                - Consider retraining with more data and cross-validation before clinical use.
                """
            )

elif nav == "Feature Importance":
    render_header()
    st.subheader("Feature importance")

    if not model_loaded:
        st.error("Model not loaded - cannot display feature importance.")
        st.stop()

    # Try to get importances from model
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feature_names = list(feature_info.keys())
            df_imp = pd.DataFrame({"feature": feature_names, "importance": importances})
            df_imp = df_imp.sort_values("importance", ascending=False).head(20)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(x="importance", y="feature", data=df_imp, palette="viridis", ax=ax)
            ax.set_title("Model Feature Importances")
            st.pyplot(fig)
        else:
            st.warning("Model has no feature_importances_. Showing permutation importance requires a dataset and is skipped.")
    except Exception as e:
        st.error("Could not compute feature importance: " + str(e))

elif nav == "About":
    render_header()
    st.subheader("About this app")
    st.markdown(
        """
        - Built for demonstration and screening use.
        - The predictive model must be supplied in `models/parkinsons_model.pkl`.
        - The scaler must be supplied in `models/scaler.pkl`.
        - To deploy: push the repository to GitHub and connect the Streamlit Cloud app (add the secrets if necessary).
        """
    )
    st.markdown("## References and caveats")
    st.markdown(
        """
        - The dataset used to train the model may have class imbalance. If so, consider stratified CV and calibration.
        - For production: add authentication, logging, health checks and unit tests.
        """
    )