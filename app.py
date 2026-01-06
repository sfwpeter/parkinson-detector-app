# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import json
import sys

# Fix the import issue by adding the project directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import from models module
from models.load_models import load_models, get_model_info
from models.preprocessing import (
    validate_input,
    prepare_features,
    get_prediction_explanation,
    FEATURE_COLUMNS,
)

# Configure page
st.set_page_config(
    page_title="Parkinson's Disease Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    . prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .positive {
        background-color: #ffebee;
        border-left: 4px solid #d32f2f;
    }
    .negative {
        background-color: #e8f5e9;
        border-left: 4px solid #388e3c;
    }
    .metric-card {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

def load_feature_info():
    """Load feature information from JSON file."""
    try:
        feature_file = Path(__file__).parent / "data" / "data_feature_info.json"
        with open(feature_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Could not load feature info: {str(e)}")
        return None

def create_input_form():
    """Create the input form for user data (uses FEATURE_COLUMNS from preprocessing).

    This implementation keeps inputs consistent with the model's expected features.
    """
    st.subheader("📊 Enter Patient Voice Measurements")

    raw_feature_info = load_feature_info() or {}
    feature_info = raw_feature_info.get('features', {}) if isinstance(raw_feature_info, dict) else {}

    input_data = {}

    # layout: two columns, distribute features
    cols = st.columns(2)
    for i, feat in enumerate(FEATURE_COLUMNS):
        col = cols[i % 2]
        default = 0.0
        # sensible defaults for some known features
        defaults = {
            'MDVP:Fo(Hz)': 150.0,
            'MDVP:Fhi(Hz)': 180.0,
            'MDVP:Flo(Hz)': 120.0,
            'MDVP:Jitter(%)': 0.005,
            'MDVP:Jitter(Abs)': 0.00005,
            'MDVP:RAP': 0.003,
            'MDVP:PPQ': 0.003,
            'Jitter:DDP': 0.01,
            'MDVP:Shimmer': 0.03,
            'MDVP:Shimmer(dB)': 0.3,
            'Shimmer:APQ3': 0.015,
            'Shimmer:APQ5': 0.018,
            'MDVP:APQ': 0.02,
            'Shimmer:DDA': 0.04,
            'NHR': 0.02,
            'HNR': 21.0,
            'RPDE': 0.5,
            'DFA': 0.71,
            'D2': 2.3,
            'spread1': -5.0,
            'spread2': 0.2,
            'PPE': 0.2,
        }
        if feat in defaults:
            default = defaults[feat]

        help_text = feature_info.get(feat, {}).get('description') if isinstance(feature_info, dict) else None

        with col:
            input_data[feat] = st.number_input(
                label=feat,
                value=float(default),
                help=help_text,
                format="%.6f",
            )

    return input_data

def display_predictions(model, scaler, input_data):
    """Display prediction results."""
    st.subheader("🔍 Prediction Results")
    
    # Prepare features
    features_df = prepare_features(input_data, scaler)
    
    if features_df is None:
        st.error("Failed to prepare features for prediction")
        return
    
    # Make prediction
    try:
        prediction = model.predict(features_df)[0]
        
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_df)[0]
        else:
            probabilities = [1 - prediction, prediction]
        
        # Get explanation
        explanation = get_prediction_explanation(prediction, probabilities)
        
        # Display results
        col1, col2 = st. columns([2, 1])
        
        with col1:
            if prediction == 1:
                st.markdown(
                    f'<div class="prediction-box positive">'
                    f'{explanation["status"]}<br>'
                    f'{explanation["description"]}'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="prediction-box negative">'
                    f'{explanation["status"]}<br>'
                    f'{explanation["description"]}'
                    f'</div>',
                    unsafe_allow_html=True
                )
        
        with col2:
            st.metric(
                "Confidence Level",
                f"{explanation['confidence']:.2f}%",
                delta=None
            )
        
        # Display probability gauge
        st.subheader("📈 Probability Distribution")
        
        fig = go.Figure(data=[
            go.Bar(
                x=['Negative', 'Positive'],
                y=[probabilities[0] * 100, probabilities[1] * 100],
                marker_color=['#4CAF50', '#f44336'],
                text=[f'{probabilities[0]*100:.2f}%', f'{probabilities[1]*100:.2f}%'],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            height=400,
            showlegend=False,
            yaxis_title="Probability (%)",
            xaxis_title="Prediction Class"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display feature values table
        st.subheader("📋 Input Features Summary")
        
        features_display = pd.DataFrame({
            'Feature':  list(input_data.keys()),
            'Value': list(input_data.values())
        }).round(6)
        
        st.dataframe(features_display, use_container_width=True)
    
    except Exception as e: 
        st.error(f"Error making prediction: {str(e)}")

def display_model_info(model):
    """Display information about the loaded model."""
    st.subheader("ℹ️ Model Information")
    
    model_info = get_model_info(model)
    
    if model_info: 
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Model Type", model_info['model_type'])
        
        with col2:
            st.metric("Feature Importance", "Available" if model_info['has_feature_importance'] else "N/A")
        
        with col3:
            st.metric("Probability Support", "Yes" if model_info['has_predict_proba'] else "No")
        
        # Display feature importance if available
        if model_info['has_feature_importance']:
            st.subheader("🎯 Feature Importance")
            
            importance_df = pd.DataFrame({
                'Feature':  FEATURE_COLUMNS,
                'Importance': model_info['feature_importances']
            }).sort_values('Importance', ascending=False).head(10)
            
            fig = go.Figure(data=[
                go.Bar(
                    x=importance_df['Importance'],
                    y=importance_df['Feature'],
                    orientation='h',
                    marker_color='#2196F3'
                )
            ])
            
            fig. update_layout(
                height=400,
                xaxis_title="Importance",
                yaxis_title="Feature",
                showlegend=False
            )
            
            st. plotly_chart(fig, use_container_width=True)

def main():
    """Main application function."""
    
    # Sidebar
    with st.sidebar:
        st.title("🧠 Parkinson's Detector")
        st.markdown("---")
        
        page = st.radio(
            "Navigation",
            ["🔍 Prediction", "ℹ️ About", "📚 Help"]
        )
        
        st.markdown("---")
        st.markdown("""
        ### About This App
        This application uses machine learning to help identify 
        potential characteristics of Parkinson's Disease based on 
        voice measurements.
        
        **⚠️ Disclaimer**: This app is for educational purposes only 
        and should not be used as a medical diagnosis tool.  Always 
        consult with a healthcare professional. 
        """)
    
    # Main content
    if page == "🔍 Prediction":
        st.title("🧠 Parkinson's Disease Detection")
        st.markdown("Predict Parkinson's disease based on voice measurements")
        
        # Load models
        with st.spinner("Loading models..."):
            model, scaler = load_models()
        
        if model is None:
            st. error("""
            ⚠️ Error loading models! 
            
            Please ensure you have: 
            1. Trained the model in the Jupyter notebook
            2. Saved it as `models/parkinsons_model.pkl`
            3. Saved the scaler as `models/scaler.pkl`
            
            See the notebook for training instructions.
            """)
            return
        
        st.success("✅ Models loaded successfully!")
        
        # Create tabs
        tab1, tab2 = st.tabs(["Input & Prediction", "Model Info"])
        
        with tab1:
            # Create input form
            input_data = create_input_form()
            
            # Validation
            errors, warnings = validate_input(input_data)
            
            if errors: 
                st.warning("⚠️ **Validation Warnings:**")
                for error in errors:
                    st.warning(f"- {error}")
            
            # Predict button
            col1, col2 = st. columns([1, 3])
            
            with col1:
                if st.button("🔍 Make Prediction", use_container_width=True):
                    display_predictions(model, scaler, input_data)
        
        with tab2:
            display_model_info(model)
    
    elif page == "ℹ️ About": 
        st.title("About Parkinson's Disease")
        
        st.markdown("""
        ## What is Parkinson's Disease? 
        
        Parkinson's disease is a progressive neurological disorder that affects movement.  
        It develops gradually, often starting with barely noticeable tremor in just one hand.  
        But while tremor may be the most well-known sign of Parkinson's, the disorder also 
        commonly causes stiffness and slows movement. 
        
        ## Key Features
        
        - **Progressive**: Symptoms worsen over time
        - **Neurological**: Affects the nervous system
        - **Movement Disorder**: Primary impact on motor function
        - **Variable**: Different people experience different symptoms
        
        ## Early Detection
        
        Early detection can help with treatment planning and symptom management. 
        Voice analysis has shown promise as a non-invasive screening tool.
        
        ## Voice Changes in Parkinson's
        
        People with Parkinson's often experience changes in voice quality due to:
        - Reduced vocal fold movement
        - Weakness in voice muscles
        - Rigidity affecting speech muscles
        
        This app analyzes various voice characteristics that correlate with Parkinson's disease.
        """)
    
    elif page == "📚 Help":
        st.title("Help & Feature Descriptions")
        
        st.markdown("""
        ## Feature Descriptions
        
        ### Frequency Features
        - **MDVP: Fo(Hz)**: Average vocal fundamental frequency
        - **MDVP:Fhi(Hz)**: Maximum vocal fundamental frequency  
        - **MDVP:Flo(Hz)**: Minimum vocal fundamental frequency
        
        ### Jitter Features
        Jitter measures the variation in fundamental frequency: 
        - **MDVP:Jitter(%)**: Percentage of F0 variations
        - **MDVP:Jitter(Abs)**: Absolute jitter in seconds
        - **MDVP:RAP**: Relative Absolute Pitch
        - **MDVP:PPQ**: Pitch Period Perturbation Quotient
        - **Jitter:DDP**: Jitter - Differential Divergence Period
        
        ### Shimmer Features
        Shimmer measures the variation in amplitude:
        - **MDVP:Shimmer**:  Amplitude variation
        - **MDVP: Shimmer(dB)**: Amplitude variation in dB
        - **Shimmer:APQ3/APQ5**: Amplitude Perturbation Quotient
        - **Shimmer:DDA**: Differential Divergence Amplitude
        
        ### Noise & Harmonic Features
        - **NHR**: Noise-to-Harmonic Ratio
        - **HNR**:  Harmonics-to-Noise Ratio
        
        ### Nonlinear Features
        - **RPDE**:  Recurrence Period Density Entropy
        - **DFA**: Detrended Fluctuation Analysis
        - **D2**:  Correlation Dimension
        
        ### Recurrence Features
        - **spread1/spread2**: First/Second Recurrence Plot Spread
        - **PPE**: Pitch Period Entropy
        """)

if __name__ == "__main__":
    main()