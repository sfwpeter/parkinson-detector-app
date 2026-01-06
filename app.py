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
    page_title="Prempehs Parkinson's Disease Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple, clean CSS
st.markdown("""
    <style>
    /* Clean, simple styling */
    .main {
        padding: 2rem;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #e9ecef;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #4a6fa5;
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #385d8a;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Number input styling */
    .stNumberInput input {
        border: 2px solid #e0e0e0;
        border-radius: 6px;
    }
    
    .stNumberInput input:focus {
        border-color: #4a6fa5;
        box-shadow: 0 0 0 3px rgba(74, 111, 165, 0.2);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: white;
        border-bottom: 2px solid #4a6fa5;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
    
    /* Dataframe */
    .dataframe {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Prediction boxes */
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid;
    }
    
    .positive {
        background-color: #ffeaea;
        border-left-color: #dc3545;
    }
    
    .negative {
        background-color: #e8f5e9;
        border-left-color: #28a745;
    }
    
    /* Headers */
    h1 {
        color: #2c3e50;
        margin-bottom: 1.5rem;
    }
    
    h2 {
        color: #34495e;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    h3 {
        color: #2c3e50;
    }
    
    /* Alerts */
    .stAlert {
        border-radius: 8px;
    }
    
    /* Make text readable */
    p, li {
        line-height: 1.6;
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
    """Create the input form for user data."""
    st.subheader("📊 Enter Patient Voice Measurements")

    raw_feature_info = load_feature_info() or {}
    feature_info = raw_feature_info.get('features', {}) if isinstance(raw_feature_info, dict) else {}

    input_data = {}

    # Two columns layout
    cols = st.columns(2)
    for i, feat in enumerate(FEATURE_COLUMNS):
        col = cols[i % 2]
        default = 0.0
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
                key=f"input_{feat}"
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
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if prediction == 1:
                st.markdown(
                    f'<div class="prediction-box positive">'
                    f'<h3>⚠️ {explanation["status"]}</h3>'
                    f'<p>{explanation["description"]}</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="prediction-box negative">'
                    f'<h3>✅ {explanation["status"]}</h3>'
                    f'<p>{explanation["description"]}</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
        
        with col2:
            st.metric(
                "Confidence Level",
                f"{explanation['confidence']:.2f}%",
                delta=None
            )
        
        # Display probability chart
        st.subheader("📈 Probability Distribution")
        
        fig = go.Figure(data=[
            go.Bar(
                x=['Negative', 'Positive'],
                y=[probabilities[0] * 100, probabilities[1] * 100],
                marker_color=['#28a745', '#dc3545'],
                text=[f'{probabilities[0]*100:.1f}%', f'{probabilities[1]*100:.1f}%'],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            height=300,
            showlegend=False,
            yaxis_title="Probability (%)",
            xaxis_title="Prediction Class",
            margin=dict(l=20, r=20, t=20, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display feature values
        st.subheader("📋 Input Features")
        
        features_display = pd.DataFrame({
            'Feature': list(input_data.keys()),
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
            st.metric("Feature Importance", 
                     "Yes" if model_info['has_feature_importance'] else "No")
        
        with col3:
            st.metric("Probability", 
                     "Yes" if model_info['has_predict_proba'] else "No")
        
        # Display feature importance if available
        if model_info['has_feature_importance']:
            st.subheader("🎯 Feature Importance")
            
            importance_df = pd.DataFrame({
                'Feature': FEATURE_COLUMNS,
                'Importance': model_info['feature_importances']
            }).sort_values('Importance', ascending=False).head(10)
            
            fig = go.Figure(data=[
                go.Bar(
                    x=importance_df['Importance'],
                    y=importance_df['Feature'],
                    orientation='h',
                    marker_color='#4a6fa5'
                )
            ])
            
            fig.update_layout(
                height=300,
                xaxis_title="Importance",
                yaxis_title="Feature",
                margin=dict(l=20, r=20, t=20, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)

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
            st.error("""
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
            
            if errors or warnings: 
                st.warning("⚠️ **Validation Warnings:**")
                for error in errors:
                    st.warning(f"- {error}")
            
            # Predict button
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