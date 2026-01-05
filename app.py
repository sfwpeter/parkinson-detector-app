# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import json
import sys

# Fix the import issue by adding the parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Now import from models module
from models. load_models import load_models, get_model_info
from models.preprocessing import (
    validate_input, 
    prepare_features, 
    get_prediction_explanation,
    FEATURE_COLUMNS
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
        feature_file = Path(__file__).parent / "data" / "feature_info.json"
        with open(feature_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Could not load feature info: {str(e)}")
        return None

def create_input_form():
    """Create the input form for user data."""
    st.subheader("📊 Enter Patient Voice Measurements")
    
    feature_info = load_feature_info()
    
    # Create columns for better layout
    input_data = {}
    
    # Organize inputs in groups
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Frequency Features")
        input_data['MDVP:Fo(Hz)'] = st.number_input(
            "MDVP:Fo(Hz) - Average Fundamental Frequency",
            min_value=50.0, max_value=300.0, value=150.0, step=0.1,
            help="Average vocal fundamental frequency in Hz"
        )
        
        input_data['MDVP:Fhi(Hz)'] = st.number_input(
            "MDVP: Fhi(Hz) - Maximum Fundamental Frequency",
            min_value=50.0, max_value=600.0, value=180.0, step=0.1,
            help="Maximum vocal fundamental frequency in Hz"
        )
        
        input_data['MDVP:Flo(Hz)'] = st.number_input(
            "MDVP:Flo(Hz) - Minimum Fundamental Frequency",
            min_value=50.0, max_value=300.0, value=120.0, step=0.1,
            help="Minimum vocal fundamental frequency in Hz"
        )
        
        st.markdown("### Jitter Features")
        input_data['MDVP:Jitter(%)'] = st.number_input(
            "MDVP:Jitter(%) - Jitter Percentage",
            min_value=0.0, max_value=0.1, value=0.005, step=0.0001,
            help="Percentage of F0 variations"
        )
        
        input_data['MDVP: Jitter(Abs)'] = st.number_input(
            "MDVP: Jitter(Abs) - Absolute Jitter",
            min_value=0.0, max_value=0.001, value=0.00005, step=0.000001,
            help="Absolute jitter in seconds"
        )
        
        input_data['MDVP: RAP'] = st.number_input(
            "MDVP:RAP",
            min_value=0.0, max_value=0.05, value=0.003, step=0.0001,
            help="Relative Absolute Pitch"
        )
        
        input_data['MDVP:PPQ'] = st.number_input(
            "MDVP:PPQ",
            min_value=0.0, max_value=0.05, value=0.003, step=0.0001,
            help="Pitch Period Perturbation Quotient"
        )
        
        input_data['Jitter:DDP'] = st.number_input(
            "Jitter:DDP",
            min_value=0.0, max_value=0.1, value=0.01, step=0.0001,
            help="Jitter - Differential Divergence Period"
        )
    
    with col2:
        st.markdown("### Shimmer Features")
        input_data['MDVP:Shimmer'] = st.number_input(
            "MDVP: Shimmer",
            min_value=0.0, max_value=0.2, value=0.03, step=0.001,
            help="Amplitude variation"
        )
        
        input_data['MDVP: Shimmer(dB)'] = st.number_input(
            "MDVP: Shimmer(dB)",
            min_value=0.0, max_value=2.0, value=0.3, step=0.01,
            help="Amplitude variation in dB"
        )
        
        input_data['Shimmer:APQ3'] = st.number_input(
            "Shimmer:APQ3",
            min_value=0.0, max_value=0.1, value=0.015, step=0.001,
            help="Shimmer - Amplitude Perturbation Quotient"
        )
        
        input_data['Shimmer:APQ5'] = st.number_input(
            "Shimmer:APQ5",
            min_value=0.0, max_value=0.1, value=0.018, step=0.001,
            help="Shimmer - Amplitude Perturbation Quotient"
        )
        
        input_data['MDVP:APQ'] = st.number_input(
            "MDVP:APQ",
            min_value=0.0, max_value=0.1, value=0.02, step=0.001,
            help="Amplitude Perturbation Quotient"
        )
        
        input_data['Shimmer:DDA'] = st. number_input(
            "Shimmer:DDA",
            min_value=0.0, max_value=0.2, value=0.04, step=0.001,
            help="Shimmer - Differential Divergence Amplitude"
        )
        
        st.markdown("### Noise & Harmonic Features")
        input_data['NHR'] = st.number_input(
            "NHR - Noise-to-Harmonic Ratio",
            min_value=0.0, max_value=0.5, value=0.02, step=0.001,
            help="Ratio of noise to harmonic components"
        )
        
        input_data['HNR'] = st.number_input(
            "HNR - Harmonics-to-Noise Ratio",
            min_value=0.0, max_value=35.0, value=21.0, step=0.1,
            help="Ratio of harmonic to noise components"
        )
    
    # Additional features in another set of columns
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("### Nonlinear Features")
        input_data['RPDE'] = st.number_input(
            "RPDE - Recurrence Period Density Entropy",
            min_value=0.0, max_value=1.0, value=0.5, step=0.01,
            help="Nonlinear measure of signal variability"
        )
        
        input_data['DFA'] = st.number_input(
            "DFA - Detrended Fluctuation Analysis",
            min_value=0.4, max_value=0.9, value=0.71, step=0.01,
            help="Fractal dimension of the signal"
        )
        
        input_data['D2'] = st.number_input(
            "D2 - Correlation Dimension",
            min_value=1.0, max_value=4.0, value=2.3, step=0.1,
            help="Correlation dimension of signal"
        )
    
    with col4:
        st. markdown("### Recurrence Features")
        input_data['spread1'] = st.number_input(
            "spread1 - First Recurrence Plot Spread",
            min_value=-8.0, max_value=0.0, value=-5.0, step=0.1,
            help="Nonlinear measure of recurrence"
        )
        
        input_data['spread2'] = st.number_input(
            "spread2 - Second Recurrence Plot Spread",
            min_value=0.0, max_value=0.5, value=0.2, step=0.01,
            help="Nonlinear measure of recurrence"
        )
        
        input_data['PPE'] = st.number_input(
            "PPE - Pitch Period Entropy",
            min_value=0.0, max_value=1.0, value=0.2, step=0.01,
            help="Entropy of pitch period"
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