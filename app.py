# app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json
import sys
import io
import time

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

# ---------- Page Configuration ----------
st.set_page_config(
    page_title="Parkinson's Disease Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Custom CSS for black-blue theme ----------
st.markdown("""
    <style>
    /* General background - dark theme */
    .stApp {
        background-color: #0a1929;
        color: #e6f7ff;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Main content area */
    .main .block-container {
        background-color: #0a1929;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5 {
        color: #3399ff !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 600;
    }
    
    /* Text color */
    p, div, span {
        color: #cce7ff !important;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #0066cc;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.75em 1.5em;
        font-size: 16px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #0052a3;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 102, 204, 0.3);
    }
    
    /* Input fields */
    .stNumberInput>div>div>input {
        background-color: #1e3a5f;
        color: #e6f7ff;
        border: 1px solid #3399ff;
        border-radius: 4px;
    }
    
    .stNumberInput label {
        color: #99ccff !important;
        font-weight: 500;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1e3a5f;
        padding: 4px;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #1e3a5f;
        color: #99ccff;
        border-radius: 4px;
        padding: 8px 16px;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #0066cc !important;
        color: white !important;
    }
    
    /* Sidebar - dark blue */
    section[data-testid="stSidebar"] {
        background-color: #001e3c !important;
        border-right: 1px solid #3399ff;
    }
    
    section[data-testid="stSidebar"] * {
        color: #e6f7ff !important;
    }
    
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        color: #66b3ff !important;
    }
    
    section[data-testid="stSidebar"] .stRadio>div {
        background-color: #1e3a5f;
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
    }
    
    /* Welcome section */
    .welcome-box {
        background: linear-gradient(135deg, #001e3c 0%, #003366 100%);
        padding: 2.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        border: 1px solid #3399ff;
        box-shadow: 0 4px 20px rgba(51, 153, 255, 0.15);
    }
    
    .welcome-box h1 {
        color: white !important;
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    /* Feature cards */
    .feature-card {
        background-color: #1e3a5f;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #3399ff;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }
    
    .feature-card h3, .feature-card h4 {
        color: #66b3ff !important;
    }
    
    /* Prediction boxes */
    .prediction-box {
        padding: 25px;
        border-radius: 10px;
        margin: 15px 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    
    .positive {
        background-color: #331111;
        border-left: 5px solid #ff4d4d;
        color: #ffcccc !important;
    }
    
    .positive h3 {
        color: #ff9999 !important;
    }
    
    .negative {
        background-color: #113311;
        border-left: 5px solid #4dff4d;
        color: #ccffcc !important;
    }
    
    .negative h3 {
        color: #99ff99 !important;
    }
    
    /* Metric cards */
    .metric-card {
        background-color: #1e3a5f;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #3399ff;
        margin: 10px 0;
        text-align: center;
    }
    
    /* Progress bars */
    .stProgress > div > div > div > div {
        background-color: #3399ff;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #1e3a5f;
        color: #66b3ff !important;
        border-radius: 6px;
        border: 1px solid #3399ff;
    }
    
    /* Dataframes */
    .dataframe {
        background-color: #1e3a5f !important;
    }
    
    /* Alert boxes */
    .stAlert {
        background-color: #1e3a5f;
        border-left: 4px solid #3399ff;
        border-radius: 8px;
    }
    
    .stAlert [data-testid="stMarkdownContainer"] p {
        color: #cce7ff !important;
    }
    
    /* File uploader */
    .stFileUploader>div>div {
        background-color: #1e3a5f;
        border: 2px dashed #3399ff;
        border-radius: 8px;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1.5rem;
        color: #99ccff !important;
        font-size: 0.9rem;
        margin-top: 3rem;
        border-top: 1px solid #3399ff;
    }
    
    /* Fix all text visibility */
    .stDataFrame, .stTable {
        color: #cce7ff !important;
    }
    
    .stDataFrame td, .stDataFrame th {
        color: #cce7ff !important;
        background-color: #1e3a5f !important;
    }
    
    /* Fix plotly charts background */
    .js-plotly-plot {
        background-color: #1e3a5f !important;
    }
    
    /* Info boxes */
    .stInfo {
        background-color: #1e3a5f !important;
        border-left: 4px solid #3399ff !important;
    }
    
    /* Warning boxes */
    .stWarning {
        background-color: #332211 !important;
        border-left: 4px solid #ff9900 !important;
    }
    
    /* Error boxes */
    .stError {
        background-color: #331111 !important;
        border-left: 4px solid #ff3333 !important;
    }
    
    /* Success boxes */
    .stSuccess {
        background-color: #113311 !important;
        border-left: 4px solid #33cc33 !important;
    }
    
    /* Radio buttons in sidebar */
    section[data-testid="stSidebar"] .stRadio label {
        color: #e6f7ff !important;
        padding: 8px;
        border-radius: 6px;
        transition: all 0.2s ease;
    }
    
    section[data-testid="stSidebar"] .stRadio label:hover {
        background-color: #2d4a75;
    }
    
    /* Markdown text */
    .stMarkdown {
        color: #cce7ff !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-color: #3399ff transparent transparent transparent !important;
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

# ---------- Utility: circular gauge ----------
def show_gauge(container, percent:int, color:str, label:str):
    """Display gauge chart"""
    percent = max(0, min(100, int(round(percent))))
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=percent,
        number={'valueformat': "d", 'suffix': '%', 'font': {'size': 28, 'color': '#ffffff'}},
        title={'text': label, 'font': {'size': 16, 'color': '#99ccff'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': '#3399ff', 'tickfont': {'color': '#ffffff'}},
            'bar': {'color': color},
            'bgcolor': "#1e3a5f",
            'borderwidth': 2,
            'bordercolor': "#3399ff",
            'steps': [
                {'range': [0, 30], 'color': "#113311"},
                {'range': [30, 70], 'color': "#332211"},
                {'range': [70, 100], 'color': "#331111"}
            ],
            'threshold': {
                'line': {'color': color, 'width': 4},
                'thickness': 0.85,
                'value': percent
            }
        }
    ))
    
    fig.update_layout(
        margin=dict(l=30, r=30, t=40, b=20),
        height=340,
        paper_bgcolor='#0a1929',
        font={'color': "#ffffff"}
    )
    container.plotly_chart(fig, use_container_width=True)

# ---------- Footer Function ----------
def show_footer():
    """Display copyright footer on all pages"""
    st.markdown("""
    <div class="footer">
        <p style="margin: 0.2rem;">© 2025 Parkinson's Disease Detector</p>
        <p style="margin: 0.2rem;">Developed by Peter Boakye Prempeh ❤️ for medical research</p>
        <p style="margin: 0.2rem; font-size: 0.8rem; color: #66b3ff !important;">For educational purposes only</p>
    </div>
    """, unsafe_allow_html=True)

def create_input_form():
    """Create the input form for user data with enhanced layout and descriptions."""
    st.subheader("📊 Enter Voice Measurements")
    
    raw_feature_info = load_feature_info() or {}
    feature_info = raw_feature_info.get('features', {}) if isinstance(raw_feature_info, dict) else {}
    
    # Feature categories for better organization
    feature_categories = {
        "Frequency Features": [
            'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)'
        ],
        "Jitter Features": [
            'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP',
            'MDVP:PPQ', 'Jitter:DDP'
        ],
        "Shimmer Features": [
            'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3',
            'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA'
        ],
        "Noise & Harmonic Features": [
            'NHR', 'HNR'
        ],
        "Nonlinear & Complexity Features": [
            'RPDE', 'DFA', 'D2', 'spread1', 'spread2', 'PPE'
        ]
    }
    
    input_data = {}
    
    # Create tabs for different feature categories
    tabs = st.tabs(list(feature_categories.keys()))
    
    for tab_idx, (category, features) in enumerate(feature_categories.items()):
        with tabs[tab_idx]:
            st.markdown(f"### {category}")
            cols = st.columns(2)
            for i, feat in enumerate(features):
                col = cols[i % 2]
                default = 0.0
                # Sensible defaults for features
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
                        label=f"**{feat}**",
                        value=float(default),
                        help=help_text,
                        format="%.6f",
                        key=f"input_{feat}"
                    )
    
    return input_data

def display_predictions(model, scaler, input_data, container=None):
    """Display prediction results with enhanced visualization."""
    if container is None:
        container = st
    
    container.subheader("🔍 Prediction Results")
    
    # Prepare features
    features_df = prepare_features(input_data, scaler)
    
    if features_df is None:
        container.error("Failed to prepare features for prediction")
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
        
        # Create a full-width container for the main results
        results_container = container.container()
        
        with results_container:
            # Top section with prediction and gauge side by side
            col1, col2 = st.columns([1.5, 1])  # Adjusted ratio
            
            with col1:
                if prediction == 1:
                    st.markdown(
                        f'<div class="prediction-box positive">'
                        f'<h3>⚠️ Parkinson\'s Disease Risk Detected</h3>'
                        f'<p><strong>Confidence Level:</strong> {explanation["confidence"]:.2f}%</p>'
                        f'<p>{explanation["description"]}</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="prediction-box negative">'
                        f'<h3>✅ No Parkinson\'s Disease Risk Detected</h3>'
                        f'<p><strong>Confidence Level:</strong> {explanation["confidence"]:.2f}%</p>'
                        f'<p>{explanation["description"]}</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
            
            with col2:
                # Add some vertical spacing
                st.markdown("<br>", unsafe_allow_html=True)
                # Gauge
                conf_percent = explanation['confidence']
                gauge_color = "#ff4d4d" if prediction == 1 else "#4dff4d"
                show_gauge(st, conf_percent, gauge_color, "Confidence Level")
        
        # Display probability distribution in its own section
        st.subheader("📈 Probability Distribution")
        
        fig = go.Figure(data=[
            go.Bar(
                x=['Negative', 'Positive'],
                y=[probabilities[0] * 100, probabilities[1] * 100],
                marker_color=['#33cc33', '#ff3333'],
                text=[f'{probabilities[0]*100:.2f}%', f'{probabilities[1]*100:.2f}%'],
                textposition='auto',
                textfont=dict(size=14, color='#ffffff')
            )
        ])
        
        fig.update_layout(
            height=350,  # Slightly reduced height
            showlegend=False,
            yaxis_title="Probability (%)",
            xaxis_title="Prediction Class",
            title="Probability Distribution of Prediction Classes",
            title_font=dict(size=16, color='#66b3ff'),
            paper_bgcolor='#0a1929',
            plot_bgcolor='#1e3a5f',
            font=dict(color='#ffffff'),
            xaxis=dict(tickfont=dict(color='#ffffff')),
            yaxis=dict(tickfont=dict(color='#ffffff')),
            margin=dict(l=50, r=50, t=50, b=50)  # Add margins
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance and extreme values
        if hasattr(model, 'feature_importances_'):
            extreme_features = []
            for feat in FEATURE_COLUMNS:
                val = input_data.get(feat, 0)
                # Check if value is significantly different from normal range
                if feat in ['MDVP:Jitter(%)', 'MDVP:Shimmer', 'NHR'] and val > 0.1:
                    extreme_features.append(feat)
            
            if extreme_features:
                st.warning(f"⚠️ **Note:** High values detected in: {', '.join(extreme_features)}")
        
        # Display feature values table
        with st.expander("📋 View Input Features Summary", expanded=False):
            features_display = pd.DataFrame({
                'Feature': list(input_data.keys()),
                'Value': list(input_data.values())
            }).round(6)
            
            st.dataframe(features_display, use_container_width=True)
        
        # Recommendations based on prediction
        st.markdown("---")
        st.subheader("💡 Recommendations")
        
        if prediction == 1:
            st.warning("""
            **Based on this prediction, we recommend:**
            
            - 🏥 **Consult a neurologist** or movement disorder specialist
            - 📋 **Request comprehensive evaluation** including medical history
            - 🔍 **Consider additional tests** if recommended by your doctor
            - 📝 **Document symptoms** including progression over time
            - 👨‍👩‍👧‍👦 **Bring a family member** for additional observations
            - ⏰ **Early consultation** can lead to better management
            
            **Remember:** This is a screening tool, not a diagnosis. Only qualified healthcare professionals can diagnose Parkinson's Disease.
            """)
        else:
            st.success("""
            **Based on this prediction:**
            
            - ✅ **Low likelihood** of Parkinson's Disease based on voice analysis
            - 🔄 **Continue monitoring** - consult a doctor if symptoms appear
            - 🎤 **Maintain voice health** with good vocal hygiene
            - 📅 **Consider periodic screenings** if you have risk factors
            - 👀 **Watch for symptoms** such as tremors or stiffness
            - 💪 **Support neurological health** with exercise and balanced diet
            
            **Note:** A negative result does not guarantee absence of Parkinson's Disease.
            """)
    
    except Exception as e: 
        container.error(f"Error making prediction: {str(e)}")

def display_model_info(model):
    """Display information about the loaded model with enhanced visualization."""
    st.subheader("ℹ️ Model Information")
    
    model_info = get_model_info(model)
    
    if model_info: 
        # Model metrics in cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Model Type", model_info['model_type'])
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Number of Features", len(FEATURE_COLUMNS))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            prob_support = "Yes" if model_info['has_predict_proba'] else "No"
            st.metric("Probability Support", prob_support)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Model description
        st.markdown("""
        <div class="feature-card">
        <h4>About This Model</h4>
        <p>This Random Forest Classifier analyzes 22 distinct voice features to predict 
        the likelihood of Parkinson's Disease. The model was trained on acoustic 
        measurements from voice recordings, focusing on features like frequency variations 
        (jitter), amplitude variations (shimmer), and noise-to-harmonic ratios.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display feature importance if available
        if model_info['has_feature_importance']:
            st.subheader("🎯 Top 10 Most Important Features")
            
            importance_df = pd.DataFrame({
                'Feature': FEATURE_COLUMNS,
                'Importance': model_info['feature_importances']
            }).sort_values('Importance', ascending=False).head(10)
            
            fig = go.Figure(data=[
                go.Bar(
                    x=importance_df['Importance'],
                    y=importance_df['Feature'],
                    orientation='h',
                    marker_color='#3399ff',
                    text=importance_df['Importance'].round(4),
                    textposition='auto',
                    textfont=dict(color='#ffffff')
                )
            ])
            
            fig.update_layout(
                height=500,
                xaxis_title="Importance Score",
                yaxis_title="Feature",
                showlegend=False,
                title="Feature Importance Ranking",
                title_font=dict(size=16, color='#66b3ff'),
                paper_bgcolor='#0a1929',
                plot_bgcolor='#1e3a5f',
                font=dict(color='#ffffff'),
                xaxis=dict(tickfont=dict(color='#ffffff')),
                yaxis=dict(tickfont=dict(color='#ffffff'))
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Full feature importance table
            with st.expander("📊 View Complete Feature Importance Table"):
                full_importance_df = pd.DataFrame({
                    'Feature': FEATURE_COLUMNS,
                    'Importance': model_info['feature_importances']
                }).sort_values('Importance', ascending=False)
                
                st.dataframe(full_importance_df, use_container_width=True)

def batch_predictions(model, scaler, uploaded_files):
    """Handle batch predictions from multiple CSV files."""
    all_dataframes = []
    
    for file in uploaded_files:
        try:
            df = pd.read_csv(file)
            df.columns = [c.strip() for c in df.columns]
            
            # Check for required columns
            missing_cols = [col for col in FEATURE_COLUMNS if col not in df.columns]
            if missing_cols:
                st.warning(f"File '{file.name}' is missing columns: {', '.join(missing_cols)}")
                continue
            
            # Add file identifier
            df.insert(0, 'File_Source', file.name)
            all_dataframes.append(df)
            
        except Exception as e:
            st.error(f"Could not process file '{file.name}': {e}")
    
    if len(all_dataframes) == 0:
        st.error("No valid data found in uploaded files.")
        return
    
    # Combine all data
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    st.write(f"### 📁 Processing {len(combined_df)} records from {len(uploaded_files)} file(s)")
    with st.expander("👁️ View Uploaded Data"):
        st.dataframe(combined_df, use_container_width=True)
    
    # Process each record
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, row in combined_df.iterrows():
        status_text.text(f"Processing record {idx + 1} of {len(combined_df)}...")
        
        # Extract features
        input_data = {col: row[col] for col in FEATURE_COLUMNS if col in row}
        
        # Make prediction
        features_df = prepare_features(input_data, scaler)
        if features_df is not None:
            try:
                prediction = model.predict(features_df)[0]
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(features_df)[0][1]
                    confidence = prob if prediction == 1 else (1 - prob)
                else:
                    confidence = 0.5
                
                results.append({
                    'Record': idx + 1,
                    'Source_File': row.get('File_Source', 'Unknown'),
                    'Prediction': 'Parkinson\'s Risk' if prediction == 1 else 'No Risk',
                    'Confidence': f"{confidence:.2%}",
                    'Risk_Level': 'High' if prediction == 1 and confidence > 0.7 else 
                                  'Medium' if prediction == 1 else 'Low'
                })
            except Exception as e:
                st.warning(f"Could not process record {idx + 1}: {e}")
        
        progress_bar.progress((idx + 1) / len(combined_df))
    
    progress_bar.empty()
    status_text.empty()
    
    # Display results
    if results:
        results_df = pd.DataFrame(results)
        st.subheader("📊 Batch Prediction Results")
        
        # Summary statistics
        pos_count = sum(1 for r in results if 'Parkinson\'s' in r['Prediction'])
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(results))
        with col2:
            st.metric("High Risk Predictions", pos_count)
        with col3:
            st.metric("Positive Rate", f"{(pos_count/len(results)*100):.1f}%")
        
        # Results table
        st.dataframe(results_df, use_container_width=True)
        
        # Download results
        csv_buffer = io.StringIO()
        results_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv_buffer.getvalue(),
            file_name="batch_predictions.csv",
            mime="text/csv"
        )
        
        # Visualizations
        st.subheader("📈 Batch Analysis")
        tab1, tab2 = st.tabs(["Distribution", "Risk Levels"])
        
        with tab1:
            pred_counts = results_df['Prediction'].value_counts()
            fig1 = px.pie(
                values=pred_counts.values,
                names=pred_counts.index,
                title="Prediction Distribution",
                color_discrete_sequence=['#33cc33', '#ff3333']
            )
            fig1.update_layout(
                paper_bgcolor='#0a1929',
                font=dict(color='#ffffff'),
                title_font=dict(color='#66b3ff')
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with tab2:
            risk_counts = results_df['Risk_Level'].value_counts()
            fig2 = px.bar(
                x=risk_counts.index,
                y=risk_counts.values,
                title="Risk Level Distribution",
                color=risk_counts.index,
                color_discrete_map={'High': '#ff3333', 'Medium': '#ff9900', 'Low': '#33cc33'}
            )
            fig2.update_layout(
                paper_bgcolor='#0a1929',
                plot_bgcolor='#1e3a5f',
                font=dict(color='#ffffff'),
                title_font=dict(color='#66b3ff'),
                xaxis=dict(tickfont=dict(color='#ffffff')),
                yaxis=dict(tickfont=dict(color='#ffffff'))
            )
            st.plotly_chart(fig2, use_container_width=True)

def main():
    """Main application function with enhanced navigation."""
    
    # Initialize session state for page navigation
    if 'page' not in st.session_state:
        st.session_state.page = "🏠 Home"
    
    # Sidebar Navigation
    with st.sidebar:
        st.markdown("""
        <div style="padding: 1.5rem; background: linear-gradient(135deg, #001e3c 0%, #003366 100%); 
                    border-radius: 10px; margin-bottom: 1.5rem; border: 1px solid #3399ff;">
            <h2 style="margin: 0; color: #ffffff !important;">🧠 Parkinson's Detector</h2>
            <p style="margin: 0.5rem 0 0 0; color: #99ccff !important;">AI-Powered Voice Analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        page = st.radio(
            "Navigation",
            ["🏠 Home", "🔍 Prediction", "📁 Batch Analysis", "📊 Model Insights", "ℹ️ About", "📚 Help"],
            index=["🏠 Home", "🔍 Prediction", "📁 Batch Analysis", "📊 Model Insights", "ℹ️ About", "📚 Help"].index(st.session_state.page),
            label_visibility="collapsed"
        )
        
        # Update session state
        if page != st.session_state.page:
            st.session_state.page = page
        
        st.markdown("---")
        
        # Quick stats/status section
        st.markdown("### 📊 Quick Stats")
        st.markdown("""
        <div style="background-color: #1e3a5f; padding: 1rem; border-radius: 8px; border-left: 4px solid #3399ff;">
        - 22 Voice Features analyzed
        - Random Forest algorithm
        - Real-time predictions
        - Batch processing supported
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("""
        <div style="padding: 1rem; background-color: #331111; border-radius: 8px; border-left: 4px solid #ff3333;">
        <h4 style="color: #ff9999 !important;">⚠️ Important Notice</h4>
        <p style="font-size: 0.8rem; color: #ffcccc !important;">
        This application is for <strong>educational and research purposes</strong> only. 
        It is not a medical diagnostic tool. Always consult healthcare professionals 
        for medical advice and diagnosis.
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Load models once
    if 'models_loaded' not in st.session_state:
        with st.spinner("🚀 Loading AI models..."):
            model, scaler = load_models()
            st.session_state.model = model
            st.session_state.scaler = scaler
            st.session_state.models_loaded = True
    
    model = st.session_state.model
    scaler = st.session_state.scaler
    
    # Main content based on selected page
    if page == "🏠 Home":
        # Welcome Section
        st.markdown("""
            <div class="welcome-box">
                <h1>🧠 Welcome to Parkinson's Disease Detector</h1>
                <p style="font-size: 1.2rem; margin-top: 1rem;">
                    An advanced AI-powered tool for early Parkinson's Disease screening using voice analysis
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # About the App Section
        st.markdown("## 🎯 About This Application")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="feature-card">
                <h3>🤖 How It Works</h3>
                <p>This application uses <strong>machine learning</strong> to analyze 22 distinct voice features 
                and predict the likelihood of Parkinson's Disease. Our model examines:</p>
                <ul>
                    <li>Frequency variations (jitter measurements)</li>
                    <li>Amplitude variations (shimmer measurements)</li>
                    <li>Harmonic and noise ratios</li>
                    <li>Non-linear complexity measures</li>
                </ul>
                <p>The system provides instant predictions with confidence scores and detailed explanations.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-card">
                <h3>✨ Key Features</h3>
                <ul>
                    <li><strong>Manual Entry:</strong> Input voice features with detailed descriptions</li>
                    <li><strong>Batch Processing:</strong> Upload CSV files for multiple predictions</li>
                    <li><strong>Real-time Analysis:</strong> Instant predictions with visualizations</li>
                    <li><strong>Model Transparency:</strong> View feature importance and insights</li>
                    <li><strong>Educational Resources:</strong> Learn about Parkinson's Disease indicators</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # About Parkinson's Disease
        st.markdown("## 🧬 Understanding Parkinson's Disease")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="feature-card">
                <h4>📋 What is Parkinson's?</h4>
                <p>Parkinson's Disease (PD) is a progressive neurodegenerative disorder affecting movement control. 
                It occurs when dopamine-producing neurons in the brain deteriorate.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-card">
                <h4>⚠️ Common Symptoms</h4>
                <ul style="font-size: 0.9rem;">
                    <li>Tremor (shaking)</li>
                    <li>Bradykinesia (slowed movement)</li>
                    <li>Muscle rigidity</li>
                    <li>Postural instability</li>
                    <li><strong>Speech changes</strong></li>
                    <li>Handwriting changes</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="feature-card">
                <h4>🎤 Voice & Parkinson's</h4>
                <p>Voice changes are often early PD indicators:</p>
                <ul style="font-size: 0.9rem;">
                    <li>Hypophonia (soft speech)</li>
                    <li>Monotone voice</li>
                    <li>Hoarseness</li>
                    <li>Articulation issues</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Call to Action
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 1.5rem; background-color: #1e3a5f; border-radius: 10px; border: 1px solid #3399ff;">
                <h3 style="color: #66b3ff !important;">Ready to Get Started?</h3>
                <p>Choose a prediction method to begin your analysis.</p>
            </div>
            """, unsafe_allow_html=True)
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("🔍 Single Prediction", use_container_width=True):
                    st.session_state.page = "🔍 Prediction"
                    st.rerun()
            with col_btn2:
                if st.button("📁 Batch Analysis", use_container_width=True):
                    st.session_state.page = "📁 Batch Analysis"
                    st.rerun()
        
        # Footer
        show_footer()
    
    elif page == "🔍 Prediction":
        st.title("🔍 Parkinson's Disease Prediction")
        
        if model is None:
            st.error("""
            ⚠️ Error loading models! 
            
            Please ensure you have: 
            1. Trained the model in the Jupyter notebook
            2. Saved it as `models/parkinsons_model.pkl`
            3. Saved the scaler as `models/scaler.pkl`
            
            See the notebook for training instructions.
            """)
            show_footer()
            return
        
        st.success("✅ AI models loaded successfully!")
        
        # Create input form
        input_data = create_input_form()
        
        # Validation
        errors, warnings = validate_input(input_data)
        
        if errors: 
            st.warning("⚠️ **Validation Warnings:**")
            for error in errors:
                st.warning(f"- {error}")
        
        # Predict button
        col1, col2 = st.columns([1, 3])
        
        with col1:
            if st.button("🔍 Make Prediction", use_container_width=True):
                display_predictions(model, scaler, input_data)
        
        # Footer
        show_footer()
    
    elif page == "📁 Batch Analysis":
        st.title("📁 Batch Analysis")
        
        if model is None:
            st.error("Models not loaded. Please go to the Prediction page first.")
            show_footer()
            return
        
        st.markdown("""
        <div class="feature-card">
        <h3>📋 Batch Processing Instructions</h3>
        <p>Upload one or more CSV files containing voice measurements. Each file should include:</p>
        <ul>
            <li>All 22 required voice features as columns</li>
            <li>Column names matching the feature names exactly</li>
            <li>Numerical values for all features</li>
            <li>Each row represents one patient/sample</li>
        </ul>
        <p>Download the sample template below to ensure proper formatting.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sample CSV download
        sample_data = pd.DataFrame({feat: [0.0] for feat in FEATURE_COLUMNS})
        csv_buffer = io.StringIO()
        sample_data.to_csv(csv_buffer, index=False)
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Download Sample CSV",
                data=csv_buffer.getvalue(),
                file_name="sample_voice_features.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose CSV files",
            type=["csv"],
            accept_multiple_files=True,
            help="Upload one or more CSV files with voice feature data"
        )
        
        if uploaded_files:
            if st.button("🚀 Process Batch Files", use_container_width=True):
                batch_predictions(model, scaler, uploaded_files)
        
        show_footer()
    
    elif page == "📊 Model Insights":
        st.title("📊 Model Insights & Analysis")
        
        if model is None:
            st.error("Models not loaded. Please go to the Prediction page first.")
            show_footer()
            return
        
        display_model_info(model)
        
        # Additional analysis
        st.markdown("---")
        st.subheader("🔍 Model Performance Analysis")
        
        # Create tabs for different analyses
        tab1, tab2, tab3 = st.tabs(["Feature Relationships", "Model Architecture", "Prediction Patterns"])
        
        with tab1:
            st.markdown("""
            <div class="feature-card">
            <h4>Feature Relationships in Parkinson's Disease</h4>
            <p>Key patterns observed in Parkinson's patients:</p>
            <ul>
                <li><strong>Increased Jitter & Shimmer:</strong> Indicates vocal instability</li>
                <li><strong>Higher NHR values:</strong> More noise in vocal signal</li>
                <li><strong>Lower HNR values:</strong> Reduced harmonic clarity</li>
                <li><strong>Altered nonlinear features:</strong> Changes in voice complexity</li>
            </ul>
            <p>These patterns help the model distinguish between healthy and affected voices.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with tab2:
            st.markdown("""
            <div class="feature-card">
            <h4>Model Architecture Details</h4>
            <p><strong>Algorithm:</strong> Random Forest Classifier</p>
            <p><strong>Number of Trees:</strong> 100 (default)</p>
            <p><strong>Max Depth:</strong> Optimized for voice feature analysis</p>
            <p><strong>Feature Selection:</strong> All 22 voice features used</p>
            <p><strong>Cross-validation:</strong> 5-fold cross-validation applied</p>
            <p><strong>Performance Metrics:</strong> Accuracy, Precision, Recall, F1-score</p>
            </div>
            """, unsafe_allow_html=True)
        
        with tab3:
            st.markdown("""
            <div class="feature-card">
            <h4>Understanding Prediction Patterns</h4>
            <p>The model analyzes combinations of features to make predictions:</p>
            <ul>
                <li><strong>Multiple feature interactions:</strong> Considers how features relate</li>
                <li><strong>Threshold-based decisions:</strong> Different combinations trigger predictions</li>
                <li><strong>Confidence scoring:</strong> Probability estimates for each prediction</li>
                <li><strong>Feature contribution:</strong> Each feature's impact on final decision</li>
            </ul>
            <p>This multi-faceted approach improves prediction accuracy and reliability.</p>
            </div>
            """, unsafe_allow_html=True)
        
        show_footer()
    
    elif page == "ℹ️ About": 
        st.title("ℹ️ About Parkinson's Disease & This Application")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="feature-card">
            <h3>🧬 What is Parkinson's Disease?</h3>
            <p>Parkinson's disease is a progressive neurological disorder that affects movement control.  
            It develops gradually, often starting with subtle symptoms like mild tremors or stiffness.</p>
            
            <h4>Key Characteristics:</h4>
            <ul>
                <li><strong>Progressive:</strong> Symptoms worsen over time</li>
                <li><strong>Neurological:</strong> Affects the nervous system</li>
                <li><strong>Movement Disorder:</strong> Primary impact on motor function</li>
                <li><strong>Variable:</strong> Different people experience different symptoms</li>
            </ul>
            
            <h4>Early Detection Benefits:</h4>
            <p>Early detection allows for:</p>
            <ul>
                <li>Better treatment planning</li>
                <li>Symptom management strategies</li>
                <li>Improved quality of life</li>
                <li>Clinical trial eligibility</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-card">
            <h3>🎤 Voice Analysis in Parkinson's</h3>
            <p>Voice changes are among the earliest and most sensitive indicators of Parkinson's Disease.</p>
            
            <h4>Common Voice Changes:</h4>
            <ul>
                <li><strong>Hypophonia:</strong> Reduced vocal loudness</li>
                <li><strong>Monotone speech:</strong> Reduced pitch variation</li>
                <li><strong>Hoarseness:</strong> Rough or breathy voice quality</li>
                <li><strong>Articulation issues:</strong> Imprecise consonant production</li>
                <li><strong>Reduced stress patterns:</strong> Monotonic speech rhythm</li>
            </ul>
            
            <h4>Why Voice Analysis Works:</h4>
            <ul>
                <li><strong>Non-invasive:</strong> No needles or scans required</li>
                <li><strong>Objective:</strong> Quantitative measurements</li>
                <li><strong>Sensitive:</strong> Detects subtle changes</li>
                <li><strong>Accessible:</strong> Can be done remotely</li>
                <li><strong>Cost-effective:</strong> Lower than traditional methods</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Technology Stack
        st.markdown("---")
        st.subheader("🛠️ Technology Stack")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
            <h4>🤖 Machine Learning</h4>
            <ul style="list-style: none; padding-left: 0;">
                <li>• scikit-learn</li>
                <li>• Random Forest</li>
                <li>• Feature Engineering</li>
                <li>• Model Validation</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
            <h4>🌐 Web Application</h4>
            <ul style="list-style: none; padding-left: 0;">
                <li>• Streamlit</li>
                <li>• Plotly Charts</li>
                <li>• Interactive Forms</li>
                <li>• Real-time Updates</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
            <h4>📊 Data Processing</h4>
            <ul style="list-style: none; padding-left: 0;">
                <li>• pandas</li>
                <li>• NumPy</li>
                <li>• Data Validation</li>
                <li>• Batch Processing</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Disclaimer
        st.markdown("---")
        st.error("""
        ### ⚠️ Important Medical Disclaimer
        
        **This application is for EDUCATIONAL AND RESEARCH PURPOSES ONLY.**
        
        - **NOT** a medical diagnostic tool
        - **NOT** a substitute for professional medical advice
        - **NOT** approved for clinical use
        - Results should be interpreted by qualified healthcare professionals
        - Always consult with a neurologist or movement disorder specialist for diagnosis
        - False positives and false negatives are possible
        
        **Use this tool responsibly and in conjunction with professional medical care.**
        """)
        
        show_footer()
    
    elif page == "📚 Help":
        st.title("📚 Help & Feature Descriptions")
        
        # Feature Categories with descriptions
        feature_categories = {
            "Frequency Features": {
                "description": "Measurements of vocal fundamental frequency",
                "features": {
                    "MDVP:Fo(Hz)": "Average vocal fundamental frequency (pitch)",
                    "MDVP:Fhi(Hz)": "Maximum vocal fundamental frequency",
                    "MDVP:Flo(Hz)": "Minimum vocal fundamental frequency"
                }
            },
            "Jitter Features": {
                "description": "Cycle-to-cycle variations in fundamental frequency",
                "features": {
                    "MDVP:Jitter(%)": "Percentage of pitch period variations",
                    "MDVP:Jitter(Abs)": "Absolute jitter in microseconds",
                    "MDVP:RAP": "Relative Average Perturbation",
                    "MDVP:PPQ": "Pitch Period Perturbation Quotient",
                    "Jitter:DDP": "Jitter - Differential Divergence Period"
                }
            },
            "Shimmer Features": {
                "description": "Cycle-to-cycle variations in amplitude",
                "features": {
                    "MDVP:Shimmer": "Amplitude variation between cycles",
                    "MDVP:Shimmer(dB)": "Amplitude variation in decibels",
                    "Shimmer:APQ3": "Amplitude Perturbation Quotient (3 cycles)",
                    "Shimmer:APQ5": "Amplitude Perturbation Quotient (5 cycles)",
                    "MDVP:APQ": "Average Amplitude Perturbation Quotient",
                    "Shimmer:DDA": "Differential Divergence Amplitude"
                }
            },
            "Noise & Harmonic Features": {
                "description": "Measurements of signal quality and noise",
                "features": {
                    "NHR": "Noise-to-Harmonic Ratio (higher = more noise)",
                    "HNR": "Harmonics-to-Noise Ratio (higher = clearer voice)"
                }
            },
            "Nonlinear & Complexity Features": {
                "description": "Advanced measures of voice signal complexity",
                "features": {
                    "RPDE": "Recurrence Period Density Entropy",
                    "DFA": "Detrended Fluctuation Analysis",
                    "D2": "Correlation Dimension",
                    "spread1": "Nonlinear measure of fundamental frequency variation",
                    "spread2": "Secondary nonlinear measure",
                    "PPE": "Pitch Period Entropy"
                }
            }
        }
        
        # Create tabs for each category
        tabs = st.tabs(list(feature_categories.keys()))
        
        for tab_idx, (category, info) in enumerate(feature_categories.items()):
            with tabs[tab_idx]:
                st.markdown(f"### {category}")
                st.markdown(f"*{info['description']}*")
                
                # Create cards for each feature
                for feature_name, description in info['features'].items():
                    with st.expander(f"**{feature_name}**"):
                        st.markdown(f"**Description:** {description}")
                        
                        # Add typical ranges if available
                        typical_ranges = {
                            "MDVP:Fo(Hz)": "100-300 Hz (varies by gender/age)",
                            "MDVP:Jitter(%)": "<0.5% normal, >0.8% may indicate issues",
                            "MDVP:Shimmer": "<3% normal, >5% may indicate issues",
                            "NHR": "<0.1 normal, >0.2 may indicate issues",
                            "HNR": ">20 dB normal, <15 dB may indicate issues"
                        }
                        
                        if feature_name in typical_ranges:
                            st.markdown(f"**Typical Range:** {typical_ranges[feature_name]}")
        
        # Usage Guide
        st.markdown("---")
        st.subheader("🎯 How to Use This Application")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="feature-card">
            <h4>🔍 Single Prediction</h4>
            <ol>
                <li>Go to <strong>Prediction</strong> page</li>
                <li>Enter voice feature values</li>
                <li>Click <strong>Make Prediction</strong></li>
                <li>View results and recommendations</li>
            </ol>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-card">
            <h4>📁 Batch Analysis</h4>
            <ol>
                <li>Go to <strong>Batch Analysis</strong> page</li>
                <li>Download sample CSV template</li>
                <li>Prepare your data file(s)</li>
                <li>Upload CSV file(s)</li>
                <li>Process and view results</li>
            </ol>
            </div>
            """, unsafe_allow_html=True)
        
        show_footer()

if __name__ == "__main__":
    main()