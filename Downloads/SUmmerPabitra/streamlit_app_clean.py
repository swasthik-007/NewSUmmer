import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="🧠 Disease Detection System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-result {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        margin: 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load the best model and preprocessing objects"""
    models_dir = 'c:/Users/mohan/Downloads/SUmmerPabitra/models'
    
    try:
        loaded_objects = {}
        
        # Load model info
        info_path = os.path.join(models_dir, 'model_info.pkl')
        if os.path.exists(info_path):
            try:
                loaded_objects['model_info'] = joblib.load(info_path)
                print(f"Successfully loaded model_info from {info_path}")
            except Exception as e:
                print(f"Error loading model_info: {e}")
                loaded_objects['model_info'] = None
        else:
            print(f"Model info file not found: {info_path}")
            loaded_objects['model_info'] = None
        
        # Load preprocessing objects
        scaler_path = os.path.join(models_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            try:
                loaded_objects['scaler'] = joblib.load(scaler_path)
                print(f"Successfully loaded scaler from {scaler_path}")
            except Exception as e:
                print(f"Error loading scaler: {e}")
                loaded_objects['scaler'] = None
        else:
            print(f"Scaler file not found: {scaler_path}")
            loaded_objects['scaler'] = None
        
        encoder_path = os.path.join(models_dir, 'label_encoder.pkl')
        if os.path.exists(encoder_path):
            try:
                loaded_objects['label_encoder'] = joblib.load(encoder_path)
                print(f"Successfully loaded label_encoder from {encoder_path}")
            except Exception as e:
                print(f"Error loading label_encoder: {e}")
                loaded_objects['label_encoder'] = None
        else:
            print(f"Label encoder file not found: {encoder_path}")
            loaded_objects['label_encoder'] = None
        
        # Load best model
        best_model_path = os.path.join(models_dir, 'best_model.pkl')
        if os.path.exists(best_model_path):
            try:
                loaded_objects['best_model'] = joblib.load(best_model_path)
                print(f"Successfully loaded best_model from {best_model_path}")
            except Exception as e:
                print(f"Error loading best_model: {e}")
                loaded_objects['best_model'] = None
        else:
            print(f"Best model file not found: {best_model_path}")
            loaded_objects['best_model'] = None
        
        return loaded_objects
        
    except Exception as e:
        print(f"General error loading models: {str(e)}")
        st.error(f"Error loading models: {str(e)}")
        return None

class DiseasePredictor:
    def __init__(self):
        """Initialize the disease predictor"""
        self.models_dir = 'c:/Users/mohan/Downloads/SUmmerPabitra/models'
        
        # Load models using cached function
        loaded_models = load_models()
        
        if loaded_models:
            self.scaler = loaded_models['scaler']
            self.label_encoder = loaded_models['label_encoder']
            self.model_info = loaded_models['model_info']
            self.best_model = loaded_models['best_model']
            self.models_loaded = True
        else:
            self.scaler = None
            self.label_encoder = None
            self.model_info = None
            self.best_model = None
            self.models_loaded = False        
    def are_models_loaded(self):
        """Check if all required models are loaded"""
        return (self.models_loaded and 
                self.scaler is not None and 
                self.label_encoder is not None and 
                self.best_model is not None)
    
    def predict_disorder(self, input_data):
        """Predict disorder using the best model"""
        try:
            # Check if models are loaded
            if not self.are_models_loaded():
                st.error("Models are not properly loaded. Please check if all model files exist.")
                return None, None
            
            # Define feature names in the correct order
            feature_names = ['HeartRate', 'SpO2', 'Glucose', 'Alpha', 'Beta', 'Gamma', 'Delta', 'Theta']
            
            # Prepare input data
            input_df = pd.DataFrame([input_data], columns=feature_names)
            
            # Scale the input
            input_scaled = self.scaler.transform(input_df)
            
            # Make prediction
            prediction = self.best_model.predict(input_scaled)[0]
            
            # Get prediction probabilities if available
            if hasattr(self.best_model, 'predict_proba'):
                probabilities = self.best_model.predict_proba(input_scaled)[0]
            else:
                probabilities = None
            
            # Convert prediction to disorder name
            disorder_name = self.label_encoder.inverse_transform([prediction])[0]
            
            return disorder_name, probabilities
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            return None, None

    def debug_model_status(self):
        """Debug method to show model loading status"""
        st.sidebar.markdown("### 🔧 Debug Info")
        st.sidebar.write(f"Models loaded: {self.models_loaded}")
        st.sidebar.write(f"Scaler loaded: {self.scaler is not None}")
        st.sidebar.write(f"Label encoder loaded: {self.label_encoder is not None}")
        st.sidebar.write(f"Best model loaded: {self.best_model is not None}")
        if self.model_info:
            st.sidebar.write(f"Model info: {self.model_info}")
        st.sidebar.write(f"All models ready: {self.are_models_loaded()}")
    
def main():
    """Main Streamlit application"""
    
    # Initialize predictor
    predictor = DiseasePredictor()
    
    # Header
    st.markdown('<h1 class="main-header">🧠 Mental Health Disorder Detection System</h1>', unsafe_allow_html=True)
    st.markdown("---")
      # Load models
    if not predictor.are_models_loaded():
        st.error("❌ Failed to load models. Please train the models first!")
        st.info("💡 Run `python quick_train.py` to train and save the best model.")
        
        # Show training button
        if st.button("🚀 Train Models Now"):
            with st.spinner("Training models... This may take a few minutes."):
                import subprocess
                result = subprocess.run(["python", "quick_train.py"], 
                                      capture_output=True, text=True, cwd="c:/Users/mohan/Downloads/SUmmerPabitra")
                if result.returncode == 0:
                    st.success("✅ Models trained successfully! Please refresh the page.")
                    st.experimental_rerun()
                else:
                    st.error(f"❌ Training failed: {result.stderr}")
        return
      # Sidebar for navigation
    st.sidebar.title("🎛️ Navigation")
    page = st.sidebar.selectbox("Choose a page:", ["🔮 Prediction", "📊 Data Analysis", "ℹ️ About"])
    
    # Add debug info if models are loaded
    if predictor.are_models_loaded():
        predictor.debug_model_status()
    
    if page == "🔮 Prediction":
        prediction_page(predictor)
    elif page == "📊 Data Analysis":
        data_analysis_page()
    elif page == "ℹ️ About":
        about_page()

def prediction_page(predictor):
    """Prediction page"""
    st.header("🔮 Disorder Prediction")
    st.markdown("Enter the patient's health metrics and brain wave data to predict potential mental health disorders.")
    
    # Display best model info
    st.subheader("🏆 Best Performing Algorithm")
    if predictor.model_info:
        best_model_name = predictor.model_info.get('best_model_name', 'Best Model')
        best_accuracy = predictor.model_info.get('best_accuracy', 0.0)
        st.success(f"Using: **{best_model_name}** (Accuracy: {best_accuracy:.4f})")
    else:
        st.info("Using: **Best Available Model**")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏥 Health Metrics")
        heart_rate = st.number_input("Heart Rate (BPM)", min_value=40, max_value=200, value=80, step=1)
        spo2 = st.number_input("SpO2 (%)", min_value=80, max_value=100, value=98, step=1)
        glucose = st.number_input("Glucose (mg/dL)", min_value=60, max_value=300, value=120, step=1)
        
    with col2:
        st.subheader("🧠 Brain Wave Data")
        alpha = st.number_input("Alpha Waves", min_value=0.0, max_value=30.0, value=10.0, step=0.1)
        beta = st.number_input("Beta Waves", min_value=0.0, max_value=40.0, value=15.0, step=0.1)
        gamma = st.number_input("Gamma Waves", min_value=0.0, max_value=60.0, value=30.0, step=0.1)
        delta = st.number_input("Delta Waves", min_value=0.0, max_value=15.0, value=3.0, step=0.1)
        theta = st.number_input("Theta Waves", min_value=0.0, max_value=20.0, value=6.0, step=0.1)
    
    # Prediction button
    if st.button("🔍 Predict Disorder", key="predict_btn"):
        # Prepare input data
        input_data = [heart_rate, spo2, glucose, alpha, beta, gamma, delta, theta]
        
        # Make prediction using best model
        with st.spinner("🔄 Analyzing data..."):
            disorder, probabilities = predictor.predict_disorder(input_data)
        
        if disorder:
            # Display prediction result
            st.markdown(f'<div class="prediction-result">🎯 Predicted Disorder: <strong>{disorder}</strong></div>', 
                       unsafe_allow_html=True)
            
            # Display probabilities if available
            if probabilities is not None:
                st.subheader("📊 Prediction Confidence")
                
                disorder_classes = predictor.label_encoder.classes_
                prob_df = pd.DataFrame({
                    'Disorder': disorder_classes,
                    'Probability': probabilities
                })
                prob_df = prob_df.sort_values('Probability', ascending=False)
                
                # Create probability chart
                fig = px.bar(prob_df, x='Disorder', y='Probability', 
                           title="Prediction Probabilities for All Disorders",
                           color='Probability', color_continuous_scale='viridis')
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
                
                # Show top 3 predictions
                st.subheader("🏆 Top 3 Predictions")
                for i, (_, row) in enumerate(prob_df.head(3).iterrows()):
                    confidence = row['Probability'] * 100
                    st.metric(f"#{i+1} {row['Disorder']}", f"{confidence:.1f}%")
            
            # Health status interpretation
            st.subheader("🩺 Health Status Interpretation")
            
            # Health metrics analysis
            col1, col2, col3 = st.columns(3)
            
            with col1:
                hr_status = "Normal" if 60 <= heart_rate <= 100 else "Abnormal"
                hr_color = "green" if hr_status == "Normal" else "red"
                st.markdown(f"**Heart Rate:** <span style='color:{hr_color}'>{hr_status}</span>", unsafe_allow_html=True)
                
            with col2:
                spo2_status = "Normal" if spo2 >= 95 else "Low"
                spo2_color = "green" if spo2_status == "Normal" else "orange"
                st.markdown(f"**SpO2:** <span style='color:{spo2_color}'>{spo2_status}</span>", unsafe_allow_html=True)
                
            with col3:
                glucose_status = "Normal" if 70 <= glucose <= 140 else "Abnormal"
                glucose_color = "green" if glucose_status == "Normal" else "red"
                st.markdown(f"**Glucose:** <span style='color:{glucose_color}'>{glucose_status}</span>", unsafe_allow_html=True)
            
            # Recommendations based on disorder
            st.subheader("💡 Recommendations")
            recommendations = get_disorder_recommendations(disorder)
            for rec in recommendations:
                st.info(f"• {rec}")

def data_analysis_page():
    """Data analysis and visualization page"""
    st.header("📈 Data Analysis & Insights")
    
    try:
        # Load the enhanced dataset
        df = pd.read_csv('c:/Users/mohan/Downloads/SUmmerPabitra/enhanced_dataset.csv')
        
        # Dataset overview
        st.subheader("📊 Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Samples", len(df))
        with col2:
            st.metric("Features", 8)  # Health + Brain wave features
        with col3:
            st.metric("Disorder Types", df['Disorder'].nunique())
        with col4:
            st.metric("Data Quality", f"{100 - (df.isnull().sum().sum() / df.size * 100):.1f}%")
        
        # Disorder distribution
        st.subheader("🧩 Disorder Distribution")
        disorder_counts = df['Disorder'].value_counts()
        
        fig_pie = px.pie(values=disorder_counts.values, names=disorder_counts.index,
                        title="Distribution of Mental Health Disorders")
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Feature correlations
        st.subheader("🔗 Feature Correlations")
        numeric_cols = ['HeartRate', 'SpO2', 'Glucose', 'Alpha', 'Beta', 'Gamma', 'Delta', 'Theta']
        correlation_matrix = df[numeric_cols].corr()
        
        fig_heatmap = px.imshow(correlation_matrix, 
                               title="Feature Correlation Heatmap",
                               color_continuous_scale='RdBu',
                               aspect='auto')
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Brain wave patterns by disorder
        st.subheader("🧠 Brain Wave Patterns by Disorder")
        brain_waves = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Theta']
        
        for wave in brain_waves:
            fig_brain = px.box(df, x='Disorder', y=wave, 
                              title=f"{wave} Wave Distribution by Disorder")
            fig_brain.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_brain, use_container_width=True)
        
        # Health metrics analysis
        st.subheader("🏥 Health Metrics Analysis")
        health_metrics = ['HeartRate', 'SpO2', 'Glucose']
        
        for metric in health_metrics:
            fig_health = px.box(df, x='Disorder', y=metric, 
                               title=f"{metric} Distribution by Disorder")
            fig_health.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_health, use_container_width=True)
        
        # Summary statistics
        st.subheader("📈 Summary Statistics")
        st.dataframe(df[numeric_cols].describe().round(2), use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")

def about_page():
    """About page with system information"""
    st.header("ℹ️ About the System")
    
    st.markdown("""
    ## 🧠 Mental Health Disorder Detection System
    
    This system uses the **best performing machine learning algorithm** to predict mental health disorders based on physiological and neurological data.
    
    ### 🎯 Supported Disorders
    - **Addictive Disorder**: Substance use and behavioral addictions
    - **Trauma**: Post-traumatic stress and trauma-related disorders  
    - **Mood**: Depression, bipolar disorder, and mood-related conditions
    - **Obsessive**: Obsessive-compulsive disorder and related conditions
    - **Schizophrenia**: Psychotic disorders and related conditions
    - **Anxiety Disorder**: Generalized anxiety, panic disorder, and related conditions
    
    ### 🏆 Algorithm Selection Process
    
    The system automatically trains multiple algorithms and selects the best performing one:
    
    1. **Deterministic Algorithms**: Decision Tree, Logistic Regression, SVM, Naive Bayes
    2. **Stochastic Algorithms**: Random Forest, Gradient Boosting, Neural Network
    3. **Best Model Selection**: Highest accuracy algorithm is automatically chosen
    
    ### 📊 Input Features
    
    #### Health Metrics
    - **Heart Rate**: Beats per minute (40-200 BPM)
    - **SpO2**: Blood oxygen saturation (80-100%)
    - **Glucose**: Blood glucose level (60-300 mg/dL)
    
    #### Brain Wave Data (EEG)
    - **Alpha Waves**: 8-12 Hz, relaxed awareness
    - **Beta Waves**: 13-30 Hz, active thinking
    - **Gamma Waves**: 30-100 Hz, cognitive processing
    - **Delta Waves**: 0.5-4 Hz, deep sleep
    - **Theta Waves**: 4-8 Hz, meditation and creativity
    
    ### 🎯 How It Works
    
    1. **Model Training**: Multiple algorithms are trained and compared
    2. **Best Model Selection**: Highest accuracy model is automatically selected
    3. **Data Input**: Enter health metrics and brain wave measurements
    4. **Prediction**: Best model provides disorder classification
    5. **Confidence Scores**: Shows probability for each disorder type
    6. **Recommendations**: Provides appropriate next steps
    
    ### ⚠️ Important Disclaimers
    
    - This system is for **educational and research purposes only**
    - **Not a substitute for professional medical diagnosis**
    - Always consult qualified healthcare professionals for medical advice
    - Results should be interpreted in conjunction with clinical assessment
    
    ---
    
    **Developed for educational purposes in machine learning and mental health applications.**
    """)

def get_disorder_recommendations(disorder):
    """Get recommendations based on predicted disorder"""
    recommendations = {
        'Addictive Disorder': [
            "Consider consultation with an addiction specialist",
            "Explore support groups and counseling services", 
            "Monitor substance use patterns",
            "Implement healthy coping mechanisms"
        ],
        'Trauma': [
            "Seek trauma-informed therapy (EMDR, CBT)",
            "Practice grounding and mindfulness techniques",
            "Consider support groups for trauma survivors",
            "Prioritize sleep hygiene and stress management"
        ],
        'Mood': [
            "Consult with a mental health professional",
            "Consider mood tracking and lifestyle modifications",
            "Explore therapy options (CBT, DBT)",
            "Monitor sleep patterns and exercise regularly"
        ],
        'Obsessive': [
            "Seek specialized OCD treatment (ERP therapy)",
            "Practice mindfulness and acceptance techniques",
            "Consider support groups for OCD",
            "Implement structured daily routines"
        ],
        'Schizophrenia': [
            "Immediate consultation with a psychiatrist",
            "Consider comprehensive psychiatric evaluation",
            "Ensure medication compliance if prescribed",
            "Family support and psychoeducation important"
        ],
        'Anxiety Disorder': [
            "Practice relaxation and breathing techniques",
            "Consider cognitive behavioral therapy (CBT)",
            "Regular exercise and stress management",
            "Limit caffeine and maintain regular sleep schedule"
        ]
    }
    
    return recommendations.get(disorder, ["Consult with a healthcare professional for personalized advice"])

if __name__ == "__main__":
    main()
