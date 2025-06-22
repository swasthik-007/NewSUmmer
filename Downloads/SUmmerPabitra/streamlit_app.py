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
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
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

class DiseasePredictor:
    def __init__(self):
        """Initialize the disease predictor"""
        self.models_dir = 'c:/Users/mohan/Downloads/SUmmerPabitra/models'
        self.models = {}
        self.scaler = None
        self.label_encoder = None
        self.model_info = None
        self.feature_names = []
        self.disorder_classes = []
        
    @st.cache_resource
    def load_models(_self):
        """Load all saved models and preprocessing objects"""
        try:
            # Load model info
            info_path = os.path.join(_self.models_dir, 'model_info.pkl')
            _self.model_info = joblib.load(info_path)
            _self.feature_names = _self.model_info['feature_names']
            _self.disorder_classes = _self.model_info['disorder_classes']
            
            # Load preprocessing objects
            scaler_path = os.path.join(_self.models_dir, 'scaler.pkl')
            _self.scaler = joblib.load(scaler_path)
            
            encoder_path = os.path.join(_self.models_dir, 'label_encoder.pkl')
            _self.label_encoder = joblib.load(encoder_path)
            
            # Load all models
            model_files = [
                'decision_tree_model.pkl',
                'logistic_regression_model.pkl',
                'svm_model.pkl',
                'naive_bayes_model.pkl',
                'random_forest_model.pkl',
                'gradient_boosting_model.pkl',
                'neural_network_model.pkl',
                'genetic_algorithm_model.pkl'
            ]
            
            for model_file in model_files:
                model_path = os.path.join(_self.models_dir, model_file)
                if os.path.exists(model_path):
                    model_name = model_file.replace('_model.pkl', '')
                    _self.models[model_name] = joblib.load(model_path)
              return True
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            return False
    
    def predict_disorder(self, input_data):
        """Predict disorder using the best model"""
        try:
            # Define feature names in the correct order
            feature_names = ['HeartRate', 'SpO2', 'Glucose', 'Alpha', 'Beta', 'Gamma', 'Delta', 'Theta']
            
            # Prepare input data
            input_df = pd.DataFrame([input_data], columns=feature_names)
            
            # Scale the input
            input_scaled = self.scaler.transform(input_df)
            
            # Load best model
            best_model_path = os.path.join(self.models_dir, 'best_model.pkl')
            if os.path.exists(best_model_path):
                model = joblib.load(best_model_path)
            else:
                # Fallback to random forest if best model not found
                model = self.models.get('random_forest', None)
                if model is None:
                    raise Exception("No models available")
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            
            # Get prediction probabilities if available
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(input_scaled)[0]
            else:
                probabilities = None
            
            # Convert prediction to disorder name
            disorder_name = self.label_encoder.inverse_transform([prediction])[0]
            
            return disorder_name, probabilities
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.error(f"Input data shape: {len(input_data)}")
            st.error(f"Expected features: {feature_names}")
            return None, None
    
    def get_model_accuracies(self):
        """Get model accuracies from saved info"""
        if self.model_info and 'model_accuracies' in self.model_info:
            return self.model_info['model_accuracies']
        return {}

def main():
    """Main Streamlit application"""
    
    # Initialize predictor
    predictor = DiseasePredictor()
    
    # Header
    st.markdown('<h1 class="main-header">🧠 Mental Health Disorder Detection System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load models
    if not predictor.load_models():
        st.error("❌ Failed to load models. Please make sure you have run the training script first!")
        st.info("💡 Run `python train_models.py` to train and save the models.")
        return
    
    # Sidebar for navigation
    st.sidebar.title("🎛️ Navigation")
    page = st.sidebar.selectbox("Choose a page:", ["🔮 Prediction", "📊 Model Performance", "📈 Data Analysis", "ℹ️ About"])
    
    if page == "🔮 Prediction":
        prediction_page(predictor)
    elif page == "📊 Model Performance":
        model_performance_page(predictor)
    elif page == "📈 Data Analysis":
        data_analysis_page()
    elif page == "ℹ️ About":
        about_page()

def prediction_page(predictor):
    """Prediction page"""
    st.header("🔮 Disorder Prediction")
    st.markdown("Enter the patient's health metrics and brain wave data to predict potential mental health disorders.")
    
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
      # Display best model info
    st.subheader("🏆 Best Performing Algorithm")
    if predictor.model_info:
        best_model_name = predictor.model_info.get('best_model_name', 'Random Forest')
        best_accuracy = predictor.model_info.get('best_accuracy', 0.0)
        st.success(f"Using: **{best_model_name}** (Accuracy: {best_accuracy:.4f})")
    else:
        st.info("Using: **Best Available Model**")
    
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
                
                prob_df = pd.DataFrame({
                    'Disorder': predictor.disorder_classes,
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

def model_performance_page(predictor):
    """Model performance comparison page"""
    st.header("📊 Model Performance Analysis")
    
    accuracies = predictor.get_model_accuracies()
    
    if accuracies:
        # Create performance comparison chart
        model_names = list(accuracies.keys())
        accuracy_values = list(accuracies.values())
        
        # Map model names to categories
        categories = []
        for model in model_names:
            if model in ['decision_tree', 'logistic_regression', 'svm', 'naive_bayes']:
                categories.append('Deterministic')
            elif model in ['random_forest', 'gradient_boosting', 'neural_network']:
                categories.append('Stochastic')
            else:
                categories.append('Meta-heuristic')
        
        # Create DataFrame for plotting
        perf_df = pd.DataFrame({
            'Model': model_names,
            'Accuracy': accuracy_values,
            'Category': categories
        })
        
        # Sort by accuracy
        perf_df = perf_df.sort_values('Accuracy', ascending=True)
        
        # Create horizontal bar chart
        fig = px.bar(perf_df, x='Accuracy', y='Model', color='Category',
                    title="Model Performance Comparison",
                    orientation='h',
                    color_discrete_map={
                        'Deterministic': '#FF6B6B',
                        'Stochastic': '#4ECDC4', 
                        'Meta-heuristic': '#45B7D1'
                    })
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance metrics table
        st.subheader("📋 Detailed Performance Metrics")
        st.dataframe(perf_df.sort_values('Accuracy', ascending=False), use_container_width=True)
        
        # Best model highlight
        best_model = perf_df.loc[perf_df['Accuracy'].idxmax()]
        st.success(f"🏆 Best Performing Model: **{best_model['Model']}** ({best_model['Category']}) with **{best_model['Accuracy']:.4f}** accuracy")
        
        # Algorithm category analysis
        st.subheader("🔍 Algorithm Category Analysis")
        category_performance = perf_df.groupby('Category')['Accuracy'].agg(['mean', 'max', 'min']).round(4)
        st.dataframe(category_performance, use_container_width=True)
        
    else:
        st.warning("⚠️ No model performance data available.")

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
            st.metric("Features", len(df.columns) - 3)  # Excluding SerialNo, Timestamp, Disorder
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
        
        # Create subplot for each brain wave
        fig_brain = make_subplots(rows=2, cols=3, 
                                 subplot_titles=brain_waves,
                                 specs=[[{"type": "box"}, {"type": "box"}, {"type": "box"}],
                                       [{"type": "box"}, {"type": "box"}, {"type": "xy"}]])
        
        for i, wave in enumerate(brain_waves):
            row = (i // 3) + 1
            col = (i % 3) + 1
            
            for disorder in df['Disorder'].unique():
                disorder_data = df[df['Disorder'] == disorder][wave]
                fig_brain.add_box(y=disorder_data, name=f"{disorder}",
                                 row=row, col=col, showlegend=(i==0))
        
        fig_brain.update_layout(height=600, title="Brain Wave Patterns by Disorder Type")
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
    
    This system uses advanced machine learning algorithms to predict mental health disorders based on physiological and neurological data.
    
    ### 🎯 Supported Disorders
    - **Addictive Disorder**: Substance use and behavioral addictions
    - **Trauma**: Post-traumatic stress and trauma-related disorders  
    - **Mood**: Depression, bipolar disorder, and mood-related conditions
    - **Obsessive**: Obsessive-compulsive disorder and related conditions
    - **Schizophrenia**: Psychotic disorders and related conditions
    - **Anxiety Disorder**: Generalized anxiety, panic disorder, and related conditions
    
    ### 🔬 Algorithm Types Implemented
    
    #### 1. Deterministic Algorithms
    - **Decision Tree**: Rule-based classification
    - **Logistic Regression**: Linear probability modeling
    - **Support Vector Machine (SVM)**: Margin-based classification
    - **Naive Bayes**: Probabilistic classification
    
    #### 2. Stochastic Algorithms  
    - **Random Forest**: Ensemble of decision trees
    - **Gradient Boosting**: Sequential improvement modeling
    - **Neural Network**: Multi-layer perceptron
    
    #### 3. Meta-heuristic Algorithms
    - **Genetic Algorithm**: Evolutionary feature selection and optimization
    
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
    
    ### 🎯 How Predictions Work
    
    1. **Data Collection**: Input physiological and brain wave measurements
    2. **Preprocessing**: Normalize and scale input features
    3. **Model Prediction**: Apply selected algorithm for classification
    4. **Result Interpretation**: Provide disorder prediction with confidence scores
    5. **Recommendations**: Suggest appropriate next steps based on results
    
    ### ⚠️ Important Disclaimers
    
    - This system is for **educational and research purposes only**
    - **Not a substitute for professional medical diagnosis**
    - Always consult qualified healthcare professionals for medical advice
    - Results should be interpreted in conjunction with clinical assessment
    
    ### 🔧 Technical Details
    
    - **Framework**: Streamlit for web interface
    - **ML Library**: Scikit-learn for machine learning
    - **Visualization**: Plotly for interactive charts
    - **Data Processing**: Pandas and NumPy
    - **Model Persistence**: Joblib for model storage
    
    ### 📈 Performance Metrics
    
    The system evaluates models using:
    - **Accuracy**: Overall correct predictions
    - **Cross-validation**: Robust performance estimation
    - **Confidence Scores**: Prediction reliability
    
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
