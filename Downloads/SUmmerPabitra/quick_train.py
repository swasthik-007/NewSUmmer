import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def train_and_find_best_model():
    """Train all models and find the best one"""
    print("🚀 Training Disease Detection Models...")
    
    # Load data
    df = pd.read_csv('c:/Users/mohan/Downloads/SUmmerPabitra/enhanced_dataset.csv')
    print(f"Dataset loaded: {df.shape}")
    
    # Prepare features
    feature_columns = ['HeartRate', 'SpO2', 'Glucose', 'Alpha', 'Beta', 'Gamma', 'Delta', 'Theta']
    X = df[feature_columns]
    
    # Encode target
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['Disorder'])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(random_state=42, probability=True),
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
    }
    
    # Train and evaluate models
    results = {}
    trained_models = {}
    
    print("\n📊 Model Training Results:")
    print("-" * 50)
    
    for name, model in models.items():
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        
        results[name] = {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        trained_models[name] = model
        
        print(f"{name:.<20} {accuracy:.4f} (CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f})")
    
    # Find best model
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    best_model = trained_models[best_model_name]
    best_accuracy = results[best_model_name]['accuracy']
    
    print(f"\n🏆 Best Model: {best_model_name} (Accuracy: {best_accuracy:.4f})")
    
    # Create models directory
    import os
    models_dir = 'c:/Users/mohan/Downloads/SUmmerPabitra/models'
    os.makedirs(models_dir, exist_ok=True)
    
    # Save best model and preprocessing objects
    joblib.dump(best_model, f'{models_dir}/best_model.pkl')
    joblib.dump(scaler, f'{models_dir}/scaler.pkl')
    joblib.dump(label_encoder, f'{models_dir}/label_encoder.pkl')
    
    # Save model info
    model_info = {
        'best_model_name': best_model_name,
        'best_accuracy': best_accuracy,
        'feature_names': feature_columns,
        'disorder_classes': label_encoder.classes_.tolist(),
        'all_results': results
    }
    
    joblib.dump(model_info, f'{models_dir}/model_info.pkl')
    
    print(f"✅ Best model saved to {models_dir}/")
    
    return best_model_name, best_accuracy, results

if __name__ == "__main__":
    best_model_name, best_accuracy, results = train_and_find_best_model()
