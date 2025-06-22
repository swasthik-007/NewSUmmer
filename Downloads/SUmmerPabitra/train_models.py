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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class ModelTrainerAndSaver:
    def __init__(self, data_path):
        """Initialize the model trainer"""
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.model_accuracies = {}
        
    def load_data(self):
        """Load the enhanced dataset"""
        print("Loading enhanced dataset...")
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset shape: {self.df.shape}")
        print("Columns:", self.df.columns.tolist())
        print("\nDisorder distribution:")
        print(self.df['Disorder'].value_counts())
        
    def prepare_features(self):
        """Prepare features and target variables"""
        print("\nPreparing features...")
        
        # Select feature columns (excluding SerialNo, Timestamp, and Disorder)
        feature_columns = ['HeartRate', 'SpO2', 'Glucose', 'Alpha', 'Beta', 'Gamma', 'Delta', 'Theta']
        
        self.X = self.df[feature_columns]
        self.y = self.label_encoder.fit_transform(self.df['Disorder'])
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        print(f"Number of classes: {len(self.label_encoder.classes_)}")
        print(f"Class labels: {self.label_encoder.classes_}")
        
    def train_deterministic_models(self):
        """Train and save deterministic models"""
        print("\n=== TRAINING DETERMINISTIC MODELS ===")
        
        deterministic_models = {
            'decision_tree': DecisionTreeClassifier(random_state=42, max_depth=10),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'svm': SVC(random_state=42, probability=True),
            'naive_bayes': GaussianNB()
        }
        
        for name, model in deterministic_models.items():
            print(f"\nTraining {name}...")
            
            # Train the model
            model.fit(self.X_train_scaled, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test_scaled)
            
            # Calculate accuracy
            accuracy = accuracy_score(self.y_test, y_pred)
            self.model_accuracies[name] = accuracy
            
            print(f"{name} Accuracy: {accuracy:.4f}")
            
            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5)
            print(f"{name} CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # Save the model
            self.models[name] = model
            
    def train_stochastic_models(self):
        """Train and save stochastic models"""
        print("\n=== TRAINING STOCHASTIC MODELS ===")
        
        stochastic_models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=15),
            'gradient_boosting': GradientBoostingClassifier(random_state=42, n_estimators=100),
            'neural_network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
        }
        
        for name, model in stochastic_models.items():
            print(f"\nTraining {name}...")
            
            # Train the model
            model.fit(self.X_train_scaled, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test_scaled)
            
            # Calculate accuracy
            accuracy = accuracy_score(self.y_test, y_pred)
            self.model_accuracies[name] = accuracy
            
            print(f"{name} Accuracy: {accuracy:.4f}")
            
            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5)
            print(f"{name} CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # Save the model
            self.models[name] = model
            
    def genetic_algorithm_optimization(self):
        """Simple genetic algorithm for feature selection and model optimization"""
        print("\n=== META-HEURISTIC: GENETIC ALGORITHM ===")
        
        def evaluate_individual(individual):
            """Evaluate fitness of feature subset"""
            features = [i for i, bit in enumerate(individual) if bit == 1]
            if len(features) == 0:
                return 0.0
            
            X_selected = self.X_train_scaled[:, features]
            X_test_selected = self.X_test_scaled[:, features]
            
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X_selected, self.y_train)
            y_pred = model.predict(X_test_selected)
            
            return accuracy_score(self.y_test, y_pred)
        
        n_features = self.X_train.shape[1]
        population_size = 20
        generations = 15
        
        # Initialize population
        population = []
        for _ in range(population_size):
            individual = np.random.randint(0, 2, n_features)
            # Ensure at least one feature is selected
            if sum(individual) == 0:
                individual[np.random.randint(0, n_features)] = 1
            population.append(individual)
        
        best_fitness = 0
        best_individual = None
        
        # Evolution
        for generation in range(generations):
            # Evaluate fitness
            fitness_scores = [evaluate_individual(ind) for ind in population]
            
            # Track best
            current_best_idx = np.argmax(fitness_scores)
            if fitness_scores[current_best_idx] > best_fitness:
                best_fitness = fitness_scores[current_best_idx]
                best_individual = population[current_best_idx].copy()
            
            # Selection and reproduction
            new_population = []
            for _ in range(population_size):
                # Tournament selection
                tournament_size = 3
                tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
                tournament_fitness = [fitness_scores[i] for i in tournament_indices]
                winner_idx = tournament_indices[np.argmax(tournament_fitness)]
                
                # Create offspring with mutation
                offspring = population[winner_idx].copy()
                mutation_rate = 0.1
                for i in range(len(offspring)):
                    if np.random.random() < mutation_rate:
                        offspring[i] = 1 - offspring[i]
                
                # Ensure at least one feature is selected
                if sum(offspring) == 0:
                    offspring[np.random.randint(0, n_features)] = 1
                    
                new_population.append(offspring)
            
            population = new_population
            
            if generation % 5 == 0:
                print(f"Generation {generation}: Best fitness = {best_fitness:.4f}")
        
        # Train final model with best features
        best_features = [i for i, bit in enumerate(best_individual) if bit == 1]
        X_train_ga = self.X_train_scaled[:, best_features]
        X_test_ga = self.X_test_scaled[:, best_features]
        
        ga_model = RandomForestClassifier(n_estimators=100, random_state=42)
        ga_model.fit(X_train_ga, self.y_train)
        y_pred_ga = ga_model.predict(X_test_ga)
        ga_accuracy = accuracy_score(self.y_test, y_pred_ga)
        
        print(f"Genetic Algorithm Best Accuracy: {ga_accuracy:.4f}")
        print(f"Selected features: {[self.X.columns[i] for i in best_features]}")
        
        # Save GA model and selected features
        self.models['genetic_algorithm'] = ga_model
        self.model_accuracies['genetic_algorithm'] = ga_accuracy
        
        # Save feature selection info
        ga_info = {
            'model': ga_model,
            'selected_features': best_features,
            'feature_names': [self.X.columns[i] for i in best_features]
        }
        
        return ga_info
        
    def save_all_models(self):
        """Save all trained models and preprocessing objects"""
        print("\n=== SAVING MODELS ===")
        
        # Save individual models
        for name, model in self.models.items():
            model_path = f'c:/Users/mohan/Downloads/SUmmerPabitra/models/{name}_model.pkl'
            joblib.dump(model, model_path)
            print(f"Saved {name} model to {model_path}")
        
        # Save preprocessing objects
        scaler_path = 'c:/Users/mohan/Downloads/SUmmerPabitra/models/scaler.pkl'
        joblib.dump(self.scaler, scaler_path)
        print(f"Saved scaler to {scaler_path}")
        
        encoder_path = 'c:/Users/mohan/Downloads/SUmmerPabitra/models/label_encoder.pkl'
        joblib.dump(self.label_encoder, encoder_path)
        print(f"Saved label encoder to {encoder_path}")
        
        # Save feature names
        feature_info = {
            'feature_names': self.X.columns.tolist(),
            'disorder_classes': self.label_encoder.classes_.tolist(),
            'model_accuracies': self.model_accuracies
        }
        
        info_path = 'c:/Users/mohan/Downloads/SUmmerPabitra/models/model_info.pkl'
        joblib.dump(feature_info, info_path)
        print(f"Saved model info to {info_path}")
        
        # Save best model (highest accuracy)
        best_model_name = max(self.model_accuracies, key=self.model_accuracies.get)
        best_model = self.models[best_model_name]
        best_model_path = 'c:/Users/mohan/Downloads/SUmmerPabitra/models/best_model.pkl'
        joblib.dump(best_model, best_model_path)
        print(f"Saved best model ({best_model_name}) to {best_model_path}")
        
        return best_model_name, self.model_accuracies[best_model_name]
        
    def generate_model_report(self):
        """Generate comprehensive model performance report"""
        print("\n" + "="*60)
        print("MODEL TRAINING REPORT")
        print("="*60)
        
        print("\nModel Performance Summary:")
        print("-" * 40)
        for model_name, accuracy in sorted(self.model_accuracies.items(), key=lambda x: x[1], reverse=True):
            print(f"{model_name:.<25} {accuracy:.4f}")
        
        best_model = max(self.model_accuracies, key=self.model_accuracies.get)
        print(f"\nBest Model: {best_model} ({self.model_accuracies[best_model]:.4f})")
        
        # Algorithm category analysis
        deterministic = ['decision_tree', 'logistic_regression', 'svm', 'naive_bayes']
        stochastic = ['random_forest', 'gradient_boosting', 'neural_network']
        meta_heuristic = ['genetic_algorithm']
        
        print(f"\nCategory Analysis:")
        print("-" * 20)
        
        det_models = {k: v for k, v in self.model_accuracies.items() if k in deterministic}
        if det_models:
            best_det = max(det_models, key=det_models.get)
            print(f"Best Deterministic: {best_det} ({det_models[best_det]:.4f})")
        
        sto_models = {k: v for k, v in self.model_accuracies.items() if k in stochastic}
        if sto_models:
            best_sto = max(sto_models, key=sto_models.get)
            print(f"Best Stochastic: {best_sto} ({sto_models[best_sto]:.4f})")
        
        meta_models = {k: v for k, v in self.model_accuracies.items() if k in meta_heuristic}
        if meta_models:
            best_meta = max(meta_models, key=meta_models.get)
            print(f"Best Meta-heuristic: {best_meta} ({meta_models[best_meta]:.4f})")
        
        print(f"\nDisorder Classes:")
        print("-" * 20)
        for i, disorder in enumerate(self.label_encoder.classes_):
            count = sum(1 for label in self.y if label == i)
            print(f"  {disorder}: {count} samples")
            
    def run_complete_training(self):
        """Run complete model training pipeline"""
        print("Starting Complete Model Training Pipeline...")
        
        # Create models directory
        import os
        models_dir = 'c:/Users/mohan/Downloads/SUmmerPabitra/models'
        os.makedirs(models_dir, exist_ok=True)
        
        # Load and prepare data
        self.load_data()
        self.prepare_features()
        
        # Train all model types
        self.train_deterministic_models()
        self.train_stochastic_models()
        self.genetic_algorithm_optimization()
        
        # Save models
        best_model_name, best_accuracy = self.save_all_models()
        
        # Generate report
        self.generate_model_report()
        
        print(f"\n✅ Training completed successfully!")
        print(f"📁 Models saved in: {models_dir}")
        print(f"🏆 Best model: {best_model_name} (Accuracy: {best_accuracy:.4f})")
        
        return self.models, self.model_accuracies

def main():
    """Main function to train and save all models"""
    trainer = ModelTrainerAndSaver('c:/Users/mohan/Downloads/SUmmerPabitra/enhanced_dataset.csv')
    models, accuracies = trainer.run_complete_training()
    return models, accuracies

if __name__ == "__main__":
    models, accuracies = main()
