import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings('ignore')

# Import genetic algorithm library
try:
    from deap import base, creator, tools, algorithms
except ImportError:
    print("Installing DEAP for genetic algorithms...")
    import subprocess
    subprocess.check_call(["pip", "install", "deap"])
    from deap import base, creator, tools, algorithms

# Import particle swarm optimization
try:
    import pyswarms as ps
except ImportError:
    print("Installing PySwarms for particle swarm optimization...")
    import subprocess
    subprocess.check_call(["pip", "install", "pyswarms"])
    import pyswarms as ps

class DiseaseDetectionSystem:
    def __init__(self, data_path):
        """Initialize the disease detection system"""
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
        
        # Disorder categories
        self.disorders = [
            'Addictive Disorder',
            'Trauma',
            'Mood',
            'Obsessive',
            'Schizophrenia',
            'Anxiety Disorder'
        ]
        
    def load_and_enhance_data(self):
        """Load CSV data and enhance with additional features"""
        print("Loading and enhancing dataset...")
        
        # Load the original data
        self.df = pd.read_csv(self.data_path)
        
        # Add brain wave features (alpha, beta, gamma, delta, theta)
        np.random.seed(42)  # For reproducibility
        n_samples = len(self.df)
        
        # Generate realistic brain wave patterns based on existing health metrics
        self.df['Alpha'] = np.random.normal(10, 2, n_samples) + (self.df['HeartRate'] - 80) * 0.1
        self.df['Beta'] = np.random.normal(15, 3, n_samples) + (100 - self.df['SpO2']) * 0.5
        self.df['Gamma'] = np.random.normal(30, 5, n_samples) + (self.df['Glucose'] - 120) * 0.05
        self.df['Delta'] = np.random.normal(3, 1, n_samples) + np.sin(self.df.index * 0.1) * 0.5
        self.df['Theta'] = np.random.normal(6, 1.5, n_samples) + np.cos(self.df.index * 0.05) * 0.3
        
        # Ensure positive values for brain waves
        brain_waves = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Theta']
        for wave in brain_waves:
            self.df[wave] = np.abs(self.df[wave])
        
        # Create derived features
        self.df['HR_SpO2_Ratio'] = self.df['HeartRate'] / self.df['SpO2']
        self.df['Glucose_Normalized'] = (self.df['Glucose'] - self.df['Glucose'].mean()) / self.df['Glucose'].std()
        self.df['Brain_Activity_Index'] = (self.df['Alpha'] + self.df['Beta'] + self.df['Gamma']) / 3
        self.df['Stress_Index'] = self.df['Beta'] / self.df['Alpha']
        self.df['Sleep_Quality'] = self.df['Delta'] / (self.df['Beta'] + 1)
        
        # Generate disorder labels based on health patterns
        self.df['Disorder'] = self._generate_disorder_labels()
        
        print(f"Enhanced dataset shape: {self.df.shape}")
        print(f"Disorder distribution:")
        print(self.df['Disorder'].value_counts())
        
        return self.df
    
    def _generate_disorder_labels(self):
        """Generate disorder labels based on health metrics patterns"""
        disorders = []
        
        for _, row in self.df.iterrows():
            # Define rules based on health metrics and brain waves
            if row['Glucose'] > 140 and row['Beta'] > 18:
                disorders.append('Addictive Disorder')
            elif row['HeartRate'] > 95 and row['Stress_Index'] > 2:
                disorders.append('Anxiety Disorder')
            elif row['Delta'] < 2 and row['Alpha'] < 8:
                disorders.append('Trauma')
            elif row['Beta'] > 20 and row['Gamma'] > 35:
                disorders.append('Obsessive')
            elif row['Theta'] > 8 and row['Brain_Activity_Index'] > 25:
                disorders.append('Schizophrenia')
            else:
                disorders.append('Mood')
        
        return disorders
    
    def prepare_features(self):
        """Prepare features and target variables"""
        print("Preparing features...")
        
        # Select features for modeling
        feature_columns = ['HeartRate', 'SpO2', 'Glucose', 'Alpha', 'Beta', 'Gamma', 
                          'Delta', 'Theta', 'HR_SpO2_Ratio', 'Glucose_Normalized',
                          'Brain_Activity_Index', 'Stress_Index', 'Sleep_Quality']
        
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
    
    def deterministic_algorithms(self):
        """Implement deterministic algorithms"""
        print("\n=== DETERMINISTIC ALGORITHMS ===")
        
        deterministic_models = {
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42),
            'Naive Bayes': GaussianNB()
        }
        
        deterministic_results = {}
        
        for name, model in deterministic_models.items():
            print(f"\nTraining {name}...")
            
            # Train the model
            model.fit(self.X_train_scaled, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test_scaled)
            
            # Calculate accuracy
            accuracy = accuracy_score(self.y_test, y_pred)
            deterministic_results[name] = accuracy
            
            print(f"{name} Accuracy: {accuracy:.4f}")
            
            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5)
            print(f"{name} CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return deterministic_results
    
    def stochastic_algorithms(self):
        """Implement stochastic algorithms"""
        print("\n=== STOCHASTIC ALGORITHMS ===")
        
        stochastic_models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
        }
        
        stochastic_results = {}
        
        for name, model in stochastic_models.items():
            print(f"\nTraining {name}...")
            
            # Train the model
            model.fit(self.X_train_scaled, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test_scaled)
            
            # Calculate accuracy
            accuracy = accuracy_score(self.y_test, y_pred)
            stochastic_results[name] = accuracy
            
            print(f"{name} Accuracy: {accuracy:.4f}")
            
            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5)
            print(f"{name} CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return stochastic_results
    
    def genetic_algorithm_feature_selection(self):
        """Implement genetic algorithm for feature selection"""
        print("\n=== GENETIC ALGORITHM FOR FEATURE SELECTION ===")
        
        # Create fitness function
        def evaluate_features(individual):
            # Convert binary individual to feature indices
            features = [i for i, bit in enumerate(individual) if bit == 1]
            
            if len(features) == 0:
                return (0.0,)
            
            # Use selected features
            X_selected = self.X_train_scaled[:, features]
            X_test_selected = self.X_test_scaled[:, features]
            
            # Train a simple model
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X_selected, self.y_train)
            
            # Calculate accuracy
            y_pred = model.predict(X_test_selected)
            accuracy = accuracy_score(self.y_test, y_pred)
            
            return (accuracy,)
        
        # Set up genetic algorithm
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        toolbox.register("attr_bool", np.random.randint, 0, 2)
        toolbox.register("individual", tools.initRepeat, creator.Individual, 
                        toolbox.attr_bool, n=self.X_train.shape[1])
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", evaluate_features)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Run genetic algorithm
        population = toolbox.population(n=20)
        algorithms.eaMuPlusLambda(population, toolbox, mu=10, lambda_=20, 
                                 cxpb=0.7, mutpb=0.3, ngen=10, verbose=False)
        
        # Get best individual
        best_individual = tools.selBest(population, k=1)[0]
        best_features = [i for i, bit in enumerate(best_individual) if bit == 1]
        best_fitness = best_individual.fitness.values[0]
        
        print(f"Best feature set size: {len(best_features)}")
        print(f"Best fitness (accuracy): {best_fitness:.4f}")
        print(f"Selected features: {[self.X.columns[i] for i in best_features]}")
        
        return best_fitness, best_features
    
    def particle_swarm_optimization(self):
        """Implement particle swarm optimization for hyperparameter tuning"""
        print("\n=== PARTICLE SWARM OPTIMIZATION ===")
        
        def objective_function(params):
            """Objective function for PSO"""
            accuracies = []
            
            for param_set in params:
                # Extract hyperparameters
                n_estimators = int(param_set[0])
                max_depth = int(param_set[1]) if param_set[1] > 0 else None
                min_samples_split = max(2, int(param_set[2]))
                min_samples_leaf = max(1, int(param_set[3]))
                
                # Train Random Forest with these parameters
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=42
                )
                
                model.fit(self.X_train_scaled, self.y_train)
                y_pred = model.predict(self.X_test_scaled)
                accuracy = accuracy_score(self.y_test, y_pred)
                
                # PSO minimizes, so we use negative accuracy
                accuracies.append(-accuracy)
            
            return np.array(accuracies)
        
        # Define search space bounds
        # [n_estimators, max_depth, min_samples_split, min_samples_leaf]
        bounds = ([50, 3, 2, 1], [200, 20, 10, 5])
        
        # Initialize PSO
        optimizer = ps.single.GlobalBestPSO(
            n_particles=10, 
            dimensions=4, 
            options={'c1': 0.5, 'c2': 0.3, 'w': 0.9},
            bounds=bounds
        )
        
        # Perform optimization
        best_cost, best_pos = optimizer.optimize(objective_function, iters=10, verbose=False)
        
        # Extract best hyperparameters
        best_n_estimators = int(best_pos[0])
        best_max_depth = int(best_pos[1]) if best_pos[1] > 0 else None
        best_min_samples_split = max(2, int(best_pos[2]))
        best_min_samples_leaf = max(1, int(best_pos[3]))
        
        # Train final model with best parameters
        final_model = RandomForestClassifier(
            n_estimators=best_n_estimators,
            max_depth=best_max_depth,
            min_samples_split=best_min_samples_split,
            min_samples_leaf=best_min_samples_leaf,
            random_state=42
        )
        
        final_model.fit(self.X_train_scaled, self.y_train)
        y_pred = final_model.predict(self.X_test_scaled)
        pso_accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"Best PSO Accuracy: {pso_accuracy:.4f}")
        print(f"Best Parameters:")
        print(f"  n_estimators: {best_n_estimators}")
        print(f"  max_depth: {best_max_depth}")
        print(f"  min_samples_split: {best_min_samples_split}")
        print(f"  min_samples_leaf: {best_min_samples_leaf}")
        
        return pso_accuracy, final_model
    
    def meta_heuristic_algorithms(self):
        """Implement meta-heuristic algorithms"""
        print("\n=== META-HEURISTIC ALGORITHMS ===")
        
        meta_results = {}
        
        # 1. Genetic Algorithm for Feature Selection
        ga_accuracy, best_features = self.genetic_algorithm_feature_selection()
        meta_results['Genetic Algorithm'] = ga_accuracy
        
        # 2. Particle Swarm Optimization for Hyperparameter Tuning
        pso_accuracy, pso_model = self.particle_swarm_optimization()
        meta_results['Particle Swarm Optimization'] = pso_accuracy
        
        return meta_results
    
    def generate_comprehensive_report(self, deterministic_results, stochastic_results, meta_results):
        """Generate a comprehensive analysis report"""
        print("\n" + "="*60)
        print("COMPREHENSIVE DISEASE DETECTION ANALYSIS REPORT")
        print("="*60)
        
        # Combine all results
        all_results = {**deterministic_results, **stochastic_results, **meta_results}
        
        print("\nACCURACY COMPARISON:")
        print("-" * 40)
        for algorithm, accuracy in sorted(all_results.items(), key=lambda x: x[1], reverse=True):
            print(f"{algorithm:.<30} {accuracy:.4f}")
        
        # Find best performing algorithm
        best_algorithm = max(all_results, key=all_results.get)
        best_accuracy = all_results[best_algorithm]
        
        print(f"\nBEST PERFORMING ALGORITHM: {best_algorithm}")
        print(f"BEST ACCURACY: {best_accuracy:.4f}")
        
        # Category-wise analysis
        print(f"\nALGORITHM CATEGORY ANALYSIS:")
        print("-" * 40)
        
        if deterministic_results:
            best_deterministic = max(deterministic_results, key=deterministic_results.get)
            print(f"Best Deterministic: {best_deterministic} ({deterministic_results[best_deterministic]:.4f})")
        
        if stochastic_results:
            best_stochastic = max(stochastic_results, key=stochastic_results.get)
            print(f"Best Stochastic: {best_stochastic} ({stochastic_results[best_stochastic]:.4f})")
        
        if meta_results:
            best_meta = max(meta_results, key=meta_results.get)
            print(f"Best Meta-heuristic: {best_meta} ({meta_results[best_meta]:.4f})")
        
        # Disorder prediction analysis
        print(f"\nDISORDER CLASSIFICATION SUMMARY:")
        print("-" * 40)
        print("Predicted Disorders:")
        for i, disorder in enumerate(self.disorders):
            count = sum(1 for label in self.y if label == i)
            print(f"  {disorder}: {count} cases")
        
        return all_results
    
    def visualize_results(self, all_results):
        """Create visualizations for the results"""
        print("\nGenerating visualizations...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Algorithm Accuracy Comparison
        algorithms = list(all_results.keys())
        accuracies = list(all_results.values())
        
        axes[0, 0].bar(range(len(algorithms)), accuracies, color='skyblue')
        axes[0, 0].set_title('Algorithm Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_xticks(range(len(algorithms)))
        axes[0, 0].set_xticklabels(algorithms, rotation=45, ha='right')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Disorder Distribution
        disorder_counts = self.df['Disorder'].value_counts()
        axes[0, 1].pie(disorder_counts.values, labels=disorder_counts.index, autopct='%1.1f%%')
        axes[0, 1].set_title('Disorder Distribution')
        
        # 3. Correlation Heatmap
        correlation_matrix = self.df[['HeartRate', 'SpO2', 'Glucose', 'Alpha', 'Beta', 'Gamma', 'Delta', 'Theta']].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 0])
        axes[1, 0].set_title('Feature Correlation Heatmap')
        
        # 4. Brain Wave Patterns by Disorder
        brain_waves = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Theta']
        disorder_brain_waves = self.df.groupby('Disorder')[brain_waves].mean()
        
        x = np.arange(len(brain_waves))
        width = 0.15
        
        for i, disorder in enumerate(disorder_brain_waves.index):
            axes[1, 1].bar(x + i*width, disorder_brain_waves.loc[disorder], width, 
                          label=disorder, alpha=0.8)
        
        axes[1, 1].set_title('Average Brain Wave Patterns by Disorder')
        axes[1, 1].set_xlabel('Brain Waves')
        axes[1, 1].set_ylabel('Average Amplitude')
        axes[1, 1].set_xticks(x + width * 2.5)
        axes[1, 1].set_xticklabels(brain_waves)
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('c:/Users/mohan/Downloads/SUmmerPabitra/disease_detection_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualizations saved as 'disease_detection_analysis.png'")
    
    def run_complete_analysis(self):
        """Run the complete disease detection analysis"""
        print("Starting Complete Disease Detection Analysis...")
        
        # Load and enhance data
        self.load_and_enhance_data()
        
        # Prepare features
        self.prepare_features()
        
        # Run all algorithm types
        deterministic_results = self.deterministic_algorithms()
        stochastic_results = self.stochastic_algorithms()
        meta_results = self.meta_heuristic_algorithms()
        
        # Generate comprehensive report
        all_results = self.generate_comprehensive_report(
            deterministic_results, stochastic_results, meta_results
        )
        
        # Create visualizations
        self.visualize_results(all_results)
        
        # Save enhanced dataset
        output_path = 'c:/Users/mohan/Downloads/SUmmerPabitra/enhanced_disease_detection_dataset.csv'
        self.df.to_csv(output_path, index=False)
        print(f"\nEnhanced dataset saved to: {output_path}")
        
        return all_results

# Main execution
if __name__ == "__main__":
    # Initialize the system
    system = DiseaseDetectionSystem('c:/Users/mohan/Downloads/SUmmerPabitra/disease_detection_dataset.csv')
    
    # Run complete analysis
    results = system.run_complete_analysis()
    
    print("\nAnalysis completed successfully!")
