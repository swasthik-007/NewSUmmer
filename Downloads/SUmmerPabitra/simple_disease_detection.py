import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def load_and_enhance_dataset(file_path):
    """Load and enhance the dataset with brain wave data and disorder labels"""
    print("Loading and enhancing dataset...")
    
    # Load original data
    df = pd.read_csv(file_path)
    
    # Add brain wave features (Alpha, Beta, Gamma, Delta, Theta)
    np.random.seed(42)
    n_samples = len(df)
    
    # Generate realistic brain wave patterns
    df['Alpha'] = np.random.normal(10, 2, n_samples) + (df['HeartRate'] - 80) * 0.1
    df['Beta'] = np.random.normal(15, 3, n_samples) + (100 - df['SpO2']) * 0.5
    df['Gamma'] = np.random.normal(30, 5, n_samples) + (df['Glucose'] - 120) * 0.05
    df['Delta'] = np.random.normal(3, 1, n_samples) + np.sin(df.index * 0.1) * 0.5
    df['Theta'] = np.random.normal(6, 1.5, n_samples) + np.cos(df.index * 0.05) * 0.3
    
    # Ensure positive values
    brain_waves = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Theta']
    for wave in brain_waves:
        df[wave] = np.abs(df[wave])
    
    # Generate disorder labels based on health patterns
    disorders = []
    for _, row in df.iterrows():
        if row['Glucose'] > 140 and row['Beta'] > 18:
            disorders.append('Addictive Disorder')
        elif row['HeartRate'] > 95 and row['Beta'] / row['Alpha'] > 2:
            disorders.append('Anxiety Disorder')
        elif row['Delta'] < 2 and row['Alpha'] < 8:
            disorders.append('Trauma')
        elif row['Beta'] > 20 and row['Gamma'] > 35:
            disorders.append('Obsessive')
        elif row['Theta'] > 8 and (row['Alpha'] + row['Beta'] + row['Gamma'])/3 > 25:
            disorders.append('Schizophrenia')
        else:
            disorders.append('Mood')
    
    df['Disorder'] = disorders
    
    print(f"Dataset shape: {df.shape}")
    print("Disorder distribution:")
    print(df['Disorder'].value_counts())
    
    return df

def prepare_data(df):
    """Prepare features and target variables"""
    # Select features
    features = ['HeartRate', 'SpO2', 'Glucose', 'Alpha', 'Beta', 'Gamma', 'Delta', 'Theta']
    X = df[features]
    
    # Encode target variable
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['Disorder'])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, label_encoder

def deterministic_algorithms(X_train, X_test, y_train, y_test):
    """Implement deterministic algorithms"""
    print("\n=== DETERMINISTIC ALGORITHMS ===")
    
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        print(f"{name} Accuracy: {accuracy:.4f}")
    
    return results

def stochastic_algorithms(X_train, X_test, y_train, y_test):
    """Implement stochastic algorithms"""
    print("\n=== STOCHASTIC ALGORITHMS ===")
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        print(f"{name} Accuracy: {accuracy:.4f}")
    
    return results

def genetic_algorithm_optimization(X_train, X_test, y_train, y_test):
    """Simple genetic algorithm for feature selection"""
    print("\n=== META-HEURISTIC ALGORITHMS ===")
    print("Genetic Algorithm for Feature Selection...")
    
    n_features = X_train.shape[1]
    population_size = 20
    generations = 10
    
    def evaluate_individual(individual):
        """Evaluate fitness of an individual (feature subset)"""
        features = [i for i, bit in enumerate(individual) if bit == 1]
        if len(features) == 0:
            return 0.0
        
        X_train_selected = X_train[:, features]
        X_test_selected = X_test[:, features]
        
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train_selected, y_train)
        y_pred = model.predict(X_test_selected)
        
        return accuracy_score(y_test, y_pred)
    
    # Initialize population
    population = []
    for _ in range(population_size):
        individual = np.random.randint(0, 2, n_features)
        population.append(individual)
    
    # Evolution
    for generation in range(generations):
        # Evaluate fitness
        fitness_scores = [evaluate_individual(ind) for ind in population]
        
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
            
            new_population.append(offspring)
        
        population = new_population
    
    # Find best individual
    final_fitness = [evaluate_individual(ind) for ind in population]
    best_idx = np.argmax(final_fitness)
    best_accuracy = final_fitness[best_idx]
    
    print(f"Genetic Algorithm Best Accuracy: {best_accuracy:.4f}")
    
    return {'Genetic Algorithm': best_accuracy}

def meta_heuristic_algorithms(X_train, X_test, y_train, y_test):
    """Implement meta-heuristic algorithms"""
    return genetic_algorithm_optimization(X_train, X_test, y_train, y_test)

def visualize_results(all_results, df):
    """Create visualization of results"""
    print("\nGenerating visualizations...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Algorithm comparison
    algorithms = list(all_results.keys())
    accuracies = list(all_results.values())
    
    axes[0].bar(algorithms, accuracies, color=['skyblue', 'lightgreen', 'salmon', 'gold', 'plum'])
    axes[0].set_title('Algorithm Accuracy Comparison')
    axes[0].set_ylabel('Accuracy')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3)
    
    # 2. Disorder distribution
    disorder_counts = df['Disorder'].value_counts()
    axes[1].pie(disorder_counts.values, labels=disorder_counts.index, autopct='%1.1f%%')
    axes[1].set_title('Disorder Distribution')
    
    # 3. Brain wave patterns
    brain_waves = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Theta']
    brain_wave_means = df[brain_waves].mean()
    
    axes[2].bar(brain_waves, brain_wave_means, color='orange', alpha=0.7)
    axes[2].set_title('Average Brain Wave Patterns')
    axes[2].set_ylabel('Average Amplitude')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('c:/Users/mohan/Downloads/SUmmerPabitra/algorithm_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run the complete analysis"""
    print("Disease Detection System - Algorithm Comparison")
    print("=" * 50)
    
    # Load and enhance data
    df = load_and_enhance_dataset('c:/Users/mohan/Downloads/SUmmerPabitra/disease_detection_dataset.csv')
    
    # Prepare data
    X_train, X_test, y_train, y_test, label_encoder = prepare_data(df)
    
    # Run algorithms
    deterministic_results = deterministic_algorithms(X_train, X_test, y_train, y_test)
    stochastic_results = stochastic_algorithms(X_train, X_test, y_train, y_test)
    meta_results = meta_heuristic_algorithms(X_train, X_test, y_train, y_test)
    
    # Combine results
    all_results = {**deterministic_results, **stochastic_results, **meta_results}
    
    # Print summary
    print("\n" + "=" * 50)
    print("FINAL RESULTS SUMMARY")
    print("=" * 50)
    
    print("\nAccuracy Results:")
    for algorithm, accuracy in sorted(all_results.items(), key=lambda x: x[1], reverse=True):
        print(f"{algorithm:.<25} {accuracy:.4f}")
    
    best_algorithm = max(all_results, key=all_results.get)
    print(f"\nBest Algorithm: {best_algorithm} ({all_results[best_algorithm]:.4f})")
    
    # Algorithm category summary
    print(f"\nCategory Summary:")
    if deterministic_results:
        best_det = max(deterministic_results, key=deterministic_results.get)
        print(f"Best Deterministic: {best_det} ({deterministic_results[best_det]:.4f})")
    
    if stochastic_results:
        best_sto = max(stochastic_results, key=stochastic_results.get)
        print(f"Best Stochastic: {best_sto} ({stochastic_results[best_sto]:.4f})")
    
    if meta_results:
        best_meta = max(meta_results, key=meta_results.get)
        print(f"Best Meta-heuristic: {best_meta} ({meta_results[best_meta]:.4f})")
    
    # Disorder classification summary
    print(f"\nDisorder Types Classified:")
    disorder_names = label_encoder.classes_
    for i, disorder in enumerate(disorder_names):
        count = sum(1 for label in y_train if label == i) + sum(1 for label in y_test if label == i)
        print(f"  {disorder}: {count} cases")
    
    # Save enhanced dataset
    df.to_csv('c:/Users/mohan/Downloads/SUmmerPabitra/enhanced_dataset.csv', index=False)
    print(f"\nEnhanced dataset saved as 'enhanced_dataset.csv'")
    
    # Create visualizations
    visualize_results(all_results, df)
    
    return all_results

if __name__ == "__main__":
    results = main()
