# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# Set style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 6)

print("=== NEURAL NETWORK REGRESSION ANALYSIS ===")
print("=== PRACTICUM 3: SINGLE AND MULTI-LAYER MLP ANALYSIS ===")

# =============================================================================
# 1. LOAD AND PREPARE DATASETS
# =============================================================================

print("\n1. LOADING AND PREPROCESSING DATASETS...")

# Dataset (a): Diabetes
diabetes = load_diabetes()
X_diabetes = diabetes.data
y_diabetes = diabetes.target

# Dataset (b): Realistic Synthetic Avocado
np.random.seed(42)
n_samples = 500

print("\nCreating realistic Avocado dataset...")
avocado_df = pd.DataFrame({
    'Total_Volume': np.random.lognormal(7.5, 0.8, n_samples),
    'Small_Hass': np.random.lognormal(6.5, 0.7, n_samples),
    'Large_Hass': np.random.lognormal(6.8, 0.6, n_samples),
    'type': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
    'region': np.random.choice([0, 1, 2, 3], n_samples)
})

# Generate realistic avocado prices ($0.50 - $3.00 range)
base_price = 1.2
volume_effect = -avocado_df['Total_Volume'] * 0.0003
organic_premium = avocado_df['type'] * 0.6
region_effect = (avocado_df['region'] - 1.5) * 0.1
noise = np.random.normal(0, 0.15, n_samples)

y_avocado = base_price + volume_effect + organic_premium + region_effect + noise
y_avocado = np.clip(y_avocado, 0.5, 3.0)

# Apply log transformation to features
skewed_cols = ['Total_Volume', 'Small_Hass', 'Large_Hass']
avocado_df[skewed_cols] = np.log1p(avocado_df[skewed_cols])
X_avocado = avocado_df.values

print(f"Diabetes: {X_diabetes.shape}")
print(f"Avocado: {X_avocado.shape}")
print(f"Avocado Price Range: ${y_avocado.min():.2f} - ${y_avocado.max():.2f}")

# =============================================================================
# 2. SPLIT DATA
# =============================================================================

X_dia_train, X_dia_test, y_dia_train, y_dia_test = train_test_split(
    X_diabetes, y_diabetes, test_size=0.2, random_state=42
)

X_avo_train, X_avo_test, y_avo_train, y_avo_test = train_test_split(
    X_avocado, y_avocado, test_size=0.2, random_state=42
)

print(f"\n2. DATA SPLIT:")
print(f"Diabetes - Train: {X_dia_train.shape}, Test: {X_dia_test.shape}")
print(f"Avocado - Train: {X_avo_train.shape}, Test: {X_avo_test.shape}")

# =============================================================================
# 3. SINGLE LAYER MLP WITH DYNAMIC NEURON ADDITION
# =============================================================================

print("\n" + "="*70)
print("3. SINGLE LAYER MLP ANALYSIS - DYNAMIC NEURON ADDITION")
print("="*70)

def analyze_single_layer_mlp(X_train, X_test, y_train, y_test, dataset_name):
    """
    Analyzes single-layer MLP with increasing number of neurons
    to determine if single layer is sufficient
    """
    print(f"\nANALYZING SINGLE-LAYER MLP FOR {dataset_name.upper()}:")
    print("Testing different numbers of neurons in single hidden layer...")
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Test different numbers of neurons in single hidden layer
    neuron_counts = [1, 2, 5, 10, 20, 50, 100]  # Removed 150
    results = []
    
    print("\n" + "-" * 65)
    print(f"{'Neurons':<8} | {'Train R2':<9} | {'Test R2':<9} | {'Test RMSE':<10} | {'Overfitting':<12}")
    print("-" * 65)
    
    for n in neuron_counts:
        mlp = MLPRegressor(
            hidden_layer_sizes=(n,),  # Single hidden layer
            activation='relu',
            solver='adam',
            max_iter=1500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            alpha=0.001
        )
        
        # Train the model
        mlp.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_train_pred = mlp.predict(X_train_scaled)
        y_test_pred = mlp.predict(X_test_scaled)
        
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        overfitting = train_r2 - test_r2
        
        results.append({
            'neurons': n,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'overfitting': overfitting
        })
        
        print(f"{n:<8} | {train_r2:<9.4f} | {test_r2:<9.4f} | {test_rmse:<10.4f} | {overfitting:<12.4f}")
    
    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(results)
    
    # Find optimal configuration
    best_idx = results_df['test_r2'].idxmax()
    best_result = results_df.loc[best_idx]
    
    print("\n" + "=" * 50)
    print(f"OPTIMAL SINGLE-LAYER CONFIGURATION FOR {dataset_name.upper()}:")
    print("=" * 50)
    print(f"• Optimal neurons: {best_result['neurons']}")
    print(f"• Test R2: {best_result['test_r2']:.4f}")
    print(f"• Test RMSE: {best_result['test_rmse']:.4f}")
    print(f"• Overfitting: {best_result['overfitting']:.4f}")
    
    # Find minimum sufficient neurons (within 1% of best performance)
    max_test_r2 = best_result['test_r2']
    plateau_threshold = max_test_r2 - 0.01
    sufficient_neurons = results_df[results_df['test_r2'] >= plateau_threshold]['neurons'].min()
    
    print(f"• Minimum sufficient neurons: {sufficient_neurons} (within 1% of best performance)")
    
    # Plotting the results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'{dataset_name} - Single Layer MLP Performance vs Neuron Count', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: R2 Scores
    ax1.plot(results_df['neurons'], results_df['train_r2'], 'o-', 
             label='Train R2', linewidth=2, markersize=6, color='blue')
    ax1.plot(results_df['neurons'], results_df['test_r2'], 'o-', 
             label='Test R2', linewidth=2, markersize=6, color='red')
    ax1.set_xlabel('Number of Neurons in Hidden Layer', fontsize=12)
    ax1.set_ylabel('R2 Score', fontsize=12)
    ax1.set_title('R2 Score vs Number of Neurons', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # Mark optimal point
    ax1.axvline(x=best_result['neurons'], color='green', linestyle='--', alpha=0.7, 
                label=f'Optimal: {best_result["neurons"]} neurons')
    ax1.legend(fontsize=11)
    
    # Plot 2: Test RMSE
    ax2.plot(results_df['neurons'], results_df['test_rmse'], 'o-', 
             linewidth=2, markersize=6, color='purple')
    ax2.set_xlabel('Number of Neurons in Hidden Layer', fontsize=12)
    ax2.set_ylabel('Test RMSE', fontsize=12)
    ax2.set_title('Test RMSE vs Number of Neurons', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    # Mark optimal point
    ax2.axvline(x=best_result['neurons'], color='green', linestyle='--', alpha=0.7, 
                label=f'Optimal: {best_result["neurons"]} neurons')
    ax2.legend(fontsize=11)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    return results_df, best_result, scaler

# Run single-layer analysis for both datasets
print("\n" + "="*70)
print("ANALYZING DIABETES DATASET")
print("="*70)
single_layer_dia, best_single_dia, dia_scaler = analyze_single_layer_mlp(
    X_dia_train, X_dia_test, y_dia_train, y_dia_test, "Diabetes"
)

print("\n" + "="*70)
print("ANALYZING AVOCADO DATASET")
print("="*70)
single_layer_avo, best_single_avo, avo_scaler = analyze_single_layer_mlp(
    X_avo_train, X_avo_test, y_avo_train, y_avo_test, "Avocado"
)

# =============================================================================
# 4. MULTI-LAYER MLP ANALYSIS
# =============================================================================

print("\n" + "="*70)
print("4. MULTI-LAYER MLP ANALYSIS")
print("="*70)

def analyze_multi_layer_mlp(X_train, X_test, y_train, y_test, dataset_name, scaler):
    """
    Analyzes MLP with increasing number of hidden layers
    to check if multi-layer architecture provides improvement
    """
    print(f"\nANALYZING MULTI-LAYER MLP FOR {dataset_name.upper()}:")
    print("Testing different network architectures...")
    
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Test different layer architectures (removed architectures with 150 neurons)
    layer_architectures = [
        (50,),           # 1 layer, 50 neurons
        (50, 50),        # 2 layers, 50 neurons each
        (50, 50, 50),    # 3 layers, 50 neurons each
        (100,),          # 1 layer, 100 neurons  
        (100, 50),       # 2 layers: 100 -> 50
        (100, 100),      # 2 layers, 100 neurons each
        (100, 100, 50),  # 3 layers: 100 -> 100 -> 50
    ]
    
    results = []
    
    print("\n" + "-" * 75)
    print(f"{'Architecture':<15} | {'Layers':<6} | {'Train R2':<9} | {'Test R2':<9} | {'Test RMSE':<10} | {'Overfitting':<12}")
    print("-" * 75)
    
    for architecture in layer_architectures:
        mlp = MLPRegressor(
            hidden_layer_sizes=architecture,
            activation='relu',
            solver='adam',
            max_iter=2000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            alpha=0.001
        )
        
        # Train the model
        mlp.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_train_pred = mlp.predict(X_train_scaled)
        y_test_pred = mlp.predict(X_test_scaled)
        
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        overfitting = train_r2 - test_r2
        
        results.append({
            'architecture': architecture,
            'layers': len(architecture),
            'total_neurons': sum(architecture),
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'overfitting': overfitting
        })
        
        arch_str = str(architecture)
        print(f"{arch_str:<15} | {len(architecture):<6} | {train_r2:<9.4f} | {test_r2:<9.4f} | {test_rmse:<10.4f} | {overfitting:<12.4f}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Find best multi-layer architecture
    best_multi_idx = results_df['test_r2'].idxmax()
    best_multi_result = results_df.loc[best_multi_idx]
    
    print("\n" + "=" * 50)
    print(f"BEST MULTI-LAYER ARCHITECTURE FOR {dataset_name.upper()}:")
    print("=" * 50)
    print(f"• Best architecture: {best_multi_result['architecture']}")
    print(f"• Layers: {best_multi_result['layers']}")
    print(f"• Total neurons: {best_multi_result['total_neurons']}")
    print(f"• Test R2: {best_multi_result['test_r2']:.4f}")
    print(f"• Test RMSE: {best_multi_result['test_rmse']:.4f}")
    print(f"• Overfitting: {best_multi_result['overfitting']:.4f}")
    
    # Plot multi-layer comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'{dataset_name} - Multi-Layer MLP Performance Comparison', 
                 fontsize=16, fontweight='bold')
    
    # Create architecture labels
    arch_labels = [str(arch) for arch in results_df['architecture']]
    x_pos = np.arange(len(arch_labels))
    
    # Define colors based on number of layers
    colors = []
    for layers in results_df['layers']:
        if layers == 1:
            colors.append('skyblue')
        elif layers == 2:
            colors.append('lightgreen')
        else:  # 3 layers
            colors.append('lightcoral')
    
    # Plot 1: Test R2 by Architecture
    bars1 = ax1.bar(x_pos, results_df['test_r2'], color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    ax1.set_xlabel('Network Architecture', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Test R2 Score', fontsize=12, fontweight='bold')
    ax1.set_title('Test R2 Score by Network Architecture', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(arch_labels, rotation=45, ha='right', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([max(0, results_df['test_r2'].min() - 0.05), results_df['test_r2'].max() + 0.05])
    
    # Add value labels on bars
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 2: Overfitting by Architecture
    bars2 = ax2.bar(x_pos, results_df['overfitting'], color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    ax2.set_xlabel('Network Architecture', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Overfitting (Train R2 - Test R2)', fontsize=12, fontweight='bold')
    ax2.set_title('Overfitting by Network Architecture', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(arch_labels, rotation=45, ha='right', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='skyblue', edgecolor='black', label='1 Layer'),
        Patch(facecolor='lightgreen', edgecolor='black', label='2 Layers'),
        Patch(facecolor='lightcoral', edgecolor='black', label='3 Layers')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')
    ax2.legend(handles=legend_elements, loc='upper right')
    
    # Highlight best architecture
    ax1.axvline(x=best_multi_idx, color='red', linestyle='--', alpha=0.8, linewidth=2)
    ax2.axvline(x=best_multi_idx, color='red', linestyle='--', alpha=0.8, linewidth=2)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    return results_df, best_multi_result

# Run multi-layer analysis for both datasets
print("\n" + "="*70)
print("MULTI-LAYER ANALYSIS - DIABETES DATASET")
print("="*70)
multi_layer_dia, best_multi_dia = analyze_multi_layer_mlp(
    X_dia_train, X_dia_test, y_dia_train, y_dia_test, "Diabetes", dia_scaler
)

print("\n" + "="*70)
print("MULTI-LAYER ANALYSIS - AVOCADO DATASET")
print("="*70)
multi_layer_avo, best_multi_avo = analyze_multi_layer_mlp(
    X_avo_train, X_avo_test, y_avo_train, y_avo_test, "Avocado", avo_scaler
)

# =============================================================================
# 5. COMPREHENSIVE COMPARISON AND CONCLUSIONS
# =============================================================================

print("\n" + "="*70)
print("5. SINGLE LAYER vs MULTI-LAYER COMPREHENSIVE COMPARISON")
print("="*70)

def compare_architectures(single_layer_results, best_single, multi_layer_results, best_multi, dataset_name):
    """Compare single layer vs multi-layer performance"""
    
    improvement = best_multi['test_r2'] - best_single['test_r2']
    
    print(f"\n{dataset_name.upper()} DATASET - ARCHITECTURE COMPARISON:")
    print("-" * 55)
    print(f"SINGLE-LAYER PERFORMANCE:")
    print(f"  • Architecture: ({best_single['neurons']},)")
    print(f"  • Test R2: {best_single['test_r2']:.4f}")
    print(f"  • Test RMSE: {best_single['test_rmse']:.4f}")
    
    print(f"\nMULTI-LAYER PERFORMANCE:")
    print(f"  • Architecture: {best_multi['architecture']}")
    print(f"  • Layers: {best_multi['layers']}")
    print(f"  • Test R2: {best_multi['test_r2']:.4f}")
    print(f"  • Test RMSE: {best_multi['test_rmse']:.4f}")
    
    print(f"\nCOMPARISON RESULTS:")
    print(f"  • R2 Improvement: {improvement:+.4f}")
    
    # Determine if single layer is sufficient
    if improvement > 0.02:
        print(f"  • Conclusion: MULTI-LAYER ARCHITECTURE PROVIDES SIGNIFICANT IMPROVEMENT")
        print(f"  • Recommendation: Use multi-layer architecture")
        sufficient = False
    elif improvement > 0.005:
        print(f"  • Conclusion: Multi-layer provides SLIGHT IMPROVEMENT")
        print(f"  • Recommendation: Single layer might be sufficient for some applications")
        sufficient = True
    else:
        print(f"  • Conclusion: SINGLE LAYER IS SUFFICIENT")
        print(f"  • Recommendation: Use single-layer architecture")
        sufficient = True
    
    return sufficient, improvement

print("\n" + "="*70)
print("ARCHITECTURE COMPARISON RESULTS")
print("="*70)

dia_sufficient, dia_improvement = compare_architectures(
    single_layer_dia, best_single_dia, multi_layer_dia, best_multi_dia, "Diabetes"
)

print("\n" + "-" * 70)

avo_sufficient, avo_improvement = compare_architectures(
    single_layer_avo, best_single_avo, multi_layer_avo, best_multi_avo, "Avocado"
)

# =============================================================================
# 6. FINAL CONCLUSIONS AND RECOMMENDATIONS
# =============================================================================

print("\n" + "="*70)
print("6. FINAL CONCLUSIONS AND RECOMMENDATIONS")
print("="*70)

print("\n NEURAL NETWORK ARCHITECTURE ANALYSIS SUMMARY")
print("=" * 50)

print(f"\n DIABETES DATASET:")
print(f"   • Optimal single-layer: ({best_single_dia['neurons']},) neurons")
print(f"   • Best single-layer R2: {best_single_dia['test_r2']:.4f}")
print(f"   • Best multi-layer: {best_multi_dia['architecture']}")
print(f"   • Best multi-layer R2: {best_multi_dia['test_r2']:.4f}")
print(f"   • Multi-layer improvement: {dia_improvement:+.4f}")

if dia_sufficient:
    print(f"   RECOMMENDATION: Single layer architecture is SUFFICIENT")
    print(f"   Suggested architecture: ({best_single_dia['neurons']},)")
else:
    print(f"   RECOMMENDATION: Multi-layer architecture is NECESSARY")
    print(f"   Suggested architecture: {best_multi_dia['architecture']}")

print(f"\n AVOCADO DATASET:")
print(f"   • Optimal single-layer: ({best_single_avo['neurons']},) neurons")
print(f"   • Best single-layer R2: {best_single_avo['test_r2']:.4f}")
print(f"   • Best multi-layer: {best_multi_avo['architecture']}")
print(f"   • Best multi-layer R2: {best_multi_avo['test_r2']:.4f}")
print(f"   • Multi-layer improvement: {avo_improvement:+.4f}")

if avo_sufficient:
    print(f"   RECOMMENDATION: Single layer architecture is SUFFICIENT")
    print(f"   Suggested architecture: ({best_single_avo['neurons']},)")
else:
    print(f"   RECOMMENDATION: Multi-layer architecture is NECESSARY")
    print(f"   Suggested architecture: {best_multi_avo['architecture']}")

print(f"\n KEY FINDINGS FROM DYNAMIC NEURON ANALYSIS:")
print(f"   1. Single-layer networks show performance plateau with increasing neurons")
print(f"   2. Optimal neuron count varies by dataset complexity")
print(f"   3. Multi-layer networks can capture more complex non-linear relationships")
print(f"   4. Overfitting generally increases with network complexity")

print(f"\n FINAL RECOMMENDATIONS:")
if dia_sufficient and avo_sufficient:
    print(f"   Both datasets can be effectively modeled with SINGLE-LAYER networks")
    print(f"   No need for complex multi-layer architectures for these problems")
elif not dia_sufficient and not avo_sufficient:
    print(f"   Both datasets benefit from MULTI-LAYER architectures")
    print(f"   Complex patterns in both datasets require deeper networks")
else:
    print(f"   Dataset complexity varies - architecture requirements differ")
    print(f"   Choose architecture based on specific dataset characteristics")

print(f"\n" + "="*70)
print("NEURAL NETWORK ANALYSIS COMPLETED SUCCESSFULLY!")
print("="*70)