#!/usr/bin/env python3
"""
Extract all experimental results from slurm output files
"""
import re
import pandas as pd
from pathlib import Path

# Slurm files mapping
SLURM_FILES = {
    'Autoformer': 'forecast-research/slurm-167120.out',
    'FEDformer': 'forecast-research/slurm-167126.out',
    'Informer': 'forecast-research/slurm-167123.out',
    'iTransformer': 'forecast-research/slurm-167138.out',
    'Mamba': 'forecast-research/slurm-167121.out',
}

DATASETS = ['NVIDIA', 'APPLE', 'SP500', 'NASDAQ', 'ABSA', 'SASOL', 'ANGLO_AMERICAN', 'DRD_GOLD']
HORIZONS = [3, 5, 10, 22, 50, 100]

def extract_results_from_file(filepath, model_name):
    """Extract all experimental results from a slurm file"""
    results = []

    with open(filepath, 'r') as f:
        content = f.read()

    # Split by training sessions
    sections = re.split(r'Training: \w+ on (\w+) for H=(\d+)', content)

    # Iterate through sections (skip first empty one)
    for i in range(1, len(sections), 3):
        if i+2 >= len(sections):
            break

        dataset = sections[i]
        horizon = int(sections[i+1])
        section_content = sections[i+2]

        # Find MSE/MAE/R² in this section
        mse_pattern = r'\tmse:([\d.]+), mae:([\d.]+), rmse:([\d.]+), r2:([-\d.]+)'
        match = re.search(mse_pattern, section_content)

        if match:
            mse, mae, rmse, r2 = match.groups()
            results.append({
                'Model': model_name,
                'Dataset': dataset,
                'Horizon': horizon,
                'MSE': float(mse),
                'MAE': float(mae),
                'RMSE': float(rmse),
                'R²': float(r2)
            })

    return results

def main():
    all_results = []

    print("Extracting results from slurm files...")
    print("=" * 80)

    for model_name, filepath in SLURM_FILES.items():
        print(f"\nProcessing {model_name}...")
        try:
            results = extract_results_from_file(filepath, model_name)
            print(f"  Found {len(results)} experiments")
            all_results.extend(results)
        except Exception as e:
            print(f"  ERROR: {e}")

    # Create DataFrame
    df = pd.DataFrame(all_results)

    print(f"\n{'=' * 80}")
    print(f"TOTAL EXPERIMENTS EXTRACTED: {len(df)}")
    print(f"Expected: {5 * 8 * 6} = 240")
    print(f"{'=' * 80}\n")

    # Summary by model
    print("Results by Model:")
    print(df.groupby('Model').size())
    print()

    # Summary by dataset
    print("Results by Dataset:")
    print(df.groupby('Dataset').size())
    print()

    # Summary by horizon
    print("Results by Horizon:")
    print(df.groupby('Horizon').size())
    print()

    # Save to CSV
    df.to_csv('all_experimental_results.csv', index=False)
    print("✅ Saved to: all_experimental_results.csv")

    # Create summary tables
    print("\n" + "=" * 80)
    print("MODEL RANKINGS (by average MSE)")
    print("=" * 80)
    model_summary = df.groupby('Model').agg({
        'MSE': 'mean',
        'MAE': 'mean',
        'R²': 'mean'
    }).sort_values('MSE')
    print(model_summary)

    print("\n" + "=" * 80)
    print("DATASET-WISE WINNERS")
    print("=" * 80)
    for dataset in DATASETS:
        dataset_df = df[df['Dataset'] == dataset]
        if len(dataset_df) > 0:
            best = dataset_df.loc[dataset_df['MSE'].idxmin()]
            print(f"{dataset:20s}: {best['Model']:15s} (MSE={best['MSE']:.6f})")

    return df

if __name__ == "__main__":
    df = main()
