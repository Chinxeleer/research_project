"""
Results Analysis Script for Financial Forecasting Research
============================================================
This script analyzes experiment results and creates summary tables
like those in Eden's paper.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import os


def load_experiment_results(results_dir="./results"):
    """
    Load all experiment results from the results directory
    """
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"Results directory not found: {results_dir}")
        return None

    experiments = []

    for exp_folder in results_path.iterdir():
        if exp_folder.is_dir():
            metrics_file = exp_folder / "metrics.npy"

            if metrics_file.exists():
                try:
                    # Load metrics: [MAE, MSE, RMSE, MAPE, MSPE, R²]
                    metrics = np.load(metrics_file)

                    # Parse experiment name: Model_Stock_HX_Exp
                    exp_name = exp_folder.name.replace("_Exp", "")
                    parts = exp_name.split("_")

                    if len(parts) >= 3:
                        model = parts[0]
                        stock = "_".join(parts[1:-1])  # Handle multi-word stock names
                        horizon = parts[-1].replace("H", "")

                        if len(metrics) == 6:
                            mae, mse, rmse, mape, mspe, r2 = metrics
                        else:
                            # Old format without R²
                            mae, mse, rmse, mape, mspe = metrics[:5]
                            r2 = np.nan

                        experiments.append({
                            'Model': model,
                            'Stock': stock,
                            'Horizon': int(horizon),
                            'MAE': mae,
                            'MSE': mse,
                            'RMSE': rmse,
                            'MAPE': mape,
                            'MSPE': mspe,
                            'R²': r2,
                            'Experiment': exp_name
                        })
                except Exception as e:
                    print(f"Error loading {exp_folder.name}: {e}")

    if not experiments:
        print("No experiment results found!")
        return None

    df = pd.DataFrame(experiments)
    return df


def create_summary_table(df, metric='MSE', models=None, stocks=None):
    """
    Create a summary table like Eden's paper
    """
    if models is None:
        models = df['Model'].unique()
    if stocks is None:
        stocks = df['Stock'].unique()

    results = []

    for model in models:
        for stock in stocks:
            row = {'Model': model, 'Stock': stock}

            for horizon in sorted(df['Horizon'].unique()):
                subset = df[(df['Model'] == model) &
                           (df['Stock'] == stock) &
                           (df['Horizon'] == horizon)]

                if not subset.empty:
                    value = subset[metric].values[0]
                    row[f'H={horizon}'] = value
                else:
                    row[f'H={horizon}'] = np.nan

            results.append(row)

    return pd.DataFrame(results)


def print_best_models(df):
    """
    Find best performing model for each stock and horizon
    """
    print("\n" + "="*80)
    print("BEST MODELS PER STOCK AND HORIZON (by MSE)")
    print("="*80 + "\n")

    for stock in df['Stock'].unique():
        print(f"\n{stock}:")
        print("-" * 60)

        for horizon in sorted(df['Horizon'].unique()):
            subset = df[(df['Stock'] == stock) & (df['Horizon'] == horizon)]

            if not subset.empty:
                best_idx = subset['MSE'].idxmin()
                best = subset.loc[best_idx]

                print(f"  H={horizon:3d}: {best['Model']:15s} - "
                      f"MSE: {best['MSE']:.6f}, MAE: {best['MAE']:.6f}, "
                      f"R²: {best['R²']:.4f}")


def generate_latex_table(df, metric='MSE'):
    """
    Generate LaTeX table for your thesis/paper
    """
    summary = create_summary_table(df, metric=metric)

    print("\n" + "="*80)
    print(f"LATEX TABLE ({metric})")
    print("="*80 + "\n")

    print("\\begin{table}[h]")
    print("\\centering")
    print(f"\\caption{{{metric} Comparison Across Models and Horizons}}")
    print("\\begin{tabular}{ll" + "c" * len([c for c in summary.columns if c.startswith('H=')]) + "}")
    print("\\hline")

    # Header
    horizons = [c for c in summary.columns if c.startswith('H=')]
    print("Model & Stock & " + " & ".join(horizons) + " \\\\")
    print("\\hline")

    # Data rows
    for _, row in summary.iterrows():
        values = [f"{row[h]:.5f}" if not np.isnan(row[h]) else "-" for h in horizons]
        print(f"{row['Model']} & {row['Stock']} & " + " & ".join(values) + " \\\\")

    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")


def main():
    """
    Main analysis function
    """
    print("="*80)
    print("FINANCIAL FORECASTING RESULTS ANALYSIS")
    print("="*80)

    # Load results
    print("\nLoading experiment results...")
    df = load_experiment_results("./results")

    if df is None or df.empty:
        print("\nNo results found! Please run experiments first.")
        return

    print(f"Loaded {len(df)} experiments\n")

    # Print overview
    print("="*80)
    print("EXPERIMENT OVERVIEW")
    print("="*80)
    print(f"\nModels: {', '.join(df['Model'].unique())}")
    print(f"Stocks: {', '.join(df['Stock'].unique())}")
    print(f"Horizons: {', '.join(map(str, sorted(df['Horizon'].unique())))}")

    # Print statistics
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80 + "\n")

    for metric in ['MSE', 'MAE', 'RMSE', 'R²']:
        print(f"{metric}:")
        print(f"  Mean: {df[metric].mean():.6f}")
        print(f"  Std:  {df[metric].std():.6f}")
        print(f"  Min:  {df[metric].min():.6f}")
        print(f"  Max:  {df[metric].max():.6f}")
        print()

    # Create summary tables
    print("\n" + "="*80)
    print("MSE SUMMARY TABLE")
    print("="*80 + "\n")
    mse_table = create_summary_table(df, metric='MSE')
    print(mse_table.to_string(index=False))

    print("\n" + "="*80)
    print("R² SUMMARY TABLE")
    print("="*80 + "\n")
    r2_table = create_summary_table(df, metric='R²')
    print(r2_table.to_string(index=False))

    # Find best models
    print_best_models(df)

    # Generate LaTeX tables
    generate_latex_table(df, metric='MSE')
    generate_latex_table(df, metric='R²')

    # Save to CSV for Excel/further analysis
    output_file = "experiment_results_summary.csv"
    df.to_csv(output_file, index=False)
    print(f"\n\nFull results saved to: {output_file}")

    # Save summary tables
    mse_table.to_csv("summary_mse.csv", index=False)
    r2_table.to_csv("summary_r2.csv", index=False)
    print("Summary tables saved to: summary_mse.csv, summary_r2.csv")


if __name__ == "__main__":
    main()
