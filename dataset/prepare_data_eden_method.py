"""
Data Preprocessing Script - Following Eden Modise's Methodology
=================================================================

This script implements the EXACT data preparation methodology from Eden's paper:
1. Load stock data from CSV files
2. Create percentage change target variable (indirect modeling)
3. Split into train/val/test: 2215/200/100 days
4. Normalize using ONLY training statistics (prevent data leakage)
5. Save in format compatible with Time-Series-Library

Key parameters from Eden's paper:
- Train: 2215 days
- Validation: 200 days
- Test: 100 days
- Normalization: x_hat = (x - μ_train) / σ_train
- Target: percentage change (pct_chg) instead of raw close price
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class EdenDataPreprocessor:
    """
    Implements Eden's exact preprocessing methodology
    """

    def __init__(self, train_days=2215, val_days=200, test_days=100):
        self.train_days = train_days
        self.val_days = val_days
        self.test_days = test_days

    def load_csv(self, filepath):
        """
        Load CSV file with robust error handling
        """
        try:
            df = pd.read_csv(filepath)

            # Handle different date column names
            date_cols = ['Date', 'date', 'DATE', 'Datetime', 'datetime']
            date_col = None
            for col in date_cols:
                if col in df.columns:
                    date_col = col
                    break

            if date_col is None:
                print(f"Warning: No date column found in {filepath}")
                return None

            # Parse dates
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

            # Remove rows with invalid dates or data
            df = df.dropna(subset=[date_col])

            # Remove #N/A and other invalid values
            df = df.replace('#N/A', np.nan)
            df = df.replace('N/A', np.nan)

            # Convert numeric columns
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # NOTE: Log transforms will be applied AFTER pct_chg calculation
            # to avoid calculating pct_chg from log-transformed prices

            # Drop rows with missing values or inf
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna()

            # Rename date column to standard 'date'
            df = df.rename(columns={date_col: 'date'})

            # Sort by date
            df = df.sort_values('date')
            df = df.reset_index(drop=True)

            return df

        except Exception as e:
            print(f"Error loading {filepath}: {str(e)}")
            return None

    def create_percentage_change(self, df):
        """
        Create percentage change target variable as in Eden's paper.
        This helps model focus on directional movement rather than absolute prices.

        Formula: pct_chg = (Close_t - Close_{t-1}) / Close_{t-1}
        """
        df = df.copy()
        df['pct_chg'] = df['Close'].pct_change()

        # Remove first row with NaN
        df = df.dropna()

        return df

    def split_data(self, df):
        """
        Split data into train/val/test using 80/10/10 ratio.
        Uses ALL available data from 2006-2024 for comprehensive training.

        - Train: 80% (for learning)
        - Val: 10% (for hyperparameter tuning and early stopping)
        - Test: 10% (for final evaluation)
        """
        total_days = len(df)

        # Use 80/10/10 split for all data
        train_size = int(total_days * 0.80)
        val_size = int(total_days * 0.10)
        test_size = total_days - train_size - val_size  # Remaining goes to test

        train_df = df.iloc[:train_size].copy()
        val_df = df.iloc[train_size:train_size + val_size].copy()
        test_df = df.iloc[train_size + val_size:].copy()

        print(f"\n{'='*60}")
        print(f"DATA SPLIT (80/10/10):")
        print(f"{'='*60}")
        print(f"Total samples: {total_days}")
        print(f"Train: {len(train_df)} samples (80%)")
        print(f"Val: {len(val_df)} samples (10%)")
        print(f"Test: {len(test_df)} samples (10%)")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"{'='*60}")

        return train_df, val_df, test_df

    def normalize_data(self, train_df, val_df, test_df):
        """
        Normalize using ONLY training statistics to prevent data leakage.

        Formula from Eden's paper: x_hat = (x - μ) / σ
        where μ and σ are computed from training set only.

        Features to normalize: Open, High, Low, Close, Volume, pct_chg
        Note: pct_chg MUST be normalized to match scale of other features!
        """
        features_to_normalize = ['Open', 'High', 'Low', 'Close', 'Volume', 'pct_chg']

        # Calculate training statistics
        train_mean = train_df[features_to_normalize].mean()
        train_std = train_df[features_to_normalize].std()

        # Normalize all sets using training statistics
        train_norm = train_df.copy()
        val_norm = val_df.copy()
        test_norm = test_df.copy()

        for feature in features_to_normalize:
            train_norm[feature] = (train_df[feature] - train_mean[feature]) / train_std[feature]
            val_norm[feature] = (val_df[feature] - train_mean[feature]) / train_std[feature]
            test_norm[feature] = (test_df[feature] - train_mean[feature]) / train_std[feature]

        # Store normalization params for later use
        norm_params = {
            'mean': train_mean.to_dict(),
            'std': train_std.to_dict()
        }

        return train_norm, val_norm, test_norm, norm_params

    def save_for_tslib(self, train_df, val_df, test_df, output_path, stock_name):
        """
        Save in format compatible with Time-Series-Library.

        Format required:
        - CSV file with columns: date, Open, High, Low, Close, Volume, pct_chg
        - Combined file with all splits (models will split internally)
        """
        # Combine all splits
        combined_df = pd.concat([train_df, val_df, test_df], axis=0)
        combined_df = combined_df.reset_index(drop=True)

        # Select relevant columns in correct order
        output_cols = ['date', 'Open', 'High', 'Low', 'Close', 'Volume', 'pct_chg']
        combined_df = combined_df[output_cols]

        # Save to CSV
        output_file = Path(output_path) / f"{stock_name}_normalized.csv"
        combined_df.to_csv(output_file, index=False)

        print(f"Saved normalized data to: {output_file}")

        return output_file

    def process_stock(self, input_csv, output_dir, stock_name):
        """
        Complete preprocessing pipeline for a single stock
        """
        print(f"\n{'='*60}")
        print(f"Processing: {stock_name}")
        print(f"{'='*60}")

        # Step 1: Load data
        print("Step 1: Loading CSV file...")
        df = self.load_csv(input_csv)
        if df is None:
            print(f"Failed to load {stock_name}")
            return None
        print(f"Loaded {len(df)} rows")

        # Step 2: Create percentage change (from ORIGINAL prices)
        print("\nStep 2: Creating percentage change target from original prices...")
        df = self.create_percentage_change(df)
        print(f"Created pct_chg column. New length: {len(df)} rows")

        # Step 2.5: Apply log transforms AFTER pct_chg calculation
        print("\nStep 2.5: Applying log transforms to prices and volume (NOT to pct_chg)...")
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            if col in df.columns:
                df[col] = np.log(df[col])  # log for prices
                print(f"  Applied log() to {col}")

        if 'Volume' in df.columns:
            df['Volume'] = np.log1p(df['Volume'])  # log1p for volume (handles zeros)
            print(f"  Applied log1p() to Volume")

        # Drop any inf or NaN that might have appeared from log transforms
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        print(f"  ✅ Log transforms applied. pct_chg column UNCHANGED (still original scale)")
        print(f"  Remaining rows after cleaning: {len(df)}")

        # Step 3: Split data
        print("\nStep 3: Splitting into train/val/test...")
        train_df, val_df, test_df = self.split_data(df)

        # Step 4: SKIP NORMALIZATION - Let Time-Series-Library data_loader handle it!
        print("\nStep 4: Skipping normalization (data_loader will handle this)...")
        print("  ⚠️  Data will be saved in ORIGINAL scale (unnormalized)")
        print("  ✅  Data loader will normalize automatically during training")

        # Step 5: Save UNNORMALIZED data
        print("\nStep 5: Saving processed data (unnormalized)...")
        output_file = self.save_for_tslib(
            train_df, val_df, test_df,  # ← Using original splits, NOT normalized!
            output_dir, stock_name
        )

        # Optionally calculate norm params for reference (not used by model)
        print("\nCalculating normalization params for reference...")
        features_to_normalize = ['Open', 'High', 'Low', 'Close', 'Volume', 'pct_chg']
        train_mean = train_df[features_to_normalize].mean()
        train_std = train_df[features_to_normalize].std()
        norm_params = {
            'mean': train_mean.to_dict(),
            'std': train_std.to_dict()
        }
        params_file = Path(output_dir) / f"{stock_name}_norm_params.csv"
        params_df = pd.DataFrame(norm_params)
        params_df.to_csv(params_file)
        print(f"Saved normalization params (reference only) to: {params_file}")

        # Print summary statistics
        print("\n" + "="*60)
        print(f"Summary for {stock_name}:")
        print("="*60)
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"Total rows: {len(df)}")
        print(f"Train rows: {len(train_df)}")
        print(f"Val rows: {len(val_df)}")
        print(f"Test rows: {len(test_df)}")
        print("\nData ranges (UNNORMALIZED - original scale):")
        print(f"  pct_chg: {train_df['pct_chg'].min():.6f} to {train_df['pct_chg'].max():.6f}")
        print(f"  Close: {train_df['Close'].min():.2f} to {train_df['Close'].max():.2f}")
        print("\nReference normalization params (not applied to saved data):")
        print("  Mean:", {k: f"{v:.6f}" for k, v in norm_params['mean'].items()})
        print("  Std:", {k: f"{v:.6f}" for k, v in norm_params['std'].items()})

        return {
            'output_file': output_file,
            'params_file': params_file,
            'train_size': len(train_df),
            'val_size': len(val_df),
            'test_size': len(test_df),
            'norm_params': norm_params
        }


def main():
    """
    Main function to process all stock datasets
    """
    print("="*80)
    print("DATA PREPROCESSING - EDEN'S METHODOLOGY")
    print("="*80)

    # Setup paths
    data_dir = Path(__file__).parent
    output_dir = data_dir / 'processed_data'
    output_dir.mkdir(exist_ok=True)

    # Define stock files mapping
    stock_files = {
        'NVIDIA': 'Data - NVIDIA.csv',
        'APPLE': 'Data - APPLE.csv',
        'SP500': 'Data - S&P500.csv',
        'NASDAQ': 'Data - NASDAQ.csv',
        'ABSA': 'Data - ABSA GROUP LTD.csv',
        'SASOL': 'Data - SASOL.csv',
        'DRD_GOLD': 'Data - DRD GOLD.csv',
        'ANGLO_AMERICAN': 'Data - Anglo American plc.csv'
    }

    # Initialize preprocessor with Eden's parameters
    preprocessor = EdenDataPreprocessor(
        train_days=2215,
        val_days=200,
        test_days=100
    )

    # Process each stock
    results = {}
    for stock_name, filename in stock_files.items():
        input_path = data_dir / filename

        if not input_path.exists():
            print(f"\nWarning: {filename} not found, skipping...")
            continue

        result = preprocessor.process_stock(
            input_csv=input_path,
            output_dir=output_dir,
            stock_name=stock_name
        )

        if result:
            results[stock_name] = result

    # Print final summary
    print("\n" + "="*80)
    print("PROCESSING COMPLETE!")
    print("="*80)
    print(f"\nSuccessfully processed {len(results)} stocks:")
    for stock_name in results.keys():
        print(f"  ✓ {stock_name}")

    print(f"\nAll processed data saved to: {output_dir}")
    print("\nYou can now use these files with the Time-Series-Library models.")
    print("Example usage:")
    print("  python run.py --data custom --root_path ../dataset/processed_data/ \\")
    print("    --data_path NVIDIA_normalized.csv --features M --target pct_chg \\")
    print("    --seq_len 60 --pred_len 10")


if __name__ == "__main__":
    main()
