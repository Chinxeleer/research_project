"""
Stock Market Data Preparation Module
Following the methodology from Modise's research paper
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

class StockDataPreparation:
    """
    Prepare stock data following the methodology from the reference paper:
    - Download from Yahoo Finance
    - Create percentage change target variable
    - Normalize using training data statistics
    - Split into train/val/test sets
    """
    
    def __init__(self, 
                 symbols: List[str] = None, 
                 start_date: str = '2014-01-01',
                 end_date: str = '2024-01-01',
                 train_days: int = 2215,
                 val_days: int = 200,
                 test_days: int = 100):
        """
        Initialize data preparation with parameters from the paper
        
        Args:
            symbols: List of stock symbols (default includes diverse markets)
            start_date: Start date for data collection
            end_date: End date for data collection
            train_days: Number of days for training (paper used 2215)
            val_days: Number of days for validation (paper used 200)
            test_days: Number of days for testing (paper used 100)
        """
        if symbols is None:
            # Default symbols from different markets as in the paper
            self.symbols = ['AAPL', 'PBR', 'NSRGY', 'RELIANCE.NS', 'SOL.JO', 'MTN.JO']
        else:
            self.symbols = symbols
            
        self.start_date = start_date
        self.end_date = end_date
        self.train_days = train_days
        self.val_days = val_days
        self.test_days = test_days
        
    def download_stock_data(self, symbol: str) -> pd.DataFrame:
        """
        Download stock data from Yahoo Finance
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            DataFrame with OHLCV data
        """
        print(f"Downloading data for {symbol}...")
        try:
            data = yf.download(symbol, start=self.start_date, end=self.end_date, progress=False)
            if data.empty:
                print(f"Warning: No data found for {symbol}")
                return None
            return data
        except Exception as e:
            print(f"Error downloading {symbol}: {str(e)}")
            return None
    
    def create_percentage_change(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create percentage change target variable as in the paper
        This helps the model focus on directional movement rather than absolute prices
        
        Args:
            df: DataFrame with stock data
            
        Returns:
            DataFrame with pct_chg column added
        """
        df = df.copy()
        df['pct_chg'] = df['Close'].pct_change()
        # Remove first row with NaN
        df = df.dropna()
        return df
    
    def normalize_data(self, train: pd.DataFrame, val: pd.DataFrame, 
                      test: pd.DataFrame, features: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Normalize data using training statistics to prevent data leakage
        Following equation from paper: x_hat = (x - μ) / σ
        
        Args:
            train: Training data
            val: Validation data
            test: Test data
            features: List of feature columns to normalize
            
        Returns:
            Tuple of normalized (train, val, test) DataFrames
        """
        # Calculate training statistics
        train_mean = train[features].mean()
        train_std = train[features].std()
        
        # Normalize all sets using training statistics
        train_norm = train.copy()
        val_norm = val.copy()
        test_norm = test.copy()
        
        for feature in features:
            train_norm[feature] = (train[feature] - train_mean[feature]) / train_std[feature]
            val_norm[feature] = (val[feature] - train_mean[feature]) / train_std[feature]
            test_norm[feature] = (test[feature] - train_mean[feature]) / train_std[feature]
        
        return train_norm, val_norm, test_norm, train_mean, train_std
    
    def prepare_forecasting_data(self, df: pd.DataFrame, 
                                lookback: int = 60,
                                horizon: int = 10) -> Dict:
        """
        Prepare data for time series forecasting models
        
        Args:
            df: Normalized DataFrame
            lookback: Number of past time steps to use as input
            horizon: Number of future time steps to predict
            
        Returns:
            Dictionary with X and y arrays for forecasting
        """
        # Use percentage change as target (indirect modeling as in the paper)
        target_col = 'pct_chg'
        
        # Features for prediction (can be extended)
        feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'pct_chg']
        
        X, y = [], []
        
        for i in range(lookback, len(df) - horizon + 1):
            # Input: lookback window of all features
            X.append(df[feature_cols].iloc[i-lookback:i].values)
            # Output: future percentage changes
            y.append(df[target_col].iloc[i:i+horizon].values)
        
        return {
            'X': np.array(X),
            'y': np.array(y),
            'feature_names': feature_cols,
            'target_name': target_col
        }
    
    def split_time_series(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/val/test following the paper's approach
        Train: 2215 days, Val: 200 days, Test: 100 days
        
        Args:
            df: Full dataset
            
        Returns:
            Tuple of (train, val, test) DataFrames
        """
        total_days = len(df)
        
        # Ensure we have enough data
        min_required = self.train_days + self.val_days + self.test_days
        if total_days < min_required:
            print(f"Warning: Dataset has {total_days} days, but {min_required} required. Adjusting splits...")
            # Proportionally adjust
            ratio_train = self.train_days / min_required
            ratio_val = self.val_days / min_required
            ratio_test = self.test_days / min_required
            
            train_size = int(total_days * ratio_train)
            val_size = int(total_days * ratio_val)
            test_size = total_days - train_size - val_size
        else:
            train_size = self.train_days
            val_size = self.val_days
            test_size = self.test_days
        
        train = df.iloc[:train_size]
        val = df.iloc[train_size:train_size + val_size]
        test = df.iloc[train_size + val_size:train_size + val_size + test_size]
        
        return train, val, test
    
    def process_stock(self, symbol: str, lookback: int = 60, 
                     horizons: List[int] = None) -> Dict:
        """
        Complete processing pipeline for a single stock
        
        Args:
            symbol: Stock ticker symbol
            lookback: Input window size
            horizons: List of forecast horizons to prepare
            
        Returns:
            Dictionary with all processed data
        """
        if horizons is None:
            # Default horizons from the paper
            horizons = [3, 5, 10, 22, 50, 100]
        
        # Download data
        df = self.download_stock_data(symbol)
        if df is None:
            return None
        
        # Create percentage change
        df = self.create_percentage_change(df)
        
        # Split data
        train, val, test = self.split_time_series(df)
        
        # Define features for normalization
        feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Normalize using training statistics
        train_norm, val_norm, test_norm, train_mean, train_std = self.normalize_data(
            train, val, test, feature_cols
        )
        
        # Prepare data for each horizon
        prepared_data = {}
        for horizon in horizons:
            prepared_data[f'H{horizon}'] = {
                'train': self.prepare_forecasting_data(train_norm, lookback, horizon),
                'val': self.prepare_forecasting_data(val_norm, lookback, horizon),
                'test': self.prepare_forecasting_data(test_norm, lookback, horizon)
            }
        
        return {
            'symbol': symbol,
            'raw_data': {
                'train': train,
                'val': val,
                'test': test
            },
            'normalized_data': {
                'train': train_norm,
                'val': val_norm,
                'test': test_norm
            },
            'statistics': {
                'train_mean': train_mean,
                'train_std': train_std
            },
            'prepared_data': prepared_data,
            'horizons': horizons
        }
    
    def process_all_stocks(self, lookback: int = 60, 
                          horizons: List[int] = None) -> Dict:
        """
        Process all stocks in the symbol list
        
        Args:
            lookback: Input window size
            horizons: List of forecast horizons
            
        Returns:
            Dictionary with processed data for all stocks
        """
        all_data = {}
        
        for symbol in self.symbols:
            print(f"\nProcessing {symbol}...")
            stock_data = self.process_stock(symbol, lookback, horizons)
            if stock_data is not None:
                all_data[symbol] = stock_data
                print(f"Successfully processed {symbol}")
            else:
                print(f"Failed to process {symbol}")
        
        return all_data


def save_processed_data(data: Dict, output_dir: str = './processed_data'):
    """
    Save processed data to disk for later use
    
    Args:
        data: Dictionary with all processed stock data
        output_dir: Directory to save the data
    """
    import os
    import pickle
    
    os.makedirs(output_dir, exist_ok=True)
    
    for symbol, stock_data in data.items():
        # Save as pickle for easy loading
        filename = os.path.join(output_dir, f'{symbol}_processed.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(stock_data, f)
        print(f"Saved {symbol} data to {filename}")
    
    # Also save metadata
    metadata = {
        'symbols': list(data.keys()),
        'horizons': data[list(data.keys())[0]]['horizons'] if data else [],
        'processing_date': pd.Timestamp.now().isoformat()
    }
    
    with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    print(f"Saved metadata to {output_dir}/metadata.pkl")


if __name__ == "__main__":
    # Initialize data preparation with parameters from the paper
    prep = StockDataPreparation(
        symbols=['AAPL', 'PBR', 'NSRGY', 'RELIANCE.NS', 'SOL.JO'],  # Stocks from paper
        start_date='2014-01-01',
        end_date='2024-01-01'
    )
    
    # Process all stocks with horizons from the paper
    horizons = [3, 5, 10, 22, 50, 100]
    all_data = prep.process_all_stocks(lookback=60, horizons=horizons)
    
    # Save processed data
    save_processed_data(all_data)
    
    # Print summary
    print("\n" + "="*50)
    print("Data Preparation Summary")
    print("="*50)
    for symbol, data in all_data.items():
        train_shape = data['prepared_data']['H10']['train']['X'].shape
        print(f"{symbol}: Train shape for H=10: {train_shape}")
