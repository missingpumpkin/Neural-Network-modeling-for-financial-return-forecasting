import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Dict, Tuple, Optional, Union
import os
import logging

class FinancialDataLoader:
    """
    Data loader for financial data with multiple factors for asset pricing models.
    
    This class handles downloading, preprocessing, and preparing financial data
    for deep learning models, including various factors like beta, book-to-market
    ratios, and other observable factors used in asset pricing.
    """
    
    def __init__(self, 
                 tickers: List[str],
                 start_date: str,
                 end_date: str,
                 factors: Optional[List[str]] = None,
                 cache_dir: str = "data/cache"):
        """
        Initialize the financial data loader.
        
        Args:
            tickers: List of stock tickers to download data for
            start_date: Start date for historical data (YYYY-MM-DD)
            end_date: End date for historical data (YYYY-MM-DD)
            factors: List of factors to include (default: None, which uses basic factors)
            cache_dir: Directory to cache downloaded data
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.factors = factors or ["Close", "Volume", "High", "Low", "Open"]
        self.cache_dir = cache_dir
        
        os.makedirs(cache_dir, exist_ok=True)
        
        self.raw_data = {}
        self.processed_data = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
    
    def download_data(self, force_download: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Download financial data for the specified tickers.
        
        Args:
            force_download: If True, download data even if cached version exists
            
        Returns:
            Dictionary mapping tickers to their respective DataFrames
        """
        for ticker in self.tickers:
            cache_file = os.path.join(self.cache_dir, f"{ticker}_{self.start_date}_{self.end_date}.csv")
            
            if os.path.exists(cache_file) and not force_download:
                self.raw_data[ticker] = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                for col in self.raw_data[ticker].columns:
                    self.raw_data[ticker][col] = pd.to_numeric(self.raw_data[ticker][col], errors='coerce')
                logging.info(f"Loaded {ticker} data from cache")
            else:
                try:
                    data = yf.download(ticker, start=self.start_date, end=self.end_date)
                    self.raw_data[ticker] = data
                    
                    data.to_csv(cache_file)
                    logging.info(f"Downloaded {ticker} data and saved to cache")
                except Exception as e:
                    logging.error(f"Error downloading {ticker} data: {e}")
        
        return self.raw_data
    
    def calculate_returns(self, period: str = 'daily') -> Dict[str, pd.Series]:
        """
        Calculate returns for each ticker.
        
        Args:
            period: Return calculation period ('daily', 'weekly', 'monthly')
            
        Returns:
            Dictionary mapping tickers to their return series
        """
        returns = {}
        
        for ticker, data in self.raw_data.items():
            if not isinstance(data, pd.DataFrame):
                logging.warning(f"Data for {ticker} is not a DataFrame, skipping return calculation")
                continue
                
            if 'Close' not in data.columns:
                logging.warning(f"Close price not found for {ticker}, skipping return calculation")
                continue
            
            try:
                prices = data['Close'].astype(float)
                prices = prices.dropna()  # Remove any NaN values after conversion
                
                if len(prices) == 0:
                    logging.warning(f"No valid price data for {ticker} after conversion, skipping")
                    continue
                
                if period == 'daily':
                    returns[ticker] = prices.pct_change().dropna()
                elif period == 'weekly':
                    returns[ticker] = prices.resample('W').last().pct_change().dropna()
                elif period == 'monthly':
                    returns[ticker] = prices.resample('M').last().pct_change().dropna()
                else:
                    raise ValueError(f"Unsupported period: {period}")
                
                logging.info(f"Calculated {period} returns for {ticker}: {len(returns[ticker])} data points")
            except Exception as e:
                logging.error(f"Error calculating returns for {ticker}: {e}")
                continue
        
        if not returns:
            raise ValueError("No valid return data could be calculated for any ticker")
            
        return returns
    
    def calculate_factors(self) -> pd.DataFrame:
        """
        Calculate or extract factors for asset pricing models.
        
        This method calculates various factors used in asset pricing models,
        such as market beta, size, value (book-to-market), momentum, etc.
        
        Returns:
            DataFrame with tickers as index and factors as columns
        """
        
        factor_data = {}
        
        for ticker, data in self.raw_data.items():
            ticker_factors = {}
            
            returns = data['Close'].pct_change().dropna()
            
            ticker_factors['volatility'] = returns.std()
            
            if len(returns) >= 63:  # Approximately 3 months of trading days
                ticker_factors['momentum'] = (returns.iloc[-1] / returns.iloc[-63]) - 1
            
            if 'Volume' in data.columns and 'Close' in data.columns:
                ticker_factors['size'] = data['Close'].iloc[-1] * data['Volume'].iloc[-1]
            
            factor_data[ticker] = ticker_factors
        
        return pd.DataFrame.from_dict(factor_data, orient='index')
    
    def prepare_data(self, 
                    window_size: int = 20, 
                    prediction_horizon: int = 1,
                    train_ratio: float = 0.7,
                    val_ratio: float = 0.15,
                    include_factors: bool = True,
                    normalize: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training, validation, and testing.
        
        Args:
            window_size: Number of time steps to use as input features
            prediction_horizon: Number of time steps ahead to predict
            train_ratio: Ratio of data to use for training
            val_ratio: Ratio of data to use for validation
            include_factors: Whether to include calculated factors
            normalize: Whether to normalize features
            
        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        returns_dict = self.calculate_returns()
        
        if include_factors:
            factors_df = self.calculate_factors()
        
        X_all, y_all = [], []
        
        for ticker, returns in returns_dict.items():
            if len(returns) <= window_size + prediction_horizon:
                logging.warning(f"Ticker {ticker} has insufficient data points, skipping")
                continue
                
            for i in range(window_size, len(returns) - prediction_horizon):
                features = returns[i - window_size:i].values
                
                features_array = np.array(features).reshape(-1, 1)  # Shape: (window_size, 1)
                
                # Add static factors if available
                if include_factors and ticker in factors_df.index:
                    ticker_factors = factors_df.loc[ticker].values
                    for factor_value in ticker_factors:
                        factor_column = np.full((window_size, 1), factor_value)
                        features_array = np.hstack((features_array, factor_column))
                
                vol_5d = 0.0
                if i >= window_size + 5:
                    vol_5d = returns[i-5:i].std()
                vol_column = np.full((window_size, 1), vol_5d)
                features_array = np.hstack((features_array, vol_column))
                
                ma_diff = 0.0
                if i >= window_size + 10:
                    ma_5d = returns[i-5:i].mean()
                    ma_10d = returns[i-10:i].mean()
                    ma_diff = ma_5d - ma_10d
                ma_diff_column = np.full((window_size, 1), ma_diff)
                features_array = np.hstack((features_array, ma_diff_column))
                
                features = features_array  # Shape: (window_size, num_features)
                
                target = returns.iloc[i + prediction_horizon - 1]
                
                X_all.append(features)
                y_all.append(target)
        
        if not X_all:
            raise ValueError("No valid data samples could be created. Check your data and parameters.")
        
        X_all = np.array(X_all)
        y_all = np.array(y_all)
        
        if normalize:
            # Calculate mean and std on training data only to prevent data leakage
            n_samples = len(X_all)
            train_idx = int(n_samples * train_ratio)
            
            for i in range(X_all.shape[-1]):
                train_data = X_all[:train_idx, :, i] if len(X_all.shape) == 3 else X_all[:train_idx, i]
                
                # Calculate mean and std
                mean = np.mean(train_data)
                std = np.std(train_data)
                
                if std == 0:
                    std = 1e-8
                
                if len(X_all.shape) == 3:  # (samples, seq_len, features)
                    X_all[:, :, i] = (X_all[:, :, i] - mean) / std
                else:  # (samples, features)
                    X_all[:, i] = (X_all[:, i] - mean) / std
                
                logging.info(f"Normalized feature {i}: mean={mean:.6f}, std={std:.6f}")
        
        n_samples = len(X_all)
        train_idx = int(n_samples * train_ratio)
        val_idx = train_idx + int(n_samples * val_ratio)
        
        indices = np.random.permutation(n_samples)
        X_all = X_all[indices]
        y_all = y_all[indices]
        
        self.X_train = X_all[:train_idx]
        self.y_train = y_all[:train_idx]
        self.X_val = X_all[train_idx:val_idx]
        self.y_val = y_all[train_idx:val_idx]
        self.X_test = X_all[val_idx:]
        self.y_test = y_all[val_idx:]
        
        logging.info(f"Data preparation complete: {len(self.X_train)} training samples, "
                    f"{len(self.X_val)} validation samples, {len(self.X_test)} test samples")
        logging.info(f"Feature shape: {self.X_train.shape}")
        
        return self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test
