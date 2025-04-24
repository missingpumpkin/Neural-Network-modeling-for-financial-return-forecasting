import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import logging

class ModelEvaluator:
    """
    Evaluator class for deep learning models for asset pricing.
    
    This class handles the evaluation of trained models using various
    financial and statistical metrics.
    """
    
    def __init__(self, 
                model: torch.nn.Module,
                device: Optional[str] = None,
                results_dir: str = "results"):
        """
        Initialize the model evaluator.
        
        Args:
            model: Trained PyTorch model to evaluate
            device: Device to use for evaluation ('cuda' or 'cpu')
            results_dir: Directory to save evaluation results
        """
        self.model = model
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions using the model.
        
        Args:
            X: Input features
            
        Returns:
            Model predictions
        """
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        self.model.eval()
        
        with torch.no_grad():
            y_pred = self.model(X_tensor)
        
        return y_pred.cpu().numpy()
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of evaluation metrics
        """
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        
        direction_true = np.sign(y_true)
        direction_pred = np.sign(y_pred)
        directional_accuracy = np.mean(direction_true == direction_pred)
        
        ic = np.corrcoef(y_true, y_pred)[0, 1]
        
        sharpe_ratio = np.mean(y_pred) / np.std(y_pred) if np.std(y_pred) > 0 else 0
        
        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'directional_accuracy': float(directional_accuracy),
            'information_coefficient': float(ic),
            'sharpe_ratio': float(sharpe_ratio)
        }
    
    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X: Test features
            y_true: True values
            
        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = self.predict(X)
        
        metrics = self.calculate_metrics(y_true, y_pred)
        
        for metric_name, metric_value in metrics.items():
            logging.info(f"{metric_name}: {metric_value:.6f}")
        
        return metrics
    
    def plot_predictions(self, 
                        y_true: np.ndarray, 
                        y_pred: np.ndarray,
                        title: str = "Actual vs Predicted Returns",
                        save_path: Optional[str] = None) -> None:
        """
        Plot actual vs predicted values.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            save_path: Path to save the plot (if None, plot is displayed)
        """
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)
        
        plt.figure(figsize=(10, 6))
        
        plt.scatter(y_true, y_pred, alpha=0.5)
        
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.xlabel("Actual Returns")
        plt.ylabel("Predicted Returns")
        plt.title(title)
        
        metrics = self.calculate_metrics(y_true, y_pred)
        metrics_text = "\n".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        plt.figtext(0.02, 0.02, metrics_text, fontsize=9)
        
        if save_path:
            plt.savefig(save_path)
            logging.info(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_returns_over_time(self,
                              y_true: np.ndarray,
                              y_pred: np.ndarray,
                              dates: Optional[np.ndarray] = None,
                              title: str = "Returns Over Time",
                              save_path: Optional[str] = None) -> None:
        """
        Plot actual and predicted returns over time.
        
        Args:
            y_true: True returns
            y_pred: Predicted returns
            dates: Dates for x-axis (if None, indices are used)
            title: Plot title
            save_path: Path to save the plot (if None, plot is displayed)
        """
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)
        
        plt.figure(figsize=(12, 6))
        
        x = dates if dates is not None else np.arange(len(y_true))
        
        plt.plot(x, y_true, label="Actual Returns", alpha=0.7)
        plt.plot(x, y_pred, label="Predicted Returns", alpha=0.7)
        
        plt.xlabel("Time" if dates is None else "Date")
        plt.ylabel("Returns")
        plt.title(title)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
            logging.info(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_cumulative_returns(self,
                               y_true: np.ndarray,
                               y_pred: np.ndarray,
                               dates: Optional[np.ndarray] = None,
                               title: str = "Cumulative Returns",
                               save_path: Optional[str] = None) -> None:
        """
        Plot cumulative actual and predicted returns.
        
        Args:
            y_true: True returns
            y_pred: Predicted returns
            dates: Dates for x-axis (if None, indices are used)
            title: Plot title
            save_path: Path to save the plot (if None, plot is displayed)
        """
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)
        
        cum_true = np.cumprod(1 + y_true) - 1
        cum_pred = np.cumprod(1 + y_pred) - 1
        
        plt.figure(figsize=(12, 6))
        
        x = dates if dates is not None else np.arange(len(y_true))
        
        plt.plot(x, cum_true, label="Actual Cumulative Returns", alpha=0.7)
        plt.plot(x, cum_pred, label="Predicted Cumulative Returns", alpha=0.7)
        
        plt.xlabel("Time" if dates is None else "Date")
        plt.ylabel("Cumulative Returns")
        plt.title(title)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
            logging.info(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def analyze_factor_importance(self, 
                                 X: np.ndarray, 
                                 y_true: np.ndarray,
                                 factor_names: List[str],
                                 title: str = "Factor Importance Analysis",
                                 save_path: Optional[str] = None) -> Dict[str, float]:
        """
        Analyze the importance of different factors.
        
        This is a simple analysis that measures how model performance changes
        when each factor is removed.
        
        Args:
            X: Input features
            y_true: True values
            factor_names: Names of factors (must match the number of features)
            title: Plot title
            save_path: Path to save the plot (if None, plot is displayed)
            
        Returns:
            Dictionary mapping factor names to importance scores
        """
        if len(X.shape) == 3:
            X_2d = X[:, -1, :]
        else:
            X_2d = X
        
        baseline_pred = self.predict(X)
        baseline_metrics = self.calculate_metrics(y_true, baseline_pred)
        baseline_mse = baseline_metrics['mse']
        
        importance_scores = {}
        
        for i, factor_name in enumerate(factor_names):
            X_modified = X_2d.copy()
            feature_mean = np.mean(X_modified[:, i])
            X_modified[:, i] = feature_mean
            
            modified_pred = self.predict(X_modified)
            modified_metrics = self.calculate_metrics(y_true, modified_pred)
            modified_mse = modified_metrics['mse']
            
            importance = (modified_mse - baseline_mse) / baseline_mse
            importance_scores[factor_name] = importance
        
        sorted_factors = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        
        plt.figure(figsize=(10, 6))
        
        names = [item[0] for item in sorted_factors]
        scores = [item[1] for item in sorted_factors]
        
        plt.barh(names, scores)
        
        plt.xlabel("Relative Importance")
        plt.ylabel("Factor")
        plt.title(title)
        
        if save_path:
            plt.savefig(save_path)
            logging.info(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        return importance_scores
