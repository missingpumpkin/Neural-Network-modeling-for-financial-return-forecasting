import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
import logging
import time
import os
from tqdm import tqdm

class ModelTrainer:
    """
    Trainer class for deep learning models for asset pricing.
    
    This class handles the training, validation, and early stopping
    of deep learning models for financial return forecasting.
    """
    
    def __init__(self,
                model: nn.Module,
                optimizer: optim.Optimizer = None,
                loss_fn: nn.Module = None,
                device: Optional[str] = None,
                checkpoint_dir: str = "checkpoints"):
        """
        Initialize the model trainer.
        
        Args:
            model: PyTorch model to train
            optimizer: PyTorch optimizer (default: Adam with lr=0.001)
            loss_fn: PyTorch loss function (default: MSELoss)
            device: Device to use for training ('cuda' or 'cpu')
            checkpoint_dir: Directory to save model checkpoints
        """
        self.model = model
        
        self.optimizer = optimizer or optim.Adam(model.parameters(), lr=0.001)
        
        self.loss_fn = loss_fn or nn.MSELoss()
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch': []
        }
    
    def prepare_data_loaders(self,
                           X_train: np.ndarray,
                           y_train: np.ndarray,
                           X_val: Optional[np.ndarray] = None,
                           y_val: Optional[np.ndarray] = None,
                           batch_size: int = 32,
                           shuffle: bool = True) -> Tuple[DataLoader, Optional[DataLoader]]:
        """
        Prepare PyTorch DataLoaders for training and validation.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            batch_size: Batch size for training
            shuffle: Whether to shuffle the training data
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train the model for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            self.optimizer.zero_grad()
            y_pred = self.model(X_batch)
            loss = self.loss_fn(y_pred, y_batch)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate the model on validation data.
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                y_pred = self.model(X_batch)
                loss = self.loss_fn(y_pred, y_batch)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        
        return avg_loss
    
    def train(self,
             X_train: np.ndarray,
             y_train: np.ndarray,
             X_val: Optional[np.ndarray] = None,
             y_val: Optional[np.ndarray] = None,
             epochs: int = 100,
             batch_size: int = 32,
             patience: int = 10,
             min_delta: float = 0.001,
             verbose: bool = True,
             save_best_only: bool = True) -> Dict[str, List[float]]:
        """
        Train the model with early stopping.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            epochs: Maximum number of epochs to train
            batch_size: Batch size for training
            patience: Number of epochs with no improvement to wait before stopping
            min_delta: Minimum change in validation loss to qualify as improvement
            verbose: Whether to print progress
            save_best_only: Whether to save only the best model
            
        Returns:
            Training history dictionary
        """
        train_loader, val_loader = self.prepare_data_loaders(
            X_train, y_train, X_val, y_val, batch_size
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            start_time = time.time()
            
            train_loss = self.train_epoch(train_loader)
            
            val_loss = None
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    if save_best_only:
                        self.save_checkpoint(f"best_model.pt")
                else:
                    patience_counter += 1
            
            if not save_best_only:
                self.save_checkpoint(f"model_epoch_{epoch}.pt")
            
            self.history['train_loss'].append(train_loss)
            self.history['epoch'].append(epoch)
            self.history['learning_rate'].append(self.get_lr())
            
            if val_loss is not None:
                self.history['val_loss'].append(val_loss)
            
            if verbose:
                epoch_time = time.time() - start_time
                val_str = f", val_loss: {val_loss:.6f}" if val_loss is not None else ""
                print(f"Epoch {epoch}/{epochs} - {epoch_time:.2f}s - train_loss: {train_loss:.6f}{val_str}")
            
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
        
        return self.history
    
    def get_lr(self) -> float:
        """
        Get the current learning rate.
        
        Returns:
            Current learning rate
        """
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
        return 0.0  # Default return if no param groups (should never happen)
    
    def save_checkpoint(self, filename: str) -> None:
        """
        Save model checkpoint.
        
        Args:
            filename: Name of checkpoint file
        """
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, filepath)
        logging.info(f"Saved checkpoint to {filepath}")
    
    def load_checkpoint(self, filename: str) -> None:
        """
        Load model checkpoint.
        
        Args:
            filename: Name of checkpoint file
        """
        filepath = os.path.join(self.checkpoint_dir, filename)
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.history = checkpoint['history']
            logging.info(f"Loaded checkpoint from {filepath}")
        else:
            logging.error(f"Checkpoint file {filepath} not found")
