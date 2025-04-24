import os
import argparse
import logging
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import time
from typing import Dict, List, Tuple, Optional, Union

from src.data.data_loader import FinancialDataLoader
from src.models.deep_asset_pricing import DeepFactorNetwork, LSTMFactorNetwork, TemporalFactorNetwork, HybridFactorNetwork
from src.training.trainer import ModelTrainer
from src.evaluation.evaluator import ModelEvaluator

# Configure logging with enhanced formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("asset_pricing_model.log"),
        logging.StreamHandler()
    ]
)

def parse_args():
    """Parse command line arguments with enhanced options."""
    parser = argparse.ArgumentParser(description='Deep Learning for Asset Pricing')
    
    # Data parameters
    data_group = parser.add_argument_group('Data Parameters')
    data_group.add_argument('--tickers', type=str, default='AAPL,MSFT,GOOGL,AMZN,META',
                        help='Comma-separated list of stock tickers')
    data_group.add_argument('--start_date', type=str, default='2010-01-01',
                        help='Start date for historical data (YYYY-MM-DD)')
    data_group.add_argument('--end_date', type=str, default='2023-12-31',
                        help='End date for historical data (YYYY-MM-DD)')
    data_group.add_argument('--window_size', type=int, default=20,
                        help='Number of time steps to use as input features')
    data_group.add_argument('--prediction_horizon', type=int, default=1,
                        help='Number of time steps ahead to predict')
    data_group.add_argument('--train_ratio', type=float, default=0.7,
                        help='Ratio of data to use for training')
    data_group.add_argument('--val_ratio', type=float, default=0.15,
                        help='Ratio of data to use for validation')
    data_group.add_argument('--normalize', action='store_true', default=True,
                        help='Whether to normalize features')
    
    # Model parameters
    model_group = parser.add_argument_group('Model Parameters')
    model_group.add_argument('--model_type', type=str, default='lstm',
                        choices=['deep', 'lstm', 'temporal', 'hybrid'],
                        help='Type of model to use')
    model_group.add_argument('--hidden_dim', type=int, default=64,
                        help='Dimension of hidden layers')
    model_group.add_argument('--num_layers', type=int, default=2,
                        help='Number of layers in LSTM/Hybrid models')
    model_group.add_argument('--dropout_rate', type=float, default=0.2,
                        help='Dropout rate for regularization')
    
    # Training parameters
    train_group = parser.add_argument_group('Training Parameters')
    train_group.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train')
    train_group.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    train_group.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for optimizer')
    train_group.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for regularization')
    train_group.add_argument('--patience', type=int, default=10,
                        help='Patience for early stopping')
    train_group.add_argument('--clip_grad', type=float, default=1.0,
                        help='Gradient clipping value (0 to disable)')
    train_group.add_argument('--scheduler', type=str, default='plateau',
                        choices=['plateau', 'cosine', 'step', 'none'],
                        help='Learning rate scheduler type')
    
    # System parameters
    sys_group = parser.add_argument_group('System Parameters')
    sys_group.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    sys_group.add_argument('--gpu', action='store_true',
                        help='Use GPU for training if available')
    sys_group.add_argument('--mode', type=str, default='train',
                        choices=['train', 'evaluate', 'predict', 'hyperopt'],
                        help='Mode to run the script in')
    sys_group.add_argument('--log_interval', type=int, default=10,
                        help='Interval for logging training progress')
    sys_group.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save model checkpoints')
    sys_group.add_argument('--results_dir', type=str, default='results',
                        help='Directory to save results')
    
    return parser.parse_args()

def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across all libraries."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def create_model(model_type: str, input_dim: int, window_size: int, 
                hidden_dim: int, num_layers: int, dropout_rate: float) -> torch.nn.Module:
    """Create a model based on the specified type with enhanced configuration."""
    if model_type == 'deep':
        return DeepFactorNetwork(
            input_dim=input_dim,
            hidden_dims=[hidden_dim, hidden_dim // 2],
            output_dim=1,
            dropout_rate=dropout_rate,
            use_batch_norm=True
        )
    elif model_type == 'lstm':
        return LSTMFactorNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=1,
            dropout_rate=dropout_rate,
            bidirectional=False
        )
    elif model_type == 'temporal':
        return TemporalFactorNetwork(
            input_dim=input_dim,
            seq_len=window_size,
            num_filters=[64, 128, 64],
            kernel_sizes=[3, 3, 3],
            output_dim=1,
            dropout_rate=dropout_rate
        )
    elif model_type == 'hybrid':
        # For hybrid model, split features into temporal and static
        # This is a heuristic approach - in practice, domain knowledge should guide this split
        temporal_dim = input_dim // 2
        static_dim = input_dim - temporal_dim
        return HybridFactorNetwork(
            temporal_input_dim=temporal_dim,
            static_input_dim=static_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            fc_dims=[32, 16],
            output_dim=1,
            dropout_rate=dropout_rate
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def create_optimizer(model: torch.nn.Module, args: argparse.Namespace) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler]]:
    """Create optimizer and learning rate scheduler with enhanced options."""
    # Create optimizer with weight decay
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Create learning rate scheduler based on user choice
    if args.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=args.patience // 2,
            verbose=True
        )
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.learning_rate / 10
        )
    elif args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.patience,
            gamma=0.5
        )
    else:  # 'none'
        scheduler = None
    
    return optimizer, scheduler

def train_model(model: torch.nn.Module, data_loader: FinancialDataLoader, 
               args: argparse.Namespace, device: torch.device) -> Dict:
    """Train the model with enhanced monitoring and features."""
    # Configure loss function
    loss_fn = torch.nn.MSELoss()
    
    # Configure optimizer and scheduler
    optimizer, scheduler = create_optimizer(model, args)
    
    # Create trainer with enhanced configuration
    trainer = ModelTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        clip_grad_value=args.clip_grad if args.clip_grad > 0 else None,
        scheduler=scheduler
    )
    
    # Check for existing checkpoint to resume training
    checkpoint_path = os.path.join(args.checkpoint_dir, f'{args.model_type}_latest.pt')
    if os.path.exists(checkpoint_path):
        logging.info(f"Resuming training from checkpoint: {checkpoint_path}")
        trainer.load_checkpoint(checkpoint_path)
    
    # Train model with enhanced monitoring
    logging.info(f"Training {args.model_type.upper()} model with {data_loader.X_train.shape[-1]} input features...")
    logging.info(f"Training set size: {len(data_loader.X_train)}, Validation set size: {len(data_loader.X_val)}")
    
    start_time = time.time()
    history = trainer.train(
        X_train=data_loader.X_train,
        y_train=data_loader.y_train,
        X_val=data_loader.X_val,
        y_val=data_loader.y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        verbose=True,
        save_best_only=True,
        log_interval=args.log_interval
    )
    training_time = time.time() - start_time
    
    # Log training results
    logging.info(f"Training completed in {training_time:.2f} seconds")
    logging.info(f"Best validation loss: {min(history['val_loss']):.6f}")
    logging.info(f"Final training loss: {history['train_loss'][-1]:.6f}")
    
    # Save training metrics to CSV
    metrics_df = pd.DataFrame(history)
    metrics_df.to_csv(f'{args.results_dir}/{args.model_type}_training_metrics.csv', index=False)
    
    # Plot training history with enhanced visualization
    visualize_training_history(history, args)
    
    return history

def visualize_training_history(history: Dict, args: argparse.Namespace) -> None:
    """Create enhanced visualizations of training history."""
    plt.figure(figsize=(12, 10))
    
    # Plot loss curves
    plt.subplot(3, 1, 1)
    plt.plot(history['epoch'], history['train_loss'], 'b-', label='Training Loss')
    if 'val_loss' in history:
        plt.plot(history['epoch'], history['val_loss'], 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{args.model_type.upper()} Model Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot learning rate
    plt.subplot(3, 1, 2)
    plt.plot(history['epoch'], history['learning_rate'], 'g-')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True, alpha=0.3)
    
    # Plot training time per epoch
    if 'epoch_time' in history:
        plt.subplot(3, 1, 3)
        plt.plot(history['epoch'], history['epoch_time'], 'm-')
        plt.xlabel('Epoch')
        plt.ylabel('Time (seconds)')
        plt.title('Training Time per Epoch')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{args.results_dir}/{args.model_type}_training_history.png', dpi=300)
    plt.close()

def evaluate_model(model: torch.nn.Module, data_loader: FinancialDataLoader, 
                  args: argparse.Namespace, device: torch.device) -> Dict:
    """Evaluate the model with enhanced metrics and visualizations."""
    # Create evaluator
    evaluator = ModelEvaluator(
        model=model,
        device=device,
        results_dir=args.results_dir
    )
    
    # Evaluate model on test set
    logging.info("Evaluating model on test set...")
    metrics = evaluator.evaluate(data_loader.X_test, data_loader.y_test)
    
    # Generate predictions
    y_pred = evaluator.predict(data_loader.X_test)
    
    # Plot predictions
    evaluator.plot_predictions(
        y_true=data_loader.y_test,
        y_pred=y_pred,
        title=f"{args.model_type.capitalize()} Model: Actual vs Predicted Returns",
        save_path=f'{args.results_dir}/{args.model_type}_predictions.png'
    )
    
    # Plot returns over time
    evaluator.plot_returns_over_time(
        y_true=data_loader.y_test,
        y_pred=y_pred,
        title=f"{args.model_type.capitalize()} Model: Returns Over Time",
        save_path=f'{args.results_dir}/{args.model_type}_returns_over_time.png'
    )
    
    # Plot cumulative returns
    evaluator.plot_cumulative_returns(
        y_true=data_loader.y_test,
        y_pred=y_pred,
        title=f"{args.model_type.capitalize()} Model: Cumulative Returns",
        save_path=f'{args.results_dir}/{args.model_type}_cumulative_returns.png'
    )
    
    # Analyze factor importance
    factor_names = [f"Factor_{i}" for i in range(data_loader.X_test.shape[-1])]
    evaluator.analyze_factor_importance(
        X=data_loader.X_test,
        y_true=data_loader.y_test,
        factor_names=factor_names,
        title=f"{args.model_type.capitalize()} Model: Factor Importance",
        save_path=f'{args.results_dir}/{args.model_type}_factor_importance.png'
    )
    
    return metrics

def predict_with_model(model: torch.nn.Module, data_loader: FinancialDataLoader, 
                      args: argparse.Namespace, device: torch.device) -> np.ndarray:
    """Generate predictions using the trained model."""
    # Create evaluator
    evaluator = ModelEvaluator(
        model=model,
        device=device,
        results_dir=args.results_dir
    )
    
    # Generate predictions on test data
    logging.info("Generating predictions...")
    y_pred = evaluator.predict(data_loader.X_test)
    
    # Save predictions to CSV
    pred_df = pd.DataFrame({
        'actual': data_loader.y_test,
        'predicted': y_pred.flatten()
    })
    pred_df.to_csv(f'{args.results_dir}/{args.model_type}_predictions.csv', index=False)
    
    return y_pred

def main():
    """Main function to run the asset pricing model with enhanced organization."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Create necessary directories
    os.makedirs('data/cache', exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Parse tickers
    tickers = args.tickers.split(',')
    
    # Initialize data loader
    data_loader = FinancialDataLoader(
        tickers=tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        cache_dir='data/cache'
    )
    
    # Download and prepare data
    logging.info("Downloading financial data...")
    data_loader.download_data()
    
    logging.info("Preparing data for model...")
    data_loader.prepare_data(
        window_size=args.window_size,
        prediction_horizon=args.prediction_horizon,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        include_factors=True,
        normalize=args.normalize
    )
    
    # Get input dimension from prepared data
    input_dim = data_loader.X_train.shape[-1]
    logging.info(f"Input dimension: {input_dim}")
    
    # Create model
    logging.info(f"Creating {args.model_type} model...")
    model = create_model(
        model_type=args.model_type, 
        input_dim=input_dim, 
        window_size=args.window_size, 
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout_rate=args.dropout_rate
    )
    model.to(device)
    
    # Execute based on mode
    if args.mode == 'train':
        # Train model
        history = train_model(model, data_loader, args, device)
        
        # Load best model for evaluation
        best_checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
        if os.path.exists(best_checkpoint_path):
            checkpoint = torch.load(best_checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logging.info(f"Loaded best model from {best_checkpoint_path}")
        
        # Evaluate model
        metrics = evaluate_model(model, data_loader, args, device)
        
    elif args.mode == 'evaluate':
        # Load best model
        best_checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
        if os.path.exists(best_checkpoint_path):
            checkpoint = torch.load(best_checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logging.info(f"Loaded best model from {best_checkpoint_path}")
        else:
            logging.warning("No checkpoint found. Using untrained model for evaluation.")
        
        # Evaluate model
        metrics = evaluate_model(model, data_loader, args, device)
        
    elif args.mode == 'predict':
        # Load best model
        best_checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
        if os.path.exists(best_checkpoint_path):
            checkpoint = torch.load(best_checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logging.info(f"Loaded best model from {best_checkpoint_path}")
        else:
            logging.warning("No checkpoint found. Using untrained model for prediction.")
        
        # Generate predictions
        y_pred = predict_with_model(model, data_loader, args, device)
        
    elif args.mode == 'hyperopt':
        logging.info("Hyperparameter optimization mode not fully implemented yet")
        # This would be implemented with libraries like Optuna or Ray Tune
        # for automated hyperparameter optimization
    
    logging.info("Process completed successfully")

if __name__ == "__main__":
    main()
