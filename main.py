import os
import argparse
import logging
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from src.data.data_loader import FinancialDataLoader
from src.models.deep_asset_pricing import DeepFactorNetwork, LSTMFactorNetwork, TemporalFactorNetwork, HybridFactorNetwork
from src.training.trainer import ModelTrainer
from src.evaluation.evaluator import ModelEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("asset_pricing_model.log"),
        logging.StreamHandler()
    ]
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Deep Learning for Asset Pricing')
    
    parser.add_argument('--tickers', type=str, default='AAPL,MSFT,GOOGL,AMZN,META',
                        help='Comma-separated list of stock tickers')
    parser.add_argument('--start_date', type=str, default='2010-01-01',
                        help='Start date for historical data (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default='2023-12-31',
                        help='End date for historical data (YYYY-MM-DD)')
    
    parser.add_argument('--model_type', type=str, default='lstm',
                        choices=['deep', 'lstm', 'temporal', 'hybrid'],
                        help='Type of model to use')
    parser.add_argument('--window_size', type=int, default=20,
                        help='Number of time steps to use as input features')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Dimension of hidden layers')
    
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for optimizer')
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience for early stopping')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for training if available')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'evaluate', 'predict'],
                        help='Mode to run the script in')
    
    return parser.parse_args()

def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def create_model(model_type, input_dim, window_size, hidden_dim):
    """Create a model based on the specified type."""
    if model_type == 'deep':
        return DeepFactorNetwork(
            input_dim=input_dim,
            hidden_dims=[hidden_dim, hidden_dim // 2],
            output_dim=1,
            dropout_rate=0.2
        )
    elif model_type == 'lstm':
        return LSTMFactorNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            output_dim=1,
            dropout_rate=0.2
        )
    elif model_type == 'temporal':
        return TemporalFactorNetwork(
            input_dim=input_dim,
            seq_len=window_size,
            num_filters=[64, 128, 64],
            kernel_sizes=[3, 3, 3],
            output_dim=1,
            dropout_rate=0.2
        )
    elif model_type == 'hybrid':
        temporal_dim = input_dim // 2
        static_dim = input_dim - temporal_dim
        return HybridFactorNetwork(
            temporal_input_dim=temporal_dim,
            static_input_dim=static_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            fc_dims=[32, 16],
            output_dim=1,
            dropout_rate=0.2
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def main():
    """Main function to run the asset pricing model."""
    args = parse_args()
    
    set_seed(args.seed)
    
    os.makedirs('data/cache', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    logging.info(f"Using device: {device}")
    
    tickers = args.tickers.split(',')
    
    data_loader = FinancialDataLoader(
        tickers=tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        cache_dir='data/cache'
    )
    
    logging.info("Downloading financial data...")
    data_loader.download_data()
    
    logging.info("Preparing data for model...")
    X_train, y_train, X_val, y_val, X_test, y_test = data_loader.prepare_data(
        window_size=args.window_size,
        prediction_horizon=1,
        train_ratio=0.7,
        val_ratio=0.15,
        include_factors=True
    )
    
    input_dim = X_train.shape[-1]
    logging.info(f"Input dimension: {input_dim}")
    
    logging.info(f"Creating {args.model_type} model...")
    model = create_model(args.model_type, input_dim, args.window_size, args.hidden_dim)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    if args.mode == 'train':
        # Configure loss function
        loss_fn = torch.nn.MSELoss()
        
        # Configure optimizer with learning rate scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=args.patience // 2
        )
        
        # Create trainer with enhanced configuration
        trainer = ModelTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            checkpoint_dir='checkpoints'
        )
        
        # Check for existing checkpoint to resume training
        checkpoint_path = os.path.join('checkpoints', f'{args.model_type}_latest.pt')
        if os.path.exists(checkpoint_path):
            logging.info(f"Resuming training from checkpoint: {checkpoint_path}")
            trainer.load_checkpoint(checkpoint_path)
        
        # Train model with enhanced monitoring
        logging.info(f"Training {args.model_type.upper()} model with {input_dim} input features...")
        logging.info(f"Training set size: {len(X_train)}, Validation set size: {len(X_val)}")
        
        start_time = datetime.now()
        history = trainer.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            verbose=True,
            save_best_only=True
        )
        training_time = datetime.now() - start_time
        
        # Log training results
        logging.info(f"Training completed in {training_time}")
        logging.info(f"Best validation loss: {min(history['val_loss']):.6f}")
        logging.info(f"Final training loss: {history['train_loss'][-1]:.6f}")
        
        # Save training metrics to CSV
        metrics_df = pd.DataFrame(history)
        metrics_df.to_csv(f'results/{args.model_type}_training_metrics.csv', index=False)
        
        # Plot training history with enhanced visualization
        plt.figure(figsize=(12, 8))
        
        # Plot loss curves
        plt.subplot(2, 1, 1)
        plt.plot(history['epoch'], history['train_loss'], 'b-', label='Training Loss')
        if 'val_loss' in history:
            plt.plot(history['epoch'], history['val_loss'], 'r-', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{args.model_type.upper()} Model Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot learning rate
        plt.subplot(2, 1, 2)
        plt.plot(history['epoch'], history['learning_rate'], 'g-')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'results/{args.model_type}_training_history.png', dpi=300)
        plt.close()
    
    # Load best model for evaluation
    checkpoint_path = os.path.join('checkpoints', 'best_model.pt')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"Loaded best model from {checkpoint_path}")
    
    # Create evaluator
    evaluator = ModelEvaluator(
        model=model,
        device=device,
        results_dir='results'
    )
    
    if args.mode in ['evaluate', 'train']:
        # Evaluate model on test set
        logging.info("Evaluating model on test set...")
        metrics = evaluator.evaluate(X_test, y_test)
        
        # Generate predictions
        y_pred = evaluator.predict(X_test)
        
        # Plot predictions
        evaluator.plot_predictions(
            y_true=y_test,
            y_pred=y_pred,
            title=f"{args.model_type.capitalize()} Model: Actual vs Predicted Returns",
            save_path='results/predictions.png'
        )
        
        # Plot returns over time
        evaluator.plot_returns_over_time(
            y_true=y_test,
            y_pred=y_pred,
            title=f"{args.model_type.capitalize()} Model: Returns Over Time",
            save_path='results/returns_over_time.png'
        )
        
        # Plot cumulative returns
        evaluator.plot_cumulative_returns(
            y_true=y_test,
            y_pred=y_pred,
            title=f"{args.model_type.capitalize()} Model: Cumulative Returns",
            save_path='results/cumulative_returns.png'
        )
        
        # Analyze factor importance
        factor_names = [f"Factor_{i}" for i in range(X_test.shape[-1])]
        evaluator.analyze_factor_importance(
            X=X_test,
            y_true=y_test,
            factor_names=factor_names,
            title=f"{args.model_type.capitalize()} Model: Factor Importance",
            save_path='results/factor_importance.png'
        )
    
    if args.mode == 'predict':
        # This is a placeholder for making predictions on new data
        logging.info("Prediction mode not fully implemented yet")

if __name__ == "__main__":
    main()
