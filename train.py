import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import os
import time

try:
    # Import custom Dataset and collate_fn from data_preprocessing module
    from data_preprocessing import load_datasets, StoreLocationDataset, collate_fn
    from model import SiteSelectionModel
    from evaluate import compute_metrics
except ImportError as e:
    print(f"Error importing module: {e}")
    print("Ensure data_preprocessing.py, model.py, evaluate.py are in the same directory or in PYTHONPATH.")
    raise

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create necessary directories for saving models and cached data
os.makedirs("data", exist_ok=True)
os.makedirs("cache", exist_ok=True)

# Default training parameters
DEFAULT_HIDDEN_DIM = 128  # Hidden layer dimension (Reduced)
DEFAULT_LSTM_LAYERS = 1
DEFAULT_DROPOUT = 0.4  # Dropout to prevent overfitting (Increased)
DEFAULT_LR = 1e-4  # Initial learning rate
DEFAULT_WEIGHT_DECAY = 1e-5  # L2 regularization
DEFAULT_EPOCHS = 50  # Number of training epochs, combined with early stopping
DEFAULT_PATIENCE = 15  # Early stopping patience: number of consecutive epochs without validation loss improvement (Increased)
DEFAULT_BATCH_SIZE = 32  # Batch size
K_NEIGHBORS_DENSITY = 5  # Number of k-neighbors used for pseudo-time series generation in data preprocessing


def train_model(train_samples, val_samples, num_total_classes, grid_embed_map, brand_embed_map,
                brand_embed_dim, grid_input_dim,
                hidden_dim=DEFAULT_HIDDEN_DIM, lstm_layers=DEFAULT_LSTM_LAYERS,
                dropout=DEFAULT_DROPOUT, lr=DEFAULT_LR, weight_decay=DEFAULT_WEIGHT_DECAY,
                epochs=DEFAULT_EPOCHS, patience=DEFAULT_PATIENCE, device_name='cpu',
                model_save_path="best_model_semantic.pth"):
    """
    Trains the store location prediction model.
    Parameters:
        train_samples (list): List of training dataset samples.
        val_samples (list): List of validation dataset samples.
        num_total_classes (int): Total number of grid IDs (i.e., dimension of model output layer).
        grid_embed_map (dict): Mapping from grid ID to its combined embedding vector.
        brand_embed_map (dict): Mapping from brand key to its embedding vector.
        brand_embed_dim (int): Dimension of brand embeddings.
        grid_input_dim (int): Dimension of grid input features.
        hidden_dim (int): LSTM hidden layer dimension.
        lstm_layers (int): Number of LSTM layers.
        dropout (float): Dropout rate.
        lr (float): Learning rate.
        weight_decay (float): Weight decay (L2 regularization).
        epochs (int): Maximum number of training epochs.
        patience (int): Early stopping patience value.
        device_name (str): Training device ('cuda' or 'cpu').
        model_save_path (str): Path to save the best model.
    Returns:
        torch.nn.Module: The trained best model.
    """
    # Set training device
    device = torch.device(device_name if torch.cuda.is_available() and device_name == 'cuda' else 'cpu')
    print(f"Using device: {device}")

    # Initialize model
    model = SiteSelectionModel(
        num_grids=num_total_classes,  # Dimension of model output layer
        brand_embed_dim=brand_embed_dim,
        grid_input_dim=grid_input_dim,
        hidden_dim=hidden_dim,
        lstm_layers=lstm_layers,
        dropout=dropout
    ).to(device)  # Move model to specified device

    # Define loss function, optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for multi-class classification
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)  # T_max is total training epochs

    # Create DataLoaders for efficient batch loading
    train_dataset = StoreLocationDataset(train_samples, grid_embed_map, brand_embed_map)
    val_dataset = StoreLocationDataset(val_samples, grid_embed_map, brand_embed_map)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=DEFAULT_BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=DEFAULT_BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    best_val_loss = float('inf')  # Initialize best validation loss to infinity
    epochs_no_improve = 0  # Counter for epochs without validation loss improvement
    trained_model = None # To store the best model instance

    print("Starting training...")
    for epoch in range(1, epochs + 1):
        model.train()  # Set model to training mode
        total_loss = 0
        start_time = time.time()  # Record start time of each epoch

        # Iterate through training data loader
        # Unpack batch_pseudo_seq_gids, although not directly used in training loss calculation
        for batch_idx, (brand_vecs, seq_grid_embeds_padded, target_idxs, seq_lengths, _) in enumerate(train_loader):
            # Move data to specified device
            brand_vecs = brand_vecs.to(device)
            seq_grid_embeds_padded = seq_grid_embeds_padded.to(device)
            target_idxs = target_idxs.to(device)  # Original grid_id
            seq_lengths = seq_lengths.to(device)

            optimizer.zero_grad()  # Clear gradients

            # Forward pass
            logits = model(brand_vecs, seq_grid_embeds_padded, seq_lengths)

            # Calculate loss
            # CrossEntropyLoss expects target_idxs to be 0-indexed if num_classes is total grids.
            # If grid_ids are 1-indexed, adjust them here: target_idxs - 1
            # Assuming grid_ids are already 0-indexed or handled by the model's output layer
            loss = criterion(logits, target_idxs)
            total_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()

        scheduler.step()  # Update learning rate

        avg_loss = total_loss / len(train_loader)  # Calculate average training loss

        # Validation phase
        model.eval()  # Set model to evaluation mode
        # Call compute_metrics function to calculate validation metrics
        # Pass num_total_classes and mask_existing_grids=True for masking
        val_metrics = compute_metrics(
            model,
            val_loader,
            device,
            num_total_classes=num_total_classes,
            mask_existing_grids=True
        )
        val_loss = val_metrics['val_loss']
        acc_at_1 = val_metrics['acc_at_k'][1]
        acc_at_5 = val_metrics['acc_at_k'][5]
        acc_at_10 = val_metrics['acc_at_k'][10]
        mrr = val_metrics['mrr']

        end_time = time.time()
        epoch_duration = end_time - start_time

        print(
            f"Epoch {epoch}/{epochs} ({epoch_duration:.2f}s): Train Loss = {avg_loss:.4f}, Val Loss = {val_loss:.4f}, "
            f"Hit@1 = {acc_at_1:.3f}, Hit@5 = {acc_at_5:.3f}, Hit@10 = {acc_at_10:.3f}, MRR = {mrr:.3f}")

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_path)  # Save current best model
            trained_model = model # Store reference to the best model
            print(f"Saving best model, validation loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            print(f"Validation loss did not improve, sustained for {epochs_no_improve} epochs.")
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    print("Training complete.")
    # Load the best model weights for final evaluation if it was saved
    if trained_model is None and os.path.exists(model_save_path):
        print(f"Loading best model from: {model_save_path}")
        trained_model = SiteSelectionModel(
            num_grids=num_total_classes,
            brand_embed_dim=brand_embed_dim,
            grid_input_dim=grid_input_dim,
            hidden_dim=hidden_dim,
            lstm_layers=lstm_layers,
            dropout=dropout
        )
        trained_model.load_state_dict(torch.load(model_save_path, map_location=device))
        trained_model.to(device)
    elif trained_model is None:
        print("Warning: No best model found. Returning the last trained model.")
        trained_model = model

    return trained_model


if __name__ == '__main__':
    # Define CSV file paths
    train_csv_path = 'data/train_data.csv'
    test_csv_path = 'data/test_data.csv'
    grid_csv_path = 'data/grid_coordinates_with_summary.csv'
    model_output_path = 'data/best_model_semantic.pth'  # Path to save the best model

    USE_CUDA_IF_AVAILABLE = True  # Set to True to prefer CUDA, otherwise use CPU

    # Load and preprocess data
    # load_datasets returns train/val/test samples, grid and brand embedding maps, total classes, and embedding dimensions
    train_samples, val_samples, test_samples, num_total_classes, grid_embed_map, brand_embed_map, brand_embed_dim, grid_combined_embed_dim = \
        load_datasets(train_csv_path, test_csv_path, grid_csv_path, K_NEIGHBORS_DENSITY)

    # Data validity check
    if not train_samples or not val_samples:
        print("Error: Training or validation samples are empty. Script terminated.")
        exit()
    if num_total_classes == 0:
        print("Error: Number of classes is 0. Script terminated.")
        exit()
    if not brand_embed_map:
        print("Error: Brand embedding map is empty. Script terminated.")
        exit()
    if not grid_embed_map:
        print("Error: Grid embedding map is empty. Script terminated.")
        exit()

    print(f"Data loading complete. Total classes (num_grids): {num_total_classes}")
    print(f"Number of training samples: {len(train_samples)}, Number of validation samples: {len(val_samples)}, Number of test samples: {len(test_samples)}")
    print(f"Brand embedding dimension: {brand_embed_dim}, Grid input feature dimension: {grid_combined_embed_dim}")

    # Train model
    trained_model = train_model(
        train_samples,
        val_samples,
        num_total_classes,
        grid_embed_map,
        brand_embed_map,
        brand_embed_dim=brand_embed_dim,
        grid_input_dim=grid_combined_embed_dim,
        hidden_dim=DEFAULT_HIDDEN_DIM,
        lstm_layers=DEFAULT_LSTM_LAYERS,
        dropout=DEFAULT_DROPOUT,
        lr=DEFAULT_LR,
        weight_decay=DEFAULT_WEIGHT_DECAY,
        epochs=DEFAULT_EPOCHS,
        patience=DEFAULT_PATIENCE,
        device_name='cuda' if USE_CUDA_IF_AVAILABLE else 'cpu',
        model_save_path=model_output_path
    )

    # Evaluate final model on test set
    if trained_model and test_samples:
        print("\nEvaluating final model on test set...")
        device_for_eval = torch.device('cuda' if USE_CUDA_IF_AVAILABLE and torch.cuda.is_available() else 'cpu')
        trained_model.to(device_for_eval)  # Move model to evaluation device

        # Create test DataLoader
        test_dataset = StoreLocationDataset(test_samples, grid_embed_map, brand_embed_map)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=DEFAULT_BATCH_SIZE, shuffle=False, collate_fn=collate_fn
        )

        # Calculate final test metrics
        final_metrics = compute_metrics(
            trained_model,
            test_loader,
            device_for_eval,
            num_total_classes=num_total_classes,
            mask_existing_grids=True # Enable masking for test set
        )

        print("\nFinal test set evaluation results:")
        print(f"  Test Loss: {final_metrics['val_loss']:.4f}")
        for k, acc in final_metrics['acc_at_k'].items():
            print(f"  Hit@{k}: {acc:.3f}")
        print(f"  MRR: {final_metrics['mrr']:.3f}")
    else:
        print("Warning: Cannot evaluate model on test set, either no trained model or no test samples.")
