import torch
import numpy as np
import torch.nn as nn  # Import nn module to use CrossEntropyLoss


def compute_metrics(model, data_loader, device, k_values=[1, 5, 10], num_total_classes=None, mask_existing_grids=False):
    """
    Calculates evaluation metrics: Top-K accuracy and MRR.
    Parameters:
        model (torch.nn.Module): The trained model.
        data_loader (iterable): Data loader, each iteration returns (brand_vecs, seq_grid_embeds_padded, target_idxs, seq_lengths, batch_pseudo_seq_gids).
        device (torch.device): 'cuda' or 'cpu'.
        k_values (list of int): List of K values for Top-K accuracy calculation.
        num_total_classes (int, optional): Total number of possible grid classes. Required if mask_existing_grids is True.
        mask_existing_grids (bool): If True, logits corresponding to grids already in the sequence will be masked (set to a very low value).
    Returns:
        dict: Contains acc_at_k (dict), mrr (float), val_loss (float).
    """
    model.eval()  # Set model to evaluation mode
    correct_at_k = {k: 0 for k in k_values}  # Initialize Top-K hit counts
    reciprocal_ranks = []  # Store reciprocal ranks for each sample
    total_samples = 0  # Record total number of samples
    total_loss = 0  # Record total loss

    criterion = nn.CrossEntropyLoss()  # Use the same loss function as training to calculate validation loss

    if mask_existing_grids and num_total_classes is None:
        raise ValueError("num_total_classes must be provided if mask_existing_grids is True.")

    with torch.no_grad():  # Disable gradient calculation during evaluation
        # Unpack batch_pseudo_seq_gids from data_loader
        for batch_idx, (brand_vecs, seq_grid_embeds_padded, target_idxs, seq_lengths,
                        batch_pseudo_seq_gids) in enumerate(data_loader):
            # Move data to specified device
            brand_vecs = brand_vecs.to(device)
            seq_grid_embeds_padded = seq_grid_embeds_padded.to(device)
            target_idxs = target_idxs.to(device)  # Original grid_id
            seq_lengths = seq_lengths.to(device)

            # Forward pass
            logits = model(brand_vecs, seq_grid_embeds_padded, seq_lengths)

            # --- Masking Logic for "No Overlap" Constraint ---
            if mask_existing_grids:
                # Create a mask for each sample in the batch
                mask = torch.zeros_like(logits, dtype=torch.bool)  # Initialize with False
                for i, pseudo_seq_gids in enumerate(batch_pseudo_seq_gids):
                    # Convert pseudo_seq_gids to a tensor and ensure they are valid indices
                    # Filter out any grid_ids that might be out of bounds for the logits tensor
                    valid_gids = [gid for gid in pseudo_seq_gids if 0 <= gid < num_total_classes]
                    if valid_gids:
                        # Set positions corresponding to existing grids to True in the mask
                        mask[i, valid_gids] = True

                # Apply the mask: set logits of existing grids to a very small negative number
                # This makes their probability close to zero after softmax
                logits[mask] = -1e9  # Use a sufficiently small number

            # Calculate validation loss
            loss = criterion(logits, target_idxs)
            total_loss += loss.item() * brand_vecs.size(0)  # Accumulate loss, multiplied by batch size

            # Convert logits to probability distribution
            probs = torch.softmax(logits, dim=-1)

            # Iterate through each sample in the batch
            for i in range(brand_vecs.size(0)):
                true_idx = target_idxs[i].item()  # Get the true target grid ID

                # Get predicted rankings sorted by probability in descending order
                ranking = torch.argsort(probs[i], descending=True)

                # Find the position (rank) of the true target in the sorted ranking (1-indexed)
                # Use .cpu().numpy() to ensure operations are on CPU to avoid complex indexing issues with CUDA tensors
                # Find where the true_idx is in the ranking
                rank_array = (ranking == true_idx).nonzero(as_tuple=True)[0].cpu().numpy()
                if rank_array.size > 0:  # Ensure target was found
                    rank = rank_array[0].item() + 1  # +1 to convert to 1-indexed rank
                else:  # If target not found in predictions (should not happen under normal circumstances)
                    rank = num_total_classes  # Set to max rank

                # Update Top-K hit counts
                for k in k_values:
                    if rank <= k:
                        correct_at_k[k] += 1

                # Update MRR (Mean Reciprocal Rank)
                reciprocal_ranks.append(1.0 / rank)

            total_samples += brand_vecs.size(0)  # Accumulate total samples

    # Calculate final metrics
    acc_at_k_final = {k: correct_at_k[k] / total_samples if total_samples > 0 else 0.0 for k in k_values}
    mrr_final = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    val_loss_avg = total_loss / total_samples if total_samples > 0 else 0.0

    return {'acc_at_k': acc_at_k_final, 'mrr': mrr_final, 'val_loss': val_loss_avg}


if __name__ == '__main__':
    print("evaluate.py: This script primarily provides the compute_metrics function.")
    print("To perform independent testing, you need to:")
    print("1. Load a trained model.")
    print("2. Prepare a data loader (e.g., validation or test set).")
    print("3. Call compute_metrics(model, data_loader, device).")

    # Example: Create a dummy model and data for testing
    # Requires importing model and data preprocessing related classes and functions
    from model import SiteSelectionModel
    from data_preprocessing import StoreLocationDataset, collate_fn, load_datasets

    # Dummy data loading (requires actual files to exist)
    train_csv_path = 'data/train_data.csv'
    test_csv_path = 'data/test_data.csv'
    grid_csv_path = 'data/grid_coordinates_with_summary.csv'
    K_NEIGHBORS_DENSITY = 5  # Must match the k value used during training

    try:
        # Try to load datasets, will raise an exception if files don't exist or processing fails
        train_samples, val_samples, test_samples, num_total_classes, grid_embed_map, brand_embed_map, brand_embed_dim, grid_combined_embed_dim = \
            load_datasets(train_csv_path, test_csv_path, grid_csv_path, K_NEIGHBORS_DENSITY)
    except Exception as e:
        print(f"Failed to load datasets, cannot proceed with independent test example: {e}")
        print("Please ensure train_data.csv, test_data.csv, grid_coordinates_with_summary.csv files exist.")
        exit()

    if not test_samples:
        print("No test samples, cannot perform evaluation.")
        exit()

    device_mock = torch.device("cpu")  # Example test uses CPU

    # Dummy model parameters (must be consistent with parameters used during training)
    DEFAULT_HIDDEN_DIM = 128  # Consistent with train.py
    DEFAULT_LSTM_LAYERS = 1
    DEFAULT_DROPOUT = 0.4  # Consistent with train.py

    # Ensure brand_embed_dim_mock is obtained from actual data, or provide a reasonable default
    if brand_embed_map:
        brand_embed_dim_mock = list(brand_embed_map.values())[0].shape[0]
    else:
        brand_embed_dim_mock = 768  # Fallback if brand_embed_map is empty (BERT base output dim)

    # Initialize dummy model
    mock_model = SiteSelectionModel(
        num_grids=num_total_classes,
        brand_embed_dim=brand_embed_dim_mock,
        grid_input_dim=grid_combined_embed_dim,
        hidden_dim=DEFAULT_HIDDEN_DIM,
        lstm_layers=DEFAULT_LSTM_LAYERS,
        dropout=DEFAULT_DROPOUT
    ).to(device_mock)

    # Create dummy data loader
    mock_test_dataset = StoreLocationDataset(test_samples, grid_embed_map, brand_embed_map)
    mock_test_loader = torch.utils.data.DataLoader(
        mock_test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn
    )

    print("\nRunning dummy evaluation...")
    # Pass num_total_classes and mask_existing_grids=True for dummy evaluation
    mock_metrics = compute_metrics(mock_model, mock_test_loader, device_mock, num_total_classes=num_total_classes,
                                   mask_existing_grids=True)
    print("Dummy evaluation results:")
    print(f"  Dummy Test Loss: {mock_metrics['val_loss']:.4f}")
    for k, acc in mock_metrics['acc_at_k'].items():
        print(f"  Dummy Hit@{k}: {acc:.3f}")
    print(f"  Dummy MRR: {mock_metrics['mrr']:.3f}")
