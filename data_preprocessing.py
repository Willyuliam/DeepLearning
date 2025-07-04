import pandas as pd
import numpy as np
import ast
from sklearn.metrics.pairwise import euclidean_distances
from transformers import AutoTokenizer, AutoModel
import torch
import os
import time
from utils import get_cache_path, save_cache, load_cache, check_cache_freshness

# Global constants
MODEL_NAME = "shibing624/text2vec-base-chinese"
CACHE_MAX_AGE_SECONDS = 3600 * 24 * 7  # Cache validity: 7 days

# Global tokenizer and model initialization to avoid repeated loading
_tokenizer = None
_model = None
_device = None


def _init_embedding_model():
    """
    Initializes the text embedding model (Sentence-BERT).
    Loads the model on the first call to encode_texts.
    """
    global _tokenizer, _model, _device
    if _tokenizer is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading embedding model to {_device}...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModel.from_pretrained(MODEL_NAME).to(_device)
        print("Embedding model loaded.")


def encode_texts(text_list, batch_size=64, max_length=512):
    """
    Encodes a list of texts into vector representations (Mean Pooling) in batches.
    Parameters:
        text_list (list): List of text strings to be encoded.
        batch_size (int): Batch size for encoding.
        max_length (int): Maximum text length, longer texts will be truncated.
    Returns:
        np.ndarray: Array of text embedding vectors.
    """
    _init_embedding_model()  # Ensure the model is loaded
    embeddings = []
    _model.eval()  # Set model to evaluation mode

    for i in range(0, len(text_list), batch_size):
        batch_texts = text_list[i:i + batch_size]
        # Tokenize and encode batch texts
        encoded = _tokenizer(batch_texts, padding=True, truncation=True,
                             max_length=max_length, return_tensors='pt')
        # Move encoded inputs to the specified device
        encoded = {k: v.to(_device) for k, v in encoded.items()}

        with torch.no_grad():  # Disable gradient calculation during inference
            output = _model(**encoded)
            token_embeddings = output.last_hidden_state  # Get embeddings for all tokens
            # Calculate Mean Pooling: weighted average of token embeddings by attention_mask
            attention_mask = encoded['attention_mask'].unsqueeze(-1).expand(token_embeddings.size())
            sum_embeddings = torch.sum(token_embeddings * attention_mask, dim=1)
            sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)  # Avoid division by zero
            batch_embeddings = sum_embeddings / sum_mask
            embeddings.append(batch_embeddings.cpu().numpy())  # Move results back to CPU and convert to numpy

    return np.vstack(embeddings) if embeddings else np.array([])  # Stack embeddings from all batches


def calculate_density_and_sort(grid_ids, grid_coords_map, k_neighbors=5):
    """
    Sorts a list of grid IDs based on density calculation to generate a pseudo-time series.
    Density is defined as the inverse of the average distance from the grid to its k nearest neighbors (of the same brand).
    Smaller average distance means higher density and earlier in the sorted sequence.
    Parameters:
        grid_ids (list): List of grid IDs belonging to the same brand.
        grid_coords_map (dict): Mapping from grid ID to its normalized coordinates.
        k_neighbors (int): Number of nearest neighbors to consider for density calculation.
    Returns:
        list: List of grid IDs sorted from high to low density.
    """
    if len(grid_ids) <= 1:  # If too few grids to form a sequence, return as is
        return grid_ids

    # If the number of grids is less than k_neighbors, set effective_k_neighbors to actual number - 1
    effective_k_neighbors = min(k_neighbors, len(grid_ids) - 1)
    if effective_k_neighbors <= 0:  # Only 1 grid or less, cannot calculate density
        return grid_ids

    # Get coordinates for all grids
    coords = np.array([grid_coords_map[gid] for gid in grid_ids])

    # Calculate Euclidean distances between all grids
    distances = euclidean_distances(coords, coords)

    density_scores = []
    for i in range(len(grid_ids)):
        # Get distances from current grid to all other grids (excluding itself)
        dists_to_others = np.delete(distances[i], i)

        # Get the effective_k_neighbors smallest distances and calculate their average
        k_nearest_dists = np.sort(dists_to_others)[:effective_k_neighbors]
        avg_dist = np.mean(k_nearest_dists)

        # Density is the inverse of the average distance. Add a small epsilon to avoid division by zero.
        # If average distance is 0 (meaning overlapping grids), set density to infinity
        density = 1.0 / (avg_dist + 1e-9) if avg_dist > 0 else float('inf')
        density_scores.append(density)

    # Pair grid IDs with their density scores and sort in descending order of density
    sorted_pairs = sorted(zip(grid_ids, density_scores), key=lambda x: x[1], reverse=True)
    sorted_grid_ids = [gid for gid, score in sorted_pairs]

    return sorted_grid_ids


class StoreLocationDataset(torch.utils.data.Dataset):
    """
    Custom PyTorch Dataset class for loading store location data.
    """

    def __init__(self, samples, grid_embed_map, brand_embed_map):
        """
        Initializes the dataset.
        Parameters:
            samples (list): List of (brand_key, pseudo_seq_gids, target_gid) tuples.
            grid_embed_map (dict): Mapping from grid ID to its combined embedding vector.
            brand_embed_map (dict): Mapping from brand key (brand_name, brand_type) to its embedding vector.
        """
        self.samples = samples
        self.grid_embed_map = grid_embed_map
        self.brand_embed_map = brand_embed_map

        # Get embedding dimensions for handling missing cases with zero vectors
        # Ensure data exists at initialization, otherwise provide default values
        self.default_grid_embed_dim = 0
        if grid_embed_map:
            first_grid_id = next(iter(grid_embed_map))
            self.default_grid_embed_dim = grid_embed_map[first_grid_id].shape[0]

        self.default_brand_embed_dim = 0
        if brand_embed_map:
            first_brand_key = next(iter(brand_embed_map))
            self.default_brand_embed_dim = brand_embed_map[first_brand_key].shape[0]

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieves a single sample by index.
        Returns:
            tuple: (brand_vec_tensor, seq_grid_embeds_tensor, target_idx_tensor, seq_length, pseudo_seq_gids)
            Note: pseudo_seq_gids is the raw list of grid IDs in the sequence, not padded.
        """
        brand_key, pseudo_seq_gids, target_gid = self.samples[idx]

        # Get brand embedding vector. If brand key does not exist, use a zero vector as default.
        brand_vec = self.brand_embed_map.get(brand_key, np.zeros(self.default_brand_embed_dim))

        # Get combined embedding vectors for each grid in the sequence.
        # If grid ID does not exist, use a zero vector as default.
        seq_grid_embeds = [
            self.grid_embed_map.get(gid, np.zeros(self.default_grid_embed_dim))
            for gid in pseudo_seq_gids
        ]

        # Convert numpy arrays to PyTorch tensors
        brand_vec_tensor = torch.tensor(brand_vec, dtype=torch.float32)
        seq_grid_embeds_tensor = torch.tensor(np.array(seq_grid_embeds), dtype=torch.float32)
        target_idx_tensor = torch.tensor(target_gid, dtype=torch.long)  # Target grid ID (not 0-indexed)

        # Return the original pseudo_seq_gids
        return brand_vec_tensor, seq_grid_embeds_tensor, target_idx_tensor, len(pseudo_seq_gids), pseudo_seq_gids


def collate_fn(batch):
    """
    Custom batching function for DataLoader, handles padding of variable-length sequences.
    Parameters:
        batch (list): A list of samples generated by DataLoader.
    Returns:
        tuple: Batched tensors (brand_vecs, padded_seq_grid_embeds, target_idxs, seq_lengths, batch_pseudo_seq_gids).
    """
    # Unpack each sample in the batch, including batch_pseudo_seq_gids
    brand_vecs, seq_grid_embeds_list, target_idxs, seq_lengths, batch_pseudo_seq_gids = zip(*batch)

    # Stack brand vectors
    brand_vecs_tensor = torch.stack(brand_vecs)

    # Pad grid sequence embeddings to have the same length
    max_len = max(seq_lengths)
    # Get feature dimension to create padding tensor
    feature_dim = seq_grid_embeds_list[0].size(1) if seq_grid_embeds_list else 0

    padded_seq_grid_embeds = []
    for seq_embeds in seq_grid_embeds_list:
        if seq_embeds.size(0) < max_len:
            # Calculate number of zero vectors needed for padding and concatenate
            padding = torch.zeros(max_len - seq_embeds.size(0), feature_dim, dtype=torch.float32)
            padded_seq_grid_embeds.append(torch.cat((seq_embeds, padding), dim=0))
        else:
            padded_seq_grid_embeds.append(seq_embeds)

    # Stack padded sequence embeddings
    padded_seq_grid_embeds_tensor = torch.stack(padded_seq_grid_embeds)

    # Convert target indices and sequence lengths to tensors
    target_idxs_tensor = torch.tensor(target_idxs, dtype=torch.long)
    seq_lengths_tensor = torch.tensor(seq_lengths, dtype=torch.long)

    # Return the raw list of batch_pseudo_seq_gids (not tensorized)
    return brand_vecs_tensor, padded_seq_grid_embeds_tensor, target_idxs_tensor, seq_lengths_tensor, list(batch_pseudo_seq_gids)


def load_datasets(train_data_path, test_data_path, grid_coords_path, k_neighbors_density=5, val_split_ratio=0.15):
    """
    Loads, preprocesses data, generates pseudo-time series, and creates grid and brand embeddings.
    Parameters:
        train_data_path (str): Path to the training data CSV file.
        test_data_path (str): Path to the test data CSV file.
        grid_coords_path (str): Path to the grid coordinates and semantic information CSV file.
        k_neighbors_density (int): Number of nearest neighbors to consider for density calculation in pseudo-time series generation.
        val_split_ratio (float): Ratio of validation set size to total training samples.
    Returns:
        tuple: (train_samples, val_samples, test_samples, num_total_classes, grid_embed_map, brand_embed_map, brand_embed_dim, grid_combined_embed_dim)
    """
    cache_path = get_cache_path(f"{train_data_path}_{test_data_path}_{grid_coords_path}_{k_neighbors_density}_{val_split_ratio}", "processed_data_v2")
    data_files = [train_data_path, test_data_path, grid_coords_path]

    if check_cache_freshness(cache_path, data_files, CACHE_MAX_AGE_SECONDS):
        cached_data = load_cache(cache_path)
        if cached_data:
            print("Loading datasets and embeddings from cache...")
            return cached_data

    print("Cache expired or not found, reprocessing data...")
    _init_embedding_model() # Ensure embedding model is loaded

    # Load data
    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)
    grid_coords_df = pd.read_csv(grid_coords_path)

    # 1. Process grid coordinates and POI semantic embeddings
    coords_map = {} # grid_id -> (normalized_x, normalized_y)
    grid_semantic_map = {} # grid_id -> (top_poi_embed, pois_summary_embed)
    top_poi_texts = []
    pois_summary_texts = []
    grid_ids_list_for_encoding = [] # Record grid_ids to be encoded, maintaining order

    # Calculate normalization parameters
    x_coords = grid_coords_df['longitude'].values
    y_coords = grid_coords_df['latitude'].values
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    x_range = x_max - x_min
    y_range = y_max - y_min

    for idx, row in grid_coords_df.iterrows():
        gid = int(row['grid_id'])
        x, y = row['longitude'], row['latitude']
        # Normalize coordinates to [0, 1]
        normalized_x = (x - x_min) / (x_range + 1e-9)
        normalized_y = (y - y_min) / (y_range + 1e-9)
        coords_map[gid] = (normalized_x, normalized_y)
        grid_ids_list_for_encoding.append(gid)

        # Ensure texts are string type to avoid encoding errors
        top_poi_texts.append(str(row["top_poi"]))
        pois_summary_texts.append(str(row["pois_summary"]))

    # Encode POI texts
    print("Encoding POI texts...")
    top_poi_embeds = encode_texts(top_poi_texts)
    pois_summary_embeds = encode_texts(pois_summary_texts)

    # Ensure semantic embedding dimension is valid
    semantic_embed_dim = top_poi_embeds.shape[1] if top_poi_embeds.size > 0 else 0
    if semantic_embed_dim == 0:
        print("Warning: Semantic embedding dimension is 0, POI texts might be empty or encoding failed.")

    for i, gid in enumerate(grid_ids_list_for_encoding):
        grid_semantic_map[gid] = (top_poi_embeds[i], pois_summary_embeds[i])

    # 2. Build complete grid embeddings (coordinates + semantics)
    # The final input dimension for grids will be coordinate dimension (2) + two semantic embedding dimensions (top_poi_embed + pois_summary_embed)
    grid_combined_embed_dim = 2 + 2 * semantic_embed_dim # 2 for coords, 2*semantic_dim for top_poi and summary
    grid_embed_map = {} # grid_id -> combined vector (normalized coords + top_poi_embed + pois_summary_embed)

    for gid in grid_ids_list_for_encoding:
        coords = coords_map[gid] # Normalized coordinates
        top_poi_e, pois_summary_e = grid_semantic_map[gid]
        # Concatenate coordinates and semantic embeddings
        combined_vec = np.concatenate([np.array(coords), top_poi_e, pois_summary_e])
        grid_embed_map[gid] = combined_vec

    print(f"Total grids: {len(grid_embed_map)}, Grid combined embedding dimension: {grid_combined_embed_dim}")

    # Determine total number of grid classes required by the model (num_grids)
    # Find the maximum grid_id across all datasets and add 1 (assuming grid_id starts from 0)
    all_grid_ids = list(coords_map.keys())
    num_total_classes = max(all_grid_ids) + 1 if all_grid_ids else 0
    print(f"Total number of grid ID classes (num_total_classes): {num_total_classes}")


    # 3. Process brand embeddings
    brand_names = set(train_df['brand_name'].tolist() + test_df['brand_name'].tolist())
    brand_types = set(train_df['brand_type'].tolist() + test_df['brand_type'].tolist())

    # Combine brand names and types to create texts
    brand_texts = [f"{name} {type}" for name in brand_names for type in brand_types]
    brand_keys = [(name, type) for name in brand_names for type in brand_types]

    print("Encoding brand texts...")
    # Encode brand texts
    if brand_texts:
        brand_embeds = encode_texts(brand_texts)
        brand_embed_dim = brand_embeds.shape[1]
        brand_embed_map = {brand_keys[i]: brand_embeds[i] for i in range(len(brand_keys))}
    else:
        brand_embed_dim = 0
        brand_embed_map = {}
        print("Warning: No brand texts available for encoding.")

    print(f"Total brands: {len(brand_embed_map)}, Brand embedding dimension: {brand_embed_dim}")


    # 4. Generate training and test samples
    all_train_samples_raw = []
    test_samples = []

    # Process training data
    print("Generating training/validation samples...")
    for idx, row in train_df.iterrows():
        brand_name = row['brand_name']
        brand_type = row['brand_type']
        brand_key = (brand_name, brand_type)
        grid_list_str = row['grid_id_list']

        try:
            raw_grid_ids = ast.literal_eval(grid_list_str)
            raw_grid_ids = [int(gid) for gid in raw_grid_ids if int(gid) in coords_map]
        except (ValueError, SyntaxError):
            print(f"Warning: Could not parse train grid_id_list: {grid_list_str}, skipping this row.")
            continue

        if len(raw_grid_ids) < 2: # Needs at least one prefix and one target
            continue

        # Generate pseudo-time series using density sorting
        pseudo_seq_gids = calculate_density_and_sort(raw_grid_ids, coords_map, k_neighbors=k_neighbors_density)

        # Create (prefix, target) pairs from the pseudo-sequence
        for i in range(1, len(pseudo_seq_gids)):
            prefix_gids = pseudo_seq_gids[:i]
            target_gid = pseudo_seq_gids[i]
            all_train_samples_raw.append((brand_key, prefix_gids, target_gid)) # prefix_gids is the current part of the pseudo-time series

    # Shuffle and split into training and validation sets
    np.random.shuffle(all_train_samples_raw)
    split_idx = int(len(all_train_samples_raw) * (1 - val_split_ratio))
    print(f"Total training samples: {len(all_train_samples_raw)}, Training set size: {split_idx}, Validation set size: {len(all_train_samples_raw) - split_idx}")
    train_samples = all_train_samples_raw[:split_idx]
    val_samples = all_train_samples_raw[split_idx:]

    # Process test data
    print("Generating test samples...")
    for idx, row in test_df.iterrows():
        brand_name = row['brand_name']
        brand_type = row['brand_type']
        brand_key = (brand_name, brand_type)
        grid_list_str = row['grid_id_list']

        try:
            raw_grid_ids = ast.literal_eval(grid_list_str)
            raw_grid_ids = [int(gid) for gid in raw_grid_ids if int(gid) in coords_map]
        except (ValueError, SyntaxError):
            print(f"Warning: Could not parse test grid_id_list: {grid_list_str}, skipping this row.")
            continue

        if len(raw_grid_ids) < 2:
            continue

        # Generate pseudo-time series for test data
        pseudo_seq_gids = calculate_density_and_sort(raw_grid_ids, coords_map, k_neighbors=k_neighbors_density)

        # For testing, we predict the last element given all previous elements
        prefix_gids = pseudo_seq_gids[:-1]
        target_gid = pseudo_seq_gids[-1]
        test_samples.append((brand_key, prefix_gids, target_gid)) # prefix_gids is the current part of the pseudo-time series

    print(f"Total test samples: {len(test_samples)}")

    processed_data = (train_samples, val_samples, test_samples, num_total_classes,
                      grid_embed_map, brand_embed_map, brand_embed_dim, grid_combined_embed_dim)
    save_cache(processed_data, cache_path)
    return processed_data


