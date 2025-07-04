import torch
import torch.nn as nn


class SiteSelectionModel(nn.Module):
    def __init__(self, num_grids, brand_embed_dim, grid_input_dim, hidden_dim, lstm_layers=1, dropout=0.3):
        """
        Store Location Prediction Model.

        Parameters:
        num_grids (int): Total number of grid IDs (for the final output layer, i.e., max_grid_id + 1).
        brand_embed_dim (int): Dimension of brand embeddings.
        grid_input_dim (int): Dimension of input features for each grid (normalized coordinates + top_poi_embedding + pois_summary_embedding).
        hidden_dim (int): Dimension of LSTM hidden layer, also the output dimension of all projection layers.
        lstm_layers (int): Number of LSTM layers.
        dropout (float): Dropout rate.
        """
        super(SiteSelectionModel, self).__init__()
        self.num_grids = num_grids
        self.brand_embed_dim = brand_embed_dim
        self.grid_input_dim = grid_input_dim
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers

        # Brand vector mapping layer: maps brand embeddings to LSTM's initial hidden and cell state dimensions
        self.brand_to_h = nn.Sequential(
            nn.Linear(self.brand_embed_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),  # Add layer normalization
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.brand_to_c = nn.Sequential(
            nn.Linear(self.brand_embed_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),  # Add layer normalization
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Grid input projection layer: projects raw grid features (coordinates + semantics) to hidden_dim
        # This is crucial for processing combined grid features to match LSTM input dimensions
        self.grid_input_projection = nn.Sequential(
            nn.Linear(self.grid_input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),  # Add layer normalization
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # LSTM layer: processes sequential data
        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,  # LSTM input dimension is the projected grid feature dimension
            hidden_size=self.hidden_dim,
            num_layers=self.lstm_layers,
            batch_first=True,  # Input/output tensor shape is (batch_size, seq_len, feature_dim)
            dropout=dropout if self.lstm_layers > 1 else 0  # Apply Dropout only for multi-layer LSTM
        )

        # Self-attention mechanism: captures dependencies between different positions in the sequence
        # Note: MultiheadAttention requires embed_dim to be divisible by num_heads
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,  # Embedding dimension for attention mechanism
            num_heads=8,  # Number of attention heads for multi-head attention, ensure hidden_dim % num_heads == 0
            dropout=dropout,
            batch_first=True  # Input/output tensor shape is (batch_size, seq_len, feature_dim)
        )
        self.norm1 = nn.LayerNorm(self.hidden_dim)  # Layer normalization after attention
        self.dropout1 = nn.Dropout(dropout)  # Dropout after attention

        # Output layer: predicts logits for the next grid ID
        # Output dimension is num_grids, corresponding to all possible grid IDs
        self.output_layer = nn.Linear(self.hidden_dim, self.num_grids)

    def forward(self, brand_vecs, seq_grid_embeds_padded, seq_lengths):
        """
        Forward pass.

        Parameters:
        brand_vecs (Tensor): Brand embedding vectors, shape (batch_size, brand_embed_dim).
        seq_grid_embeds_padded (Tensor): Padded grid sequence embeddings, shape (batch_size, max_seq_len, grid_input_dim).
        seq_lengths (Tensor): Original sequence lengths, shape (batch_size).

        Returns:
        Tensor: Logits for predicting the next grid, shape (batch_size, num_grids).
        """
        batch_size = brand_vecs.size(0)

        # Get initial hidden and cell states for LSTM from brand vectors
        # LSTM expects initial state shape: (num_layers * num_directions, batch_size, hidden_size)
        h0 = self.brand_to_h(brand_vecs).unsqueeze(0).repeat(self.lstm_layers, 1, 1)
        c0 = self.brand_to_c(brand_vecs).unsqueeze(0).repeat(self.lstm_layers, 1, 1)

        # Project grid input features to hidden_dim
        projected_seq_grid_embeds = self.grid_input_projection(seq_grid_embeds_padded)

        # Pack padded sequences for efficient LSTM processing and to ignore padding
        # Note: seq_lengths needs to be a tensor on CPU
        # Ensure seq_lengths is of long type
        packed_seq = nn.utils.rnn.pack_padded_sequence(
            projected_seq_grid_embeds, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # LSTM processing
        lstm_out, _ = self.lstm(packed_seq, (h0, c0))

        # Unpack padded sequences, restoring original shape
        lstm_unpacked, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        # Apply self-attention mechanism (with residual connection and normalization)
        # Query, Key, Value all come from LSTM output
        # MultiheadAttention input requirement (batch_first=True): query, key, value shapes are (batch_size, seq_len, embed_dim)
        # key_padding_mask shape (batch_size, seq_len)

        # Create key_padding_mask
        # True means the position will be ignored (padded position)
        max_seq_len = projected_seq_grid_embeds.size(1)
        # Generate an all-False mask
        key_padding_mask = torch.arange(max_seq_len, device=seq_lengths.device).unsqueeze(0) >= seq_lengths.unsqueeze(1)

        attn_out, _ = self.self_attention(
            query=lstm_unpacked,
            key=lstm_unpacked,
            value=lstm_unpacked,
            key_padding_mask=key_padding_mask  # Apply mask
        )
        attn_out = self.norm1(lstm_unpacked + self.dropout1(attn_out))

        # Get the output of the last effective time step for each sequence
        # This is the key feature for predicting the next grid
        final_hidden_states = []
        for i, length in enumerate(seq_lengths):
            # length - 1 because sequence length is 1-indexed, while tensor indexing is 0-indexed
            # Ensure length > 0 to avoid index errors
            if length > 0:
                final_hidden_states.append(attn_out[i, length - 1, :])
            else:  # If sequence length is 0, append a zero vector
                final_hidden_states.append(torch.zeros(self.hidden_dim, device=attn_out.device))

        # Stack the final hidden states of all samples into a batch
        final_hidden_states_tensor = torch.stack(final_hidden_states)

        # Predict logits for the next grid
        logits = self.output_layer(final_hidden_states_tensor)

        return logits