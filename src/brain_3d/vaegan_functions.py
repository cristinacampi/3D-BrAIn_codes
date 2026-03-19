"""Variational Autoencoder-GAN (VAE-GAN) implementation for MEA signal synthesis."""

import torch
import math
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
from torch.nn.utils import spectral_norm
import os
import csv
import torch.optim as optim
import matplotlib.pyplot as plt


class VAEEncoder(nn.Module):
    """Encoder based on Transformer and convolutional layers to map input sequences to latent distribution.

    Uses a combination of embedding projection, positional encoding, Transformer blocks, and 
    convolutional layers to compress sequences into a latent space representation with mean and 
    log-variance parameters for the variational inference.

    Args:
        input_dim (int): Length of the input sequence.
        feature_dim (int): Dimensionality of the input features at each time step.
        emb_dim (int): Dimensionality of the embedding space.
        latent_dim (int): Dimensionality of the latent space.
        num_heads (int): Number of attention heads in the Transformer Encoder blocks.
        num_layers (int, optional): Number of Transformer Encoder blocks. Defaults to 6.
        dropout_rate (float, optional): Dropout rate applied throughout the network. Defaults to 0.2.

    Returns:
        tuple: (z, mu, logvar) where:
            - z (Tensor): sampled latent vector of shape [batch_size, seq_len, latent_dim]
            - mu (Tensor): mean of latent distribution of shape [batch_size, seq_len, latent_dim]
            - logvar (Tensor): log-variance of latent distribution of shape [batch_size, seq_len, latent_dim]
    """
    def __init__(self, input_dim, feature_dim, emb_dim, latent_dim, num_heads, num_layers=6, dropout_rate=0.2):
        super(VAEEncoder, self).__init__()
        self.input_dim = input_dim

        # Embedding projection
        self.emb_projection = nn.Linear(feature_dim, emb_dim)

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(input_dim, emb_dim)  
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(emb_dim, num_heads, dropout_rate) for _ in range(num_layers)
        ])
        
        # Convolutional layers for feature extraction
        self.conv_block = nn.Sequential(
            nn.Conv1d(emb_dim, 512, kernel_size=7, stride=1, padding=3),        
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout_rate),

            nn.Conv1d(512, 256, kernel_size=5, stride=1, padding=2),        
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout_rate),

            nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1),        
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout_rate),
        )
        
        # Output layers for mean and log-variance
        self.mu_layer = nn.Linear(128, latent_dim)
        self.logvar_layer = nn.Linear(128, latent_dim)

    def Reparameterize(self, mu, logvar):
        """Reparameterization trick to sample from the latent distribution.

        Args:
            mu (Tensor): Mean of the latent distribution.
            logvar (Tensor): Log-variance of the latent distribution.

        Returns:
            Tensor: Sampled latent vector of shape [seq_len, batch_size, latent_dim].
        """
        std = torch.exp(0.5 * logvar)  
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """Forward pass through encoder.
        
        Args:
            x (Tensor): Input tensor of shape [batch_size, seq_len, feature_dim]
            
        Returns:
            tuple: (z, mu, logvar)
        """
        # Project features to embedding dimension
        x = self.emb_projection(x)    # [batch_size, seq_len, emb_dim]
        
        # Permute for transformer (expects [seq_len, batch_size, emb_dim])
        x = x.permute(1, 0, 2)
        x = self.positional_encoding(x)
        
        # Apply transformer blocks with residual connections
        for block in self.transformer_blocks:
            x = block(x) + x
           
        # Prepare for convolution [batch_size, emb_dim, seq_len]
        x = x.permute(1, 2, 0)
        x = self.conv_block(x)
        
        # Permute back and extract mu/logvar [seq_len, batch_size, 128]
        x = x.permute(2, 0, 1)

        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        
        z = self.Reparameterize(mu, logvar)
        z = z.permute(1, 0, 2)  # [batch_size, seq_len, latent_dim]
        
        return z, mu, logvar


class LinearConvDecoder(nn.Module):
    """Decoder combining linear and convolutional layers to reconstruct sequences.

    Decodes latent representations back to signal space using linear projection,
    Transformer blocks for structure learning, and convolutional upsampling layers.

    Args:
        output_dim (int): Dimensionality of the output sequence.
        latent_dim (int): Dimensionality of the latent representation.
        emb_dim (int): Dimensionality of the intermediate embedding features.
        num_heads (int, optional): Number of attention heads. Defaults to 8.
        num_layers (int, optional): Number of Transformer layers. Defaults to 1.
        dropout_rate (float, optional): Dropout rate applied throughout the network. Defaults to 0.1.

    Returns:
        Tensor: Reconstructed sequence tensor of shape [batch_size, seq_len, 1].
    """
    def __init__(self, output_dim, latent_dim, emb_dim, num_heads=8, num_layers=1, dropout_rate=0.1):
        super(LinearConvDecoder, self).__init__()
        
        # Positional Encoding
        self.positional_encoding = PositionalEncoding(output_dim, emb_dim)  
        
        # Linear projection from latent to embedding dimension
        self.linear_block = nn.Sequential(
            nn.Linear(latent_dim, emb_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
        )

        # Transformer Decoder blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerDecoderBlock(emb_dim, num_heads, dropout_rate) for _ in range(num_layers)
        ])

        # Convolutional upsampling layers
        self.conv_block = nn.Sequential(
            nn.Conv1d(emb_dim, 512, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout_rate),

            nn.Conv1d(512, 256, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout_rate),
            
            nn.Conv1d(256, 128, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout_rate),  
            
            nn.Conv1d(128, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout_rate), 
            
            nn.Conv1d(64, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout_rate), 

            nn.Conv1d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout_rate),

            nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout_rate), 

            nn.Conv1d(8, 1, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm1d(1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout_rate), 
        )

    def forward(self, x):
        """Forward pass through decoder.
        
        Args:
            x (Tensor): Latent representation of shape [batch_size, seq_len, latent_dim]
            
        Returns:
            Tensor: Reconstructed signal of shape [batch_size, seq_len, 1]
        """
        x = self.linear_block(x)  # [batch_size, seq_len, emb_dim]
        
        x = x.permute(1, 0, 2)   # [seq_len, batch_size, emb_dim]
        x = self.positional_encoding(x)
        
        for block in self.transformer_blocks:
            x = block(x) + x
        
        x = x.permute(1, 2, 0)  # [batch_size, emb_dim, seq_len]
        x = self.conv_block(x)  # [batch_size, 1, seq_len]
        x = x.permute(0, 2, 1)  # [batch_size, seq_len, 1]
        
        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer models.
    
    Injects positional information into sequences without modifying the embedding dimension,
    crucial for Transformers which lack inherent position awareness.

    Args:
        seq_len (int): The sequence length for which positional encodings are precomputed.
        emb_dim (int): Dimension of the features to which positional encodings will be applied.

    Returns:
        Tensor: Input tensor with positional encodings added.
    """
    def __init__(self, seq_len, emb_dim):
        super(PositionalEncoding, self).__init__()
        self.seq_len = seq_len

        # Initialize positional encoding tensor [seq_len, emb_dim]
        pe = torch.zeros(seq_len, emb_dim)  
        
        # Generate position indices [seq_len, 1]
        position = torch.arange(seq_len).unsqueeze(1)  
        
        # Compute scaling term for sinusoidal frequencies 
        div_term = torch.exp(torch.arange(0, emb_dim, 2) * (-math.log(10000.0) / emb_dim))
        
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Handle odd embedding dimension
        if emb_dim % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter, won't be updated during training)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Add positional encodings to input signal.

        Args:
            x (Tensor): Input tensor of shape [seq_len, batch_size, feature_dim].

        Returns:
            Tensor: Signal with positional encodings added, same shape as input.
        """
        pe = self.pe[:self.seq_len, :].unsqueeze(1).repeat(1, x.size(1), 1)
        return x + pe


class TransformerEncoderBlock(nn.Module):
    """Transformer Encoder Block with multi-head attention and feed-forward network.

    Standard Transformer encoder layer with layer normalization, multi-head self-attention,
    and feed-forward sub-networks with residual connections.

    Args:
        emb_dim (int, optional): Dimensionality of the input embeddings. Defaults to 256.
        num_heads (int, optional): Number of attention heads. Must divide emb_dim. Defaults to 4.
        dropout_rate (float, optional): Dropout rate for attention and feed-forward layers. Defaults to 0.1.

    Returns:
        Tensor: Output tensor with the same shape as input [seq_len, batch_size, emb_dim].
    """
    def __init__(self, emb_dim=256, num_heads=4, dropout_rate=0.1):
        super(TransformerEncoderBlock, self).__init__()
        
        # Pre-attention normalization
        self.norm_before_attn = nn.LayerNorm(emb_dim)

        # Multi-head attention
        self.attention = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads, dropout=dropout_rate)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Pre-FFN normalization
        self.norm_before_ffn = nn.LayerNorm(emb_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(emb_dim * 2, emb_dim),
        )
        self.norm2 = nn.LayerNorm(emb_dim)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        """Forward pass for encoder block.

        Args:
            x (Tensor): Input tensor of shape [seq_len, batch_size, emb_dim].
            
        Returns:
            Tensor: Output tensor of shape [seq_len, batch_size, emb_dim].
        """
        # Attention with pre-normalization
        x_norm = self.norm_before_attn(x)
        attn_output, _ = self.attention(x_norm, x_norm, x_norm)
        x = x + self.dropout1(attn_output)
        
        # FFN with pre-normalization
        x_norm = self.norm_before_ffn(x)
        ffn_output = self.ffn(x_norm)
        x = x + self.dropout2(ffn_output)
        
        return self.norm2(x)


class TransformerDecoderBlock(nn.Module):
    """Transformer Decoder Block with multi-head attention and feed-forward network.

    Similar to encoder but typically used with different attention mechanisms.
    Implements layer normalization, attention, and FFN with residual connections.

    Args:
        emb_dim (int): Dimensionality of the input embeddings.
        num_heads (int): Number of attention heads. Must divide emb_dim.
        dropout_rate (float, optional): Dropout rate. Defaults to 0.1.

    Returns:
        Tensor: Output tensor with the same shape as input [seq_len, batch_size, emb_dim].
    """
    def __init__(self, emb_dim, num_heads, dropout_rate=0.1):
        super(TransformerDecoderBlock, self).__init__()

        # Attention components
        self.norm_before_attn = nn.LayerNorm(emb_dim)
        self.attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.norm1 = nn.LayerNorm(emb_dim)
        
        # FFN components
        self.norm2 = nn.LayerNorm(emb_dim)
        self.ff = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(emb_dim * 2, emb_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        """Forward pass for decoder block.
        
        Args:
            x (Tensor): Input tensor of shape [seq_len, batch_size, emb_dim].
            
        Returns:
            Tensor: Output tensor of shape [seq_len, batch_size, emb_dim].
        """
        x_norm = self.norm_before_attn(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_out)
        
        x_norm = self.norm1(x)
        ff_out = self.ff(x_norm)
        x = x + ff_out
        
        return self.norm2(x)
