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
import numpy as np

from .GanFunctions import Discriminator


class VAEEncoder(nn.Module):
    """Encoder based on Transformer and convolutional layers to map input sequences to latent distribution.

    Uses a combination of embedding projection, positional encoding, Transformer blocks, and
    convolutional layers to compress sequences into a latent space representation with mean and
    log-variance parameters for the variational inference.

    Parameters
    ----------
    input_dim : int
        Length of the input sequence.
    feature_dim : int
        Dimensionality of the input features at each time step.
    emb_dim : int
        Dimensionality of the embedding space.
    latent_dim : int
        Dimensionality of the latent space.
    num_heads : int
        Number of attention heads in the Transformer Encoder blocks.
    num_layers : int, optional
        Number of Transformer Encoder blocks. Defaults to 6.
    dropout_rate : float, optional
        Dropout rate applied throughout the network. Defaults to 0.2.

    Returns
    -------
    tuple
        Sampled latent vector, latent mean, and latent log-variance.
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

        Parameters
        ----------
        mu : Tensor
            Mean of the latent distribution.
        logvar : Tensor
            Log-variance of the latent distribution.

        Returns
        -------
        Tensor
            Sampled latent vector of shape [seq_len, batch_size, latent_dim].
        """
        Std = torch.exp(0.5 * logvar)
        Eps = torch.randn_like(Std)
        return mu + Eps * Std

    def forward(self, x):
        """Forward pass through encoder.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape [batch_size, seq_len, feature_dim]

        Returns
        -------
        tuple
            (z, mu, logvar)
        """
        # Project features to embedding dimension
        x = self.emb_projection(x)    # [batch_size, seq_len, emb_dim]

        # Permute for transformer (expects [seq_len, batch_size, emb_dim])
        x = x.permute(1, 0, 2)
        x = self.positional_encoding(x)

        # Apply transformer blocks with residual connections
        for Block in self.transformer_blocks:
            x = Block(x) + x

        # Prepare for convolution [batch_size, emb_dim, seq_len]
        x = x.permute(1, 2, 0)
        x = self.conv_block(x)

        # Permute back and extract mu/logvar [seq_len, batch_size, 128]
        x = x.permute(2, 0, 1)

        Mu = self.mu_layer(x)
        Logvar = self.logvar_layer(x)

        Z = self.Reparameterize(Mu, Logvar)
        Z = Z.permute(1, 0, 2)  # [batch_size, seq_len, latent_dim]

        return Z, Mu, Logvar


class LinearConvDecoder(nn.Module):
    """Decoder combining linear and convolutional layers to reconstruct sequences.

    Decodes latent representations back to signal space using linear projection,
    Transformer blocks for structure learning, and convolutional upsampling layers.

    Parameters
    ----------
    output_dim : int
        Dimensionality of the output sequence.
    latent_dim : int
        Dimensionality of the latent representation.
    emb_dim : int
        Dimensionality of the intermediate embedding features.
    num_heads : int, optional
        Number of attention heads. Defaults to 8.
    num_layers : int, optional
        Number of Transformer layers. Defaults to 1.
    dropout_rate : float, optional
        Dropout rate applied throughout the network. Defaults to 0.1.

    Returns
    -------
    Tensor
        Reconstructed sequence tensor of shape [batch_size, seq_len, 1].
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

        Parameters
        ----------
        x : Tensor
            Latent representation of shape [batch_size, seq_len, latent_dim]

        Returns
        -------
        Tensor
            Reconstructed signal of shape [batch_size, seq_len, 1]
        """
        x = self.linear_block(x)  # [batch_size, seq_len, emb_dim]

        x = x.permute(1, 0, 2)   # [seq_len, batch_size, emb_dim]
        x = self.positional_encoding(x)

        for Block in self.transformer_blocks:
            x = Block(x) + x

        x = x.permute(1, 2, 0)  # [batch_size, emb_dim, seq_len]
        x = self.conv_block(x)  # [batch_size, 1, seq_len]
        x = x.permute(0, 2, 1)  # [batch_size, seq_len, 1]

        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer models.

    Injects positional information into sequences without modifying the embedding dimension,
    crucial for Transformers which lack inherent position awareness.

    Parameters
    ----------
    seq_len : int
        The sequence length for which positional encodings are precomputed.
    emb_dim : int
        Dimension of the features to which positional encodings will be applied.

    Returns
    -------
    Tensor
        Input tensor with positional encodings added.
    """
    def __init__(self, seq_len, emb_dim):
        super(PositionalEncoding, self).__init__()
        self.seq_len = seq_len

        # Initialize positional encoding tensor [seq_len, emb_dim]
        Pe = torch.zeros(seq_len, emb_dim)

        # Generate position indices [seq_len, 1]
        Position = torch.arange(seq_len).unsqueeze(1)

        # Compute scaling term for sinusoidal frequencies
        DivTerm = torch.exp(torch.arange(0, emb_dim, 2) * (-math.log(10000.0) / emb_dim))

        # Apply sine to even indices and cosine to odd indices
        Pe[:, 0::2] = torch.sin(Position * DivTerm)

        # Handle odd embedding dimension
        if emb_dim % 2 != 0:
            Pe[:, 1::2] = torch.cos(Position * DivTerm[:-1])
        else:
            Pe[:, 1::2] = torch.cos(Position * DivTerm)

        # Register as buffer (not a parameter, won't be updated during training)
        self.register_buffer('pe', Pe)

    def forward(self, x):
        """Add positional encodings to input signal.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape [seq_len, batch_size, feature_dim].

        Returns
        -------
        Tensor
            Signal with positional encodings added, same shape as input.
        """
        Pe = self.pe[:self.seq_len, :].unsqueeze(1).repeat(1, x.size(1), 1)
        return x + Pe


class TransformerEncoderBlock(nn.Module):
    """Transformer Encoder Block with multi-head attention and feed-forward network.

    Standard Transformer encoder layer with layer normalization, multi-head self-attention,
    and feed-forward sub-networks with residual connections.

    Parameters
    ----------
    emb_dim : int, optional
        Dimensionality of the input embeddings. Defaults to 256.
    num_heads : int, optional
        Number of attention heads. Must divide emb_dim. Defaults to 4.
    dropout_rate : float, optional
        Dropout rate for attention and feed-forward layers. Defaults to 0.1.

    Returns
    -------
    Tensor
        Output tensor with the same shape as input [seq_len, batch_size, emb_dim].
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

        Parameters
        ----------
        x : Tensor
            Input tensor of shape [seq_len, batch_size, emb_dim].

        Returns
        -------
        Tensor
            Output tensor of shape [seq_len, batch_size, emb_dim].
        """
        # Attention with pre-normalization
        XNorm = self.norm_before_attn(x)
        AttnOutput, _ = self.attention(XNorm, XNorm, XNorm)
        x = x + self.dropout1(AttnOutput)

        # FFN with pre-normalization
        XNorm = self.norm_before_ffn(x)
        FfnOutput = self.ffn(XNorm)
        x = x + self.dropout2(FfnOutput)

        return self.norm2(x)


class TransformerDecoderBlock(nn.Module):
    """Transformer Decoder Block with multi-head attention and feed-forward network.

    Similar to encoder but typically used with different attention mechanisms.
    Implements layer normalization, attention, and FFN with residual connections.

    Parameters
    ----------
    emb_dim : int
        Dimensionality of the input embeddings.
    num_heads : int
        Number of attention heads. Must divide emb_dim.
    dropout_rate : float, optional
        Dropout rate. Defaults to 0.1.

    Returns
    -------
    Tensor
        Output tensor with the same shape as input [seq_len, batch_size, emb_dim].
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

        Parameters
        ----------
        x : Tensor
            Input tensor of shape [seq_len, batch_size, emb_dim].

        Returns
        -------
        Tensor
            Output tensor of shape [seq_len, batch_size, emb_dim].
        """
        XNorm = self.norm_before_attn(x)
        AttnOut, _ = self.attn(XNorm, XNorm, XNorm)
        x = x + self.dropout(AttnOut)

        XNorm = self.norm1(x)
        FfOut = self.ff(XNorm)
        x = x + FfOut

        return self.norm2(x)


class VAE_GANTrainer:
    """Trainer for VAE-GAN signal reconstruction and adversarial refinement.

    The class intentionally matches the historical ``main_vaegan.py`` API:
    ``pre_train``, ``train``, ``test``, ``visualize_data``, and
    ``_plot_mean_std_data``.
    """

    def __init__(
        self,
        feature_dim,
        input_dim,
        latent_dim,
        emb_dim,
        batch_size,
        num_heads,
        n_enc_blocks,
        lr_d,
        lr_E,
        lr_Dec,
        gen_steps,
        disc_steps,
        lr_grad_penalty,
        lambda_rec,
        beta,
    ):
        self.feature_dim = feature_dim
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.emb_dim = emb_dim
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.n_enc_blocks = n_enc_blocks
        self.gen_steps = gen_steps
        self.disc_steps = disc_steps
        self.gp_lambda = lr_grad_penalty
        self.lambda_rec = lambda_rec
        self.beta = beta
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = VAEEncoder(
            input_dim=input_dim,
            feature_dim=feature_dim,
            emb_dim=emb_dim,
            latent_dim=latent_dim,
            num_heads=num_heads,
            num_layers=n_enc_blocks,
        ).to(self.device)
        self.decoder = LinearConvDecoder(
            output_dim=input_dim,
            latent_dim=latent_dim,
            emb_dim=emb_dim,
            num_heads=num_heads,
        ).to(self.device)
        self.discriminator = Discriminator(seq_len=input_dim, feature_dim=feature_dim).to(self.device)

        self.optimizer_E = optim.Adam(self.encoder.parameters(), lr=lr_E, betas=(0.5, 0.999))
        self.optimizer_Dec = optim.Adam(self.decoder.parameters(), lr=lr_Dec, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))

        self.reconstruction_loss = nn.MSELoss()
        self.pretrain_losses = []
        self.gen_losses = []
        self.dis_losses = []
        self.test_losses = []

    def _move_batch(self, Batch):
        return Batch.float().to(self.device)

    def _encode_decode(self, Batch):
        Z, Mu, Logvar = self.encoder(Batch)
        Reconstructed = self.decoder(Z)
        return Reconstructed, Mu, Logvar

    def _kl_loss(self, Mu, Logvar):
        return -0.5 * torch.mean(1 + Logvar - Mu.pow(2) - Logvar.exp())

    def _vae_loss(self, Batch):
        Reconstructed, Mu, Logvar = self._encode_decode(Batch)
        RecLoss = self.reconstruction_loss(Reconstructed, Batch)
        KlLoss = self._kl_loss(Mu, Logvar)
        return self.lambda_rec * RecLoss + self.beta * KlLoss, RecLoss, KlLoss, Reconstructed

    def _to_discriminator_shape(self, Batch):
        return Batch.permute(1, 0, 2)

    def _gradient_penalty(self, RealData, FakeData):
        RealDisc = self._to_discriminator_shape(RealData)
        FakeDisc = self._to_discriminator_shape(FakeData)
        Epsilon = torch.rand(RealDisc.size(), device=self.device)
        Interpolated = Epsilon * RealDisc + (1 - Epsilon) * FakeDisc
        Interpolated.requires_grad_(True)

        Validity = self.discriminator(Interpolated)
        Gradients = torch.autograd.grad(
            outputs=Validity,
            inputs=Interpolated,
            grad_outputs=torch.ones_like(Validity),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        Gradients = Gradients.reshape(Gradients.size(1), -1)
        return ((Gradients.norm(2, dim=1) - 1) ** 2).mean()

    def pre_train(self, data_loader, epochs, patience=10):
        """Pre-train encoder and decoder with reconstruction and KL losses."""
        BestLoss = float("inf")
        NoImprovement = 0

        for Epoch in range(epochs):
            self.encoder.train()
            self.decoder.train()
            EpochLosses = []

            for Batch in data_loader:
                Batch = self._move_batch(Batch)
                self.optimizer_E.zero_grad()
                self.optimizer_Dec.zero_grad()

                Loss, RecLoss, KlLoss, _ = self._vae_loss(Batch)
                Loss.backward()
                self.optimizer_E.step()
                self.optimizer_Dec.step()
                EpochLosses.append(Loss.item())

            AvgLoss = float(np.mean(EpochLosses))
            self.pretrain_losses.append(AvgLoss)
            print(f"Pre-train epoch [{Epoch + 1}/{epochs}], Loss: {AvgLoss:.4f}")

            if AvgLoss < BestLoss:
                BestLoss = AvgLoss
                NoImprovement = 0
            else:
                NoImprovement += 1
            if NoImprovement >= patience:
                print("Early stopping pre-training")
                break

    def _train_generator_step(self, Batch):
        self.optimizer_E.zero_grad()
        self.optimizer_Dec.zero_grad()

        VaeLoss, RecLoss, KlLoss, Reconstructed = self._vae_loss(Batch)
        FakeValidity = self.discriminator(self._to_discriminator_shape(Reconstructed))
        GenLoss = VaeLoss - torch.mean(FakeValidity)
        GenLoss.backward()
        self.optimizer_E.step()
        self.optimizer_Dec.step()
        return GenLoss.item(), Reconstructed.detach()

    def _train_discriminator_step(self, Batch, Reconstructed):
        self.optimizer_D.zero_grad()

        RealValidity = self.discriminator(self._to_discriminator_shape(Batch))
        FakeValidity = self.discriminator(self._to_discriminator_shape(Reconstructed))
        GradientPenalty = self._gradient_penalty(Batch, Reconstructed)
        DiscLoss = -torch.mean(RealValidity) + torch.mean(FakeValidity) + self.gp_lambda * GradientPenalty
        DiscLoss.backward()
        self.optimizer_D.step()
        return DiscLoss.item()

    def train(self, train_loader, validation_loader=None, epochs=10, patience=10):
        """Train encoder, decoder, and discriminator adversarially."""
        BestLoss = float("inf")
        NoImprovement = 0

        for Epoch in range(epochs):
            self.encoder.train()
            self.decoder.train()
            self.discriminator.train()
            GenLosses = []
            DiscLosses = []

            for Batch in train_loader:
                Batch = self._move_batch(Batch)

                Reconstructed = None
                for _ in range(self.gen_steps):
                    GenLoss, Reconstructed = self._train_generator_step(Batch)
                    GenLosses.append(GenLoss)

                for _ in range(self.disc_steps):
                    DiscLoss = self._train_discriminator_step(Batch, Reconstructed)
                    DiscLosses.append(DiscLoss)

            AvgGenLoss = float(np.mean(GenLosses))
            AvgDiscLoss = float(np.mean(DiscLosses))
            self.gen_losses.append(AvgGenLoss)
            self.dis_losses.append(AvgDiscLoss)
            print(
                f"Epoch [{Epoch + 1}/{epochs}], "
                f"Gen Loss: {AvgGenLoss:.4f}, Disc Loss: {AvgDiscLoss:.4f}"
            )

            MonitorLoss = self._validation_loss(validation_loader) if validation_loader is not None else AvgGenLoss
            if MonitorLoss < BestLoss:
                BestLoss = MonitorLoss
                NoImprovement = 0
            else:
                NoImprovement += 1
            if NoImprovement >= patience:
                print("Early stopping adversarial training")
                break

    def _validation_loss(self, validation_loader):
        self.encoder.eval()
        self.decoder.eval()
        Losses = []
        with torch.no_grad():
            for Batch in validation_loader:
                Batch = self._move_batch(Batch)
                Loss, _, _, _ = self._vae_loss(Batch)
                Losses.append(Loss.item())
        return float(np.mean(Losses)) if Losses else float("inf")

    def test(self, test_loader):
        """Evaluate reconstruction loss on a test loader."""
        TestLoss = self._validation_loss(test_loader)
        self.test_losses.append(TestLoss)
        print(f"Test reconstruction loss: {TestLoss:.4f}")
        return TestLoss

    def visualize_data(self, data_loader, mode="test"):
        """Return reconstructed samples for all batches in ``data_loader``."""
        self.encoder.eval()
        self.decoder.eval()
        ReconstructedBatches = []

        with torch.no_grad():
            for Batch in data_loader:
                Batch = self._move_batch(Batch)
                Reconstructed, _, _ = self._encode_decode(Batch)
                ReconstructedBatches.append(Reconstructed.cpu())

        ReconstructedData = torch.cat(ReconstructedBatches, dim=0)
        print(f"{mode} reconstructed data shape: {tuple(ReconstructedData.shape)}")
        return ReconstructedData

    def _plot_mean_std_data(self, RealData, GeneratedData, Filename):
        """Plot mean and standard deviation for real and reconstructed data."""
        RealData = RealData.detach().cpu().squeeze()
        GeneratedData = GeneratedData.detach().cpu().squeeze()

        AveReal = torch.mean(RealData, dim=0)
        SdReal = torch.std(RealData, dim=0)
        AveGenerated = torch.mean(GeneratedData, dim=0)
        SdGenerated = torch.std(GeneratedData, dim=0)

        plt.figure(figsize=(10, 6))
        plt.plot(AveReal, label="average real data", linewidth=2)
        plt.plot(AveGenerated, label="average generated data", linewidth=2, linestyle="--")
        plt.fill_between(range(self.input_dim), AveReal - SdReal, AveReal + SdReal, alpha=0.2, label="Std real")
        plt.fill_between(
            range(self.input_dim),
            AveGenerated - SdGenerated,
            AveGenerated + SdGenerated,
            alpha=0.2,
            label="Std generated",
        )
        plt.xlabel("signal index", fontsize=12)
        plt.ylabel("average value", fontsize=12)
        plt.title("real-generated data comparison", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.savefig(Filename)
        plt.close()
        print(f"\nPlot saved in {Filename}")
