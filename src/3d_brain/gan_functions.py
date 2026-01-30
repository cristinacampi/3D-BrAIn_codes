import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import math
import torch.nn as nn
from torch.nn.functional import conv2d
import csv
import datetime

#dynamic filename for dataframe
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def GenerateSinusoidalNoise(batch_size, seq_len, feature_dim, freq_range=(1, 5), amplitude=1.0):
    """Generate random noise with sinusoidal component.

    Args:
        batch_size (int): batch size.
        seq_len (int): signal length.
        feature_dim (int): number of features.
        freq_range (tuple, optional): frequency range for the sinusoidal component (min, max). Defaults to (1, 5).
        amplitude (float, optional): amplitude of the sinusoidal component. Defaults to 1.0.

    Returns:
        torch.Tensor: noise tensor of shape [seq_len, batch_size, feature_dim].
    """
    # Gaussian noise
    noise = torch.randn(seq_len, batch_size, feature_dim)  # Shape [seq_len, batch_size, feature_dim]

    # Sinuosoid component
    freq = torch.rand(batch_size) * (freq_range[1] - freq_range[0]) + freq_range[0]  # random frequency for each batch
    time = torch.linspace(0, 1, steps=seq_len).unsqueeze(1).to(noise.device)  # time axis [seq_len, 1]

    sinusoidal_component = amplitude * torch.sin(2 * math.pi * freq.unsqueeze(0) * time)  # Sinusoid
    sinusoidal_component = sinusoidal_component.unsqueeze(-1).expand(-1, -1, feature_dim)  # add feature_dim

    noise += sinusoidal_component

    return noise

def TrainTestSplit(dataset, val_size=0.1, test_size=0.2, shuffle=True, seed=None):
    """Split dataset into training, validation, and test sets.

    Args:
        dataset (torch.utils.data.Dataset): dataset to split
        val_size (float, optional): validation set percentage. Defaults to 0.1.
        test_size (float, optional): test set percentage. Defaults to 0.2.
        shuffle (bool, optional): if True, shuffle the dataset indices. Defaults to True.
        seed (int, optional): random seed for reproducibility. Defaults to None.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset) as torch Subset objects.
    """    
    if seed is not None:
        np.random.seed(seed)

    indices = np.arange(len(dataset))
    
    # Indices shuffling
    if shuffle:
        np.random.shuffle(indices)

    # dataset dimensions
    test_size = int(len(dataset) * test_size)
    val_size = int(len(dataset) * val_size)
    train_size = len(dataset) - test_size - val_size

    # indices split
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # dataset split
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    return train_dataset, val_dataset, test_dataset


class MEADataset(Dataset):
    """MEA (Microelectrode Array) dataset for loading and normalizing signal data.
    
    Args:
        data_path (str): path to CSV file containing MEA data
        
    Returns:
        Dataset: PyTorch Dataset object with normalized signal data
    """
    def __init__(self, data_path):
        
        mea_data = pd.read_csv(data_path, sep=None, engine='python', decimal=',', on_bad_lines='skip')
        self.signal_data = mea_data.iloc[:, :-2].values.astype(np.float32)
        
        # Row normlaization
        self.normalized_data = (self.signal_data - np.mean(self.signal_data, axis=1, keepdims=True)) / np.std(self.signal_data, axis=1, keepdims=True)

        # add a 1-dimension to represent the single feature 
        self.normalized_data = self.normalized_data[:, :, np.newaxis]  # (shape [num_samples, 99, 1])

    def __len__(self):
        return len(self.signal_data)
    
    def __getitem__(self, idx):
        # Return signal data and condition based on channel for CGAN
        signal = torch.tensor(self.normalized_data[idx], dtype=torch.float32)
        #condition = torch.tensor([1.0] if self.channels[idx] == 10 else [0.0], dtype=torch.float32)
        #return signal, condition
        return signal
    
    def plot_data(self):
        # Plot original and normalized data
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot original data
        axes[0].plot(self.signal_data.T)
        axes[0].set_title('Original Data')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Amplitude')
        
        # Plot normalized data
        axes[1].plot(self.normalized_data.T)
        axes[1].set_title('Normalized Data')
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Amplitude')
        
        plt.tight_layout()
        plt.show()

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer models.
    
    Args:
        feature_dim (int): dimension of input features
        max_len (int): maximum sequence length
        freq_scaling (float, optional): frequency scaling factor. Defaults to 1.
        
    Returns:
        Tensor: input tensor with positional encodings added
    """
    def __init__(self, feature_dim, max_len, freq_scaling=1):
        super(PositionalEncoding, self).__init__()
        self.feature_dim = feature_dim

        position = torch.arange(max_len).unsqueeze(1)  # [max_len, 1]
        
        div_term = torch.exp(torch.arange(0, feature_dim, 2) * (-math.log(10000.0* freq_scaling) / feature_dim))
        pe = torch.zeros(max_len, feature_dim)  # [99, embed_dim]
        
        pe[:, 0::2] = torch.sin(position * div_term * freq_scaling)
        # Gestisci il caso in cui embed_dim sia dispari
        if feature_dim % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term[:-1] * freq_scaling)
        else:
            pe[:, 1::2] = torch.cos(position * div_term * freq_scaling)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :].unsqueeze(1)  
        #pe = self.pe[:x.size(0), :].unsqueeze(1)
        #pe = pe[:, :, :x.size(2)]  # Match feature dim of `x`
        #x = x + pe

        #x = x + self.pe[:x.size(1), :].unsqueeze(0)

        return x
    
class TransformerEncoderBlock(nn.Module):
    """Transformer encoder block with multi-head self-attention and feed-forward network.
    
    Args:
        input_dim (int): input dimension
        emb_dim (int, optional): embedding dimension. Defaults to 240.
        num_heads (int, optional): number of attention heads. Defaults to 3.
        dropout_rate (float, optional): dropout rate. Defaults to 0.1.
        
    Returns:
        Tensor: output tensor of shape [seq_len, batch_size, emb_dim]
    """
    def __init__(self, input_dim, emb_dim=240, num_heads=3, dropout_rate=0.1):
        super(TransformerEncoderBlock, self).__init__()
        
        self.attention = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(emb_dim)    #input_dim
        self.dropout1 = nn.Dropout(dropout_rate)  # Dropout after attention block 
        
        self.norm2 = nn.LayerNorm(emb_dim)    #input_dim
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, 512),  #input_dim
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Dropout in feed-forward network
            nn.Linear(512, emb_dim),        #input_dim
        )
        self.dropout2 = nn.Dropout(dropout_rate)  # Final dropout after feed-forward

    def central_attention_mask(seq_len, focus_start, focus_end):
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
        mask[focus_start:focus_end, :] = False  # attention only in central window 
        mask[:, focus_start:focus_end] = False  # attention only to central window 
        return mask

    def gaussian_attention_mask(seq_len, center, std_dev):
        indices = torch.arange(seq_len).float()
        weights = torch.exp(-0.5 * ((indices - center) / std_dev) ** 2)
        mask = 1.0 - weights.unsqueeze(0) * weights.unsqueeze(1)  
        return mask

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    focus_start, focus_end = 30, 70  # central region with spike #TODO: make parameter
    mask = central_attention_mask(seq_len=99, focus_start=focus_start, focus_end=focus_end).to(device)
    
    center, std_dev = 49, 10  # Gaussian center and deviation standard
    gaussian_mask = gaussian_attention_mask(seq_len=99, center=center, std_dev=std_dev).to(device)


    def forward(self, x, mask):
        #attn_output, _ = self.attention(x, x, x)
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        #attn_output, _ = self.attention(x, x, x, attn_mask=gaussian_mask)
        x = self.norm1(x + self.dropout1(attn_output))

        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        
        return x

class MEAEncoder(nn.Module):
    """Encoder for MEA signals using stacked Transformer blocks.
    
    Args:
        emb_dim (int): embedding dimension
        latent_dim (int): latent space dimension
        num_heads (int): number of attention heads
        num_layers (int): number of Transformer layers
        dropout_rate (float): dropout rate
        
    Returns:
        Tensor: encoded representation of shape [seq_len, batch_size, latent_dim]
    """
    def __init__(self, emb_dim, latent_dim, num_heads, num_layers, dropout_rate):
        super(MEAEncoder, self).__init__()
        
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(latent_dim, emb_dim, num_heads, dropout_rate) for _ in range(num_layers)
        ])
        
        self.reduction = nn.Sequential(
                    nn.Linear(emb_dim, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, latent_dim)
                )


    def forward(self, x, gaussian_mask):
        for block in self.transformer_blocks:
            x = block(x, gaussian_mask)
        
        return self.reduction(x)

class MEADecoder(nn.Module):
    """Decoder for reconstructing MEA signals from latent representation.
    
    Args:
        latent_dim (int): dimension of latent input
        output_dim (int): output dimension (sequence length)
        p_drop (float, optional): dropout probability. Defaults to 0.3.
        
    Returns:
        Tensor: reconstructed signal of shape [seq_len, batch_size, output_dim]
    """
    def __init__(self, latent_dim, output_dim, p_drop=0.3):
        super(MEADecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),  
            nn.Dropout(p_drop),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.decoder(z)
    
class MEAGenerator(nn.Module):
    """Generator for synthesizing MEA signals using encoder-decoder architecture.
    
    Args:
        input_dim (int): sequence length
        latent_dim (int): latent space dimension
        feature_dim (int): feature dimension
        emb_dim (int): embedding dimension
        freq_scaling (float, optional): frequency scaling for positional encoding. Defaults to 1.5.
        num_heads (int, optional): number of attention heads. Defaults to 3.
        num_encoder_layers (int, optional): number of encoder layers. Defaults to 4.
        dropout_rate (float, optional): dropout rate. Defaults to 0.3.
        p_drop (float, optional): decoder dropout. Defaults to 0.3.
        dim_decoder (int, optional): decoder dimension. Defaults to 36.
        
    Returns:
        Tensor: generated signal of shape [seq_len, batch_size, feature_dim]
    """
    def __init__(self, input_dim, latent_dim, feature_dim, emb_dim, freq_scaling=1.5, num_heads=3, num_encoder_layers=4,  dropout_rate=0.3, p_drop=0.3,  dim_decoder=36):
        super(MEAGenerator, self).__init__()
        # (*)Positional Encoding
        self.positional_encoding = PositionalEncoding(feature_dim, input_dim, freq_scaling=freq_scaling)  
        #Embedding
        self.emb_projection = nn.Linear(feature_dim + latent_dim, emb_dim)  # Proietta a emb dimension

        #Encoder: transformer
        self.encoder = MEAEncoder(emb_dim, latent_dim, num_heads, num_encoder_layers, dropout_rate=dropout_rate)
        #Decoder
        self.decoder = MEADecoder(latent_dim, feature_dim, p_drop=p_drop)

        #Adaptive pooling
        self.avg_pool = nn.AdaptiveAvgPool1d(latent_dim)  # adpat the sequence to the latent dimension
        self.max_pool = nn.AdaptiveMaxPool1d(latent_dim)
        
        # Mapping to get the correct dimension for decoder
        self.map_layer = nn.Linear(1, dim_decoder)  # Map dimensione from 1 a 36 

    
    def forward(self, x=None, noise=None):
        # if x is not provided, the generator get noise as input
        if x is None:
            assert noise is not None
            # creation of the temoporal axis [seq_len, 1]
            time = torch.linspace(0, 1, steps=noise.size(0)).unsqueeze(1).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

            # Gaussian impulse generation
            gaussian_impulse = 1.5 * torch.exp(-0.5 * ((time - 0.5) / 0.2) ** 2)  
            
            # gaussian impulse expansion w.r.t. batch_size and feature dimension
            gaussian_impulse = gaussian_impulse.unsqueeze(2)  
            gaussian_impulse = gaussian_impulse.expand(-1, noise.size(1), -1)  # [seq_len, batch_size, 1]
            x = gaussian_impulse.expand(-1, -1, 1)  # [seq_len, batch_size, feature_dim]

                                
        x = torch.cat([x, noise], dim=2) #concatenation along dim feature
            
        x = self.positional_encoding(x)  
        
        x = x.permute(1, 0, 2)
        x = self.emb_projection(x)  # linear projection to 240 dimensions
        #produce output [batch_size, 99, emb_dim=240]
        #print(f"Input x shape after proj: {x.shape}")
        x = x.permute(1, 0, 2)

        center, std_dev = x.size(0) // 2, 6
        gaussian_mask = TransformerEncoderBlock.gaussian_attention_mask(x.size(0), center, std_dev).to(x.device)
        
        # transformer needs input as [99, batch_size, 240]
        z = self.encoder(x, gaussian_mask)
        # output [99, batch_size, latent_dim]
        #print(f"Input z shape after encoder: {z.shape}")
    
        # reshape for pooling application
        max_pooled = self.max_pool(z)
        avg_pooled= self.avg_pool(z)
        z = 0.7*max_pooled + 0.3*avg_pooled
        #print(f"z shape after pool: {z.shape}")

        output = self.decoder(z)
        #print(f"z shape after Decoder: {output.shape}")
        return output

class Discriminator(nn.Module):
    """Discriminator for distinguishing real from generated MEA signals.
    
    Args:
        seq_len (int, optional): sequence length. Defaults to 99.
        feature_dim (int, optional): feature dimension. Defaults to 1.
        hidden_dim (int, optional): hidden layer dimension. Defaults to 128.
        
    Returns:
        Tensor: classification output (real/fake probability) of shape [batch_size, 1]
    """
    def __init__(self, seq_len=99, feature_dim=1, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(feature_dim, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),  # Appiattisce l'output della convoluzione
            nn.Linear(hidden_dim * seq_len, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)  # Output scalare
        )

    def forward(self, x):
        """
        Input `x` dovrebbe avere la forma [seq_len, batch_size, feature_dim]
        """
        x = x.permute(1, 2, 0)  # [batch_size, feature_dim, seq_len]
        x = self.conv(x)
        return self.fc(x)


class GANTrainer:
    """Trainer class for adversarial training of Generator and Discriminator networks.
    
    Implements Wasserstein GAN with gradient penalty for training MEA signal generation.
    
    Args:
        feature_dim (int): feature dimension
        input_dim (int): sequence length
        latent_dim (int): latent space dimension
        emb_dim (int): embedding dimension
        num_heads (int): number of attention heads
        learning_rate_G (float, optional): generator learning rate. Defaults to 0.0001.
        learning_rate_D (float, optional): discriminator learning rate. Defaults to 0.0002.
        n_gen_steps (int, optional): number of generator training steps per iteration. Defaults to 5.
        gp_lambda (float, optional): gradient penalty weight. Defaults to 10.
        
    Returns:
        None
    """
    
    def __init__(self, feature_dim, input_dim, latent_dim, emb_dim, num_heads, learning_rate_G=0.0001, learning_rate_D=0.0002, n_gen_steps=5, gp_lambda=10):
        """Initialize GAN trainer with generator and discriminator networks."""
        self.generator = MEAGenerator(input_dim, latent_dim,  feature_dim, emb_dim, num_heads)
        self.discriminator = Discriminator(input_dim, feature_dim)
        
        self.learning_rate_G = learning_rate_G
        self.learning_rate_D = learning_rate_D
        
        self.feature_dim = feature_dim
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.emb_dim = emb_dim

        self.n_gen_steps = n_gen_steps
        self.gp_lambda = gp_lambda
        
        # Ottimizzatori: aggiornano pesi G e D per gestire discesa gradiente e minimizzare loss
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.learning_rate_G, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.learning_rate_D, betas=(0.5, 0.999))

        #LR opt
        self.scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_G, mode='min', factor=0.5, patience=2)
        self.scheduler_D = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_D, mode='min', factor=0.5, patience=5)

        self.reconstruction_loss = nn.MSELoss()
        self.gen_losses = []
        self.dis_losses = []
        self.reconstruction_losses = []

    def preprocess_batch(self,batch):
        """
        Preprocessa il batch per avere la forma [99, batch_size, feature_dim].
        """
        return batch.permute(1, 0, 2) 
      
    def wasserstein_loss(self, y_true, y_pred):
        """Calculate Wasserstein loss.
        
        Args:
            y_true (Tensor): ground truth labels
            y_pred (Tensor): predicted values
            
        Returns:
            Tensor: Wasserstein loss value
        """
        return torch.mean(y_true * y_pred)
    
    def ssim_loss(self, img1, img2, window_size=11, reduction='mean'):
        """Calculate Structural Similarity Index (SSIM) loss.
        
        Args:
            img1 (Tensor): first image/signal
            img2 (Tensor): second image/signal
            window_size (int, optional): window size for SSIM calculation. Defaults to 11.
            reduction (str, optional): loss reduction method ('mean' or other). Defaults to 'mean'.
            
        Returns:
            Tensor: SSIM loss value
        """
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        def gaussian(window_size, sigma):
            gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
                                for x in range(window_size)])
            return gauss / gauss.sum()

        def create_window(window_size, channel):
            _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
            _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
            window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
            return window

        (_, channel, height, width) = img1.shape()
        window = create_window(window_size, channel).to(img1.device)

        mu1 = conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1 - ssim_map.mean() if reduction == 'mean' else 1 - ssim_map

    
    def gradient_penalty(self, discriminator, real_data, fake_data):
        """Calculate gradient penalty for Wasserstein GAN with gradient penalty.
        
        Args:
            discriminator (nn.Module): discriminator network
            real_data (Tensor): real data samples
            fake_data (Tensor): generated fake data samples
            
        Returns:
            Tensor: gradient penalty value
        """
        batch_size = real_data.size(1)
        epsilon = torch.rand(real_data.size(), device=real_data.device)
        
        interpolated = epsilon * real_data + (1 - epsilon) * fake_data
        interpolated.requires_grad_(True)

        interpolated_validity = discriminator(interpolated)
        gradients = torch.autograd.grad(
                        outputs=interpolated_validity,
                        inputs=interpolated,
                        grad_outputs=torch.ones_like(interpolated_validity),
                        create_graph=True,
                        retain_graph=True,
                        only_inputs=True
                    )[0]

        gradients = gradients.reshape(batch_size, -1)

        penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return penalty


    def train_step(self, batch):
        """Perform one training step for both generator and discriminator.
        
        Args:
            batch (Tensor): training batch
            
        Returns:
            tuple: (generator_loss, discriminator_loss)
        """
        real_data = self.preprocess_batch(batch)
        batch_size = real_data.size(1)
        
        # Generator training
        for _ in range(self.n_gen_steps):
            for param in self.discriminator.parameters():
                param.requires_grad = False

            self.optimizer_G.zero_grad()
            
            #noise = torch.randn(self.input_dim, batch_size, self.latent_dim).to(real_data.device)
            #spike = torch.zeros_like(noise)
            #spike[50] = 1.0  # Picco centrale
            #noise = noise + spike
            # Generazione del rumore sinusoidale
            noise = GenerateSinusoidalNoise(batch_size=batch_size, seq_len=self.input_dim, feature_dim=self.latent_dim)

            #dati ricostruiti a partire da dati reali
            reconstructed_data = self.generator(x=real_data, noise=noise)
            
            reconstruction_loss = self.reconstruction_loss(reconstructed_data, real_data)
            self.reconstruction_losses.append(reconstruction_loss.item())  
            
            #dati ricostruiti a partire da noise
            #print("fake")
            fake_data = self.generator(noise=noise)
            gen_validity = self.discriminator(fake_data)
            
            #gen_loss = -torch.mean(gen_validity) + reconstruction_loss
            #gen_loss = -torch.mean(gen_validity)
            gen_loss = -torch.mean(gen_validity) + 0.5 * reconstruction_loss
            #gen_loss = -self.wasserstein_loss(gen_validity, torch.ones_like(gen_validity))
            gen_loss.backward()
            
            #total_gen_loss = gen_loss + reconstruction_loss
            #total_gen_loss.backward()

            self.optimizer_G.step()

        # Discriminator training
        for param in self.discriminator.parameters():
            param.requires_grad = True

        self.optimizer_D.zero_grad()
        
        real_validity = self.discriminator(real_data)
        fake_validity = self.discriminator(fake_data.detach())
        real_loss = -torch.mean(real_validity)
        fake_loss = torch.mean(fake_validity)
        
        gp = self.gradient_penalty(self.discriminator, real_data.data.detach(), fake_data.data.detach())
        disc_loss = real_loss + fake_loss + self.gp_lambda * gp
        #disc_loss = self.wasserstein_loss(real_validity, torch.ones_like(real_validity)) - self.wasserstein_loss(fake_validity, torch.zeros_like(fake_validity))
        disc_loss.backward(retain_graph=True)
            
        self.optimizer_D.step()

        return gen_loss.item(), disc_loss.item()

    
    def train(self, data_loader, epochs=10, patience=10):
        """Train GAN model.
        
        Args:
            data_loader (DataLoader): training data loader
            epochs (int, optional): number of training epochs. Defaults to 10.
            patience (int, optional): early stopping patience. Defaults to 10.
            
        Returns:
            None
        """
        best_loss = float('inf')
        no_improvement = 0

        for epoch in range(epochs):
            self.gen_losses = []
            self.dis_losses = []
            self.reconstruction_losses = []

            for batch in data_loader:
                gen_loss, disc_loss = self.train_step(batch)
                self.gen_losses.append(gen_loss)
                self.dis_losses.append(disc_loss)
            
            avg_gen_loss = sum(self.gen_losses) / len(self.gen_losses)
            avg_disc_loss = sum(self.dis_losses) / len(self.dis_losses)
            avg_reconstruction_loss = np.mean(self.reconstruction_losses)
            print(f"Epoch [{epoch+1}/{epochs}], Gen Loss: {avg_gen_loss:.4f}, Disc Loss: {avg_disc_loss:.4f}, Reconstruction Loss: {avg_reconstruction_loss:.4f}")

            # Scheduler for learning rates
            self.scheduler_G.step(avg_gen_loss)
            self.scheduler_D.step(avg_disc_loss)
            
            
            # Adjust the number of generator steps dynamically based on the change in losses
            if avg_disc_loss < avg_gen_loss:
                self.n_gen_steps = min(self.n_gen_steps + 1, 10)    #aumenta fino ad un massimo
                print(f"Increasing generator steps to {self.n_gen_steps}")
                
            elif avg_disc_loss > avg_gen_loss and self.n_gen_steps > 1:
                self.n_gen_steps = max(self.n_gen_steps - 1, 1)     #diminuisce fino ad un minimo
                print(f"Decreasing generator steps to {self.n_gen_steps}")

            lr_G = self.optimizer_G.param_groups[0]['lr']
            lr_D = self.optimizer_D.param_groups[0]['lr']
            print(f"Learning Rate - Generator: {lr_G:.6f}, Discriminator: {lr_D:.6f}")
                

            if avg_gen_loss < best_loss:
                best_loss = avg_gen_loss
                no_improvement = 0
                #save Gen and Disc weights
                torch.save(self.generator.state_dict(), 'best_generator.pth')
                torch.save(self.discriminator.state_dict(), 'best_discriminator.pth')
            else:
                no_improvement += 1
            
            if no_improvement >= patience:
                print("Early stopping")
                break
        
     

    def save_losses(self, filename):
        """Save training losses to CSV file.
        
        Args:
            filename (str): path to save CSV file
            
        Returns:
            None
        """
        with open(filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Generator Loss", "Discriminator Loss"])
            for epoch, (gen_loss, disc_loss) in enumerate(zip(self.gen_losses, self.dis_losses), start=1):
                writer.writerow([epoch, gen_loss, disc_loss])
        print(f"Training losses saved to {filename}")
    
    def plot_losses(self, filename=None):
        """Plot generator and discriminator losses.
        
        Args:
            filename (str, optional): path to save plot. If None, only displays plot. Defaults to None.
            
        Returns:
            None
        """
        plt.figure(figsize=(12, 6))
        plt.plot(self.gen_losses, label="Generator Loss")
        plt.plot(self.dis_losses, label="Discriminator Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Generator vs Discriminator Losses")
        plt.legend()
        plt.grid(True)
        
        if filename:
            plt.savefig(filename)
            print(f"Loss plot saved as {filename}")
        
        plt.show()
    
    def visualize_generated_data(self, real_data, num_samples=5):
        """Visualize generated and real samples side-by-side.
        
        Args:
            real_data (Tensor): real data samples
            num_samples (int, optional): number of samples to visualize. Defaults to 5.
            
        Returns:
            Tensor: generated data samples
        """
        self.generator.eval()
        #metrics = EvaluationMetrics()
        
        with torch.no_grad():
            #[latent_dim, batch_size, 1]
            #noise = generate_sinusoidal_noise(batch_size=batch_size, seq_len=self.input_dim, feature_dim=self.latent_dim)

            noise = torch.randn(self.input_dim, num_samples, self.latent_dim).to(real_data.device)
            generated_data = self.generator(real_data[:num_samples].permute(1,0,2), noise)
            gen_data = generated_data

            # Converte in numpy per il plotting e rimuove feature_dim se =1
            generated_data = generated_data.squeeze(-1).cpu().numpy()
            real_data = real_data[:num_samples].squeeze(-1).cpu().numpy()
            #print(f"Forma di real data prima di fig: {real_data[:num_samples].permute(1,0).shape}")  #.permute(1,0,2)
        
        return gen_data
    

    def save_generated_samples(self, real_data, filename=None, num_samples=5):
        """Save generated samples to numpy file.
        
        Args:
            real_data (Tensor): real data samples (used for context)
            filename (str, optional): path to save numpy file. Defaults to timestamped filename.
            num_samples (int, optional): number of samples to generate. Defaults to 5.
            
        Returns:
            None
        """
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(self.latent_dim, num_samples, self.feature_dim).to(real_data.device)
            generated_data = self.generator(real_data[:num_samples].permute(1,0,2), noise)
            generated_data = generated_data.squeeze(-1).cpu().numpy()
        
        np.save(filename, generated_data)
        print(f"Generated samples saved to {filename}")