
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
import numpy as np

def gaussian(window_size, sigma):
    """Create gaussian kernel for SSIM computation"""
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    """Create 2D gaussian window for SSIM computation"""
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim_loss(img1, img2, window_size=11, size_average=True, val_range=None):
    """
    Compute SSIM between two images

    Args:
        img1, img2: Input images (B, C, H, W)
        window_size: Size of the gaussian window (default: 11x11 as per paper)
        size_average: Whether to average over all pixels
        val_range: Dynamic range of pixel values

    Returns:
        SSIM value
    """
    # Determine value range
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    window = create_window(window_size, channel).to(img1.device)

    # Compute means
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # Compute variances and covariance
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    # SSIM constants
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    # Compute SSIM map
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIMLoss(nn.Module):
    """SSIM Loss function for training autoencoder"""
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average

    def forward(self, img1, img2):
        # Return 1 - SSIM to use as loss (minimize)
        return 1 - ssim_loss(img1, img2, self.window_size, self.size_average)

class ConvolutionalAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for anomaly detection with SSIM loss
    Architecture based on Bergmann et al. (2019) with extension for 256x256 images
    """
    def __init__(self, latent_dim=128):
        super(ConvolutionalAutoencoder, self).__init__()

        # Encoder: 256x256x1 -> 128D latent space
        self.encoder = nn.Sequential(
            # Block 1: 256 -> 128
            nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),

            # Block 2: 128 -> 64
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),

            # Block 3: 64 -> 32
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),

            # Block 4: 32 -> 16
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),

            # Block 5: 16 -> 8
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),

            # Block 6: 8 -> 4
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),

            # Flatten and fully connected to latent space
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, latent_dim),
        )

        # Decoder: 128D latent space -> 256x256x1
        self.decoder = nn.Sequential(
            # Linear expansion
            nn.Linear(latent_dim, 512 * 4 * 4),
            nn.ReLU(True),
            nn.Unflatten(1, (512, 4, 4)),

            # Block 1: 4 -> 8
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),

            # Block 2: 8 -> 16
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),

            # Block 3: 16 -> 32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),

            # Block 4: 32 -> 64
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),

            # Block 5: 64 -> 128
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),

            # Block 6: 128 -> 256
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Output in [0, 1] range
        )

    def forward(self, x):
        """Forward pass through encoder and decoder"""
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def encode(self, x):
        """Encode input to latent space"""
        return self.encoder(x)

    def decode(self, z):
        """Decode latent representation to image"""
        return self.decoder(z)

def anomaly_score_ssim(original, reconstructed, window_size=11):
    """
    Compute anomaly score using SSIM between original and reconstructed images

    Args:
        original: Original input image
        reconstructed: Reconstructed image from autoencoder  
        window_size: SSIM window size

    Returns:
        Anomaly score (1 - SSIM, higher means more anomalous)
    """
    return 1 - ssim_loss(original, reconstructed, window_size=window_size, size_average=False)

if __name__ == "__main__":
    # Test the model
    model = ConvolutionalAutoencoder(latent_dim=128)
    test_input = torch.randn(1, 1, 256, 256)
    output = model(test_input)

    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test SSIM loss
    ssim_criterion = SSIMLoss(window_size=11)
    loss = ssim_criterion(test_input, output)
    print(f"SSIM loss: {loss.item():.4f}")
