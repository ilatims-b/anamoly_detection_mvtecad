
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import time
import os
from tqdm import tqdm

from ssim_autoencoder import ConvolutionalAutoencoder, SSIMLoss
from dataset_utils import get_dataloaders, TransistorDataset

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)

        optimizer.zero_grad()

        # Forward pass
        reconstructed = model(images)
        loss = criterion(images, reconstructed)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Update progress bar
        pbar.set_postfix({'Loss': f'{loss.item():.6f}'})

    return total_loss / len(train_loader)

def save_sample_reconstructions(model, test_loader, device, save_dir, epoch, num_samples=5):
    """Save sample reconstructions for visualization"""
    model.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= num_samples:
                break

            images = batch['image'].to(device)
            reconstructed = model(images)

            # Convert to numpy for visualization
            original = images[0].cpu().squeeze().numpy()
            recon = reconstructed[0].cpu().squeeze().numpy()

            # Plot comparison
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(original, cmap='gray')
            axes[0].set_title('Original')
            axes[0].axis('off')

            axes[1].imshow(recon, cmap='gray')
            axes[1].set_title('Reconstructed')
            axes[1].axis('off')

            plt.tight_layout()
            plt.savefig(save_dir / f'epoch_{epoch}_sample_{i}.png', dpi=150)
            plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train SSIM Autoencoder for Anomaly Detection')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of transistor dataset')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100 as per paper)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size (default: 128 as per paper)')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Learning rate (default: 2e-4 as per paper)')
    parser.add_argument('--latent_dim', type=int, default=128,
                        help='Latent space dimension (default: 128 as per paper)')
    parser.add_argument('--window_size', type=int, default=11,
                        help='SSIM window size (default: 11x11 as per paper)')
    parser.add_argument('--save_dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Save model every N epochs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading dataset...")
    train_loader, test_loader = get_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_size=(256, 256)
    )

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Initialize model
    model = ConvolutionalAutoencoder(latent_dim=args.latent_dim).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss function and optimizer (as per paper)
    criterion = SSIMLoss(window_size=args.window_size)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    print("Starting training...")
    train_losses = []
    best_loss = float('inf')

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        # Train
        avg_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        train_losses.append(avg_loss)

        print(f'Epoch {epoch}/{args.epochs}, Average Loss: {avg_loss:.6f}')

        # Save sample reconstructions
        if epoch % args.save_interval == 0 or epoch == 1:
            save_sample_reconstructions(
                model, test_loader, device, 
                save_dir / 'samples', epoch
            )

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'args': args,
            }, save_dir / 'best_model.pth')

        # Save checkpoint
        if epoch % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'args': args,
            }, save_dir / f'checkpoint_epoch_{epoch}.pth')

    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_losses[-1],
        'args': args,
    }, save_dir / 'final_model.pth')

    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.title('Training Loss (SSIM)')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM Loss')
    plt.grid(True)
    plt.savefig(save_dir / 'training_loss.png', dpi=150)
    plt.close()

    # Save training history
    np.save(save_dir / 'train_losses.npy', np.array(train_losses))

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Results saved to: {save_dir}")

if __name__ == "__main__":
    main()
