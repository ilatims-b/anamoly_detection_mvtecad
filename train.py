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

from models.ssim_autoencoder import ConvolutionalAutoencoder, SSIMLoss
from models.padim import PaDiM
from data.data_utils import get_dataloaders


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch_ssim(model, train_loader, criterion, optimizer, scheduler, device, epoch):
    """Train SSIM autoencoder for one epoch"""
    model.train()
    total_loss = 0.0

    pbar = tqdm(train_loader, desc=f'SSIM Epoch {epoch}')
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
        pbar.set_postfix({'Loss': f'{loss.item():.6f}', 'LR': f'{scheduler.get_last_lr()[0]:.2e}'})

    # Step scheduler after each epoch
    scheduler.step()
    
    return total_loss / len(train_loader)


def train_padim(model, train_loader, device):
    """Train PaDiM model (fitting phase)"""
    print("Fitting PaDiM model on training data...")
    model.eval()  # PaDiM doesn't require gradient computation
    
    # Fit the model using the training data
    model.fit(train_loader)
    
    print("PaDiM model fitting completed.")
    return 0.0  # PaDiM doesn't have a traditional loss


def save_sample_reconstructions_ssim(model, test_loader, device, save_dir, epoch, num_samples=5):
    """Save sample reconstructions for SSIM autoencoder visualization"""
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


def save_sample_anomaly_maps_padim(model, test_loader, device, save_dir, num_samples=5):
    """Save sample anomaly maps for PaDiM visualization"""
    model.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= num_samples:
                break

            images = batch['image'].to(device)
            labels = batch['label'].numpy()
            masks = batch['mask'].numpy()

            # Generate anomaly maps
            anomaly_maps = model(images)

            # Convert to numpy for visualization
            original = images[0].cpu().numpy()
            if original.shape[0] == 3:
                original = np.transpose(original, (1, 2, 0))  # [H, W, 3]

            anomaly_map = anomaly_maps[0].cpu().squeeze().numpy()
            gt_mask = masks[0].squeeze()
            label = labels[0]

            # Plot comparison
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(original)  # RGB, no cmap
            axes[0].set_title('Original')
            axes[0].axis('off')

            axes[1].imshow(anomaly_map, cmap='hot')
            axes[1].set_title('PaDiM Anomaly Map')
            axes[1].axis('off')

            axes[2].imshow(gt_mask, cmap='gray')
            axes[2].set_title(f'Ground Truth (Label: {label})')
            axes[2].axis('off')

            plt.tight_layout()
            plt.savefig(save_dir / f'padim_sample_{i}_label_{label}.png', dpi=150)
            plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train SSIM Autoencoder or PaDiM for Anomaly Detection')
    
    # Model selection
    parser.add_argument('--model', type=str, choices=['ssim', 'padim'], default='ssim',
                        help='Model to train: ssim (autoencoder) or padim')
    
    # PaDiM specific arguments
    parser.add_argument('--backbone', type=str, default='resnet18', 
                        choices=['resnet18', 'wide_resnet50_2'],
                        help='Backbone architecture for PaDiM')
    parser.add_argument('--max_features', type=int, default=100,
                        help='Maximum number of features for PaDiM dimension reduction')
    
    # Resume training arguments (mainly for SSIM)
    parser.add_argument('--resume', type=str, default='', 
                        help='Path to checkpoint to resume from (SSIM only)')
    parser.add_argument('--start_epoch', type=int, default=0, 
                        help='Starting epoch for resumed training (SSIM only)')
    
    # Early stopping arguments (for SSIM)
    parser.add_argument('--early_stopping_patience', type=int, default=20, 
                        help='Patience for early stopping (SSIM only)')
    
    # Learning rate scheduler arguments (for SSIM)
    parser.add_argument('--lr_step_size', type=int, default=50, 
                        help='Step size for LR scheduler (SSIM only)')
    parser.add_argument('--lr_gamma', type=float, default=0.5, 
                        help='Gamma for LR scheduler (SSIM only)')
    
    # Original arguments
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of transistor dataset')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (SSIM only, default: 100)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Learning rate (SSIM only, default: 2e-4)')
    parser.add_argument('--latent_dim', type=int, default=128,
                        help='Latent space dimension (SSIM only, default: 128)')
    parser.add_argument('--window_size', type=int, default=11,
                        help='SSIM window size (SSIM only, default: 11x11)')
    parser.add_argument('--save_dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Save model every N epochs (SSIM only)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Setup device
    if args.device == 'auto':
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Create save directory
    save_dir = Path(args.save_dir) / args.model
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading dataset...")
    is_grayscale = args.model in ['ssim', 'mse']
    train_loader, test_loader = get_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_size=(256, 256),
        to_grayscale=is_grayscale
    )

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    if args.model == 'ssim':
        # SSIM Autoencoder Training
        print("Training SSIM Autoencoder...")
        
        # Initialize early stopping variables
        patience_counter = 0
        best_loss = float('inf')

        # Initialize model
        model = ConvolutionalAutoencoder(latent_dim=args.latent_dim).to(device)
        print(f"SSIM Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Loss function and optimizer
        criterion = SSIMLoss(window_size=args.window_size)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

        # Resume from checkpoint if provided
        start_epoch = 0
        if args.resume:
            checkpoint_path = Path(args.resume)
            if checkpoint_path.exists():
                print(f"Loading checkpoint from {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                best_loss = checkpoint.get('best_loss', float('inf'))
                
                if 'scheduler_state_dict' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
                print(f"Resuming training from epoch {start_epoch}")
                print(f"Previous best loss: {best_loss:.6f}")
            else:
                print(f"Checkpoint {checkpoint_path} not found, starting training from scratch.")

        # Training loop
        print("Starting SSIM training...")
        train_losses = []
        start_time = time.time()

        for epoch in range(start_epoch, args.epochs):
            # Train
            avg_loss = train_epoch_ssim(model, train_loader, criterion, optimizer, scheduler, device, epoch)
            train_losses.append(avg_loss)

            current_lr = scheduler.get_last_lr()[0]
            print(f'Epoch {epoch}/{args.epochs}, Average Loss: {avg_loss:.6f}, LR: {current_lr:.2e}')

            # Early stopping and best model saving
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': avg_loss,
                    'best_loss': best_loss,
                    'model_type': 'ssim',
                    'args': args,
                }, save_dir / 'best_model.pth')
                patience_counter = 0
                print(f"New best SSIM model saved with loss: {best_loss:.6f}")
            else:
                patience_counter += 1

            # Early stopping check
            if patience_counter >= args.early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch}")
                print(f"No improvement for {args.early_stopping_patience} epochs")
                break

            # Save sample reconstructions
            if epoch % args.save_interval == 0 or epoch == start_epoch:
                save_sample_reconstructions_ssim(
                    model, test_loader, device, 
                    save_dir / 'samples', epoch
                )

            # Save checkpoint
            if epoch % args.save_interval == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': avg_loss,
                    'best_loss': best_loss,
                    'model_type': 'ssim',
                    'args': args,
                }, save_dir / f'checkpoint_epoch_{epoch}.pth')

        # Save final model
        final_epoch = epoch if 'epoch' in locals() else args.epochs - 1
        torch.save({
            'epoch': final_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': train_losses[-1] if train_losses else float('inf'),
            'best_loss': best_loss,
            'model_type': 'ssim',
            'args': args,
        }, save_dir / 'final_model.pth')

        # Plot training progress
        if train_losses:
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 1, 1)
            plt.plot(range(start_epoch, start_epoch + len(train_losses)), train_losses)
            plt.title('SSIM Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('SSIM Loss')
            plt.grid(True)
            
            plt.subplot(2, 1, 2)
            lrs = []
            temp_scheduler = optim.lr_scheduler.StepLR(
                optim.Adam([torch.tensor(0.0)], lr=args.lr), 
                step_size=args.lr_step_size, 
                gamma=args.lr_gamma
            )
            for i in range(start_epoch, start_epoch + len(train_losses)):
                lrs.append(temp_scheduler.get_last_lr()[0])
                temp_scheduler.step()
            
            plt.plot(range(start_epoch, start_epoch + len(train_losses)), lrs)
            plt.title('Learning Rate Schedule')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.yscale('log')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(save_dir / 'training_progress.png', dpi=150)
            plt.close()

            np.save(save_dir / 'train_losses.npy', np.array(train_losses))

        total_time = time.time() - start_time
        print(f"SSIM training completed in {total_time:.2f} seconds")
        print(f"Best loss: {best_loss:.6f}")

    elif args.model == 'padim':
        # PaDiM Training
        print("Training PaDiM...")
        
        # Initialize PaDiM model
        model = PaDiM(
            backbone=args.backbone,
            layers=("layer1", "layer2", "layer3"),
            max_features=args.max_features
        ).to(device)
        
        print(f"PaDiM Model with backbone: {args.backbone}")
        print(f"Feature dimension reduction to: {args.max_features}")

        start_time = time.time()
        
        # Train PaDiM (fitting phase)
        train_padim(model, train_loader, device)
        
        # Save the fitted model
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_type': 'padim',
            'backbone': args.backbone,
            'max_features': args.max_features,
            'args': args,
        }, save_dir / 'padim_model.pth')
        
        print("PaDiM model saved.")
        
        # Save sample anomaly maps
        save_sample_anomaly_maps_padim(
            model, test_loader, device, 
            save_dir / 'samples'
        )
        
        total_time = time.time() - start_time
        print(f"PaDiM training completed in {total_time:.2f} seconds")

    print(f"Results saved to: {save_dir}")


if __name__ == "__main__":
    main()