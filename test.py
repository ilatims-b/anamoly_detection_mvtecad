import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from tqdm import tqdm
import cv2
from sklearn.metrics import roc_auc_score, roc_curve

from models.ssim_autoencoder import ConvolutionalAutoencoder, SSIMLoss, anomaly_score_ssim
from models.l2_autoencoder import L2Autoencoder
from models.padim import PaDiM
from data.data_utils import get_dataloaders

# Import project-specific evaluation utilities
from utils.pro_curve_util import compute_pro
from utils.roc_curve_util import compute_classification_roc
from utils.generic_util import trapezoid


def compute_anomaly_maps_ssim(model, images, device):
    """
    Compute pixel-wise anomaly maps using SSIM autoencoder
    """
    model.eval()
    with torch.no_grad():
        reconstructed = model(images)
        
        # Use SSIM-based anomaly score
        anomaly_maps = []
        for i in range(images.size(0)):
            orig = images[i:i+1]
            recon = reconstructed[i:i+1]
            # Get SSIM map (1 - SSIM gives anomaly score)
            ssim_map = 1 - anomaly_score_ssim(orig, recon, window_size=11)
            anomaly_maps.append(ssim_map)
        anomaly_maps = torch.cat(anomaly_maps, dim=0)
    
    return anomaly_maps


def compute_anomaly_maps_mse(model, images, device):
    """
    Compute pixel-wise anomaly maps using MSE autoencoder
    """
    model.eval()
    with torch.no_grad():
        reconstructed = model(images)
        # Use MSE-based anomaly score (pixel-wise squared error)
        anomaly_maps = F.mse_loss(reconstructed, images, reduction='none')
        anomaly_maps = torch.mean(anomaly_maps, dim=1, keepdim=True)  # Average over channels
    return anomaly_maps


def compute_anomaly_maps_padim(model, images, device):
    """
    Compute pixel-wise anomaly maps using PaDiM
    """
    model.eval()
    with torch.no_grad():
        # PaDiM directly outputs anomaly maps
        anomaly_maps = model(images)
    
    return anomaly_maps


def compute_anomaly_maps(model, images, device, method='ssim'):
    """
    Compute pixel-wise anomaly maps based on method
    """
    if method == 'ssim':
        return compute_anomaly_maps_ssim(model, images, device)
    elif method == 'mse':
        return compute_anomaly_maps_mse(model, images, device)
    elif method == 'padim':
        return compute_anomaly_maps_padim(model, images, device)
    else:
        raise ValueError(f"Unknown method: {method}")


def evaluate_model(model, test_loader, device, save_dir, method='ssim'):
    """
    Evaluate the trained model using project-specific evaluation utilities
    """
    model.eval()

    # Collect all predictions and ground truths
    all_image_scores = []
    all_pixel_scores = []
    all_image_labels = []
    all_pixel_labels = []
    all_paths = []

    print(f"Evaluating {method.upper()} model...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader)):
            images = batch['image'].to(device)
            labels = batch['label'].cpu().numpy()
            masks = batch['mask'].cpu().numpy()
            paths = batch['path']

            anomaly_maps = compute_anomaly_maps(model, images, device, method=method)

            for i in range(images.size(0)):
                anomaly_map = anomaly_maps[i].cpu().squeeze().numpy()
                gt_mask = masks[i].squeeze()
                image_label = labels[i]
                
                if anomaly_map.ndim != 2:
                    if anomaly_map.ndim == 0: anomaly_map = np.zeros((256, 256))
                    elif anomaly_map.ndim == 1: anomaly_map = anomaly_map.reshape(256, 256)
                    elif anomaly_map.ndim == 3: anomaly_map = anomaly_map.squeeze()

                if gt_mask.ndim != 2:
                    if gt_mask.ndim == 0: gt_mask = np.zeros((256, 256))
                    elif gt_mask.ndim == 1: gt_mask = gt_mask.reshape(256, 256)
                    elif gt_mask.ndim == 3: gt_mask = gt_mask.squeeze()
                
                gt_mask_binary = (gt_mask > 0.5).astype(np.uint8)

                if anomaly_map.shape != gt_mask_binary.shape:
                    gt_mask_binary = cv2.resize(gt_mask_binary, 
                                              (anomaly_map.shape[1], anomaly_map.shape[0]), 
                                              interpolation=cv2.INTER_NEAREST)

                image_score = np.max(anomaly_map)
                all_image_scores.append(image_score)
                all_pixel_scores.append(anomaly_map)
                all_image_labels.append(image_label)
                all_pixel_labels.append(gt_mask_binary)
                all_paths.extend(paths)

    all_image_scores = np.array(all_image_scores)
    all_image_labels = np.array(all_image_labels)

    print(f"Total images processed: {len(all_image_scores)}")
    print(f"Normal images: {np.sum(all_image_labels == 0)}")
    print(f"Anomalous images: {np.sum(all_image_labels == 1)}")

    results = {}

    if len(np.unique(all_image_labels)) > 1:
        try:
            i_fpr, i_tpr = compute_classification_roc(
                anomaly_maps=all_pixel_scores,
                scoring_function=np.max,
                ground_truth_labels=all_image_labels.tolist()
            )
            i_auroc = trapezoid(i_fpr, i_tpr)
            results['I-AUROC'] = i_auroc
            print(f"Image-level AUROC: {i_auroc:.4f}")

            plt.figure(figsize=(8, 6))
            plt.plot(i_fpr, i_tpr, label=f'{method.upper()} ROC Curve (AUC = {i_auroc:.4f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{method.upper()} ROC Curve - Image Level')
            plt.legend()
            plt.grid(True)
            plt.savefig(save_dir / f'{method}_roc_curve_image.png', dpi=150)
            plt.close()
        except Exception as e:
            print(f"Could not compute I-AUROC: {e}")
            results['I-AUROC'] = 0.0
    else:
        print("Cannot compute I-AUROC: Only one class present in image labels")
        results['I-AUROC'] = 0.0

    try:
        pixel_scores_flat = [s.flatten() for s in all_pixel_scores]
        pixel_labels_flat = [l.flatten() for l in all_pixel_labels]
        pixel_scores_flat = np.concatenate(pixel_scores_flat)
        pixel_labels_flat = np.concatenate(pixel_labels_flat)
        
        print(f"Pixel-level evaluation:")
        print(f"Total pixels: {len(pixel_scores_flat)}")
        print(f"Anomalous pixels: {np.sum(pixel_labels_flat)}")
        print(f"Normal pixels: {len(pixel_labels_flat) - np.sum(pixel_labels_flat)}")

        if len(np.unique(pixel_labels_flat)) > 1:
            try:
                p_auroc = roc_auc_score(pixel_labels_flat, pixel_scores_flat)
                results['P-AUROC'] = p_auroc
                print(f"Pixel-level AUROC: {p_auroc:.4f}")

                p_fpr, p_tpr, _ = roc_curve(pixel_labels_flat, pixel_scores_flat)
                plt.figure(figsize=(8, 6))
                plt.plot(p_fpr, p_tpr, label=f'{method.upper()} Pixel ROC Curve (AUC = {p_auroc:.4f})')
                plt.plot([0, 1], [0, 1], 'k--', label='Random')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'{method.upper()} ROC Curve - Pixel Level')
                plt.legend()
                plt.grid(True)
                plt.savefig(save_dir / f'{method}_roc_curve_pixel.png', dpi=150)
                plt.close()
            except Exception as e:
                print(f"Could not compute P-AUROC: {e}")
                results['P-AUROC'] = 0.0
        else:
            print("Cannot compute P-AUROC: Only one class present in pixel labels")
            results['P-AUROC'] = 0.0
    except Exception as e:
        print(f"Could not compute P-AUROC: {e}")
        results['P-AUROC'] = 0.0

    try:
        anomalous_indices = np.where(all_image_labels == 1)[0]
        if len(anomalous_indices) > 0:
            anomalous_pixel_scores = [all_pixel_scores[i] for i in anomalous_indices]
            anomalous_pixel_labels = [all_pixel_labels[i] for i in anomalous_indices]
            
            all_fprs, all_pros = compute_pro(anomalous_pixel_scores, anomalous_pixel_labels)
            integration_limit = 0.3
            au_pro = trapezoid(all_fprs, all_pros, x_max=integration_limit)
            au_pro /= integration_limit
            results['AUPRO'] = au_pro
            print(f"AUPRO score: {au_pro:.4f}")
        else:
            print("No anomalous images found for PRO computation")
            results['AUPRO'] = 0.0
    except Exception as e:
        print(f"Could not compute PRO score: {e}")
        results['AUPRO'] = 0.0

    save_sample_results(model, test_loader, device, save_dir, method=method, num_samples=10)

    return results


def save_sample_results(model, test_loader, device, save_dir, method='ssim', num_samples=10):
    """Save sample results with proper shape handling"""
    model.eval()
    samples_dir = Path(save_dir) / 'test_samples'
    samples_dir.mkdir(exist_ok=True)

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= num_samples: break

            images = batch['image'].to(device)
            labels = batch['label'].numpy()
            masks = batch['mask'].numpy()

            if method in ['ssim', 'mse']:
                reconstructed = model(images)
                anomaly_maps = compute_anomaly_maps(model, images, device, method=method)
                
                original = images[0].cpu().squeeze().numpy()
                recon = reconstructed[0].cpu().squeeze().numpy()
                anomaly_map = anomaly_maps[0].cpu().squeeze().numpy()
                gt_mask = masks[0].squeeze()
                label = labels[0]

                fig, axes = plt.subplots(2, 2, figsize=(12, 12))
                axes[0, 0].imshow(original, cmap='gray')
                axes[0, 0].set_title('Original')
                axes[0, 0].axis('off')
                axes[0, 1].imshow(recon, cmap='gray')
                axes[0, 1].set_title('Reconstructed')
                axes[0, 1].axis('off')
                axes[1, 0].imshow(anomaly_map, cmap='hot')
                axes[1, 0].set_title(f'Anomaly Map ({method.upper()})')
                axes[1, 0].axis('off')
                axes[1, 1].imshow(gt_mask, cmap='gray')
                axes[1, 1].set_title(f'Ground Truth (Label: {label})')
                axes[1, 1].axis('off')

            elif method == 'padim':
                anomaly_maps = compute_anomaly_maps(model, images, device, method=method)
                original = images[0].cpu().numpy()
                if original.shape[0] == 3:
                    original = np.transpose(original, (1, 2, 0))
                
                anomaly_map = anomaly_maps[0].cpu().squeeze().numpy()
                gt_mask = masks[0].squeeze()
                label = labels[0]

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(original)
                axes[0].set_title('Original')
                axes[0].axis('off')
                axes[1].imshow(anomaly_map, cmap='hot')
                axes[1].set_title('PaDiM Anomaly Map')
                axes[1].axis('off')
                axes[2].imshow(gt_mask, cmap='gray')
                axes[2].set_title(f'Ground Truth (Label: {label})')
                axes[2].axis('off')

            plt.tight_layout()
            plt.savefig(samples_dir / f'sample_{i:03d}_{method}_label_{label}.png', dpi=150)
            plt.close()


def main():
    parser = argparse.ArgumentParser(description='Test SSIM Autoencoder or PaDiM for Anomaly Detection')
    
    parser.add_argument('--model', type=str, choices=['ssim', 'padim', 'mse'], default='ssim',
                        help='Model to test: ssim, mse, or padim')
    
    parser.add_argument('--data_root', type=str, required=True, help='Root directory of dataset')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--save_dir', type=str, default='./test_results', help='Directory to save test results')
    
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')

    args = parser.parse_args()

    if args.device == 'auto':
        if torch.backends.mps.is_available(): device = torch.device("mps")
        elif torch.cuda.is_available(): device = torch.device("cuda")
        else: device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    save_dir = Path(args.save_dir) / args.model
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)

    if args.model in ['ssim', 'mse']:
        latent_dim = checkpoint.get('args', {}).latent_dim if 'args' in checkpoint and hasattr(checkpoint['args'], 'latent_dim') else 128
        
        if args.model == 'ssim':
            model = ConvolutionalAutoencoder(latent_dim=latent_dim).to(device)
        else:
            model = L2Autoencoder(latent_dim=latent_dim).to(device)
            
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'epoch' in checkpoint:
            print(f"{args.model.upper()} model loaded from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.6f}")
        else:
            print(f"{args.model.upper()} model loaded successfully")

    elif args.model == 'padim':
        model = PaDiM(
            backbone=checkpoint.get('backbone', 'resnet18'),
            layers=("layer1", "layer2", "layer3"),
            max_features=checkpoint.get('max_features', 100)
        ).to(device)
        
        state_dict = checkpoint['model_state_dict']
        mean_buffer = state_dict.pop('mean', None)
        icov_buffer = state_dict.pop('icov', None)
        model.load_state_dict(state_dict, strict=False)
        
        if mean_buffer is not None: model.mean = mean_buffer
        if icov_buffer is not None: model.icov = icov_buffer
        
        print(f"PaDiM model loaded with backbone: {checkpoint.get('backbone', 'resnet18')}, max_features: {checkpoint.get('max_features', 100)}")

    print("Loading test dataset...")
    is_grayscale = args.model in ['ssim', 'mse']
    _, test_loader = get_dataloaders(
        data_root=args.data_root,
        batch_size=1,
        num_workers=args.num_workers,
        target_size=(256, 256),
        to_grayscale=is_grayscale
    )

    print(f"Test samples: {len(test_loader.dataset)}")

    results = evaluate_model(model, test_loader, device, save_dir, method=args.model)

    results_file = save_dir / 'evaluation_results.txt'
    with open(results_file, 'w') as f:
        f.write(f"{args.model.upper()} Evaluation Results\n")
        f.write("=" * 40 + "\n")
        for metric, value in results.items():
            f.write(f"{metric}: {value:.4f}\n")

    print(f"\n{args.model.upper()} Evaluation Results:")
    print("=" * 40)
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    print(f"\nResults saved to: {save_dir}")


if __name__ == "__main__":
    main()