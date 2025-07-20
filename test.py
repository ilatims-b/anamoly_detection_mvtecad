
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
import cv2

from ssim_autoencoder import ConvolutionalAutoencoder, SSIMLoss, anomaly_score_ssim
from dataset_utils import get_dataloaders

def compute_anomaly_maps(model, images, device, method='ssim'):
    """
    Compute pixel-wise anomaly maps

    Args:
        model: Trained autoencoder model
        images: Input images tensor
        device: Device to run on
        method: Method for anomaly score computation ('ssim' or 'mse')

    Returns:
        Anomaly maps tensor
    """
    model.eval()
    with torch.no_grad():
        reconstructed = model(images)

        if method == 'ssim':
            # Use SSIM-based anomaly score
            anomaly_maps = []
            for i in range(images.size(0)):
                # Compute SSIM map for each image
                orig = images[i:i+1]
                recon = reconstructed[i:i+1]

                # Get SSIM map (not averaged)
                ssim_map = 1 - anomaly_score_ssim(orig, recon, window_size=11)
                anomaly_maps.append(ssim_map)

            anomaly_maps = torch.cat(anomaly_maps, dim=0)

        elif method == 'mse':
            # Use MSE-based anomaly score
            anomaly_maps = F.mse_loss(reconstructed, images, reduction='none')
            anomaly_maps = torch.mean(anomaly_maps, dim=1, keepdim=True)  # Average over channels

        else:
            raise ValueError(f"Unknown method: {method}")

    return anomaly_maps

def compute_pro_score(anomaly_maps, gt_masks, num_thresholds=100):
    """
    Compute Per-Region Overlap (PRO) score

    Args:
        anomaly_maps: Predicted anomaly maps (B, 1, H, W)
        gt_masks: Ground truth masks (B, 1, H, W)  
        num_thresholds: Number of thresholds to evaluate

    Returns:
        PRO AUC score
    """
    # Flatten for easier computation
    anomaly_maps = anomaly_maps.cpu().numpy()
    gt_masks = gt_masks.cpu().numpy()

    max_step = num_thresholds
    expect_fpr = 0.3  # Standard FPR limit for PRO
    max_th = anomaly_maps.max()
    min_th = anomaly_maps.min()
    delta = (max_th - min_th) / max_step

    ious_mean = []
    ious_std = []
    pros_mean = []
    pros_std = []
    threds = []
    fprs = []

    binary_score_maps = np.zeros_like(anomaly_maps, dtype=np.bool_)

    for step in range(max_step):
        thred = max_th - step * delta

        # Anomalous pixels with score > threshold
        binary_score_maps = (anomaly_maps >= thred).astype(np.bool_)

        pro = []  # Per-region overlap
        iou = []  # Intersection over union

        # Iterate over each sample in batch
        for i in range(len(gt_masks)):
            gt_mask = gt_masks[i].squeeze()
            score_mask = binary_score_maps[i].squeeze()

            # Skip if no ground truth anomaly
            if gt_mask.sum() == 0:
                continue

            # Connected components in ground truth
            gt_mask_labeled = cv2.connectedComponents(gt_mask.astype(np.uint8))[1]

            temp_pro = []
            for j in range(1, gt_mask_labeled.max() + 1):
                region_mask = (gt_mask_labeled == j)
                if region_mask.sum() == 0:
                    continue

                # Per-region overlap
                overlap = score_mask & region_mask
                pro_score = overlap.sum() / region_mask.sum()
                temp_pro.append(pro_score)

            if temp_pro:
                pro.extend(temp_pro)

            # IoU calculation  
            intersection = (score_mask & gt_mask).sum()
            union = (score_mask | gt_mask).sum()
            if union > 0:
                iou.append(intersection / union)

        # FPR calculation (false positive rate on normal regions)
        normals = gt_masks.sum(axis=(1,2,3)) == 0  # Normal images
        if normals.sum() > 0:
            normal_scores = binary_score_maps[normals]
            fpr = normal_scores.sum() / (normals.sum() * binary_score_maps.shape[-1] * binary_score_maps.shape[-2])
        else:
            fpr = 0

        if pro:
            pros_mean.append(np.mean(pro))
            pros_std.append(np.std(pro))
        else:
            pros_mean.append(0)
            pros_std.append(0)

        if iou:
            ious_mean.append(np.mean(iou))
            ious_std.append(np.std(iou))
        else:
            ious_mean.append(0)
            ious_std.append(0)

        fprs.append(fpr)
        threds.append(thred)

        if fpr >= expect_fpr:
            break

    # Calculate AUC-PRO
    fprs = np.array(fprs)
    pros_mean = np.array(pros_mean)

    if len(fprs) > 1:
        pro_auc = auc(fprs, pros_mean)
    else:
        pro_auc = 0.0

    return pro_auc, fprs, pros_mean

def evaluate_model(model, test_loader, device, save_dir):
    """
    Evaluate the trained model on test data

    Args:
        model: Trained autoencoder model
        test_loader: Test data loader
        device: Device to run on
        save_dir: Directory to save results

    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()

    all_anomaly_scores = []  # Image-level anomaly scores
    all_anomaly_maps = []    # Pixel-level anomaly maps
    all_labels = []          # Ground truth labels
    all_masks = []           # Ground truth masks
    all_paths = []           # Image paths

    print("Evaluating model...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader)):
            images = batch['image'].to(device)
            labels = batch['label'].cpu().numpy()
            masks = batch['mask'].cpu().numpy()
            paths = batch['path']

            # Get reconstructions
            reconstructed = model(images)

            # Compute anomaly maps using SSIM
            anomaly_maps = compute_anomaly_maps(model, images, device, method='ssim')

            # Compute image-level anomaly scores (max of anomaly map)
            anomaly_scores = torch.max(anomaly_maps.view(anomaly_maps.size(0), -1), dim=1)[0]

            all_anomaly_scores.extend(anomaly_scores.cpu().numpy())
            all_anomaly_maps.append(anomaly_maps.cpu())
            all_labels.extend(labels)
            all_masks.extend(masks)
            all_paths.extend(paths)

    # Convert to numpy arrays
    all_anomaly_scores = np.array(all_anomaly_scores)
    all_anomaly_maps = torch.cat(all_anomaly_maps, dim=0)
    all_labels = np.array(all_labels)
    all_masks = np.array(all_masks)

    # Compute metrics
    results = {}

    # Image-level AUROC (I-AUROC)
    if len(np.unique(all_labels)) > 1:  # Need both normal and anomalous samples
        i_auroc = roc_auc_score(all_labels, all_anomaly_scores)
        results['I-AUROC'] = i_auroc
        print(f"Image-level AUROC: {i_auroc:.4f}")

        # Plot ROC curve
        fpr, tpr, _ = roc_curve(all_labels, all_anomaly_scores)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {i_auroc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Image Level')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_dir / 'roc_curve_image.png', dpi=150)
        plt.close()

    # Pixel-level evaluation (only for anomalous images)
    anomalous_indices = np.where(all_labels == 1)[0]
    if len(anomalous_indices) > 0:
        # Pixel-level AUROC (P-AUROC) 
        pixel_scores = []
        pixel_labels = []

        for idx in anomalous_indices:
            anomaly_map = all_anomaly_maps[idx].squeeze().numpy()
            gt_mask = all_masks[idx].squeeze()

            pixel_scores.extend(anomaly_map.flatten())
            pixel_labels.extend(gt_mask.flatten())

        pixel_scores = np.array(pixel_scores)
        pixel_labels = np.array(pixel_labels)

        if len(np.unique(pixel_labels)) > 1:
            p_auroc = roc_auc_score(pixel_labels, pixel_scores)
            results['P-AUROC'] = p_auroc
            print(f"Pixel-level AUROC: {p_auroc:.4f}")

        # PRO score calculation
        anomalous_maps = all_anomaly_maps[anomalous_indices]
        anomalous_masks = torch.tensor(all_masks[anomalous_indices])

        try:
            pro_auc, fprs, pros = compute_pro_score(anomalous_maps, anomalous_masks)
            results['AUPRO'] = pro_auc
            print(f"AUPRO score: {pro_auc:.4f}")
        except Exception as e:
            print(f"Could not compute PRO score: {e}")
            results['AUPRO'] = 0.0

    # Save sample results
    save_sample_results(model, test_loader, device, save_dir, num_samples=10)

    return results

def save_sample_results(model, test_loader, device, save_dir, num_samples=10):
    """Save sample results showing original, reconstructed, and anomaly map"""
    model.eval()
    save_dir = Path(save_dir)
    samples_dir = save_dir / 'test_samples'
    samples_dir.mkdir(exist_ok=True)

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= num_samples:
                break

            images = batch['image'].to(device)
            labels = batch['label'].numpy()
            masks = batch['mask'].numpy()
            paths = batch['path']

            reconstructed = model(images)
            anomaly_maps = compute_anomaly_maps(model, images, device, method='ssim')

            # Convert to numpy
            original = images[0].cpu().squeeze().numpy()
            recon = reconstructed[0].cpu().squeeze().numpy()
            anomaly_map = anomaly_maps[0].cpu().squeeze().numpy()
            gt_mask = masks[0].squeeze()
            label = labels[0]

            # Create visualization
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))

            axes[0, 0].imshow(original, cmap='gray')
            axes[0, 0].set_title('Original')
            axes[0, 0].axis('off')

            axes[0, 1].imshow(recon, cmap='gray')
            axes[0, 1].set_title('Reconstructed')
            axes[0, 1].axis('off')

            axes[1, 0].imshow(anomaly_map, cmap='hot')
            axes[1, 0].set_title('Anomaly Map (SSIM)')
            axes[1, 0].axis('off')

            axes[1, 1].imshow(gt_mask, cmap='gray')
            axes[1, 1].set_title(f'Ground Truth (Label: {label})')
            axes[1, 1].axis('off')

            plt.tight_layout()
            plt.savefig(samples_dir / f'sample_{i:03d}_label_{label}.png', dpi=150)
            plt.close()

def main():
    parser = argparse.ArgumentParser(description='Test SSIM Autoencoder for Anomaly Detection')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of transistor dataset')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--save_dir', type=str, default='./test_results',
                        help='Directory to save test results')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    args = parser.parse_args()

    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)

    # Initialize model
    if 'args' in checkpoint:
        latent_dim = checkpoint['args'].latent_dim
    else:
        latent_dim = 128  # Default

    model = ConvolutionalAutoencoder(latent_dim=latent_dim).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.6f}")

    # Load test data
    print("Loading test dataset...")
    _, test_loader = get_dataloaders(
        data_root=args.data_root,
        batch_size=1,  # Process one image at a time
        num_workers=args.num_workers,
        target_size=(256, 256)
    )

    print(f"Test samples: {len(test_loader.dataset)}")

    # Evaluate model
    results = evaluate_model(model, test_loader, device, save_dir)

    # Save results
    results_file = save_dir / 'evaluation_results.txt'
    with open(results_file, 'w') as f:
        f.write("SSIM Autoencoder Evaluation Results\n")
        f.write("=" * 40 + "\n")
        for metric, value in results.items():
            f.write(f"{metric}: {value:.4f}\n")

    print("\nEvaluation Results:")
    print("=" * 40)
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    print(f"\nResults saved to: {save_dir}")

if __name__ == "__main__":
    main()
