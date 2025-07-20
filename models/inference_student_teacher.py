"""
Inference script for Student-Teacher Anomaly Detection
Generates anomaly maps and computes evaluation metrics using the provided evaluation utilities
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import argparse
import json
import os
import sys
from tqdm import tqdm

from student_teacher_model import StudentTeacherEnsemble, MVTecADDataset

# Add the utils and eval directories to path
sys.path.append('utils')
sys.path.append('eval')

# Import evaluation utilities
try:
    from pro_curve_util import compute_pro
    from roc_curve_util import compute_classification_roc
    from generic_util import trapezoid
    EVAL_AVAILABLE = True
except ImportError:
    print("Warning: Evaluation utilities not found. Install or check paths to utils/ and eval/ directories.")
    EVAL_AVAILABLE = False


class AnomalyDetector:
    """
    Anomaly detection using trained Student-Teacher ensemble
    """
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device)

        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)

        # Initialize model
        self.model = StudentTeacherEnsemble(
            patch_size=checkpoint['patch_size'],
            num_students=checkpoint['num_students']
        ).to(self.device)

        # Load state dict
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Store metadata
        self.patch_size = checkpoint['patch_size']
        self.category = checkpoint['category']

        print(f"Loaded model: patch_size={self.patch_size}, category={self.category}")

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict_single_image(self, image_path):
        """
        Generate anomaly map for a single image
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        original_size = image.size

        # Transform for model
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Generate anomaly map
        anomaly_map = self._generate_anomaly_map(input_tensor)

        # Resize back to original size
        anomaly_map_resized = cv2.resize(anomaly_map, original_size, interpolation=cv2.INTER_LINEAR)

        return anomaly_map_resized

    def _generate_anomaly_map(self, input_tensor):
        """
        Generate dense anomaly map for input image
        """
        with torch.no_grad():
            batch_size, channels, height, width = input_tensor.shape
            patch_size = self.patch_size

            # Extract all possible patches
            patches = F.unfold(input_tensor, kernel_size=patch_size, stride=1)
            num_patches_h = height - patch_size + 1
            num_patches_w = width - patch_size + 1
            patches = patches.transpose(1, 2).reshape(-1, channels, patch_size, patch_size)

            # Compute anomaly scores for patches
            regression_errors = []
            uncertainties = []

            # Process in smaller batches to avoid OOM
            batch_size_patches = 1000
            for i in range(0, patches.size(0), batch_size_patches):
                patch_batch = patches[i:i+batch_size_patches]
                reg_error, uncertainty = self.model.compute_anomaly_scores(patch_batch)
                regression_errors.append(reg_error.cpu())
                uncertainties.append(uncertainty.cpu())

            regression_errors = torch.cat(regression_errors, dim=0)
            uncertainties = torch.cat(uncertainties, dim=0)

            # Reshape to spatial dimensions
            reg_error_map = regression_errors.view(num_patches_h, num_patches_w).numpy()
            uncertainty_map = uncertainties.view(num_patches_h, num_patches_w).numpy()

            # Normalize scores
            reg_error_normalized = (reg_error_map - reg_error_map.mean()) / (reg_error_map.std() + 1e-8)
            uncertainty_normalized = (uncertainty_map - uncertainty_map.mean()) / (uncertainty_map.std() + 1e-8)

            # Combine scores (Equation 11 in paper)
            anomaly_map = reg_error_normalized + uncertainty_normalized

            return anomaly_map

    def evaluate_dataset(self, dataset_path, category, output_dir):
        """
        Evaluate model on test dataset and generate results
        """
        print(f"Evaluating on {category} test set...")

        # Create output directories
        output_dir = Path(output_dir)
        anomaly_maps_dir = output_dir / category / 'test'
        anomaly_maps_dir.mkdir(parents=True, exist_ok=True)

        # Load test dataset
        test_dataset = MVTecADDataset(
            root_dir=dataset_path,
            category=category,
            split='test',
            transform=None  # We'll handle transform manually
        )

        # Generate anomaly maps
        anomaly_maps = []
        ground_truth_maps = []
        image_level_scores = []
        image_level_labels = []

        for idx in tqdm(range(len(test_dataset)), desc="Generating anomaly maps"):
            image, label, image_path = test_dataset[idx]

            # Generate anomaly map
            anomaly_map = self.predict_single_image(image_path)
            anomaly_maps.append(anomaly_map)

            # Load ground truth if anomalous
            if label == 1:
                # Find corresponding ground truth mask
                gt_path = self._get_ground_truth_path(image_path, dataset_path, category)
                if gt_path and os.path.exists(gt_path):
                    gt_mask = np.array(Image.open(gt_path).convert('L')) / 255.0
                else:
                    gt_mask = np.zeros_like(anomaly_map)
            else:
                gt_mask = np.zeros_like(anomaly_map)

            ground_truth_maps.append(gt_mask)

            # Image-level score (max of anomaly map)
            image_score = np.max(anomaly_map)
            image_level_scores.append(image_score)
            image_level_labels.append(label)

            # Save anomaly map
            relative_path = Path(image_path).relative_to(Path(dataset_path) / category / 'test')
            output_path = anomaly_maps_dir / relative_path.parent / f"{relative_path.stem}.tiff"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert to 0-255 range and save
            anomaly_map_uint8 = ((anomaly_map - anomaly_map.min()) / 
                                (anomaly_map.max() - anomaly_map.min() + 1e-8) * 255).astype(np.uint8)
            cv2.imwrite(str(output_path), anomaly_map_uint8)

        print(f"Generated {len(anomaly_maps)} anomaly maps")

        # Compute evaluation metrics if utilities are available
        if EVAL_AVAILABLE:
            return self._compute_metrics(anomaly_maps, ground_truth_maps, 
                                       image_level_scores, image_level_labels)
        else:
            print("Evaluation utilities not available. Skipping metric computation.")
            return {}

    def _get_ground_truth_path(self, image_path, dataset_path, category):
        """
        Get corresponding ground truth mask path
        """
        image_path = Path(image_path)

        # Extract defect type and image name
        defect_type = image_path.parent.name
        image_name = image_path.stem

        if defect_type == 'good':
            return None

        # Construct ground truth path
        gt_path = Path(dataset_path) / category / 'ground_truth' / defect_type / f"{image_name}_mask.png"
        return str(gt_path)

    def _compute_metrics(self, anomaly_maps, ground_truth_maps, 
                        image_scores, image_labels):
        """
        Compute evaluation metrics using provided utilities
        """
        print("Computing evaluation metrics...")

        metrics = {}

        try:
            # Pixel-level metrics (PRO curve)
            fprs, pros = compute_pro(anomaly_maps, ground_truth_maps)
            au_pro = trapezoid(fprs, pros, x_max=0.3) / 0.3  # Normalize by integration limit
            metrics['au_pro'] = au_pro

            print(f"AU-PRO (pixel-level): {au_pro:.3f}")

        except Exception as e:
            print(f"Error computing AU-PRO: {e}")

        try:
            # Image-level metrics (ROC curve)
            fprs_img, tprs_img = compute_classification_roc(
                anomaly_maps=anomaly_maps,
                scoring_function=np.max,
                ground_truth_labels=image_labels
            )
            au_roc = trapezoid(fprs_img, tprs_img)
            metrics['classification_au_roc'] = au_roc

            print(f"AU-ROC (image-level): {au_roc:.3f}")

        except Exception as e:
            print(f"Error computing AU-ROC: {e}")

        return metrics


def main():
    parser = argparse.ArgumentParser(description='Student-Teacher Anomaly Detection Inference')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to MVTecAD dataset')
    parser.add_argument('--category', type=str,
                       help='Category to evaluate (if not specified, use model category)')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Directory to save anomaly maps and results')
    parser.add_argument('--single_image', type=str,
                       help='Path to single image for inference')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for inference')

    args = parser.parse_args()

    # Initialize detector
    detector = AnomalyDetector(args.model_path, args.device)

    # Get category
    category = args.category if args.category else detector.category

    if args.single_image:
        # Single image inference
        print(f"Processing single image: {args.single_image}")
        anomaly_map = detector.predict_single_image(args.single_image)

        # Save result
        output_path = Path(args.output_dir) / f"{Path(args.single_image).stem}_anomaly_map.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Normalize and save
        anomaly_map_norm = ((anomaly_map - anomaly_map.min()) / 
                           (anomaly_map.max() - anomaly_map.min() + 1e-8) * 255).astype(np.uint8)
        cv2.imwrite(str(output_path), anomaly_map_norm)
        print(f"Anomaly map saved to: {output_path}")

    else:
        # Dataset evaluation
        metrics = detector.evaluate_dataset(args.dataset_path, category, args.output_dir)

        # Save metrics
        if metrics:
            metrics_path = Path(args.output_dir) / category / 'metrics.json'
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"Metrics saved to: {metrics_path}")


if __name__ == '__main__':
    main()
