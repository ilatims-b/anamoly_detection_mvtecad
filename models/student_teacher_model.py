import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
from pathlib import Path
import random


class PatchCNN(nn.Module):
    """
    Custom CNN architecture for extracting patch descriptors
    Supports different patch sizes (17, 33, 65) as described in the paper
    """
    def __init__(self, patch_size=65, output_dim=128):
        super(PatchCNN, self).__init__()
        self.patch_size = patch_size

        if patch_size == 17:
            # Architecture for 17x17 patches
            self.features = nn.Sequential(
                nn.Conv2d(3, 128, kernel_size=5, stride=1, padding=0),  # 13x13x128
                nn.LeakyReLU(negative_slope=0.005, inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 6x6x128
                nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=0),  # 2x2x128
                nn.LeakyReLU(negative_slope=0.005, inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 1x1x128 (after cropping)
                nn.Conv2d(128, output_dim, kernel_size=2, stride=1, padding=0),  # 1x1x128
                nn.LeakyReLU(negative_slope=0.005, inplace=True)
            )
        elif patch_size == 33:
            # Architecture for 33x33 patches
            self.features = nn.Sequential(
                nn.Conv2d(3, 128, kernel_size=5, stride=1, padding=0),  # 29x29x128
                nn.LeakyReLU(negative_slope=0.005, inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 14x14x128
                nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=0),  # 10x10x128
                nn.LeakyReLU(negative_slope=0.005, inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 5x5x128
                nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=0),  # 1x1x128
                nn.LeakyReLU(negative_slope=0.005, inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 1x1x256
                nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=0),  # 1x1x256
                nn.LeakyReLU(negative_slope=0.005, inplace=True),
                nn.Conv2d(256, output_dim, kernel_size=3, stride=1, padding=1),  # 1x1x128
                nn.LeakyReLU(negative_slope=0.005, inplace=True)
            )
        elif patch_size == 65:
            # Architecture for 65x65 patches (as detailed in Table 4 of the paper)
            self.features = nn.Sequential(
                nn.Conv2d(3, 128, kernel_size=5, stride=1, padding=0),  # 61x61x128
                nn.LeakyReLU(negative_slope=0.005, inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 30x30x128
                nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=0),  # 26x26x128
                nn.LeakyReLU(negative_slope=0.005, inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 13x13x128
                nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=0),  # 9x9x128
                nn.LeakyReLU(negative_slope=0.005, inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 4x4x256
                nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=0),  # 1x1x256
                nn.LeakyReLU(negative_slope=0.005, inplace=True),
                nn.Conv2d(256, output_dim, kernel_size=3, stride=1, padding=1),  # 1x1x128
                nn.LeakyReLU(negative_slope=0.005, inplace=True)
            )
        else:
            raise ValueError(f"Unsupported patch size: {patch_size}")

    def forward(self, x):
        x = self.features(x)
        # Ensure output is 1x1 spatial dimension
        if x.size(2) > 1 or x.size(3) > 1:
            x = F.adaptive_avg_pool2d(x, (1, 1))
        return x.view(x.size(0), -1)


class TeacherNetwork(nn.Module):
    """
    Teacher network that extracts descriptive features for patches
    Uses knowledge distillation from pretrained ResNet-18
    """
    def __init__(self, patch_size=65, output_dim=128):
        super(TeacherNetwork, self).__init__()
        self.patch_cnn = PatchCNN(patch_size, output_dim)

        # For knowledge distillation - decoder to match ResNet-18 output
        self.decoder = nn.Linear(output_dim, 512)  # ResNet-18 fc input size

    def forward(self, x, return_decoded=False):
        features = self.patch_cnn(x)
        if return_decoded:
            decoded = self.decoder(features)
            return features, decoded
        return features


class StudentNetwork(nn.Module):
    """
    Student network with identical architecture to teacher
    Trained to regress teacher outputs on anomaly-free data
    """
    def __init__(self, patch_size=65, output_dim=128):
        super(StudentNetwork, self).__init__()
        self.patch_cnn = PatchCNN(patch_size, output_dim)

    def forward(self, x):
        return self.patch_cnn(x)


class StudentTeacherEnsemble(nn.Module):
    """
    Ensemble of student networks for anomaly detection
    """
    def __init__(self, patch_size=65, output_dim=128, num_students=3):
        super(StudentTeacherEnsemble, self).__init__()
        self.teacher = TeacherNetwork(patch_size, output_dim)
        self.students = nn.ModuleList([
            StudentNetwork(patch_size, output_dim) for _ in range(num_students)
        ])
        self.num_students = num_students
        self.patch_size = patch_size

        # Statistics for normalization (computed from training data)
        self.register_buffer('feature_mean', torch.zeros(output_dim))
        self.register_buffer('feature_std', torch.ones(output_dim))

    def forward_teacher(self, x):
        """Forward pass through teacher network"""
        return self.teacher(x)

    def forward_students(self, x):
        """Forward pass through all student networks"""
        student_outputs = []
        for student in self.students:
            student_outputs.append(student(x))
        return torch.stack(student_outputs, dim=0)  # (num_students, batch_size, output_dim)

    def compute_anomaly_scores(self, x):
        """
        Compute anomaly scores using regression error and predictive uncertainty
        """
        # Get teacher and student predictions
        with torch.no_grad():
            teacher_output = self.forward_teacher(x)
            student_outputs = self.forward_students(x)

            # Normalize teacher output
            teacher_normalized = (teacher_output - self.feature_mean) / self.feature_std

            # Compute ensemble mean
            student_mean = student_outputs.mean(dim=0)

            # Regression error (Equation 8 in paper)
            regression_error = torch.norm(student_mean - teacher_normalized, dim=1) ** 2

            # Predictive uncertainty (Equation 10 in paper)
            student_squared_norms = torch.norm(student_outputs, dim=2) ** 2  # (num_students, batch_size)
            mean_squared_norm = torch.norm(student_mean, dim=1) ** 2  # (batch_size,)
            predictive_uncertainty = student_squared_norms.mean(dim=0) - mean_squared_norm

            return regression_error, predictive_uncertainty

    def set_normalization_params(self, mean, std):
        """Set normalization parameters computed from training data"""
        self.feature_mean.copy_(mean)
        self.feature_std.copy_(std)


class MVTecADDataset(Dataset):
    """
    MVTecAD Dataset class for loading train/test images
    """
    def __init__(self, root_dir, category, split='train', transform=None):
        self.root_dir = Path(root_dir)
        self.category = category
        self.split = split
        self.transform = transform

        # Load image paths
        if split == 'train':
            self.image_dir = self.root_dir / category / 'train' / 'good'
            self.image_paths = list(self.image_dir.glob('*.png'))
        elif split == 'test':
            test_dir = self.root_dir / category / 'test'
            self.image_paths = []
            self.labels = []

            # Good images (label 0)
            good_dir = test_dir / 'good'
            if good_dir.exists():
                good_paths = list(good_dir.glob('*.png'))
                self.image_paths.extend(good_paths)
                self.labels.extend([0] * len(good_paths))

            # Defective images (label 1)
            for defect_dir in test_dir.iterdir():
                if defect_dir.is_dir() and defect_dir.name != 'good':
                    defect_paths = list(defect_dir.glob('*.png'))
                    self.image_paths.extend(defect_paths)
                    self.labels.extend([1] * len(defect_paths))
        else:
            raise ValueError(f"Invalid split: {split}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.split == 'train':
            return image
        else:
            return image, self.labels[idx], str(image_path)
