"""
Training script for Student-Teacher Anomaly Detection
Implements the complete training pipeline including teacher pretraining and student ensemble training
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from pathlib import Path
import argparse
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

from student_teacher_model import StudentTeacherEnsemble, MVTecADDataset


class ImageNetPatchDataset(Dataset):
    """
    Dataset for generating patches from ImageNet for teacher pretraining
    """
    def __init__(self, imagenet_dir, patch_size=65, num_patches_per_epoch=10000):
        self.imagenet_dir = Path(imagenet_dir)
        self.patch_size = patch_size
        self.num_patches_per_epoch = num_patches_per_epoch

        # Find all images
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            self.image_paths.extend(self.imagenet_dir.rglob(ext))

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {imagenet_dir}")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return self.num_patches_per_epoch

    def __getitem__(self, idx):
        # Random image
        img_path = random.choice(self.image_paths)

        try:
            from PIL import Image
            image = Image.open(img_path).convert('RGB')

            # Random crop size between 4*patch_size and 16*patch_size
            crop_size = random.randint(4 * self.patch_size, 16 * self.patch_size)

            # Resize image if too small
            if min(image.size) < crop_size:
                scale = crop_size / min(image.size)
                new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
                image = image.resize(new_size, Image.LANCZOS)

            # Random crop
            left = random.randint(0, image.size[0] - crop_size)
            top = random.randint(0, image.size[1] - crop_size)
            patch = image.crop((left, top, left + crop_size, top + crop_size))

            # Resize to patch size
            patch = patch.resize((self.patch_size, self.patch_size), Image.LANCZOS)

            # Random grayscale conversion (10% probability)
            if random.random() < 0.1:
                patch = patch.convert('L').convert('RGB')

            return self.transform(patch)

        except Exception as e:
            # Return random noise if image loading fails
            return torch.randn(3, self.patch_size, self.patch_size)


def pretrain_teacher(model, imagenet_dir, patch_size=65, num_iterations=50000, 
                    batch_size=64, learning_rate=2e-4, device='cuda'):
    """
    Pretrain teacher network using knowledge distillation from ResNet-18
    """
    print(f"Pretraining teacher network (patch_size={patch_size})")

    # Load pretrained ResNet-18 for knowledge distillation
    resnet18 = models.resnet18(pretrained=True)
    # Remove final classification layer and get 512-dim features
    resnet18.fc = nn.Identity()
    resnet18 = resnet18.to(device)
    resnet18.eval()

    # Prepare dataset and dataloader
    patch_dataset = ImageNetPatchDataset(imagenet_dir, patch_size)
    dataloader = DataLoader(patch_dataset, batch_size=batch_size, shuffle=True, 
                           num_workers=4, pin_memory=True)

    # Optimizer
    optimizer = optim.Adam(model.teacher.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Training loop
    model.train()
    running_loss = 0.0
    iteration = 0

    pbar = tqdm(total=num_iterations, desc="Teacher pretraining")

    while iteration < num_iterations:
        for batch in dataloader:
            if iteration >= num_iterations:
                break

            patches = batch.to(device)

            # Forward pass through teacher
            teacher_features, teacher_decoded = model.teacher(patches, return_decoded=True)

            # Forward pass through ResNet-18
            with torch.no_grad():
                resnet_features = resnet18(patches)

            # Knowledge distillation loss
            kd_loss = F.mse_loss(teacher_decoded, resnet_features)

            # Descriptor compactness loss (correlation minimization)
            # Compute correlation matrix within the batch
            teacher_centered = teacher_features - teacher_features.mean(dim=0, keepdim=True)
            correlation_matrix = torch.mm(teacher_centered.t(), teacher_centered) / (batch_size - 1)
            correlation_matrix.fill_diagonal_(0)  # Remove diagonal elements
            compactness_loss = torch.sum(torch.abs(correlation_matrix))

            # Total loss
            total_loss = kd_loss + compactness_loss

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            iteration += 1
            pbar.update(1)

            if iteration % 1000 == 0:
                avg_loss = running_loss / 1000
                pbar.set_postfix({'Loss': f'{avg_loss:.4f}'})
                running_loss = 0.0

    pbar.close()
    print(f"Teacher pretraining completed")


def compute_normalization_stats(model, dataloader, device):
    """
    Compute mean and std of teacher features on training data for normalization
    """
    print("Computing normalization statistics...")

    model.eval()
    features_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing stats"):
            images = batch.to(device)

            # Extract features using dense convolution (full image)
            batch_size, channels, height, width = images.shape

            # Use unfold to extract all possible patches
            patch_size = model.patch_size
            patches = F.unfold(images, kernel_size=patch_size, stride=1)
            patches = patches.transpose(1, 2).reshape(-1, channels, patch_size, patch_size)

            # Forward through teacher in smaller batches to save memory
            batch_features = []
            for i in range(0, patches.size(0), 1000):
                patch_batch = patches[i:i+1000]
                features = model.forward_teacher(patch_batch)
                batch_features.append(features.cpu())

            if batch_features:
                batch_features = torch.cat(batch_features, dim=0)
                features_list.append(batch_features)

    if features_list:
        all_features = torch.cat(features_list, dim=0)
        mean = all_features.mean(dim=0)
        std = all_features.std(dim=0)

        # Ensure std is not zero
        std = torch.clamp(std, min=1e-6)

        return mean, std
    else:
        # Fallback
        output_dim = model.teacher.patch_cnn.features[-2].out_channels
        return torch.zeros(output_dim), torch.ones(output_dim)


def train_students(model, dataloader, patch_size=65, num_epochs=100, 
                  learning_rate=1e-4, device='cuda'):
    """
    Train student ensemble to regress teacher outputs on normal data
    """
    print(f"Training student ensemble (patch_size={patch_size}, num_students={model.num_students})")

    # Optimizers for each student
    optimizers = []
    for student in model.students:
        optimizer = optim.Adam(student.parameters(), lr=learning_rate, weight_decay=1e-5)
        optimizers.append(optimizer)

    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in pbar:
            images = batch.to(device)
            batch_size, channels, height, width = images.shape

            # Extract patches using unfold
            patches = F.unfold(images, kernel_size=patch_size, stride=1)
            patches = patches.transpose(1, 2).reshape(-1, channels, patch_size, patch_size)

            if patches.size(0) == 0:
                continue

            # Forward through teacher
            with torch.no_grad():
                teacher_features = model.forward_teacher(patches)
                # Normalize teacher features
                teacher_normalized = (teacher_features - model.feature_mean) / model.feature_std

            # Train each student
            batch_loss = 0.0
            for i, (student, optimizer) in enumerate(zip(model.students, optimizers)):
                student_features = student(patches)

                # MSE loss between student and normalized teacher
                loss = F.mse_loss(student_features, teacher_normalized)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss += loss.item()

            batch_loss /= model.num_students
            epoch_loss += batch_loss
            num_batches += 1

            pbar.set_postfix({'Loss': f'{batch_loss:.4f}'})

        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train Student-Teacher Anomaly Detection')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to MVTecAD dataset')
    parser.add_argument('--imagenet_path', type=str, 
                       help='Path to ImageNet data for teacher pretraining')
    parser.add_argument('--category', type=str, required=True,
                       help='MVTecAD category to train on')
    parser.add_argument('--patch_size', type=int, default=65, choices=[17, 33, 65],
                       help='Patch size for networks')
    parser.add_argument('--num_students', type=int, default=3,
                       help='Number of student networks in ensemble')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                       help='Directory to save model checkpoints')
    parser.add_argument('--skip_teacher_pretraining', action='store_true',
                       help='Skip teacher pretraining (use if already pretrained)')
    parser.add_argument('--teacher_checkpoint', type=str,
                       help='Path to pretrained teacher checkpoint')

    args = parser.parse_args()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    model = StudentTeacherEnsemble(
        patch_size=args.patch_size,
        output_dim=128,
        num_students=args.num_students
    ).to(device)

    # Create output directory
    output_dir = Path(args.output_dir) / args.category
    output_dir.mkdir(parents=True, exist_ok=True)

    # Teacher pretraining
    if not args.skip_teacher_pretraining:
        if args.imagenet_path:
            pretrain_teacher(model, args.imagenet_path, args.patch_size, device=device)
            # Save pretrained teacher
            teacher_path = output_dir / f'teacher_pretrained_patch{args.patch_size}.pth'
            torch.save(model.teacher.state_dict(), teacher_path)
            print(f"Pretrained teacher saved to {teacher_path}")
        else:
            print("Warning: No ImageNet path provided, skipping teacher pretraining")
    elif args.teacher_checkpoint:
        # Load pretrained teacher
        model.teacher.load_state_dict(torch.load(args.teacher_checkpoint, map_location=device))
        print(f"Loaded pretrained teacher from {args.teacher_checkpoint}")

    # Prepare training dataset
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = MVTecADDataset(
        root_dir=args.dataset_path,
        category=args.category,
        split='train',
        transform=transform
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True
    )

    print(f"Training dataset loaded: {len(train_dataset)} images")

    # Compute normalization statistics
    mean, std = compute_normalization_stats(model, train_dataloader, device)
    model.set_normalization_params(mean, std)
    print(f"Normalization stats computed: mean={mean[:5]}, std={std[:5]}")

    # Train students
    train_students(model, train_dataloader, args.patch_size, device=device)

    # Save trained model
    model_path = output_dir / f'student_teacher_patch{args.patch_size}.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'patch_size': args.patch_size,
        'num_students': args.num_students,
        'category': args.category,
        'feature_mean': model.feature_mean,
        'feature_std': model.feature_std
    }, model_path)

    print(f"Training completed! Model saved to {model_path}")


if __name__ == '__main__':
    main()
