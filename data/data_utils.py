
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path

class TransistorDataset(Dataset):
    """
    Dataset class for transistor anomaly detection
    Handles both training (normal images only) and testing (normal + anomalous images)
    """

    def __init__(self, root_dir, mode='train', transform=None, target_size=(256, 256)):
        """
        Args:
            root_dir: Path to dataset root directory
            mode: 'train' for training data, 'test' for testing data
            transform: Optional transform to be applied on samples
            target_size: Target image size (H, W)
        """
        self.root_dir = Path(root_dir)
        self.mode = mode
        self.target_size = target_size

        # Default transforms: convert to grayscale and normalize
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
                transforms.Resize(target_size),  # Resize to target size
                transforms.ToTensor(),  # Convert to tensor [0, 1]
            ])
        else:
            self.transform = transform

        self.images = []
        self.labels = []  # 0 for normal, 1 for anomalous
        self.masks = []   # Ground truth masks for anomalous images

        self._load_data()

    def _load_data(self):
        """Load image paths and labels based on dataset structure"""
        if self.mode == 'train':
            # Training data: only normal images from train folder
            train_dir = self.root_dir / 'train'
            if train_dir.exists():
                # Look for images in train directory
                for img_path in sorted(train_dir.glob('*.png')):
                    self.images.append(img_path)
                    self.labels.append(0)  # All training images are normal
                    self.masks.append(None)  # No masks for training

                # Also check subdirectories
                for subdir in train_dir.iterdir():
                    if subdir.is_dir():
                        for img_path in sorted(subdir.glob('*.png')):
                            self.images.append(img_path)
                            self.labels.append(0)
                            self.masks.append(None)

        elif self.mode == 'test':
            # Testing data: images from test folder with ground truth
            test_dir = self.root_dir / 'test'
            gt_dir = self.root_dir / 'gt_masks'

            if test_dir.exists():
                # Load test images
                for img_path in sorted(test_dir.glob('*.png')):
                    self.images.append(img_path)

                    # Check if ground truth mask exists
                    mask_path = gt_dir / img_path.name if gt_dir.exists() else None

                    if mask_path and mask_path.exists():
                        # Anomalous image (has ground truth mask)
                        self.labels.append(1)
                        self.masks.append(mask_path)
                    else:
                        # Normal image (no mask)
                        self.labels.append(0)
                        self.masks.append(None)

                # Also check subdirectories
                for subdir in test_dir.iterdir():
                    if subdir.is_dir():
                        for img_path in sorted(subdir.glob('*.png')):
                            self.images.append(img_path)

                            # Check for corresponding mask
                            if gt_dir.exists():
                                mask_path = gt_dir / subdir.name / img_path.name
                                if mask_path.exists():
                                    self.labels.append(1)
                                    self.masks.append(mask_path)
                                else:
                                    self.labels.append(0)
                                    self.masks.append(None)
                            else:
                                # Assume normal if no gt_masks directory
                                self.labels.append(0)
                                self.masks.append(None)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """Get sample at index idx"""
        img_path = self.images[idx]
        label = self.labels[idx]
        mask_path = self.masks[idx]

        # Load and transform image
        image = Image.open(img_path).convert('RGB')  # Ensure RGB before grayscale conversion
        image = self.transform(image)

        # Load mask if available
        if mask_path:
            mask = Image.open(mask_path).convert('L')  # Load as grayscale
            mask = transforms.Compose([
                transforms.Resize(self.target_size),
                transforms.ToTensor(),
            ])(mask)
            # Binarize mask
            mask = (mask > 0.5).float()
        else:
            # No mask available
            mask = torch.zeros(1, *self.target_size)

        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'mask': mask,
            'path': str(img_path)
        }

def create_data_augmentation(target_size=(256, 256), for_objects=True):
    """
    Create data augmentation transforms based on paper recommendations

    Args:
        target_size: Target image size
        for_objects: True for object images, False for texture images

    Returns:
        Transform composition
    """
    if for_objects:
        # For objects: random translation, rotation, and zoom
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomRotation(degrees=15),  # Small rotation
            transforms.RandomResizedCrop(target_size, scale=(0.8, 1.2)),  # Random zoom
            transforms.RandomHorizontalFlip(p=0.5),  # Horizontal flip for transistors
            transforms.ToTensor(),
        ])
    else:
        # For textures: random patches
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomCrop(target_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
        ])

def get_dataloaders(data_root, batch_size=128, num_workers=4, target_size=(256, 256)):
    """
    Create training and testing dataloaders

    Args:
        data_root: Root directory of dataset
        batch_size: Batch size for training (as per paper)
        num_workers: Number of data loading workers
        target_size: Target image size

    Returns:
        train_loader, test_loader
    """
    # Training dataset with augmentation
    train_transform = create_data_augmentation(target_size, for_objects=True)
    train_dataset = TransistorDataset(
        root_dir=data_root,
        mode='train',
        transform=train_transform,
        target_size=target_size
    )

    # Test dataset without augmentation
    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ])
    test_dataset = TransistorDataset(
        root_dir=data_root,
        mode='test',
        transform=test_transform,
        target_size=target_size
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Process test images one by one for evaluation
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader

if __name__ == "__main__":
    # Test dataset loading
    print("Testing dataset loading...")

    # Create dummy data structure for testing
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create fake directory structure
        train_dir = Path(tmp_dir) / 'train'
        test_dir = Path(tmp_dir) / 'test'
        gt_dir = Path(tmp_dir) / 'gt_masks'

        train_dir.mkdir()
        test_dir.mkdir()
        gt_dir.mkdir()

        # Create dummy images
        dummy_img = Image.new('RGB', (300, 300), color='gray')
        dummy_img.save(train_dir / 'normal_001.png')
        dummy_img.save(test_dir / 'test_001.png')

        dummy_mask = Image.new('L', (300, 300), color='white')
        dummy_mask.save(gt_dir / 'test_001.png')

        # Test dataset
        dataset = TransistorDataset(tmp_dir, mode='train')
        print(f"Training samples: {len(dataset)}")

        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Sample image shape: {sample['image'].shape}")
            print(f"Sample label: {sample['label']}")

        test_dataset = TransistorDataset(tmp_dir, mode='test')
        print(f"Test samples: {len(test_dataset)}")

    print("Dataset utilities test completed!")
