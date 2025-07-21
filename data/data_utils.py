
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path

class CustomAnomalyDataset(Dataset):
    """
    Custom anomaly dataset for your structure:
    - Training: all images in train/ are normal (label 0), no masks.
    - Test: all images in test/; label is 1 if corresponding mask in gt_masks/ has any nonzero pixel, else 0.
    """
    def __init__(self, root_dir, mode='train', transform=None, target_size=(256, 256)):
        self.root_dir = Path(root_dir)
        self.mode = mode
        self.transform = transform
        self.target_size = target_size

        self.image_paths = []
        self.labels = []
        self.mask_paths = []

        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(self.target_size),
                transforms.ToTensor(),
                transforms.Grayscale(num_output_channels=1),
                transforms.Normalize([0.5], [0.5])
            ])
        self._load_data()

    def _load_data(self):
        if self.mode == 'train':
            train_dir = self.root_dir / 'train'
            # Look for images in the train directory and its subdirectories
            image_paths = list(train_dir.glob('*.png'))
            for subdir in train_dir.iterdir():
                if subdir.is_dir():
                    image_paths.extend(list(subdir.glob('*.png')))
            
            self.image_paths = sorted(image_paths)
            self.labels = [0] * len(self.image_paths)
            self.mask_paths = [None] * len(self.image_paths)
        elif self.mode == 'test':
            test_dir = self.root_dir / 'test'
            mask_dir = self.root_dir / 'gt_masks'
            for img_path in sorted(test_dir.glob('*.png')):
                self.image_paths.append(img_path)
                mask_path = mask_dir / img_path.name
                self.mask_paths.append(mask_path)
                # Determine label by mask content
                if mask_path.exists():
                    mask = Image.open(mask_path).convert('L')
                    mask_np = np.array(mask)
                    label = 1 if np.any(mask_np > 0) else 0
                else:
                    label = 0
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        mask_path = self.mask_paths[idx]

        # Always convert to RGB first to handle single-channel source images consistently
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # The transform pipeline now correctly handles grayscale conversion.
        # No extra channel logic is needed here.

        if mask_path and mask_path.exists():
            mask = Image.open(mask_path).convert('L')
            mask = transforms.functional.resize(mask, self.target_size)
            mask = transforms.functional.to_tensor(mask)
            mask[mask > 0] = 1
        else:
            mask = torch.zeros((1, *self.target_size))

        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'mask': mask,
            'path': str(img_path)
        }

def get_dataloaders(data_root, batch_size=32, num_workers=4, target_size=(256, 256), to_grayscale=False):
    """Creates training and testing dataloaders."""
    
    train_transforms_list = [
        transforms.Resize(target_size),
    ]
    test_transforms_list = [
        transforms.Resize(target_size),
    ]

    if to_grayscale:
        train_transforms_list.append(transforms.Grayscale(num_output_channels=1))
        test_transforms_list.append(transforms.Grayscale(num_output_channels=1))

    train_transforms_list += [
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ]
    test_transforms_list += [
        transforms.ToTensor(),
    ]

    train_transform = transforms.Compose(train_transforms_list)
    test_transform = transforms.Compose(test_transforms_list)

    train_dataset = CustomAnomalyDataset(
        root_dir=data_root,
        mode='train',
        transform=train_transform,
        target_size=target_size
    )
    test_dataset = CustomAnomalyDataset(
        root_dir=data_root,
        mode='test',
        transform=test_transform,
        target_size=target_size
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return train_loader, test_loader

if __name__ == "__main__":
    # Example of how to use the dataloader
    # Replace with the actual path to your MVTec AD dataset category
    data_path = 'path/to/your/mvtec_ad/transistor'
    
    if not Path(data_path).exists():
        print(f"Path not found: {data_path}")
        print("Please update the `data_path` variable in the __main__ block.")
    else:
        train_loader, test_loader = get_dataloaders(data_path, batch_size=4)
        
        print(f"Number of training batches: {len(train_loader)}")
        print(f"Number of test batches: {len(test_loader)}")

        # Check one batch from the test loader
        try:
            sample_batch = next(iter(test_loader))
            print("\nTest batch details:")
            print("Image tensor shape:", sample_batch['image'].shape)
            print("Labels:", sample_batch['label'])
            print("Mask tensor shape:", sample_batch['mask'].shape)
            print("Image paths:", sample_batch['path'])
        except StopIteration:
            print("Test loader is empty.")
