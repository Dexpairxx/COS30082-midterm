import os
import copy
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset, Subset
from PIL import Image


class SubsetWithTransform(Dataset):
    """
    A wrapper around a Subset that applies a specific transform,
    without modifying the original dataset's transform.
    This avoids the bug where train and val share the same transform object.
    """
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        # At this point, img is a PIL Image (because we set dataset.transform = None)
        if self.transform:
            img = self.transform(img)
        return img, label


def get_data_loaders(data_dir, batch_size=32, validation_split=0.2, num_workers=4, image_size=224):
    """
    Creates train and validation dataloaders, including data augmentation.
    Uses SubsetWithTransform to safely apply different transforms to train and val sets.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory {data_dir} does not exist. Please place the 'train' folder there.")

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Strong Data Augmentation for the Train set to combat overfitting
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Validation Set: Only resizing and normalizing
    val_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Check if data_dir contains subfolders (like 'Amphibia', 'Animalia')
    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    if len(subdirs) > 0:
        print("Detected class subdirectories inside train folder.")
        # Load WITHOUT any transform first — raw PIL images
        full_dataset = datasets.ImageFolder(root=data_dir, transform=None)
        class_names = full_dataset.classes
    else:
        raise ValueError(
            f"Expected subdirectories (classes) inside {data_dir}, but found none. "
            "Please ensure you placed the contents of your Google Drive 'train' folder here."
        )

    total_size = len(full_dataset)
    val_size = int(total_size * validation_split)
    train_size = total_size - val_size

    # Split with a fixed seed for reproducibility (same split every run)
    train_subset, val_subset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Wrap each subset with its own transform — no shared state
    train_dataset = SubsetWithTransform(train_subset, train_transforms)
    val_dataset = SubsetWithTransform(val_subset, val_transforms)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    print(f"Dataset Loaded & Split!")
    print(f"Total Images: {total_size}")
    print(f"Train Size: {train_size} | Validation Size: {val_size}")
    print(f"Number of Classes: {len(class_names)}")
    print(f"Classes: {class_names}")

    return train_loader, val_loader, class_names


if __name__ == "__main__":
    print("Testing DataLoader script...")
