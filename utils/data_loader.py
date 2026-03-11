import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
from PIL import Image

class CustomImageDataset(Dataset):
    """
    Custom Dataset for loading images from subfolders. 
    (Matches the structure of the provided 'train' folder which contains subfolders like 'Amphibia', 'Animalia', etc.)
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(
            [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        )
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for f in os.listdir(class_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(class_dir, f), self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_data_loaders(data_dir, batch_size=32, validation_split=0.2, num_workers=4, image_size=224):
    """
    Creates train and validation dataloaders, including data augmentation.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory {data_dir} does not exist. Please place the 'train' folder there.")

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Check if data_dir contains subfolders (like 'Amphibia', 'Animalia')
    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    if len(subdirs) > 0:
        # Structure 1: Train folder contains class subfolders (what you showed in image 2)
        print("Detected class subdirectories inside train folder.")
        # We can just use standard ImageFolder because it already handles this
        full_dataset = datasets.ImageFolder(root=data_dir)
        class_names = full_dataset.classes
    else:
        raise ValueError(
            f"Expected subdirectories (classes) inside {data_dir}, but found none. "
            "Please ensure you placed the contents of your Google Drive 'train' folder here."
        )

    total_size = len(full_dataset)
    val_size = int(total_size * validation_split)
    train_size = total_size - val_size

    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_dataset.dataset.transform = train_transforms
    
    import copy
    val_dataset_transformed = copy.deepcopy(val_dataset)
    val_dataset_transformed.dataset.transform = val_transforms

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset_transformed, batch_size=batch_size, shuffle=False, 
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
