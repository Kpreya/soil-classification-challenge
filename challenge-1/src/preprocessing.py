import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class SoilDataset(Dataset):
    """Custom Dataset class for soil classification"""
    
    def __init__(self, df, img_dir, transform=None, is_test=False):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test
        self.class_to_idx = {"Alluvial soil": 0, "Black Soil": 1, "Clay soil": 2, "Red soil": 3}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.df.iloc[idx]["image_id"])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.is_test:
            return image, 0  # Dummy label for test set

        label = self.class_to_idx[self.df.iloc[idx]["soil_type"]]
        return image, label


def get_transforms():
    """Define data transforms for training and validation"""
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return train_transform, val_transform


def load_and_split_data(train_csv_path, test_size=0.2, random_state=42):
    """Load training data and split into train/validation sets"""
    
    train_df = pd.read_csv(train_csv_path)
    print("Training data shape:", train_df.shape)
    print("\nClass distribution:")
    print(train_df["soil_type"].value_counts())

    # Split into train and validation sets
    train_df, val_df = train_test_split(
        train_df, 
        test_size=test_size, 
        stratify=train_df["soil_type"], 
        random_state=random_state
    )
    
    return train_df, val_df


def prepare_datasets(train_df, val_df, test_csv_path, train_dir, test_dir):
    """Prepare datasets with transforms"""
    
    train_transform, val_transform = get_transforms()
    
    # Create datasets
    train_dataset = SoilDataset(train_df, train_dir, transform=train_transform)
    val_dataset = SoilDataset(val_df, train_dir, transform=val_transform)
    
    # Load test data
    test_df = pd.read_csv(test_csv_path)
    test_dataset = SoilDataset(test_df, test_dir, transform=val_transform, is_test=True)
    
    return train_dataset, val_dataset, test_dataset, test_df
