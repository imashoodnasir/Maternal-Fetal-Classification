import os
import numpy as np
import torch
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

# Custom Dataset Class
class MaternalFetalDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Preprocessing Function
def load_and_preprocess_data(data_dir, image_size=(224, 224)):
    """
    Load images and labels from the dataset directory and preprocess them.
    
    Args:
        data_dir (str): Path to the dataset directory.
        image_size (tuple): Desired image size (width, height).
    
    Returns:
        train_dataset, val_dataset: PyTorch Dataset objects for training and validation.
    """
    images = []
    labels = []
    classes = sorted(os.listdir(data_dir))  # Assuming one folder per class

    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

    # Load images and labels
    for cls_name in classes:
        cls_path = os.path.join(data_dir, cls_name)
        for file_name in os.listdir(cls_path):
            img_path = os.path.join(cls_path, file_name)
            img = np.load(img_path)  # Assuming pre-saved NumPy arrays
            images.append(img)
            labels.append(class_to_idx[cls_name])

    images = np.array(images)
    labels = np.array(labels)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42, stratify=labels)

    # Define image transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(image_size),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Assuming grayscale images
    ])

    # Create Dataset objects
    train_dataset = MaternalFetalDataset(X_train, y_train, transform=transform)
    val_dataset = MaternalFetalDataset(X_val, y_val, transform=transform)

    return train_dataset, val_dataset

# Example Usage
data_dir = "path/to/your/dataset"  # Replace with your dataset path
train_dataset, val_dataset = load_and_preprocess_data(data_dir)

print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")
