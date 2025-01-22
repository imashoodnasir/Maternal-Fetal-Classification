import torch
import torch.nn as nn
import torch.optim as optim
from cnn_module import CNNModule
from vit_module import VisionTransformer
from feature_fusion_classifier import FeatureFusionClassifier
from torch.utils.data import DataLoader, Dataset
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

# Example Dataset Class (Customize for your dataset)
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

# Define the complete model
class HybridCNNViT(nn.Module):
    def __init__(self, cnn_params, vit_params, fusion_params):
        super(HybridCNNViT, self).__init__()
        self.cnn = CNNModule(**cnn_params)
        self.vit = VisionTransformer(**vit_params)
        self.classifier = FeatureFusionClassifier(**fusion_params)

    def forward(self, x):
        # CNN Features
        cnn_features = self.cnn(x)

        # Reshape CNN output for ViT (batch_size, num_patches, input_dim)
        vit_input = cnn_features.view(cnn_features.size(0), -1, self.vit.embed_dim)
        vit_features = self.vit(vit_input)

        # Classification
        output = self.classifier(cnn_features, vit_features)
        return output

# Initialize the model
cnn_params = {'input_channels': 3, 'output_dim': 128}
vit_params = {'input_dim': 128, 'num_patches': 64, 'embed_dim': 128, 'num_heads': 4, 'mlp_dim': 256, 'num_layers': 2}
fusion_params = {'cnn_output_dim': 128, 'vit_embed_dim': 128, 'num_classes': 6}

model = HybridCNNViT(cnn_params, vit_params, fusion_params).to('cuda')

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()

# Hyperparameter tuning with Bayesian Optimization
space = [
    Real(1e-5, 1e-2, name='learning_rate'),
    Integer(10, 100, name='epochs'),
    Integer(16, 128, name='batch_size')
]

@use_named_args(space)
def objective(**params):
    learning_rate = params['learning_rate']
    epochs = params['epochs']
    batch_size = params['batch_size']

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_val_accuracy = 0
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation step
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to('cuda'), labels.to('cuda')
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = correct / total
        best_val_accuracy = max(best_val_accuracy, val_accuracy)

    return -best_val_accuracy

# Run Bayesian Optimization
result = gp_minimize(objective, space, n_calls=20, random_state=42)

print("Best Hyperparameters:")
print(f"Learning Rate: {result.x[0]}")
print(f"Epochs: {result.x[1]}")
print(f"Batch Size: {result.x[2]}")
