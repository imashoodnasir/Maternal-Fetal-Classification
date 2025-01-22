import torch
import torch.nn as nn
import torch.optim as optim
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from torch.utils.data import DataLoader
from data_preprocessing import load_and_preprocess_data
from train_and_optimize import HybridCNNViT

# Load datasets
data_dir = "path/to/your/dataset"  # Replace with your dataset path
train_dataset, val_dataset = load_and_preprocess_data(data_dir)

# Define hyperparameter search space
space = [
    Real(1e-5, 1e-2, name='learning_rate'),
    Integer(10, 50, name='epochs'),
    Integer(16, 128, name='batch_size')
]

# Initialize the model
cnn_params = {'input_channels': 3, 'output_dim': 128}
vit_params = {'input_dim': 128, 'num_patches': 64, 'embed_dim': 128, 'num_heads': 4, 'mlp_dim': 256, 'num_layers': 2}
fusion_params = {'cnn_output_dim': 128, 'vit_embed_dim': 128, 'num_classes': 6}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@use_named_args(space)
def objective(**params):
    # Extract hyperparameters
    learning_rate = params['learning_rate']
    epochs = params['epochs']
    batch_size = params['batch_size']

    # Prepare data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    model = HybridCNNViT(cnn_params, vit_params, fusion_params).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_accuracy = 0

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = correct / total
        best_val_accuracy = max(best_val_accuracy, val_accuracy)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Return negative accuracy because `gp_minimize` minimizes the objective
    return -best_val_accuracy

# Run Bayesian Optimization
result = gp_minimize(objective, space, n_calls=20, random_state=42)

# Best hyperparameters
print("\nBest Hyperparameters:")
print(f"Learning Rate: {result.x[0]}")
print(f"Epochs: {result.x[1]}")
print(f"Batch Size: {result.x[2]}")

# Save best hyperparameters
best_hyperparameters = {
    'learning_rate': result.x[0],
    'epochs': result.x[1],
    'batch_size': result.x[2]
}

# Save results for further use
torch.save(best_hyperparameters, "best_hyperparameters.pth")
