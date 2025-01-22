import torch
import torch.nn as nn

class FeatureFusionClassifier(nn.Module):
    def __init__(self, cnn_output_dim, vit_embed_dim, num_classes):
        super(FeatureFusionClassifier, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected layers for classification
        self.fc1 = nn.Linear(cnn_output_dim + vit_embed_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        # Activation function
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, cnn_features, vit_features):
        # Apply global average pooling to ViT features
        vit_features = vit_features.mean(dim=1)

        # Concatenate CNN and ViT features
        fused_features = torch.cat((cnn_features, vit_features), dim=1)

        # Fully connected layers
        x = self.relu(self.fc1(fused_features))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        # Output probabilities
        output = self.softmax(x)
        return output
