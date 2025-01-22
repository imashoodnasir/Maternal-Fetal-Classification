import torch
import torch.nn as nn

class VisionTransformer(nn.Module):
    def __init__(self, input_dim, num_patches, embed_dim, num_heads, mlp_dim, num_layers):
        super(VisionTransformer, self).__init__()
        self.num_patches = num_patches
        self.embed_dim = embed_dim

        # Linear projection of flattened patches
        self.patch_embedding = nn.Linear(input_dim, embed_dim)
        
        # Positional Encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, num_patches, embed_dim))

        # Transformer Encoder
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(embed_dim, num_heads, mlp_dim)
            for _ in range(num_layers)
        ])

        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Flatten patches and apply linear projection
        x = self.patch_embedding(x)

        # Add positional encoding
        x += self.positional_encoding

        # Pass through Transformer encoder layers
        for layer in self.encoder_layers:
            x = layer(x)

        # Apply Layer Norm
        x = self.layer_norm(x)
        return x
