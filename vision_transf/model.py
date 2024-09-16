import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    def __init__(self, patch_size, img_size, in_channels, d_model) -> None:
        super().__init__()

        self.n_patches = (img_size // patch_size)**2

        self.conv = nn.Conv2d(
            in_channels,    
            d_model,        
            kernel_size=patch_size, 
            stride=patch_size  
        )

    def forward(self, x: torch.Tensor):
        x = self.conv(x) 
        x = x.flatten(2) 
        x = x.transpose(1, 2)  
        return x

# Initialize patch embedding
patch_size = 16
img_size = 128
in_channels = 3
d_model = 512

patch_embed = PatchEmbed(patch_size, img_size, in_channels, d_model)

# Transformer Encoder Layer
encoder_layer = nn.TransformerEncoderLayer(d_model, nhead=8, dim_feedforward=2048)
encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

# ViT Class
class ViT(nn.Module):
    def __init__(self, patch_embed: PatchEmbed, encoder: nn.TransformerEncoder, num_classes: int):
        super().__init__()
        self.patch_embed = patch_embed
        self.encoder = encoder
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))  
        self.pos_embed = nn.Parameter(torch.zeros(1, patch_embed.n_patches + 1, d_model)) 
        self.fc = nn.Linear(d_model, num_classes)  
    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
        x = self.patch_embed(x)  # Get patch embeddings (batch_size, n_patches, d_model)

        # Class token prep
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, d_model)
        x = torch.cat((cls_tokens, x), dim=1)  # Concatenate class token (batch_size, n_patches + 1, d_model)

        # Add position embeddings
        x = x + self.pos_embed

        # Pass through the Transformer encoder
        x = self.encoder(x)

        # Extract the class token output
        cls_token_final = x[:, 0]  # (batch_size, d_model)

        # Final classification head
        logits = self.fc(cls_token_final)  # (batch_size, num_classes)
        return logits

num_classes = 10
vit_model = ViT(patch_embed, encoder, num_classes)

# Dummy input (batch_size=3, in_channels=3, img_size=128x128)
x = torch.randn(3, 3, 128, 128)

# Forward pass through the model
logits = vit_model(x)
print('Logits shape:', logits.shape)
