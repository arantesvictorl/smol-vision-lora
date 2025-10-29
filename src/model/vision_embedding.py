import torch
import torch.nn as nn
from configs.config import VisionConfig


class VisionEmbedding(nn.Module):
    """
    Lightweight vision embedding layer for processing image patches.
    
    Converts raw image pixels into embeddings compatible with the LLM's
    hidden dimension through patch embedding and positional encoding.
    """
    
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        
        self.patch_embed = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False
        )
        
        self.pos_embed = nn.Parameter(
            torch.randn(1, config.max_patches, config.hidden_dim) * 0.02
        )
        
        self.norm = nn.LayerNorm(config.hidden_dim)
        
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: Tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Tensor of shape (batch_size, num_patches, hidden_dim)
        """
        batch_size = pixel_values.shape[0]
        
        x = self.patch_embed(pixel_values)
        x = x.flatten(2).transpose(1, 2)
        
        num_patches = x.shape[1]
        x = x + self.pos_embed[:, :num_patches, :]
        
        x = self.norm(x)
        
        return x
    
    def save_pretrained(self, save_directory: str):
        """Save vision embedding weights."""
        import os
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_directory, "vision_embed.pt"))
        
    @classmethod
    def from_pretrained(cls, load_directory: str, config: VisionConfig):
        """Load vision embedding weights."""
        import os
        model = cls(config)
        state_dict = torch.load(os.path.join(load_directory, "vision_embed.pt"))
        model.load_state_dict(state_dict)
        return model