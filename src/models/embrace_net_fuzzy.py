"""
FuzzyEmbraceNet: Multimodal fusion architecture with Gaussian fuzzy activation.

This module implements the core architecture that extends EmbraceNet with
fuzzy logic for improved handling of emotional ambiguity.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
import torch
import torch.nn as nn
import torch.nn.functional as F


class FuzzyGaussActivation(nn.Module):
    """
    Gaussian fuzzy activation function.
    
    Implements a Gaussian membership function that provides a smooth,
    interpretable activation with fuzzy logic semantics.
    
    Args:
        sigma (float): Standard deviation of the Gaussian. Controls the
                      width of the activation region. Default: 0.7
    
    Mathematical form:
        f(x) = exp(-(x²) / (2σ²))
    """
    def __init__(self, sigma=0.7):
        super().__init__()
        self.sigma = sigma
    
    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of any shape
            
        Returns:
            Tensor: Activated output with same shape as input
        """
        return torch.exp(-(x**2) / (2 * self.sigma**2))


class EmbraceNetFuzzy(nn.Module):
    """
    FuzzyEmbraceNet: Neuro-fuzzy multimodal emotion recognition architecture.
    
    This model extends the EmbraceNet framework by incorporating Gaussian fuzzy
    activation functions to better represent uncertainty in emotion recognition.
    It fuses face, audio, and text modalities through a probabilistic selection
    mechanism and fuzzy classification head.
    
    Architecture:
        1. Projection layers: Map each modality to common latent space
        2. Weighted fusion: Learnable weights combine modality representations
        3. Fuzzy classification: Gaussian activation + dropout + linear output
    
    Args:
        K (int): Dimensionality of the common latent space. Default: 32
        num_classes (int): Number of emotion classes to predict. Default: 4
                          (anger, happiness, sadness, neutral)
    
    Input:
        video, audio, text: Tensors of shape [batch_size, num_classes]
                           containing logits or probabilities from unimodal models
    
    Output:
        logits: Tensor of shape [batch_size, num_classes] with emotion predictions
    
    Example:
        >>> model = EmbraceNetFuzzy(K=32, num_classes=4)
        >>> video = torch.randn(16, 4)  # batch=16, classes=4
        >>> audio = torch.randn(16, 4)
        >>> text = torch.randn(16, 4)
        >>> logits = model(video, audio, text)
        >>> probs = F.softmax(logits, dim=-1)
    """
    
    def __init__(self, K=32, num_classes=NUM_CLASSES):
        super().__init__()
        
        # Projection layers for each modality (video, audio, text)
        # Each maps from num_classes to K-dimensional latent space
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_classes, K),
                nn.ReLU()
            )
            for _ in range(3)
        ])
        
        # Learnable fusion weights (one per modality)
        # Softmax ensures they sum to 1.0
        self.w = nn.Parameter(torch.zeros(3))
        
        # Classification head with fuzzy activation
        self.cls = nn.Sequential(
            nn.Linear(K, 64),
            FuzzyGaussActivation(sigma=0.7),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )
    
    def forward(self, video, audio, text):
        """
        Forward pass through the fuzzy multimodal fusion network.
        
        Args:
            video (Tensor): Face modality features [batch_size, num_classes]
            audio (Tensor): Audio modality features [batch_size, num_classes]
            text (Tensor): Text modality features [batch_size, num_classes]
        
        Returns:
            Tensor: Emotion logits [batch_size, num_classes]
        
        Note:
            If a modality is missing, pass a zero vector of shape [batch_size, num_classes]
            to maintain the tri-modal structure.
        """
        # Project each modality to latent space
        # Result: [batch_size, 3, K] where 3 = number of modalities
        stacked = torch.stack([
            proj(x) for proj, x in zip(self.proj, [video, audio, text])
        ], dim=1)
        
        # Compute normalized fusion weights
        weights = F.softmax(self.w, dim=0)  # Shape: [3]
        
        # Weighted fusion across modalities
        # weights.view(1, 3, 1) broadcasts to [batch_size, 3, K]
        fused = (stacked * weights.view(1, 3, 1)).sum(dim=1)  # Shape: [batch_size, K]
        
        # Classification with fuzzy activation
        logits = self.cls(fused)  # Shape: [batch_size, num_classes]
        
        return logits
    
    def get_fusion_weights(self):
        """
        Get the current learned fusion weights for each modality.
        
        Returns:
            dict: Dictionary with keys 'video', 'audio', 'text' and their
                 corresponding normalized weights
        """
        weights = F.softmax(self.w, dim=0).detach().cpu().numpy()
        return {
            'video': float(weights[0]),
            'audio': float(weights[1]),
            'text': float(weights[2])
        }


# ================================================================
# Utility functions for model loading and inference
# ================================================================

def load_fuzzy_embracenet(checkpoint_path, K=32, num_classes=4, device=None):
    """
    Load a pretrained FuzzyEmbraceNet model from checkpoint.
    
    Args:
        checkpoint_path (str): Path to the .pt or .pth checkpoint file
        K (int): Latent space dimensionality (must match checkpoint)
        num_classes (int): Number of classes (must match checkpoint)
        device (str or torch.device): Device to load model on
    
    Returns:
        EmbraceNetFuzzy: Loaded model in eval mode
    """
    if device is None:
        device = DEVICE
    
    model = EmbraceNetFuzzy(K=K, num_classes=num_classes)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model


if __name__ == "__main__":
    # Test the model architecture
    print("Testing FuzzyEmbraceNet...")
    
    model = EmbraceNetFuzzy(K=32, num_classes=4)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Dummy input
    batch_size = 8
    video = torch.randn(batch_size, 4)
    audio = torch.randn(batch_size, 4)
    text = torch.randn(batch_size, 4)
    
    # Forward pass
    logits = model(video, audio, text)
    probs = F.softmax(logits, dim=-1)
    
    print(f"Input shapes: video={video.shape}, audio={audio.shape}, text={text.shape}")
    print(f"Output logits: {logits.shape}")
    print(f"Output probs: {probs.shape}")
    print(f"Fusion weights: {model.get_fusion_weights()}")
    print("✅ FuzzyEmbraceNet test passed!")
