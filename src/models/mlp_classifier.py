"""
Multi-layer Perceptron classifier for emotion recognition from audio embeddings.

This module implements the MLP architecture used on top of frozen HuBERT-Large
embeddings for audio-based emotion classification.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
import torch
import torch.nn as nn


class EmotionMLP(nn.Module):
    """
    Multi-layer perceptron for emotion classification from audio embeddings.
    
    This architecture is trained on top of frozen HuBERT-Large embeddings
    (1024-dimensional) for emotion recognition. It uses batch normalization
    and dropout for regularization.
    
    Architecture:
        Input (1024) → Linear(512) → BN → ReLU → Dropout
                    → Linear(256) → BN → ReLU → Dropout
                    → Linear(128) → BN → ReLU → Dropout
                    → Linear(num_classes)
    
    Args:
        input_dim (int): Dimensionality of input embeddings. Default: 1024
                        (HuBERT-Large hidden size)
        hidden_dims (tuple): Sizes of hidden layers. Default: (512, 256, 128)
        num_classes (int): Number of emotion classes. Default: 4
        dropout (float): Dropout probability. Default: 0.4
    
    Input:
        x (Tensor): Audio embeddings of shape [batch_size, input_dim]
    
    Output:
        logits (Tensor): Emotion logits of shape [batch_size, num_classes]
    
    Example:
        >>> mlp = EmotionMLP(input_dim=1024, hidden_dims=(512, 256, 128))
        >>> embeddings = torch.randn(32, 1024)  # batch=32, HuBERT embedding
        >>> logits = mlp(embeddings)
        >>> print(logits.shape)  # torch.Size([32, 4])
    """
    
    def __init__(self, 
                 input_dim=1024, 
                 hidden_dims=(512, 256, 128),
                 num_classes=NUM_CLASSES, 
                 dropout=0.4):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers with BatchNorm and Dropout
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        
        # Output layer (no activation, returns logits)
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.net = nn.Sequential(*layers)
        
        # Store configuration
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dropout = dropout
    
    def forward(self, x):
        """
        Forward pass through the MLP.
        
        Args:
            x (Tensor): Input embeddings [batch_size, input_dim]
        
        Returns:
            Tensor: Emotion logits [batch_size, num_classes]
        """
        return self.net(x)
    
    def get_num_parameters(self):
        """
        Get the total number of trainable parameters.
        
        Returns:
            int: Total parameter count
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_mlp_for_hubert(num_classes=4, dropout=0.4):
    """
    Factory function to create an MLP specifically for HuBERT-Large embeddings.
    
    Args:
        num_classes (int): Number of emotion classes
        dropout (float): Dropout probability
    
    Returns:
        EmotionMLP: Configured MLP model
    """
    return EmotionMLP(
        input_dim=1024,  # HuBERT-Large hidden size
        hidden_dims=(512, 256, 128),
        num_classes=num_classes,
        dropout=dropout
    )


if __name__ == "__main__":
    # Test the MLP architecture
    print("Testing EmotionMLP...")
    
    # Create model
    mlp = create_mlp_for_hubert(num_classes=4, dropout=0.4)
    print(f"Model parameters: {mlp.get_num_parameters():,}")
    print(f"Configuration:")
    print(f"  Input dim: {mlp.input_dim}")
    print(f"  Hidden dims: {mlp.hidden_dims}")
    print(f"  Num classes: {mlp.num_classes}")
    print(f"  Dropout: {mlp.dropout}")
    
    # Test forward pass
    batch_size = 16
    embeddings = torch.randn(batch_size, 1024)
    
    mlp.eval()
    with torch.no_grad():
        logits = mlp(embeddings)
        probs = torch.nn.functional.softmax(logits, dim=-1)
    
    print(f"\nTest forward pass:")
    print(f"  Input: {embeddings.shape}")
    print(f"  Logits: {logits.shape}")
    print(f"  Probs: {probs.shape}")
    print(f"  Probs sum: {probs.sum(dim=-1)}")  # Should be all 1.0
    
    print("\n✅ EmotionMLP test passed!")
