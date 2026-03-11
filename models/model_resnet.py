import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNet18Transfer(nn.Module):
    """
    Transfer Learning Model based on ResNet18.
    The backbone comes pre-trained on ImageNet.
    """
    def __init__(self, num_classes=10, freeze_backbone=False):
        super(ResNet18Transfer, self).__init__()
        
        weights = ResNet18_Weights.IMAGENET1K_V1
        self.model = resnet18(weights=weights)
        
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
                
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)
