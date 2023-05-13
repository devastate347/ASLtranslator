import torch
import torch.nn as nn
import torchvision.models as models

# Load pre-trained model (ResNet18)
model = models.resnet18(pretrained=True)

# Replace the last fully-connected layer to output 29 classes (ASL alphabet)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 29)

# Set model to evaluation mode
model.eval()
