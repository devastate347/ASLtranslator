import torch
import torchvision
import torchvision.transforms as transforms
import os
from torchvision.transforms import transforms
import torch.nn as nn


# Define the transformation to be applied to the input images
asl_transform = transforms.Compose([
    transforms.Resize((64, 64)), # Resize images to 64x64
    transforms.ToTensor(), # Convert images to PyTorch tensors
    transforms.Normalize((0.5,), (0.5,)) # Normalize the images
])

english_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# Define path to the unzipped ASL Alphabet dataset
asl_dataset_path = '/path/to/unzipped/asl_alphabet_dataset'

# Define the train and test dataset using the ImageFolder dataset class
asl_train_dataset = datasets.ImageFolder(os.path.join(asl_dataset_path, 'asl_alphabet_train'), transform=asl_transform)
asl_test_dataset = datasets.ImageFolder(os.path.join(asl_dataset_path, 'asl_alphabet_test'), transform=asl_transform)

# Load the English Alphabet dataset
english_dataset_path = '/path/to/unzipped/english_alphabet_dataset'

# Define the train and test dataset using the ImageFolder dataset class
english_train_dataset = datasets.ImageFolder(os.path.join(english_dataset_path, 'english_alphabet_train'), transform=english_transform)
english_test_dataset = datasets.ImageFolder(os.path.join(english_dataset_path, 'english_alphabet_test'), transform=english_transform)


# Split the datasets into training and validation sets
asl_train_set, asl_val_set = torch.utils.data.random_split(asl_train_dataset, [25140, 6960]) # 80% training, 20% validation
english_train_set, english_val_set = torch.utils.data.random_split(english_train_dataset, [22500, 2500]) # 90% training, 10% validation

# Create data loaders to load the data in batches
asl_train_loader = torch.utils.data.DataLoader(asl_train_set, batch_size=64, shuffle=True)
asl_val_loader = torch.utils.data.DataLoader(asl_val_set, batch_size=64, shuffle=False)

english_train_loader = torch.utils.data.DataLoader(english_train_set, batch_size=64, shuffle=True)
english_val_loader = torch.utils.data.DataLoader(english_val_set, batch_size=64, shuffle=False)


class ASLTranslator(nn.Module):
    def __init__(self):
        super(ASLTranslator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 29)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = nn
