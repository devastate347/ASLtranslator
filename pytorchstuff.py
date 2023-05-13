import torch
import torchvision
import torchvision.transforms as transforms
import os
from torchvision.transforms import transforms
import torch.nn as nn


# Define the transformation to be applied to the input images
transform = transforms.Compose([
    transforms.Resize((64, 64)), # Resize images to 64x64
    transforms.ToTensor(), # Convert images to PyTorch tensors
    transforms.Normalize((0.5,), (0.5,)) # Normalize the images
])

# Define path to the unzipped ASL Alphabet dataset
dataset_path = 'archive'

# Define transformations to be applied to the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Define the train and test dataset using the ImageFolder dataset class
train_dataset = datasets.ImageFolder(os.path.join(dataset_path, 'asl_alphabet_train'), transform=transform)
test_dataset = datasets.ImageFolder(os.path.join(dataset_path, 'asl_alphabet_test'), transform=transform)


# Load the ASL Alphabet dataset
dataset = torchvision.datasets.ImageFolder(
    root='archive/asl_alphabet_train', # Replace with path to the unzipped ASL Alphabet dataset
    transform=transform
)

# Split the dataset into training and validation sets
train_set, val_set = torch.utils.data.random_split(dataset, [25140, 6960]) # 80% training, 20% validation

# Create data loaders to load the data in batches
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=False)



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
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create an instance of the model
model = ASLTranslator()


import torch.optim as optim

criterion = nn.CrossEntropyLoss() # Loss function
optimizer = optim.Adam(model.parameters(), lr=0.001) # Optimizer

for epoch in range(10): # Number of epochs
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 200 == 199: # Print loss every 200 batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0
            

