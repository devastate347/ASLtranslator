import torch
import torchvision.transforms as transforms
from PIL import Image


model = torch.load('asl_model.pth', map_location=torch.device('cpu'))


transform = transforms.Compose([
    transforms.Resize((224, 224)), # Resize image to 224x224
    transforms.ToTensor(), # Convert image to PyTorch tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalize the image
])


image = Image.open('asl_image.jpg')


image_tensor = transform(image).unsqueeze(0)


output = model(image_tensor)

predicted_class = output.argmax(dim=1)


text = chr(ord('A') + predicted_class)


print(text)
