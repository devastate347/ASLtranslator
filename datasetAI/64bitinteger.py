import torch
import torchvision.transforms as transforms
from flask import Flask, request

app = Flask(__name__)

# Define the transformation to be applied to the input images
asl_transform = transforms.Compose([
    transforms.Resize((64, 64)), # Resize images to 64x64
    transforms.ToTensor(), # Convert images to PyTorch tensors
    transforms.Normalize((0.5,), (0.5,)) # Normalize the images
])

# Load the pre-trained PyTorch CNN model
model = torch.load('path/to/pretrained/model')

# Define a dictionary to map predicted classes to text labels
class_to_text = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E',
    5: 'F',
    6: 'G',
    7: 'H',
    8: 'I',
    9: 'J',
    10: 'K',
    11: 'L',
    12: 'M',
    13: 'N',
    14: 'O',
    15: 'P',
    16: 'Q',
    17: 'R',
    18: 'S',
    19: 'T',
    20: 'U',
    21: 'V',
    22: 'W',
    23: 'X',
    24: 'Y',
    25: 'Z'
    
    # add more classes and text labels as needed
}

@app.route('/translate', methods=['POST'])
def translate_asl():
    # Get the input image from the Flask API request
    image_file = request.files['image']
    
    # Preprocess the input image
    image = asl_transform(image_file)
    image = image.unsqueeze(0) # add batch dimension
    
    # Pass the preprocessed image through the CNN model to get the predicted class
    with torch.no_grad():
        model.eval()
        output = model.forward(image)
        predicted_class = torch.argmax(output).item()
    
    # Convert the predicted class to text using the lookup table
    predicted_text = class_to_text[predicted_class]
    
    # Return the predicted text as a response to the Flask API call
    return predicted_text
