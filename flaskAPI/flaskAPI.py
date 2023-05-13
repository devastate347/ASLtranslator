from flask import Flask, jsonify, request
from PIL import Image
import io

app = Flask(name)

@app.route('/translate', methods=['POST'])
def translate():
    img_bytes = io.BytesIO(request.data)
    img = Image.open(imgbytes)
    img = transform(img).unsqueeze(0)
    output = model(img)
    , predicted = torch.max(output.data, 1)
    return jsonify({'predicted_text': chr(predicted.item() + 65)})
