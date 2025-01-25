# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import torch
from torchvision import transforms
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # Serves the frontend

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


# Function to load the model
def load_model(model, path):
    model.load_state_dict(torch.load(path))
  
    model.eval()  # Set the model to evaluation mode
    print(f'Model loaded from {path}')


# Define the model class (ResNet18 for single-channel input)
def get_resnet18_model(num_classes):
    model = models.resnet18(pretrained=False)  # pretrained=False as we load custom weights
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# Specify the number of classes
num_classes = 4  # COVID-19, Normal, Lung Opacity, Viral Pneumonia

# Initialize the model and load the trained weights
model = get_resnet18_model(num_classes)


# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load the PyTorch model
load_model(model,'attack on chest model weights.pth')


# Define alphabet mapping (index 0 -> 'A', index 25 -> 'Z')
ALPHABET = [chr(i) for i in range(65, 91)]  # Generate ['A', 'B', ..., 'Z']

# Define image preprocessing
preprocess = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

@app.route('/upload-image', methods=['POST'])
def upload_image():
    try:
        # Check if an image file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400

        file = request.files['file']  # Get the uploaded file
        if file.filename == '':
            return jsonify({'error': 'No file selected for uploading'}), 400

        # Read the image file
        npimg = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Preprocess the image
        input_tensor = preprocess(frame).unsqueeze(0)  # Add batch dimension

        # Perform inference
        with torch.no_grad():
            output = model(input_tensor)  # Raw logits for each class
            probabilities = torch.softmax(output, dim=1)  # Get probabilities
            _, predicted_class = torch.max(output, 1)  # Get the class index
            confidence = probabilities[0, predicted_class].item()

        # Map the predicted class to the corresponding alphabet letter
        predicted_letter = ALPHABET[predicted_class.item()]  # Convert index to letter

        # Return the result
        return jsonify({
            'predicted_letter': predicted_letter,  # The letter (A-Z)
            'confidence': confidence  # Confidence score
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
