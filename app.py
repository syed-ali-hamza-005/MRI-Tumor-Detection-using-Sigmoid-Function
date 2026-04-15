from flask import Flask, render_template, request
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Ensure 'uploads' folder exists
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load manual trained mathematically weights
try:
    weights = np.load('manual_weights.npz')
    W = weights['W']
    b = weights['b']
except FileNotFoundError:
    print("Error: Could not find 'manual_weights.npz'. Make sure it is in the same folder as app.py")
    W, b = None, None

# 1. Sigmoid Function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 2. Prediction
def predict_tumor(img_path):
    # Open image, resize to 64*64, and convert to RGB
    img = Image.open(img_path).convert('RGB')
    img = img.resize((64, 64))

    # Convert to numbers and flatten into column vector
    img_array = np.array(img) / 255.0
    X_test = img_array.flatten().reshape(-1, 1)

    # Apply formula: Z = W*X + b
    Z = np.dot(W.T, X_test) + b
    probability = sigmoid(Z)[0][0]

    if probability > 0.5:
        return f"Tumor Detected! (Confidence: {probability*100:.2f}%)", "red"
    else:
        return f"Healthy Brain. (Confidence: {probability*100:.2f}%)", "green"

# Web Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    text_color = "black"
    image_url = None

    if request.method == 'POST':
        # Check if a file is uploaded
        if 'file' not in request.files:
            return render_template('index.html', prediction="No file selected.")

        file = request.files['file']
        if file.filename != '':
            # Save file temporarily
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Run math prediction
            if W is not None:
                prediction, text_color = predict_tumor(filepath)
                image_url = f"/static/uploads/{file.filename}"
            else:
                prediction = "Error: Model weights not loaded."
                text_color = "red"

    return render_template('index.html', prediction=prediction, color=text_color, image_url=image_url)

if __name__ == '__main__':
    app.run(debug=True)