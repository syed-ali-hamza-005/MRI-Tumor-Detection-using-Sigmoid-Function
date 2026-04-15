# 🧠 Brain Tumor Detector: Built from Scratch with Pure Math

An end-to-end Machine Learning web application that predicts the presence of a brain tumor from an MRI scan. 

Unlike standard projects that rely on high-level libraries like TensorFlow or PyTorch, the core classification model for this project was built **completely from scratch using pure linear algebra, calculus, and NumPy**. The trained mathematical weights are then served through a custom Flask web interface.

## ✨ Key Features
* **Custom Neural Network (Logistic Regression):** Implemented Gradient Descent, the Sigmoid activation function, and Log-Loss (Cross-Entropy) cost functions manually.
* **End-to-End Pipeline:** Handles everything from image flattening and matrix transformations to training, weight extraction, and inference.
* **Interactive Web Interface:** A clean, user-friendly Flask frontend that allows users to upload an MRI scan, view the uploaded image, and instantly receive a prediction with a confidence percentage.

## 🛠️ Tech Stack
* **Backend:** Python, Flask
* **Machine Learning:** NumPy (Matrix operations), Calculus (Gradient Descent)
* **Image Processing:** Pillow (PIL), Keras Preprocessing (strictly for image loading/resizing)
* **Frontend:** HTML5, CSS3

## 📂 Project Structure
```text
Brain-Tumor-Web-App/
│
├── app.py                 # The Flask server and inference math
├── manual_weights.npz     # The pre-trained weights (W) and bias (b)
├── .gitignore             # Keeps the venv and cache files out of the repo
│
├── static/
│   └── uploads/           # Temporary storage for user-uploaded MRI scans
│
└── templates/
    └── index.html         # The frontend UI