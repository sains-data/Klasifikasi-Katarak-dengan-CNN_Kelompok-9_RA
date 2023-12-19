# app.py

import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, models
from torch.utils.data import DataLoader
import torchvision

# Define the model architecture
model = models.inception_v3(pretrained=True)
num_classes = 2  # Assuming binary classification (cataract or non-cataract)
model.AuxLogits.fc = torch.nn.Linear(768, num_classes)
model.fc = torch.nn.Linear(2048, num_classes)

# Load the trained model weights
model_path = "trained_model.h5"
model.load_state_dict(torch.load(model_path))
model.eval()

# Define the transformation to apply to input images
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create a function to make predictions
def predict(image):
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
    return predicted.item()

# Streamlit app
st.title("Cataract Detection App")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Make a prediction
    prediction = predict(image)

    # Display the prediction
    class_names = ["Non-Cataract", "Cataract"]
    st.subheader("Prediction:")
    st.write(f"The model predicts: {class_names[prediction]}")
