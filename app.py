import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models

# Fungsi untuk memproses gambar dan melakukan prediksi
def process_image(image):
    # Load the image
    img = Image.open(image)
    
    # Preprocess the image
    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

    return img_tensor

# Fungsi untuk melakukan prediksi menggunakan model
def predict(image, model):
    # Process the image
    img_tensor = process_image(image)

    # Load the model
    model.eval()

    # Make prediction
    with torch.no_grad():
        output = model(img_tensor)

    # Get class probabilities
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Get the predicted class label
    predicted_class = torch.argmax(probabilities).item()

    return probabilities, predicted_class

# Load the pre-trained InceptionV3 model
model = models.inception_v3(pretrained=False)

# Modify the final fully connected layer for your specific number of classes
num_classes = 4  
model.AuxLogits.fc = torch.nn.Linear(768, num_classes)
model.fc = torch.nn.Linear(2048, num_classes)

# Load the trained model
model.load_state_dict(torch.load('trained_model_compressed.h5', map_location=torch.device('cpu')))
model.eval()

# Streamlit app
st.title("Image Classification with InceptionV3")
st.write("Upload an image for classification")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])  # Accept both JPG and PNG

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Classify the image
    probabilities, predicted_class = predict(uploaded_file, model)

    # Mendefinisikan nama kelas sesuai dengan kategori katarak
    class_names = ["Normal", "Cataract", "glaucoma", "retina_disease"]

    # Mendapatkan kelas yang paling dominan
    dominant_class_index = torch.argmax(probabilities).item()
    dominant_class_name = class_names[dominant_class_index]

    st.write(f"Predicted Class: {dominant_class_name}")
    st.write(f"Probability ({dominant_class_name}): {probabilities[dominant_class_index].item():.4f}")


# Informasi tentang pencipta aplikasi
st.markdown("---")
st.subheader("Creator:")
st.write("1. Nama: Nacre Faiz Hibatullah A.P")
st.write("   NIM: 120450091")
st.write("2. Nama: Danar Zahar Tambun")
st.write("   NIM: 120450093")
st.write("3. Nama: Atikah Yona Putri")
st.write("   NIM: 120450083")
st.write("4. Nama: Muhammad Rasyid Aditya")
st.write("   NIM: 120450089")
