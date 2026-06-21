import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

st.warning('⚠️ This is an educational prototype, not a diagnostic tool. Results may be unreliable, especially for non-dermoscopic images. Always consult a dermatologist for any skin concerns.')

# wczytaj model
@st.cache_resource
def load_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load('../models/melanoma_model.pth', map_location='cpu'))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class_names = ['benign', 'malignant']

st.title('Melanoma Classifier')
st.write('Upload a close-up image of a skin lesion')

uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded image', use_container_width=True)

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        predicted = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted].item()

    st.subheader(f'Prediction: {class_names[predicted]}')
    st.write(f'Confidence: {confidence:.2%}')

    if class_names[predicted] == 'malignant':
        st.warning('This is not a medical diagnosis. Please consult a dermatologist.')