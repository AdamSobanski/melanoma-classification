import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
from supabase import create_client
from dotenv import load_dotenv
import uuid
from datetime import datetime
import io

load_dotenv()

supabase = create_client(
    os.getenv('SUPABASE_URL'),
    os.getenv('SUPABASE_KEY')
)

st.warning('⚠️ This is an educational prototype, not a diagnostic tool. Results may be unreliable, especially for non-dermoscopic images. Always consult a dermatologist for any skin concerns.')

# wczytaj model
@st.cache_resource
def load_model():
    from huggingface_hub import hf_hub_download
    model_path = hf_hub_download(
        repo_id="Ai-Adam-Six-Sigma/melanoma-classifier",
        filename="melanoma_model.pth"
    )
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class_names = ['benign', 'malignant']

def save_to_supabase(image, prediction, confidence):
    # zapisz zdjęcie jako JPEG bytes
    img_filename = f"{uuid.uuid4()}.jpg"
    img_buffer = io.BytesIO()
    image.save(img_buffer, format='JPEG')
    img_bytes = img_buffer.getvalue()
    
    supabase.storage.from_('melanoma-images').upload(
        img_filename,
        img_bytes,
        {'content-type': 'image/jpeg'}
    )
    
    image_url = f"{os.getenv('SUPABASE_URL')}/storage/v1/object/public/melanoma-images/{img_filename}"
    
    # zapisz wynik do bazy
    supabase.table('predictions').insert({
        'prediction': prediction,
        'confidence': float(confidence),
        'image_path': image_url
    }).execute()

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
    save_to_supabase(image, class_names[predicted], confidence)
    st.write(f'Confidence: {confidence:.2%}')

    if class_names[predicted] == 'malignant':
        st.warning('This is not a medical diagnosis. Please consult a dermatologist.')