import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import models, transforms

st.title(':orange[Model Inference | Malaria Detection]') 

example_images = ['uninfected3.png', 'uninfected.png', 'uninfected4.png', 'uninfected2.png','uninfected5.png','infected5.png','infected4.png','infected3.png','infected2.png','infected.png']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@st.cache_resource()
def load_model():
    mobilenet = models.mobilenet_v2(pretrained=True)
    mobilenet.classifier = nn.Identity()  # Remove the final classification layer

    # Load pre-trained EfficientNetB0
    efficientnet = models.efficientnet_b0(pretrained=True)
    efficientnet.classifier = nn.Identity()  # Remove the final classification layer

    class FusionModel(nn.Module):
        def __init__(self, mobilenet, efficientnet, num_classes):
            super(FusionModel, self).__init__()
            self.mobilenet = mobilenet
            self.efficientnet = efficientnet
            self.fc = nn.Sequential(
                nn.Linear(1280 + 1280, 512),  # Concatenate the outputs
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )

        def forward(self, x):
            mobilenet_features = self.mobilenet(x)
            efficientnet_features = self.efficientnet(x)
            combined_features = torch.cat((mobilenet_features, efficientnet_features), dim=1)
            output = self.fc(combined_features)
            return output

    # Define the number of classes in your dataset
    num_classes = 2  # Example: 10 classes

    # Create the fusion model
    fusion_model = FusionModel(mobilenet, efficientnet, num_classes)
    fusion_model.load_state_dict(torch.load('FusionOutlierGeneralised.pth', map_location='cpu'))
    return fusion_model

@st.cache_data()
def inference(img):
    model = load_model()
    model.eval()
    img = transform(img)
    img = img.unsqueeze(0)
    st.write(img.shape)
    with torch.no_grad():
        return torch.argmax(model(img))

def get_true_label(filename):
    if 'uninfected' in filename:
        return 0  # Uninfected
    elif 'infected' in filename:
        return 1  # Infected
    else:
        return -1  # Unknown

# File uploader
img_file = st.file_uploader('Upload a file', type=['jpg', 'png'])

# Example image selector
example_image = st.selectbox('Or select an example image', example_images)

if img_file:
    st.write('File uploaded')
    img = Image.open(img_file)
    true_label = get_true_label(img_file.name)
else:
    st.write('Using example image')
    img = Image.open(example_image)
    true_label = get_true_label(example_image)

img = img.resize((224, 224))
result = inference(img)
predicted_label = result.item()

st.markdown(f"<h3 style='text-align: center;'>True Label: {'Uninfected' if true_label == 0 else 'Infected'}</h3>", unsafe_allow_html=True)
st.markdown(f"<h3 style='text-align: center;'>Predicted Label: {'Uninfected' if predicted_label == 0 else 'Infected'}</h3>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col2:
    st.image(img)
    if predicted_label == 0:
        st.markdown("<h1 style='text-align: center; color: green;'>UnInfected</h1>", unsafe_allow_html=True)
    else:
        st.markdown("<h1 style='text-align: center; color: red;'>Infected</h1>", unsafe_allow_html=True)
