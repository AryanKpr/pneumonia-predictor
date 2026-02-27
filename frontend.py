
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*13*13,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,2)
    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net()
    model.load_state_dict(torch.load(
        '/Users/ADMIN/Desktop/pneumonia nn/pneumonia_model.pth',
        map_location=device
    ))
    model.to(device)
    model.eval()
    return model, device

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((64, 64)),  # Match your training size!
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    return transform(image).unsqueeze(0)

def predict(image, model, device):
    img_tensor = preprocess_image(image).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        confidence, predicted = torch.max(probabilities, 0)
    
    result = "PNEUMONIA" if predicted.item() == 1 else "NORMAL"
    return result, confidence.item() * 100

st.title("Pneumonia Detection")

uploaded_file = st.file_uploader("Upload Chest X-Ray", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-Ray", width=300)
    
    if st.button("Analyze"):
        with st.spinner("Analyzing..."):
            model, device = load_model()
            prediction, confidence = predict(image, model, device)
            
            if prediction == "PNEUMONIA":
                st.error(f"{prediction} Detected")
            else:
                st.success(f"{prediction}")
            
            st.metric("Confidence", f"{confidence:.1f}%")