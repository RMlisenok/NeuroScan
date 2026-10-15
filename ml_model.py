# ml_model.py
import os
import torch
from torchvision import transforms
from PIL import Image
from app import app

MODEL_PATH = os.path.join(app.root_path, 'models', 'best_model.pth')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.load(MODEL_PATH, map_location=DEVICE)
model.eval()

preprocess = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

CLASS_NAMES = [
    'astrocytoma', 'ependymoma', 'glioma', 'meningioma',
    'neurocytoma', 'normal', 'oligodendroglioma', 'schwannoma'
]

def predict_mri(image_path):
    """Возвращает (класс, уверенность)."""
    image = Image.open(image_path).convert('RGB')
    img_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    return CLASS_NAMES[predicted.item()], confidence.item()