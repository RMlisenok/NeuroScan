import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

CLASS_NAMES_RU = {
    "glioma": "Глиома",
    "meningioma": "Менингиома",
    "notumor": "Опухоль не обнаружена",
    "pituitary": "Опухоль гипофиза",
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def build_model(num_classes=4, pretrained=True):
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer or model.features[-1]
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self._fwd)
        self.target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, module, inp, out):
        self.activations = out.detach()

    def _bwd(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(output.argmax(dim=1).item())
        output[0, class_idx].backward(retain_graph=True)

        grads = self.gradients[0]
        acts = self.activations[0]
        weights = grads.mean(dim=(1, 2))
        cam = torch.zeros(acts.shape[1:], dtype=torch.float32, device=acts.device)
        for i, w in enumerate(weights):
            cam += w * acts[i]
        cam = torch.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))

        probs = torch.softmax(output, dim=1)[0].detach().cpu().numpy()
        return cam, class_idx, probs


def prepare_input(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    rgb_resized = cv2.resize(rgb, (224, 224))
    tensor = _preprocess(rgb_resized).unsqueeze(0)
    return tensor, rgb_resized


def overlay_cam(img_rgb, cam, threshold=0.5, alpha=0.45):
    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    mask = (cam >= threshold).astype(np.uint8) * 255
    overlay = cv2.addWeighted(img_rgb, 1 - alpha, heatmap, alpha, 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 0, 0), 2)
    return overlay, mask, contours


def largest_bbox(contours, shape):
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return int(x), int(y), int(w), int(h)


def bbox_area_pct(bbox, shape):
    if bbox is None:
        return 0.0
    _, _, w, h = bbox
    return round(100.0 * w * h / (shape[0] * shape[1]), 2)