import os
import datetime
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from app import app, db, build_model, DEVICE
from models_db import MRIImage


CLASS_ORDER = ["glioma", "meningioma", "notumor", "pituitary"]
IMAGE_SIZE = 224


class RetrainDataset(Dataset):
    """Crop по маске + ресайз + трансформации."""

    def __init__(self, records, transform=None):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = cv2.imread(rec["original_path"])
        mask = cv2.imread(rec["mask_path"], cv2.IMREAD_GRAYSCALE)

        if img is None:
            img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8)

        if mask is not None:
            if mask.shape[:2] != img.shape[:2]:
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
            ys, xs = np.where(mask > 127)
            if len(xs) > 0 and len(ys) > 0:
                pad = 20
                x1 = max(0, xs.min() - pad)
                x2 = min(img.shape[1], xs.max() + pad)
                y1 = max(0, ys.min() - pad)
                y2 = min(img.shape[0], ys.max() + pad)
                if x2 > x1 and y2 > y1:
                    img = img[y1:y2, x1:x2]

        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        return img, rec["class_idx"]


def collect_records():
    with app.app_context():
        images = MRIImage.query.filter_by(
            is_sent_to_training=True,
            training_used=False,
        ).all()

        records = []
        used_ids = []
        for img in images:
            if img.verified_class not in CLASS_ORDER:
                continue
            if not img.edited_mask_path:
                continue

            base = os.path.dirname(os.path.abspath(__file__))
            orig = img.original_path if os.path.isabs(img.original_path) \
                else os.path.join(base, img.original_path)
            mask = img.edited_mask_path if os.path.isabs(img.edited_mask_path) \
                else os.path.join(base, img.edited_mask_path)

            if not os.path.exists(orig) or not os.path.exists(mask):
                print(f"  ⚠ Пропущен id={img.id}: файлы не найдены")
                continue

            records.append({
                "id": img.id,
                "original_path": orig,
                "mask_path": mask,
                "class_idx": CLASS_ORDER.index(img.verified_class),
            })
            used_ids.append(img.id)
        return records, used_ids


def retrain(epochs=10, lr=1e-4, batch_size=8):
    records, used_ids = collect_records()
    if len(records) < 4:
        print(f"❌ Недостаточно данных: {len(records)} снимков. Нужно минимум 4.")
        return

    print(f"✅ Найдено {len(records)} снимков для дообучения.")

    ckpt_path = app.config["MODEL_PATH"]
    if not os.path.exists(ckpt_path):
        print(f"❌ Модель не найдена: {ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    classes = ckpt["classes"]

    # Проверка: если классы в модели и в CLASS_ORDER не совпадают
    if set(classes) != set(CLASS_ORDER):
        print(f"⚠ Порядок классов в модели: {classes}")
        print(f"  Ожидалось: {CLASS_ORDER}")
        print("  Убедитесь, что порядок совпадает, иначе обучение бессмысленно.")

    model = build_model(num_classes=len(classes), pretrained=False).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.train()

    train_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    dataset = RetrainDataset(records, transform=train_tf)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)

        acc = correct / max(total, 1)
        print(f"Эпоха {epoch+1}/{epochs}: loss={total_loss/total:.4f}, acc={acc:.3f}")

    backup = ckpt_path + f".backup_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    os.rename(ckpt_path, backup)

    ckpt["state_dict"] = model.state_dict()
    ckpt["saved_at"] = datetime.datetime.now().isoformat()
    ckpt["retrained_on"] = len(records)
    torch.save(ckpt, ckpt_path)

    print(f"✅ Модель сохранена: {ckpt_path}")
    print(f"   Бэкап: {backup}")

    with app.app_context():
        for iid in used_ids:
            img = db.session.get(MRIImage, iid)
            if img:
                img.training_used = True
        db.session.commit()
    print(f"✅ Помечено использованных: {len(used_ids)}")

if __name__ == "__main__":
    retrain(epochs=10, lr=1e-4, batch_size=8)