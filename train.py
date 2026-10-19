import os
import argparse
import json
import time
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, accuracy_score)
from sklearn.model_selection import train_test_split
from ml import CLASS_NAMES_RU, build_model


# ---------- Аргументы ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data", help="Корень с train/ и test/")
    p.add_argument("--out", default="models/classifier_v2.pth", help="Куда сохранить")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-split", type=float, default=0.15,
                   help="Доля валидации, отрезаемая от train/")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--patience", type=int, default=5, help="Early stopping")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--init", default=None,
                   help="Путь к чекпоинту для дообучения (models/classifier_v2.pth)")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------- Трансформации ----------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def build_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((144, 144)),
        transforms.RandomResizedCrop(128, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=12),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


# ---------- Сплиты ----------
def make_splits(data_dir, train_tf, eval_tf, val_split, seed):
    train_root = os.path.join(data_dir, "train")
    test_root = os.path.join(data_dir, "test")

    if not os.path.isdir(train_root):
        raise SystemExit(f"Не найдена папка: {train_root}")
    if not os.path.isdir(test_root):
        raise SystemExit(f"Не найдена папка: {test_root}")

    # База — чтобы получить индексы и классы
    base_train = datasets.ImageFolder(train_root, transform=None)
    class_names = base_train.classes
    targets = np.array([y for _, y in base_train.samples])
    indices = np.arange(len(base_train))

    # Стратифицированный сплит train/val
    idx_train, idx_val = train_test_split(
        indices,
        test_size=val_split,
        stratify=targets,
        random_state=seed,
    )

    train_base = datasets.ImageFolder(train_root, transform=train_tf)
    eval_base = datasets.ImageFolder(train_root, transform=eval_tf)
    test_base = datasets.ImageFolder(test_root, transform=eval_tf)

    # Проверка: классы в test должны совпадать
    if test_base.classes != class_names:
        raise SystemExit(
            f"Классы в test/ не совпадают с train/.\n"
            f"train: {class_names}\ntest:  {test_base.classes}"
        )

    train_ds = Subset(train_base, idx_train)
    val_ds = Subset(eval_base, idx_val)
    test_ds = test_base  # весь test

    print(f"[SPLIT] train={len(train_ds)} | val={len(val_ds)} | test={len(test_ds)}")
    print(f"[SPLIT] classes: {class_names}")

    for name, idx in [("train", idx_train), ("val", idx_val)]:
        counts = np.bincount(targets[idx], minlength=len(class_names))
        print(f"  {name}: " + ", ".join(f"{c}={n}" for c, n in zip(class_names, counts)))

    # Распределение test
    test_targets = np.array([y for _, y in test_base.samples])
    test_counts = np.bincount(test_targets, minlength=len(class_names))
    print(f"  test:  " + ", ".join(f"{c}={n}" for c, n in zip(class_names, test_counts)))

    return train_ds, val_ds, test_ds, class_names


# ---------- Метрики ----------
def compute_metrics(y_true, y_pred, class_names):
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names,
                                   zero_division=0, output_dict=True)
    return {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "confusion_matrix": cm.tolist(),
        "per_class": {k: v for k, v in report.items() if k in class_names},
    }


# ---------- Эпохи ----------
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_true = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        running_loss += loss.item() * x.size(0)
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_true.extend(y.cpu().numpy())
    return running_loss / len(loader.dataset), np.array(all_true), np.array(all_preds)


# ---------- Основной цикл ----------
def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    print(f"[CFG] device={device} | epochs={args.epochs} | batch={args.batch} | lr={args.lr}")
    print(f"[CFG] data={args.data} | val_split={args.val_split}")

    train_tf, eval_tf = build_transforms()
    train_ds, val_ds, test_ds, class_names = make_splits(
        args.data, train_tf, eval_tf, args.val_split, args.seed
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    # model = build_model(num_classes=len(class_names), pretrained=True).to(device)
    if args.init:
        print(f"[INIT] загружаю веса из {args.init}")
        ckpt = torch.load(args.init, map_location=device)
        model = build_model(num_classes=len(class_names), pretrained=False)
        model.load_state_dict(ckpt["state_dict"])
        model = model.to(device)
        old_classes = ckpt.get("classes")
        print(f"[INIT] старые классы: {old_classes}")
        if old_classes != class_names:
            raise SystemExit(
                f"Классы не совпадают!\n"
                f"  чекпоинт: {old_classes}\n"
                f"  датасет:  {class_names}"
            )
    else:
        model = build_model(num_classes=len(class_names), pretrained=True).to(device)


    # Class weights — против дисбаланса (считаем только по train)
    train_targets = np.array([train_ds.dataset.samples[i][1] for i in train_ds.indices])
    counts = np.bincount(train_targets, minlength=len(class_names))
    class_weights = torch.tensor(
        (counts.sum() / (len(counts) * counts + 1e-6)), dtype=torch.float32
    ).to(device)
    print(f"[CFG] class_weights={class_weights.cpu().numpy().round(3)}")

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    best_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    history = []

    print("\n=== Старт обучения ===")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, y_true, y_pred = eval_epoch(model, val_loader, criterion, device)
        va_metrics = compute_metrics(y_true, y_pred, class_names)
        scheduler.step()

        dt = time.time() - t0
        print(f"[{epoch:02d}/{args.epochs}] "
              f"train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val_loss={va_loss:.4f} acc={va_metrics['accuracy']:.4f} "
              f"f1={va_metrics['f1_macro']:.4f} | {dt:.1f}s")

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": va_loss, "val_acc": va_metrics["accuracy"],
            "val_f1_macro": va_metrics["f1_macro"],
            "lr": optimizer.param_groups[0]["lr"],
        })

        if va_metrics["f1_macro"] > best_f1:
            best_f1 = va_metrics["f1_macro"]
            best_epoch = epoch
            patience_counter = 0

            torch.save({
                "state_dict": model.state_dict(),
                "classes": class_names,
                "classes_ru": {c: CLASS_NAMES_RU.get(c, c) for c in class_names},
                "val_metrics": va_metrics,
                "epoch": epoch,
                "args": vars(args),
                "saved_at": datetime.now().isoformat(timespec="seconds"),
            }, args.out)
            print(f"   💾 saved → {args.out} (f1={best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"[EARLY STOP] нет улучшений {args.patience} эпох.")
                break

    print(f"\n=== Лучшая модель: эпоха {best_epoch}, val_f1={best_f1:.4f} ===")

    # ---------- Финальный тест ----------
    print("\n=== Тест на data/test/ ===")
    ckpt = torch.load(args.out, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    te_loss, y_true, y_pred = eval_epoch(model, test_loader, criterion, device)
    te_metrics = compute_metrics(y_true, y_pred, class_names)

    print(f"test_loss={te_loss:.4f} | acc={te_metrics['accuracy']:.4f} | "
          f"f1_macro={te_metrics['f1_macro']:.4f} | "
          f"f1_weighted={te_metrics['f1_weighted']:.4f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    print("Confusion matrix:")
    print(np.array(te_metrics["confusion_matrix"]))

    # ---------- Отчёт ----------
    report_path = os.path.splitext(args.out)[0] + "_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "best_epoch": best_epoch,
            "best_val_f1": best_f1,
            "test_metrics": te_metrics,
            "history": history,
            "classes": class_names,
            "split_sizes": {
                "train": len(train_ds),
                "val": len(val_ds),
                "test": len(test_ds),
            },
        }, f, ensure_ascii=False, indent=2)
    print(f"\n📄 Отчёт: {report_path}")


if __name__ == "__main__":
    main()