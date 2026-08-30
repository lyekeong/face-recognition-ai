import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
SPLIT_DIR = Path("..") / "dataset_split"
OUTPUT_DIR = Path("outputs_cnn")
IMG_SIZE = (96, 96)
BATCH_SIZE = 32
EPOCHS = 80
PATIENCE = 15
BASE_LR = 3e-4
WARMUP_LR = 1e-5
MIN_LR_RATIO = 0.02
LABEL_SMOOTHING = 0.1
VAL_RATIO = 0.15
VALID_EXTS = {".jpg", ".jpeg", ".png"}
FACE_CROP_MARGIN = 0.20
DETECTOR_PATH = Path("models") / "face_detection_yunet_2023mar.onnx"
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
MODEL_STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)


def collect_images(dataset_dir):
    class_names = sorted(d.name for d in dataset_dir.iterdir() if d.is_dir())
    label_map = {name: idx for idx, name in enumerate(class_names)}
    filepaths, labels = [], []
    for name in class_names:
        for fp in sorted((dataset_dir / name).rglob("*")):
            if fp.suffix.lower() in VALID_EXTS:
                filepaths.append(fp)
                labels.append(label_map[name])
    return filepaths, labels, class_names


def align_face(img_rgb, face):
    left_eye = (face[4], face[5])
    right_eye = (face[6], face[7])
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))
    center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img_rgb, M, (img_rgb.shape[1], img_rgb.shape[0]),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def load_pixels(filepaths, labels):
    detector = cv2.FaceDetectorYN.create(str(DETECTOR_PATH), "", (320, 320))
    images = np.empty((len(filepaths), *IMG_SIZE, 3), dtype=np.uint8)
    valid_labels = []
    skipped = 0
    for fp, lb in zip(filepaths, labels):
        img_bgr = cv2.imread(str(fp))
        if img_bgr is None:
            skipped += 1
            continue
        h, w = img_bgr.shape[:2]
        detector.setInputSize((w, h))
        _, faces = detector.detect(img_bgr)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if faces is not None and len(faces) > 0:
            f = max(faces, key=lambda f: float(f[14]))
            fx, fy, fw, fh = int(f[0]), int(f[1]), int(f[2]), int(f[3])
            mh, mw = int(fh * FACE_CROP_MARGIN), int(fw * FACE_CROP_MARGIN)
            y1, y2 = max(0, fy - mh), min(h, fy + fh + mh)
            x1, x2 = max(0, fx - mw), min(w, fx + fw + mw)
            img_rgb = align_face(img_rgb, f)
            img_rgb = img_rgb[y1:y2, x1:x2]
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        img_gray = CLAHE.apply(img_gray)
        img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        images[len(valid_labels)] = cv2.resize(img_rgb, IMG_SIZE)
        valid_labels.append(lb)
    return images[: len(valid_labels)], np.array(valid_labels), skipped


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )

    def forward(self, x):
        return self.block(x)


class FaceRecognitionCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.head(self.stem(x))


class Augment(nn.Module):
    def __init__(self):
        super().__init__()
        self.aug = T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomRotation(8),
                T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.85, 1.15)),
                T.ColorJitter(brightness=0.1, contrast=0.1),
            ]
        )

    def forward(self, x):
        return self.aug(x)


def make_loaders(X, y, shuffle=False):
    X_t = torch.from_numpy(X).permute(0, 3, 1, 2).float() / 255.0
    y_t = torch.from_numpy(y).long()
    ds = TensorDataset(X_t, y_t)
    return DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=shuffle, drop_last=False, num_workers=0
    )


def build_model(num_classes, total_steps):
    model = FaceRecognitionCNN(num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=BASE_LR)
    warmup_steps = int(total_steps * 0.05)
    cosine_steps = max(total_steps - warmup_steps, 1)
    base_lr = BASE_LR
    apply_warmup = warmup_steps > 0

    def lr_multiplier(step):
        if apply_warmup and step < warmup_steps:
            return (WARMUP_LR + (base_lr - WARMUP_LR) * (step / warmup_steps)) / base_lr
        t = (step - (warmup_steps if apply_warmup else 0)) / cosine_steps
        t = min(max(t, 0.0), 1.0)
        return MIN_LR_RATIO + 0.5 * (1.0 - MIN_LR_RATIO) * (1 + np.cos(np.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_multiplier)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    return model, optimizer, scheduler, criterion


def train_one_epoch(model, loader, optimizer, criterion, augment):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        X = augment(X)
        X = (X - MODEL_MEAN) / MODEL_STD
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        X = (X - MODEL_MEAN) / MODEL_STD
        out = model(X)
        loss = criterion(out, y)
        running_loss += loss.item() * X.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return running_loss / total, correct / total


def train_model(model, optimizer, scheduler, criterion, train_loader, val_loader):
    augment = Augment().to(DEVICE)
    history = {"accuracy": [], "loss": [], "val_accuracy": [], "val_loss": []}
    best_val_acc = 0.0
    best_state = None
    steps_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, augment)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        scheduler.step()

        history["accuracy"].append(train_acc)
        history["loss"].append(train_loss)
        history["val_accuracy"].append(val_acc)
        history["val_loss"].append(val_loss)

        print(
            f"Epoch {epoch}/{EPOCHS} - loss: {train_loss:.4f} - "
            f"accuracy: {train_acc:.4f} - val_loss: {val_loss:.4f} - "
            f"val_accuracy: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                k: v.clone() for k, v in model.state_dict().items()
            }
            steps_no_improve = 0
        else:
            steps_no_improve += 1
            if steps_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best validation accuracy: {best_val_acc:.4f}")
    return history


def evaluate_model(model, X_test, y_test, class_names, output_dir):
    X_t = torch.from_numpy(X_test).permute(0, 3, 1, 2).float() / 255.0
    y_t = torch.from_numpy(y_test).long()
    loader = DataLoader(
        TensorDataset(X_t, y_t), batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    probas = []
    model.eval()
    with torch.no_grad():
        for X, _ in loader:
            X = (X.to(DEVICE) - MODEL_MEAN) / MODEL_STD
            out = torch.softmax(model(X), dim=1)
            probas.append(out.cpu().numpy())
    probs = np.concatenate(probas, axis=0)
    y_pred = np.argmax(probs, axis=1)
    mc = probs.max(axis=1)
    gate = 0.60
    gate_correct = (mc >= gate) & (y_pred == y_test)
    gate_n = int((mc >= gate).sum())
    metrics = {
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro")),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
        "gate_threshold": gate,
        "gate_precision": float(gate_correct.sum() / max(gate_n, 1)),
        "gate_recall": float(gate_correct.sum() / len(y_test)),
        "gate_decided": gate_n,
    }
    report = classification_report(
        y_test, y_pred, target_names=class_names, digits=4
    )
    (output_dir / "classification_report.txt").write_text(report)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("\n=== Test Set Evaluation ===")
    for key, value in metrics.items():
        print(f"{key:>16}: {value:.4f}")
    print("\n" + report)
    plot_confusion_matrix(y_test, y_pred, class_names, output_dir)
    return y_pred


def plot_confusion_matrix(y_true, y_pred, class_names, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("CNN Face Recognition - Confusion Matrix")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def plot_history(history, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["accuracy"], label="train")
    axes[0].plot(history["val_accuracy"], label="validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(history["loss"], label="train")
    axes[1].plot(history["val_loss"], label="validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_history.png", dpi=150)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Using device: {DEVICE}")

    print("[1/5] Loading training data...")
    train_paths, train_labels, class_names = collect_images(SPLIT_DIR / "train")
    print(f"Found {len(train_paths)} training images across {len(class_names)} people")

    print("[2/5] Splitting train/val ({:.0f}/{:.0f})...".format(
        (1 - VAL_RATIO) * 100, VAL_RATIO * 100))
    X_tr_p, X_va_p, y_tr, y_va = train_test_split(
        train_paths, train_labels, test_size=VAL_RATIO,
        stratify=train_labels, random_state=SEED,
    )
    print(f"Train={len(X_tr_p)}, Val={len(X_va_p)}")

    print("[3/5] Loading test data...")
    test_paths, test_labels, _ = collect_images(SPLIT_DIR / "test")
    print(f"Found {len(test_paths)} test images")

    print("[4/5] Loading and resizing images...")
    X_train, y_train, sk1 = load_pixels(X_tr_p, list(y_tr))
    X_val, y_val, sk2 = load_pixels(X_va_p, list(y_va))
    X_test, y_test, sk3 = load_pixels(test_paths, test_labels)
    skipped = sk1 + sk2 + sk3
    if skipped:
        print(f"Skipped {skipped} unreadable files")
    print(f"Final: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    train_loader = make_loaders(X_train, y_train, shuffle=True)
    val_loader = make_loaders(X_val, y_val)

    print("[5/5] Building and training CNN...")
    steps_per_epoch = int(np.ceil(len(X_train) / BATCH_SIZE))
    total_steps = steps_per_epoch * EPOCHS
    model, optimizer, scheduler, criterion = build_model(len(class_names), total_steps)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    history = train_model(model, optimizer, scheduler, criterion, train_loader, val_loader)
    plot_history(history, OUTPUT_DIR)

    checkpoint = {
        "state_dict": model.state_dict(),
        "class_names": class_names,
        "img_size": IMG_SIZE,
        "mean": MODEL_MEAN.cpu(),
        "std": MODEL_STD.cpu(),
    }
    torch.save(checkpoint, OUTPUT_DIR / "cnn_face_model.pth")
    print(f"Model saved to: {(OUTPUT_DIR / 'cnn_face_model.pth').resolve()}")

    print("\n[6/6] Evaluating on test set...")
    evaluate_model(model, X_test, y_test, class_names, OUTPUT_DIR)

    with open(OUTPUT_DIR / "class_names.json", "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"\nArtifacts saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
