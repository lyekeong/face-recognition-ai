"""ResNet-18/50 face classifier for the YOLO pipeline.

Everything that defines, finetunes, stores and re-loads the ResNet-18/50
classifier lives here so yolo.py only keeps the YOLOv8n-face detector, the
evaluate pipeline and the CLI.

Train/eval transforms come from preprocessing.py.
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models

from preprocessing import BASE_DIR, build_transforms

TRAIN_DIR = "../dataset_split/train"
TEST_DIR = "../dataset_split/test"
OUT_DIR = os.path.join(BASE_DIR, "output")

BACKBONES = ("resnet18", "resnet50")


def make_model(backbone, num_classes):
    if backbone == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif backbone == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    else:
        raise ValueError(f"Unknown backbone {backbone}")
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, num_classes))
    return model


def make_model_for_inference(backbone, num_classes):
    if backbone == "resnet18":
        model = models.resnet18()
    elif backbone == "resnet50":
        model = models.resnet50()
    else:
        raise ValueError(f"Unknown backbone {backbone}")
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, num_classes))
    return model


def split_train_val(train_dataset, val_ratio=0.10, seed=42):
    targets = np.array(train_dataset.targets)
    idx = np.arange(len(train_dataset))
    tr_idx, va_idx = train_test_split(
        idx, test_size=val_ratio, stratify=targets, random_state=seed)
    return Subset(train_dataset, tr_idx), Subset(train_dataset, va_idx)


def save_arch_json(path, backbone, num_classes, class_names, val_acc, test_acc):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({
            "backbone": backbone,
            "num_classes": num_classes,
            "class_names": class_names,
            "val_acc": round(float(val_acc), 4),
            "test_acc": round(float(test_acc), 4),
        }, fh, indent=2)


def load_arch():
    with open(os.path.join(OUT_DIR, "arch.json"), encoding="utf-8") as fh:
        return json.load(fh)


def align_index(backbone):
    return "resnet18" if backbone == "resnet18" else "resnet50"


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_transform, eval_transform = build_transforms()
    train_full = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=eval_transform)

    class_names = train_full.classes
    num_classes = len(class_names)

    train_ds, val_ds = split_train_val(train_full, val_ratio=0.10, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(args.backbone, num_classes).to(device)

    arch_file = os.path.join(OUT_DIR, "arch.json")
    model_path = os.path.join(OUT_DIR, "resnet18_faces.pth"
                              if args.backbone == "resnet18"
                              else "resnet50_faces.pth")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"Backbone: {args.backbone} | device: {device}")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_dataset)}")

    best_val_acc = 0.0
    best_state = None
    bad_epochs = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss, run_corr, run_n = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * x.size(0)
            run_corr += (out.argmax(1) == y).sum().item()
            run_n += x.size(0)
        scheduler.step()

        # Validation
        model.eval()
        v_corr, v_n = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                v_corr += (out.argmax(1) == y).sum().item()
                v_n += x.size(0)
        val_acc = v_corr / v_n
        train_acc = run_corr / run_n
        print(f"Epoch {epoch:3d}/{args.epochs}  loss {run_loss/run_n:.4f}  "
              f"train_acc {train_acc:.4f}  val_acc {val_acc:.4f}  "
              f"lr {scheduler.get_last_lr()[0]:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            bad_epochs = 0
            print(f"  -> best val acc so far ({best_val_acc:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch} "
                      f"(no improvement for {args.patience})")
                break

    print(f"\nTraining done in {time.time()-t0:.0f}s. Best val acc {best_val_acc:.4f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save(best_state, model_path)
    print(f"Best checkpoint saved to {model_path}")

    # Final evaluation on the held-out test set using the best checkpoint
    model.load_state_dict(best_state)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            out = model(x)
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    test_acc = float((np.array(all_preds) == np.array(all_labels)).mean())
    print(f"\nTest accuracy (best checkpoint): {test_acc:.4f}")
    print("\n========== Final evaluation report ==========")
    print(classification_report(all_labels, all_preds,
                                target_names=class_names, zero_division=0))

    save_arch_json(arch_file, args.backbone, num_classes,
                   class_names, best_val_acc, test_acc)
    print(f"Arch info saved to {arch_file}")
    print("NOTE: arch.json points the 'evaluate' subcommand and the GUI (yolo "
          "backend) to the architecture of the LAST trained model. For "
          "inference with a different backbone, retrain or edit the file.")