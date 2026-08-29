import json
import os

import cv2
import numpy as np
import torch
from torchvision import transforms
from ultralytics import YOLO
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "output")
DETECTOR_PATH = os.path.join(BASE_DIR, "yolov8n-face.pt")
TEST_DIR = "../dataset_split/test"

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
MIN_FACE_SIZE = (80, 80)
MARGIN_RATIO = 0.20
DETECT_CONF = 0.70

TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


def load_arch():
    with open(os.path.join(OUT_DIR, "arch.json"), encoding="utf-8") as fh:
        return json.load(fh)


def build_model(backbone, num_classes):
    from train_classifier import make_model_for_inference
    model = make_model_for_inference(backbone, num_classes)
    return model


def align_index(backbone):
    return "resnet18" if backbone == "resnet18" else "resnet50"


def crop_face(frame, box):
    xmin, ymin, xmax, ymax = box
    h, w = frame.shape[:2]
    pad_x = int((xmax - xmin) * MARGIN_RATIO)
    pad_y = int((ymax - ymin) * MARGIN_RATIO)
    x1, y1 = max(0, xmin - pad_x), max(0, ymin - pad_y)
    x2, y2 = min(w, xmax + pad_x), min(h, ymax + pad_y)
    crop = frame[y1:y2, x1:x2]
    return None if crop.size == 0 else crop


def detect_largest_face(frame, yolo):
    results = yolo.predict(source=frame, conf=DETECT_CONF, verbose=False)
    best = None
    best_area = 0
    for result in results:
        for box in result.boxes:
            if int(box.cls[0]) != 0:
                continue
            x = box.xyxy[0].cpu().numpy()
            area = (x[2] - x[0]) * (x[3] - x[1])
            if area > best_area:
                best_area = area
                best = x
    if best is None:
        return None
    x0, y0, x1, y1 = map(int, best)
    if (x1 - x0) < MIN_FACE_SIZE[0] or (y1 - y0) < MIN_FACE_SIZE[1]:
        return None
    return (x0, y0, x1, y1)


def main():
    arch = load_arch()
    backbone = arch["backbone"]
    num_classes = arch["num_classes"]
    class_names = arch["class_names"]
    model_path = os.path.join(OUT_DIR, f"{align_index(backbone)}_faces.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(backbone, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    yolo = YOLO(DETECTOR_PATH)

    classes = sorted([d for d in os.listdir(TEST_DIR)
                      if os.path.isdir(os.path.join(TEST_DIR, d))])
    cmap = {name: i for i, name in enumerate(class_names)}

    files, trues, preds, confs, dets = [], [], [], [], []
    for label in classes:
        folder = os.path.join(TEST_DIR, label)
        for fname in sorted(os.listdir(folder)):
            path = os.path.join(folder, fname)
            if not os.path.isfile(path):
                continue
            img = cv2.imread(path)
            if img is None:
                continue

            box = detect_largest_face(img, yolo)
            detected = box is not None
            patch = crop_face(img, box) if box is not None else img
            if patch is None:
                patch = img
                detected = False

            tensor = TRANSFORM(patch).unsqueeze(0).to(device)
            with torch.no_grad():
                proba = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
            idx = int(proba.argmax())
            true_idx = cmap.get(label, -1)

            files.append(fname)
            trues.append(true_idx)
            preds.append(idx)
            confs.append(float(proba[idx]))
            dets.append(detected)

    trues = np.array(trues)
    preds = np.array(preds)
    confs = np.array(confs)
    dets = np.array(dets)
    keep = trues >= 0
    trues, preds, confs, dets, files = (trues[keep], preds[keep], confs[keep],
                                        dets[keep], np.array(files)[keep])

    acc_all = accuracy_score(trues, preds)
    acc_det = accuracy_score(trues[dets], preds[dets]) if dets.any() else 0.0
    print(f"Model:        {backbone} ({model_path})")
    print(f"Test images:  {len(trues)}  (face detected: {dets.sum()}, "
          f"fallback whole image: {(~dets).sum()})")
    print(f"Argmax acc (all):        {acc_all:.4f}")
    print(f"Argmax acc (detected):   {acc_det:.4f}")

    print("\nClassification report (all 360):")
    print(classification_report(trues, preds,
                                target_names=class_names, zero_division=0))

    # Johnny Depp line
    d_idx = cmap.get("Johnny Depp")
    if d_idx is not None:
        m = trues == d_idx
        print(f"\nJohnny Depp: {int(m.sum())} samples, "
              f"{int((preds[m] == d_idx).sum())} correct "
              f"({(preds[m] == d_idx).mean():.3f})")

    print("\nThreshold sweep (Unknown gate):")
    for t in (0.50, 0.60, 0.70, 0.80, 0.90, 0.95):
        lab = confs >= t
        cor = int(((lab) & (preds == trues)).sum())
        print(f"  >= {t:.2f}: precision {cor / max(int(lab.sum()), 1):.4f}  "
              f"recall {cor / len(trues):.4f}  decided {int(lab.sum())}")

    os.makedirs(OUT_DIR, exist_ok=True)
    # Per-image CSV
    with open(os.path.join(OUT_DIR, "per_image.csv"), "w", newline="",
              encoding="utf-8") as fh:
        fh.write("file,true,pred,confidence,detected,gated\n")
        for f, t, pp, c, dd, g in zip(
                files, trues, preds, confs, dets, confs >= 0.60):
            fh.write(f"{f},{class_names[t]},{class_names[pp]},{c:.4f},"
                     f"{'yes' if dd else 'no'},{'yes' if g else 'no'}\n")

    # Metrics CSV
    with open(os.path.join(OUT_DIR, "results.csv"), "w", newline="",
              encoding="utf-8") as fh:
        fh.write("metric,value\n")
        fh.write(f"backbone,{backbone}\n")
        fh.write(f"accuracy_argmax_all,{acc_all:.4f}\n")
        fh.write(f"accuracy_argmax_detected,{acc_det:.4f}\n")
        report = classification_report(trues, preds,
                                       target_names=class_names,
                                       zero_division=0, output_dict=True)
        fh.write(f"precision_macro,{report['macro avg']['precision']:.4f}\n")
        fh.write(f"recall_macro,{report['macro avg']['recall']:.4f}\n")
        fh.write(f"f1_macro,{report['macro avg']['f1-score']:.4f}\n")
        for t in (0.50, 0.60, 0.70, 0.80, 0.90, 0.95):
            lab = confs >= t
            cor = int(((lab) & (preds == trues)).sum())
            fh.write(f"gate_{t:.2f}_precision,"
                     f"{cor / max(int(lab.sum()), 1):.4f}\n")
            fh.write(f"gate_{t:.2f}_recall,{cor / len(trues):.4f}\n")

    # Confusion matrix
    labels = sorted(set(trues) | set(preds))
    cm = confusion_matrix(trues, preds, labels=labels)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=[class_names[i] for i in labels])
    fig, ax = plt.subplots(figsize=(14, 12))
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title("Face Recognition Confusion Matrix (YOLO + ResNet)")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=150)
    print("\nSaved output/results.csv, output/per_image.csv, "
          "output/confusion_matrix.png")


if __name__ == "__main__":
    main()