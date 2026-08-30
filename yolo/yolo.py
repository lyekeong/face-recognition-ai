"""YOLO face recognition pipeline: YOLOv8n-face detector + ResNet classifier.

Subcommands:
  train     -- fine-tune the ResNet-18/50 classifier on dataset_split/train
               (delegates the training loop to resnet.py).
  evaluate  -- full end-to-end pipeline (YOLOv8n-face detect -> margin crop
               -> ResNet classify) on the 360-image test set; writes
               output/results.csv, per_image.csv and confusion_matrix.png.

The classifier (model + training) lives in resnet.py; face detection/crop/
transforms live in preprocessing.py.
"""
import argparse
import os

import cv2
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preprocessing import (DETECTOR_PATH, TRANSFORM, crop_face,
                           detect_largest_face)
from resnet import (BACKBONES, OUT_DIR, align_index, load_arch,
                    make_model_for_inference, train)

TEST_DIR = "../dataset_split/test"


def evaluate():
    from ultralytics import YOLO

    arch = load_arch()
    backbone = arch["backbone"]
    num_classes = arch["num_classes"]
    class_names = arch["class_names"]
    model_path = os.path.join(OUT_DIR, f"{align_index(backbone)}_faces.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model_for_inference(backbone, num_classes)
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


def main():
    parser = argparse.ArgumentParser(
        description="YOLO face recognition (detector + ResNet-18/50): "
                    "train the classifier or evaluate the full pipeline")
    sub = parser.add_subparsers(dest="command")

    p_train = sub.add_parser("train", help="Train the face classifier (YOLO crop)")
    p_train.add_argument("--backbone", choices=BACKBONES, default="resnet18")
    p_train.add_argument("--epochs", type=int, default=60)
    p_train.add_argument("--batch-size", type=int, default=32)
    p_train.add_argument("--lr", type=float, default=0.001)
    p_train.add_argument("--momentum", type=float, default=0.9)
    p_train.add_argument("--weight-decay", type=float, default=5e-4)
    p_train.add_argument("--patience", type=int, default=10)
    p_train.add_argument("--seed", type=int, default=42)

    sub.add_parser("evaluate", help="Evaluate the pipeline on the test set")

    args = parser.parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "evaluate":
        evaluate()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()