"""HOG + SVM face recognition (single refined pipeline).

This is the pipeline behind the comparison table and report metrics:
  224x224 images, HOG cell size 16, PCA-1200 (float32),
  full GridSearchCV + isotonic calibration (cv=5),
  relaxed face crops with no size gate (mode "crop", default)
  or direct whole images (mode "direct").

The original v1 baseline (128x128, gated crops) was removed; the GUI's
HOG+SVM backend now runs this pipeline (mode "crop") as its default.

Unknown-gate convention: CONF_THRESHOLD = GATE = 0.70.
Evaluation writes the reference metric files to output/:
  results.csv, per_image.csv, classification_report.txt, confusion_matrix.png
"""
import argparse
import os

import cv2
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, ConfusionMatrixDisplay)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC

from ml_preprocessing import (config, make_detector, make_hog,
                              transform_features, load_features,
                              build_and_save_preprocess, load_preprocess,
                              detect_align_crop_relaxed, preprocess_gray,
                              SEED, PROJECT_DIR, DATA_DIR, MODEL_DIR)

# Inference / evaluation configuration =====================================
CONF_THRESHOLD = 0.70
GATE = 0.70
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")

# Note: the pipeline config(), YuNet face detector, HOG descriptor and the
# feature transformers (StandardScaler + PCA) all live in ml_preprocessing.py.
# ml.py only trains, evaluates and recognizes faces.


# Face detection, HOG descriptor, crop/align helpers and the feature
# transformer / pickle build routines have moved to ml_preprocessing.py.

def load_model_and_preprocess(mode="crop"):
    pp_art = load_preprocess(mode)
    cfg = config(mode)
    model = joblib.load(os.path.join(MODEL_DIR, cfg["model"]))
    hog = make_hog(pp_art)
    return model, hog, pp_art


# Training =================================================================
def train(mode="crop"):
    cfg = config(mode)

    F_train, y_train, pp = build_and_save_preprocess(cfg)
    print(f"Train feature shape: {F_train.shape}")

    F_tr, F_va, y_tr, y_va = train_test_split(
        F_train, y_train, test_size=0.15, stratify=y_train, random_state=SEED)
    print(f"\nTuning split: train={len(F_tr)}, val={len(F_va)}")

    svc = SVC(kernel="rbf", probability=False)
    print("Grid search (full GridSearchCV)...")
    search = GridSearchCV(
        svc, cfg["params"], cv=3, scoring="accuracy", n_jobs=2, verbose=1)
    search.fit(F_tr, y_tr)
    print(f"Best params: {search.best_params_}  "
          f"(CV acc {search.best_score_:.4f})")

    best = search.best_params_
    print("\nFitting final calibrated SVM (isotonic, cv=5) on full train...")
    base = SVC(kernel="rbf", C=best["C"], gamma=best["gamma"],
               probability=False, random_state=SEED)
    model = CalibratedClassifierCV(base, method="isotonic", cv=5)
    model.fit(F_train, y_train)

    train_acc = model.score(F_train, y_train)
    print(f"Train accuracy: {train_acc:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, cfg["model"])
    joblib.dump(model, model_path, compress=3)
    print(f"Model saved to {model_path}")


# Inference ================================================================
def predict_face(frame_bgr, model, hog, pp_art, detector):
    style = pp_art.get("crop_style", "relaxed")
    if style == "direct":
        face_bgr, box = frame_bgr, None
        # direct mode has no face box: still classify the whole frame
    else:
        face_bgr, box = detect_align_crop_relaxed(frame_bgr, detector)
    gray = preprocess_gray(face_bgr, size=pp_art["image_size"])
    feat = hog.compute(gray).reshape(1, -1)
    if box is None:
        return None, 0.0, None
    feat = transform_features(feat, pp_art)
    proba = model.predict_proba(feat)[0]
    idx = int(proba.argmax())
    name = pp_art["idx_to_label"][idx]
    return name, float(proba[idx]), box


def recognize_image(path, display=True, mode="crop"):
    model, hog, pp_art = load_model_and_preprocess(mode)
    detector = make_detector()
    frame = cv2.imread(path)
    if frame is None:
        print("Cannot read image:", path)
        return None

    name, proba, box = predict_face(frame, model, hog, pp_art, detector)
    if box is None:
        print("No usable face detected in image:", path)
        return None

    if proba >= CONF_THRESHOLD:
        label = f"{name} {proba:.2f}"
        color = (0, 255, 0)
    else:
        label = "Unknown"
        name = "Unknown"
        color = (0, 0, 255)

    print(f"Image: {os.path.basename(path)} -> predicted: {name} "
          f"(conf={proba:.2f})")
    if display:
        x0, y0, x1, y1 = box
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
        cv2.putText(frame, label, (x0, max(0, y0 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow("Result", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return name


# Evaluation ===============================================================
def evaluate(mode="crop"):
    cfg = config(mode)
    pp = load_preprocess(mode)
    model = joblib.load(os.path.join(MODEL_DIR, cfg["model"]))
    hog = make_hog(pp)

    F_test, y_test_names = load_features(os.path.join(DATA_DIR, "test"), cfg, hog)
    keep = np.array([c in pp["label_to_idx"] for c in y_test_names])
    F_test, y_test_names = F_test[keep], np.array(y_test_names)[keep]
    y_test = np.array([pp["label_to_idx"][c] for c in y_test_names])
    F_test = transform_features(F_test, pp)

    y_pred = model.predict(F_test)
    proba = model.predict_proba(F_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print(f"Test samples      : {len(y_test)}")
    print(f"Accuracy (argmax) : {acc:.4f}")
    print(f"Precision (macro) : {prec:.4f}")
    print(f"Recall (macro)    : {rec:.4f}")
    print(f"F1 score (macro)  : {f1:.4f}")

    report = classification_report(y_test, y_pred,
                                   target_names=pp["classes"],
                                   zero_division=0)
    print("\nClassification report:")
    print(report)

    # Gating / unknown-rejection analysis (webcam-style)
    mc = proba.max(axis=1)
    gate_correct = (mc >= GATE) & (y_pred == y_test)
    gate_n = int((mc >= GATE).sum())
    gate_rec = gate_correct.sum() / len(y_test)
    gate_prec = gate_correct.sum() / max(gate_n, 1)
    print(f"\nUnknown gate (conf >= {GATE}):")
    print(f"  decided: {gate_n}/{len(y_test)}  precision {gate_prec:.4f}  "
          f"recall {gate_rec:.4f}\n")
    print("Threshold sweep:")
    for t in (0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90):
        lab = mc >= t
        cor = ((lab) & (y_pred == y_test)).sum()
        print(f"  >= {t:.2f}: precision {cor / max(int(lab.sum()), 1):.4f}  "
              f"recall {cor / len(y_test):.4f}  decided {int(lab.sum())}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    labels = sorted(set(y_test) | set(y_pred))
    class_names = [pp["idx_to_label"][i] for i in labels]
    conf = confusion_matrix(y_test, y_pred, labels=labels)

    metrics_path = os.path.join(OUTPUT_DIR, "results.csv")
    with open(metrics_path, "w") as fh:
        fh.write("metric,value\n")
        fh.write(f"accuracy_argmax,{acc:.4f}\n")
        fh.write(f"precision_macro,{prec:.4f}\n")
        fh.write(f"recall_macro,{rec:.4f}\n")
        fh.write(f"f1_macro,{f1:.4f}\n")
        fh.write(f"gate_threshold,{GATE}\n")
        fh.write(f"gate_precision,{gate_prec:.4f}\n")
        fh.write(f"gate_recall,{gate_rec:.4f}\n")
        fh.write(f"gate_decided,{gate_n}\n")
    print(f"Results saved to {metrics_path}")

    per_img_path = os.path.join(OUTPUT_DIR, "per_image.csv")
    with open(per_img_path, "w") as fh:
        fh.write("file,true,pred,confidence,gated\n")
        for fname, t, p, c, ai in zip(
                y_test_names, y_test, y_pred, mc, (mc >= GATE)):
            g = "recognized" if bool(ai) else "unknown"
            fh.write(f"{fname},{pp['idx_to_label'][t]},{pp['idx_to_label'][p]},"
                     f"{c:.4f},{g}\n")
    print(f"Per-image results saved to {per_img_path}")

    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as fh:
        fh.write(report)

    disp = ConfusionMatrixDisplay(confusion_matrix=conf,
                                  display_labels=class_names)
    fig, ax = plt.subplots(figsize=(14, 12))
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title(f"Face Recognition Confusion Matrix "
                 f"(HOG + SVM - {cfg['crop_style']})")
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    print(f"Confusion matrix saved to {cm_path}")


def main():
    parser = argparse.ArgumentParser(
        description="HOG + SVM face recognition (single pipeline: "
                    "train / evaluate / recognize)")
    sub = parser.add_subparsers(dest="command")

    def add_mode(p):
        p.add_argument("--mode", choices=("crop", "direct"), default="crop",
                       help="input mode (default: crop)")

    p_train = sub.add_parser("train", help="Train the SVM and save artifacts")
    add_mode(p_train)
    p_eval = sub.add_parser("evaluate", help="Evaluate a trained SVM on the test set")
    add_mode(p_eval)

    p_rec = sub.add_parser("recognize", help="Recognize a single image")
    add_mode(p_rec)
    p_rec.add_argument("image", help="Path to the image file")
    p_rec.add_argument("--no-display", action="store_true",
                       help="Do not pop up a display window (headless)")

    args = parser.parse_args()

    if args.command == "train":
        train(args.mode)
    elif args.command == "evaluate":
        evaluate(args.mode)
    elif args.command == "recognize":
        recognize_image(args.image, display=not args.no_display, mode=args.mode)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()