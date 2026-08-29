import argparse
import os
import pickle

import cv2
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, ConfusionMatrixDisplay)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ml import (HOG, align_face, FACE_CROP_MARGIN, SEED, make_detector,
                make_hog, preprocess_gray, transform_features)

# ML v2 configuration --------------------------------------------------------
# Experiment pipeline. Two input modes:
#   direct - consume the face-centered portrait directly (YOLO-style);
#   crop   - largest-confidence face + eye alignment + margin (CNN-style).
# Both avoid the v1 80px-or-whole-image fallback that capped performance.
IMAGE_SIZE = 224
HOG_CELL = 16
HOG_BLOCK = 2
HOG_BINS = 12
PCA_COMPONENTS = 1200

# Same unknown-gate convention as v1 for comparable acceptance semantics.
CONF_THRESHOLD = 0.50
GATE = 0.50

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(os.path.dirname(PROJECT_DIR), "dataset_split")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")

PARAM_GRID = {
    "C": [1.0, 10.0, 100.0],
    "gamma": ["scale", 0.001],
}


def mode_paths(mode):
    output_dir = os.path.join(PROJECT_DIR, f"output_v2_{mode}")
    return {
        "preprocess": os.path.join(MODEL_DIR, f"preprocess_v2_{mode}.pkl"),
        "model": os.path.join(MODEL_DIR, f"svm_model_v2_{mode}.joblib"),
        "output": output_dir,
        "csv": os.path.join(output_dir, "results.csv"),
        "cm": os.path.join(output_dir, "confusion_matrix.png"),
        "img": os.path.join(output_dir, "per_image.csv"),
        "report": os.path.join(output_dir, "classification_report.txt"),
    }


def detect_align_crop_relaxed(frame_bgr, detector):
    """CNN-style crop: highest-confidence face, eye-aligned, margin, any size."""
    faces = detector.detect_raw(frame_bgr)
    if faces is None or len(faces) == 0:
        return frame_bgr, None
    f = max(faces, key=lambda f: float(f[14]))
    fx, fy, fw, fh = float(f[0]), float(f[1]), float(f[2]), float(f[3])
    box = (int(fx), int(fy), int(fx + fw), int(fy + fh))

    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = align_face(img_rgb, f)
    h, w = img_rgb.shape[:2]
    mh, mw = int(fh * FACE_CROP_MARGIN), int(fw * FACE_CROP_MARGIN)
    y1 = min(max(0, int(fy - mh)), h)
    y2 = min(max(0, int(fy + fh + mh)), h)
    x1 = min(max(0, int(fx - mw)), w)
    x2 = min(max(0, int(fx + fw + mw)), w)
    face_rgb = img_rgb[y1:y2, x1:x2]
    if face_rgb.size == 0:
        return frame_bgr, box
    return cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR), box


def load_features(root, hog, mode, size=IMAGE_SIZE):
    detector = None
    if mode == "crop":
        detector = make_detector()
    classes = sorted([d for d in os.listdir(root)
                      if os.path.isdir(os.path.join(root, d))])
    feats, y = [], []
    for label in classes:
        class_dir = os.path.join(root, label)
        for fname in sorted(os.listdir(class_dir)):
            path = os.path.join(class_dir, fname)
            if not os.path.isfile(path):
                continue
            img = cv2.imread(path)
            if img is None:
                continue
            if detector is not None:
                img, _ = detect_align_crop_relaxed(img, detector)
            gray = preprocess_gray(img, size=size)
            feats.append(hog.compute(gray).reshape(-1))
            y.append(label)
    return np.vstack(feats).astype(np.float32), np.array(y)


def build_and_save_preprocess(mode, train_root=os.path.join(DATA_DIR, "train")):
    hog = HOG(cell_size=HOG_CELL, block_size=HOG_BLOCK, nbins=HOG_BINS)
    print(f"[{mode}] Extracting HOG features "
          f"({IMAGE_SIZE}x{IMAGE_SIZE}, bins={HOG_BINS}, cell={HOG_CELL})...")
    F_train, y_train_names = load_features(train_root, hog, mode)
    print(f"  train feature shape: {F_train.shape}")

    classes = sorted(set(y_train_names))
    label_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_label = {i: c for c, i in label_to_idx.items()}
    y_train = np.array([label_to_idx[c] for c in y_train_names])
    print(f"Train images: {len(F_train)}, classes: {len(classes)}")

    scaler = StandardScaler()
    F_train = scaler.fit_transform(F_train).astype(np.float32)

    pca = PCA(n_components=PCA_COMPONENTS, svd_solver="randomized",
              random_state=SEED)
    F_train = pca.fit_transform(F_train).astype(np.float32)
    print(f"PCA applied: {F_train.shape[1]} components "
          f"(explained var {pca.explained_variance_ratio_.sum():.3f})")

    preprocess = {
        "classes": classes,
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "scaler": scaler,
        "pca": pca,
        "image_size": IMAGE_SIZE,
        "hog_cell": HOG_CELL,
        "hog_block": HOG_BLOCK,
        "hog_bins": HOG_BINS,
        "mode": mode,
    }
    os.makedirs(MODEL_DIR, exist_ok=True)
    paths = mode_paths(mode)
    with open(paths["preprocess"], "wb") as fh:
        pickle.dump(preprocess, fh)
    print(f"Preprocess saved to {paths['preprocess']}")
    return F_train, y_train, preprocess


def train(mode):
    F_train, y_train, pp = build_and_save_preprocess(mode)
    print(f"Train feature shape: {F_train.shape}")

    F_tr, F_va, y_tr, y_va = train_test_split(
        F_train, y_train, test_size=0.15, stratify=y_train, random_state=SEED)
    print(f"\nTuning split: train={len(F_tr)}, val={len(F_va)}")

    print("Grid search (full GridSearchCV)...")
    search = GridSearchCV(
        SVC(kernel="rbf", probability=False),
        param_grid=PARAM_GRID,
        cv=3,
        scoring="accuracy",
        n_jobs=2,
        verbose=1,
    )
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

    paths = mode_paths(mode)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, paths["model"], compress=3)
    print(f"Model saved to {paths['model']}")


def evaluate(mode):
    paths = mode_paths(mode)
    with open(paths["preprocess"], "rb") as fh:
        pp = pickle.load(fh)
    model = joblib.load(paths["model"])
    hog = make_hog(pp)

    F_test, y_test_names = load_features(
        os.path.join(DATA_DIR, "test"), hog, mode, size=pp["image_size"])
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

    mc = proba.max(axis=1)
    print(f"\nTop-1 confidence stats on test: "
          f"mean={mc.mean():.3f} median={np.median(mc):.3f} "
          f"max={mc.max():.3f}")

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

    os.makedirs(paths["output"], exist_ok=True)
    labels = sorted(set(y_test) | set(y_pred))
    class_names = [pp["idx_to_label"][i] for i in labels]
    conf = confusion_matrix(y_test, y_pred, labels=labels)

    with open(paths["csv"], "w") as fh:
        fh.write("metric,value\n")
        fh.write(f"accuracy_argmax,{acc:.4f}\n")
        fh.write(f"precision_macro,{prec:.4f}\n")
        fh.write(f"recall_macro,{rec:.4f}\n")
        fh.write(f"f1_macro,{f1:.4f}\n")
        fh.write(f"gate_threshold,{GATE}\n")
        fh.write(f"gate_precision,{gate_prec:.4f}\n")
        fh.write(f"gate_recall,{gate_rec:.4f}\n")
        fh.write(f"gate_decided,{gate_n}\n")
    print(f"Results saved to {paths['csv']}")

    with open(paths["img"], "w") as fh:
        fh.write("file,true,pred,confidence,gated\n")
        for fname, t, p, c, ai in zip(
                y_test_names, y_test, y_pred, mc, (mc >= GATE)):
            g = "recognized" if bool(ai) else "unknown"
            fh.write(f"{fname},{pp['idx_to_label'][t]},{pp['idx_to_label'][p]},"
                     f"{c:.4f},{g}\n")
    print(f"Per-image results saved to {paths['img']}")

    with open(paths["report"], "w") as fh:
        fh.write(report)
    print(f"Classification report saved to {paths['report']}")

    disp = ConfusionMatrixDisplay(confusion_matrix=conf,
                                  display_labels=class_names)
    fig, ax = plt.subplots(figsize=(14, 12))
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title(f"Face Recognition Confusion Matrix "
                 f"(HOG + SVM v2 - {mode})")
    plt.tight_layout()
    fig.savefig(paths["cm"], dpi=150)
    print(f"Confusion matrix saved to {paths['cm']}")


def main():
    parser = argparse.ArgumentParser(
        description="HOG + SVM v2 (direct/crop) face recognition - train / evaluate")
    sub = parser.add_subparsers(dest="command")
    p_train = sub.add_parser("train")
    p_train.add_argument("--mode", choices=("direct", "crop"), default="direct")
    p_eval = sub.add_parser("evaluate")
    p_eval.add_argument("--mode", choices=("direct", "crop"), default="direct")
    args = parser.parse_args()

    if args.command == "train":
        train(args.mode)
    elif args.command == "evaluate":
        evaluate(args.mode)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()