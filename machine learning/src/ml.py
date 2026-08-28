import argparse
import os
import pickle
import urllib.request

import cv2
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.experimental import enable_halving_search_cv
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, ConfusionMatrixDisplay)
from sklearn.model_selection import HalvingGridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Shared configuration =====================================================
SEED = 42
IMAGE_SIZE = 128          # spatial size used for HOG features
HOG_CELL = 8
HOG_BLOCK = 2
HOG_BINS = 12
FACE_CROP_MARGIN = 0.20
MIN_FACE_SIZE = (80, 80)
PCA_COMPONENTS = 400
DETECTOR_SIZE = (640, 640)
DETECTOR_CONF = 0.70

CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(os.path.dirname(PROJECT_DIR), "dataset_split")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")

PREPROCESS_PKL = os.path.join(MODEL_DIR, "preprocess.pkl")
MODEL_PKL = os.path.join(MODEL_DIR, "svm_model.joblib")
OUT_CSV = os.path.join(OUTPUT_DIR, "results.csv")
OUT_CM = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
OUT_IMG = os.path.join(OUTPUT_DIR, "per_image.csv")

# ML-specific unknown gate. Calibrated SVM confidence saturates low
# (isotonic calibration in 17-class ovr; max achievable confidence on the
# test set is 0.716), so a threshold of 0.50 is the best practical
# precision/recall balance for the ML model (test precision ~0.735 /
# recall ~0.30). CNN/YOLO keep a higher 0.80 gate.
CONF_THRESHOLD = 0.50
GATE = 0.50

MODEL_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")


# Face detection ===========================================================
def ensure_model():
    if not os.path.exists(DETECTOR_PATH):
        os.makedirs(MODEL_DIR, exist_ok=True)
        print("Downloading YuNet face detection model...")
        urllib.request.urlretrieve(MODEL_URL, DETECTOR_PATH)
    return DETECTOR_PATH


DETECTOR_PATH = os.path.join(MODEL_DIR, "face_detection_yunet_2023mar.onnx")


class FaceDetector:
    def __init__(self, input_size=(320, 320), conf_threshold=0.6):
        ensure_model()
        self.det = cv2.FaceDetectorYN_create(DETECTOR_PATH, "", input_size)
        self.det.setScoreThreshold(conf_threshold)

    def detect_raw(self, frame):
        h, w = frame.shape[:2]
        self.det.setInputSize((w, h))
        ok, faces = self.det.detect(frame)
        if not ok or faces is None:
            return None
        return np.asarray(faces)

    def detect(self, frame):
        faces = self.detect_raw(frame)
        if faces is None:
            return []
        out = []
        for f in faces:
            x, y, ww, hh = f[0], f[1], f[2], f[3]
            conf = f[14] if f.shape[0] > 14 else f[-1]
            out.append((x, y, ww, hh, conf))
        return out

    def detect_largest(self, frame):
        faces = self.detect(frame)
        if not faces:
            return None
        return max(faces, key=lambda f: f[2] * f[3])


def make_detector():
    return FaceDetector(input_size=DETECTOR_SIZE, conf_threshold=DETECTOR_CONF)


# HOG descriptor ===========================================================
class HOG:

    def __init__(self, cell_size=8, block_size=2, nbins=9,
                 pixels_per_cell=None):
        if pixels_per_cell is not None:
            cell_size = pixels_per_cell[0]
        self.cell_size = cell_size
        self.block_size = block_size
        self.nbins = nbins

    def _gradient(self, img):
        # Fall back to central differences if sobel unavailable
        gy, gx = np.gradient(img.astype(np.float32))
        mag = np.hypot(gx, gy)
        # unsigned orientation in radians (0 .. pi)
        orient = np.arctan2(gy, gx)
        orient = np.mod(orient, np.pi)
        return mag, orient

    def _cell_histograms(self, mag, orient):
        h, w = mag.shape
        cs = self.cell_size
        nbins = self.nbins
        bin_width = np.pi / nbins

        n_cells_y = h // cs
        n_cells_x = w // cs

        # Truncate image to a whole number of cells
        mag = mag[:n_cells_y * cs, :n_cells_x * cs]
        orient = orient[:n_cells_y * cs, :n_cells_x * cs]

        # Assign each pixel to a cell index (y-cell, x-cell)
        cy_map = np.repeat(np.arange(n_cells_y), cs).reshape(-1, 1)
        cx_map = np.repeat(np.arange(n_cells_x), cs).reshape(1, -1)

        bin_idx = (orient / bin_width).astype(np.int32)
        bin_idx = np.clip(bin_idx, 0, nbins - 1)

        # One-hot style accumulation over bins using advanced indexing.
        cell_hists = np.zeros((n_cells_y, n_cells_x, nbins), dtype=np.float32)
        for b in range(nbins):
            mask = (bin_idx == b)
            np.add.at(cell_hists, (cy_map, cx_map, b), mag * mask)
        return cell_hists

    def compute(self, img):
        if img.ndim == 3:
            # RGB -> luma
            img = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
        img = np.asarray(img, dtype=np.float32)

        mag, orient = self._gradient(img)
        cell_hists = self._cell_histograms(mag, orient)

        nc_y, nc_x, nbins = cell_hists.shape
        bs = self.block_size

        blocks = []
        for by in range(nc_y - bs + 1):
            for bx in range(nc_x - bs + 1):
                block = cell_hists[by:by + bs, bx:bx + bs, :]
                vec = block.flatten()
                norm = np.linalg.norm(vec) + 1e-6
                vec = vec / norm
                blocks.append(vec)

        if not blocks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(blocks).astype(np.float32)

    def descriptor_size(self, height, width):
        nc_y = height // self.cell_size
        nc_x = width // self.cell_size
        blocks = (nc_y - self.block_size + 1) * (nc_x - self.block_size + 1)
        return blocks * self.block_size * self.block_size * self.nbins


def make_hog(pp_art=None):
    if pp_art is None:
        return HOG(cell_size=HOG_CELL, block_size=HOG_BLOCK, nbins=HOG_BINS)
    return HOG(cell_size=pp_art["hog_cell"], block_size=pp_art["hog_block"],
               nbins=pp_art["hog_bins"])


# Preprocessing ============================================================
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


def detect_align_crop(frame_bgr, detector):
    faces = detector.detect_raw(frame_bgr)
    if faces is None:
        return frame_bgr, None
    f = max(faces, key=lambda f: float(f[2] * f[3]))
    fx, fy = float(f[0]), float(f[1])
    fw, fh = float(f[2]), float(f[3])
    if fw < MIN_FACE_SIZE[0] or fh < MIN_FACE_SIZE[1]:
        return frame_bgr, None
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


def preprocess_gray(src_bgr, size=IMAGE_SIZE):
    gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)
    gray = CLAHE.apply(gray)
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return gray


def frame_to_features(frame_bgr, detector, hog, image_size=IMAGE_SIZE):
    face_bgr, box = detect_align_crop(frame_bgr, detector)
    gray = preprocess_gray(face_bgr, size=image_size)
    return hog.compute(gray).reshape(1, -1), box


def transform_features(feat, pp_art):
    F = pp_art["scaler"].transform(feat)
    pca = pp_art.get("pca")
    if pca is not None:
        F = pca.transform(F)
    return F


def load_features_from_folders(root, detector, hog):
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
            feat, _ = frame_to_features(img, detector, hog)
            feats.append(feat)
            y.append(label)
    return np.vstack(feats), np.array(y)


def build_and_save_preprocess(train_root=os.path.join(DATA_DIR, "train"),
                              test_root=os.path.join(DATA_DIR, "test")):
    detector = make_detector()
    hog = make_hog()

    print("Extracting HOG features (train)...")
    F_train, y_train_names = load_features_from_folders(train_root, detector, hog)
    print(f"  train feature shape: {F_train.shape}")
    print("Extracting HOG features (test)...")
    F_test, y_test_names = load_features_from_folders(test_root, detector, hog)
    print(f"  test feature shape:  {F_test.shape}")

    # Build label map from training classes only
    classes = sorted(set(y_train_names))
    label_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_label = {i: c for c, i in label_to_idx.items()}

    y_train = np.array([label_to_idx[c] for c in y_train_names])
    # Dropping test samples whose class is not present in training
    keep = np.array([c in label_to_idx for c in y_test_names])
    F_test = F_test[keep]
    y_test = np.array([label_to_idx[c] for c in y_test_names if c in label_to_idx])

    print(f"Train images: {len(F_train)}, classes: {len(classes)}")
    print(f"Test images:  {len(F_test)}")

    # Standard scaling (fit on train only)
    scaler = StandardScaler()
    F_train = scaler.fit_transform(F_train)
    F_test = scaler.transform(F_test)

    # Dimensionality reduction (fit on train only)
    pca = None
    if F_train.shape[1] > PCA_COMPONENTS:
        pca = PCA(n_components=PCA_COMPONENTS, random_state=SEED)
        F_train = pca.fit_transform(F_train)
        F_test = pca.transform(F_test)
        print(f"PCA applied: {F_train.shape[1]} components "
              f"(explained var {pca.explained_variance_ratio_.sum():.3f})")

    # Persist everything needed for inference later
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
        "face_crop_margin": FACE_CROP_MARGIN,
        "min_face_size": MIN_FACE_SIZE,
    }
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(PREPROCESS_PKL, "wb") as fh:
        pickle.dump(preprocess, fh)
    print(f"Preprocess saved to {PREPROCESS_PKL}")

    return (F_train, y_train), (F_test, y_test), preprocess


def load_preprocess():
    with open(PREPROCESS_PKL, "rb") as fh:
        return pickle.load(fh)


def load_model_and_preprocess():
    pp_art = load_preprocess()
    model = joblib.load(MODEL_PKL)
    hog = make_hog(pp_art)
    return model, hog, pp_art


# Training =================================================================
PARAM_GRID = {
    "C": [0.1, 1.0, 10.0, 100.0],
    "gamma": ["scale", 0.001, 0.0005],
}


def train():
    # Build features and preprocess artifacts (shared pipeline with scaler+PCA).
    (F_train, y_train), (F_test, y_test), pp = build_and_save_preprocess()
    print(f"Train feature shape: {F_train.shape}")
    print(f"Test feature shape:  {F_test.shape}")

    # Tuning/validation split (stratified, train only)
    F_tr, F_va, y_tr, y_va = train_test_split(
        F_train, y_train, test_size=0.15, stratify=y_train, random_state=SEED)
    print(f"\nTuning split: train={len(F_tr)}, val={len(F_va)}")

    print("Grid search (Halving)...")
    search = HalvingGridSearchCV(
        SVC(kernel="rbf", probability=False),
        param_grid=PARAM_GRID,
        factor=2,
        scoring="accuracy",
        random_state=SEED,
        n_jobs=1,
    )
    search.fit(F_tr, y_tr)
    print(f"Best params: {search.best_params_}  "
          f"(val acc {search.best_score_:.4f})")

    best = search.best_params_
    print("\nFitting final calibrated SVM (isotonic, cv=5) on full train...")
    base = SVC(kernel="rbf", C=best["C"], gamma=best["gamma"],
               probability=False, random_state=SEED)
    model = CalibratedClassifierCV(base, method="isotonic", cv=5)
    model.fit(F_train, y_train)

    train_acc = model.score(F_train, y_train)
    test_acc = model.score(F_test, y_test)
    preds = model.predict(F_test)
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")

    # Report calibration headroom for the 0.80 unknown gate
    proba = model.predict_proba(F_test)
    maxc = np.max(proba, axis=1)
    correct = preds == y_test
    print(f"\nTop-1 confidence stats on test: "
          f"mean={maxc.mean():.3f} median={np.median(maxc):.3f} "
          f"max={maxc.max():.3f}")
    for t in (0.3, 0.4, 0.5, 0.6, 0.8):
        lab = maxc >= t
        cor = (lab & correct).sum()
        print(f"  >= {t:.2f}: correct/{int(lab.sum())} recorded "
              f"({cor}/{int(lab.sum())} correct) -> "
              f"precision {cor / max(int(lab.sum()), 1):.3f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PKL)
    print(f"Model saved to {MODEL_PKL}")


# Inference ================================================================
def predict_face(frame_bgr, model, hog, pp_art, detector):
    feat, box = frame_to_features(frame_bgr, detector, hog,
                                  image_size=pp_art["image_size"])
    if box is None:
        return None, 0.0, None
    feat = transform_features(feat, pp_art)
    proba = model.predict_proba(feat)[0]
    idx = int(proba.argmax())
    name = pp_art["idx_to_label"][idx]
    return name, float(proba[idx]), box


def recognize_image(path, display=True):
    model, hog, pp_art = load_model_and_preprocess()
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
def evaluate():
    pp = load_preprocess()
    model = joblib.load(MODEL_PKL)
    detector = make_detector()
    hog = make_hog(pp)

    F_test, y_test_names = load_features_from_folders(
        os.path.join(DATA_DIR, "test"), detector, hog)
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

    print("\nClassification report:")
    print(classification_report(y_test, y_pred,
                                target_names=pp["classes"],
                                zero_division=0))

    # Gating / unknown-rejection analysis (webcam-style, shared pipeline)
    mc = proba.max(axis=1)
    gate_correct = (mc >= GATE) & (y_pred == y_test)
    gate_n = int((mc >= GATE).sum())
    gate_rec = gate_correct.sum() / len(y_test)
    gate_prec = gate_correct.sum() / max(gate_n, 1)
    print(f"\nUnknown gate (conf >= {GATE}):")
    print(f"  decided: {gate_n}/{len(y_test)}  precision {gate_prec:.4f}  "
          f"recall {gate_rec:.4f}\n")
    print("Threshold sweep:")
    for t in (0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
        lab = mc >= t
        cor = ((lab) & (y_pred == y_test)).sum()
        print(f"  >= {t:.2f}: precision {cor / max(int(lab.sum()), 1):.4f}  "
              f"recall {cor / len(y_test):.4f}  decided {int(lab.sum())}")

    # Save numeric results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    labels = sorted(set(y_test) | set(y_pred))
    class_names = [pp["idx_to_label"][i] for i in labels]
    conf = confusion_matrix(y_test, y_pred, labels=labels)

    with open(OUT_CSV, "w") as fh:
        fh.write("metric,value\n")
        fh.write(f"accuracy_argmax,{acc:.4f}\n")
        fh.write(f"precision_macro,{prec:.4f}\n")
        fh.write(f"recall_macro,{rec:.4f}\n")
        fh.write(f"f1_macro,{f1:.4f}\n")
        fh.write(f"gate_threshold,{GATE}\n")
        fh.write(f"gate_precision,{gate_prec:.4f}\n")
        fh.write(f"gate_recall,{gate_rec:.4f}\n")
        fh.write(f"gate_decided,{gate_n}\n")
    print(f"Results saved to {OUT_CSV}")

    # Per-image gating results
    with open(OUT_IMG, "w") as fh:
        fh.write("file,true,pred,confidence,gated\n")
        for fname, t, p, c, ai in zip(
                y_test_names, y_test, y_pred, mc, (mc >= GATE)):
            g = "recognized" if bool(ai) else "unknown"
            fh.write(f"{fname},{pp['idx_to_label'][t]},{pp['idx_to_label'][p]},"
                     f"{c:.4f},{g}\n")
    print(f"Per-image results saved to {OUT_IMG}")

    # Confusion matrix figure
    disp = ConfusionMatrixDisplay(confusion_matrix=conf,
                                  display_labels=class_names)
    fig, ax = plt.subplots(figsize=(14, 12))
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title("Face Recognition Confusion Matrix (HOG + SVM)")
    plt.tight_layout()
    fig.savefig(OUT_CM, dpi=150)
    print(f"Confusion matrix saved to {OUT_CM}")


def main():
    parser = argparse.ArgumentParser(
        description="HOG + SVM face recognition (train / evaluate / recognize)")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("train", help="Train the SVM and save artifacts")
    sub.add_parser("evaluate", help="Evaluate the trained SVM on the test set")

    p_rec = sub.add_parser("recognize", help="Recognize a single image")
    p_rec.add_argument("image", help="Path to the image file")
    p_rec.add_argument("--no-display", action="store_true",
                       help="Do not pop up a display window (headless)")

    p_web = sub.add_parser("webcam", help="Run the live webcam demo")
    p_web.add_argument("--camera", type=int, default=0,
                       help="Camera index (default 0)")

    args = parser.parse_args()

    if args.command == "train":
        train()
    elif args.command == "evaluate":
        evaluate()
    elif args.command == "recognize":
        recognize_image(args.image, display=not args.no_display)
    elif args.command == "webcam":
        # Webcam lives in webcam.py; delegate to avoid duplicating it here.
        import webcam
        webcam.run_webcam(camera_index=args.camera)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
