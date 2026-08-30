"""HOG + SVM feature preprocessing: YuNet face detection, eye alignment,
20% margin crops, CLAHE grayscale, the custom HOG descriptor and the
StandardScaler + PCA feature transformers.

Everything that maps a raw image (or folder of images) to model-ready
feature vectors lives here so ml.py only handles training, inference and
evaluation using the artifacts in models/preprocess_<mode>.pkl.
"""
import os
import pickle
import urllib.request

import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Shared configuration =====================================================
SEED = 42
FACE_CROP_MARGIN = 0.20
DETECTOR_SIZE = (640, 640)
DETECTOR_CONF = 0.70

CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(PROJECT_DIR), "dataset_split")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")

MODEL_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")
DETECTOR_PATH = os.path.join(MODEL_DIR, "face_detection_yunet_2023mar.onnx")

if not os.path.exists(DETECTOR_PATH):
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("Downloading YuNet face detection model...")
    urllib.request.urlretrieve(MODEL_URL, DETECTOR_PATH)

# Pipeline configuration ===================================================
PARAM_GRID = {
    "C": [1.0, 10.0, 100.0],
    "gamma": ["scale", 0.001],
}


def config(mode="crop"):
    """Resolve pipeline settings + artifact filenames for an input mode."""
    if mode not in ("crop", "direct"):
        raise ValueError(f"Unknown mode {mode!r} (use 'crop' or 'direct')")
    return {
        "crop_style": "relaxed" if mode == "crop" else "direct",
        "image_size": 224,
        "hog_cell": 16,
        "hog_block": 2,
        "hog_bins": 12,
        "pca_components": 1200,
        "params": PARAM_GRID,
        "preprocess": f"preprocess_{mode}.pkl",
        "model": f"svm_model_{mode}.joblib",
    }


# Face detection ===========================================================
class FaceDetector:
    def __init__(self, input_size=(320, 320), conf_threshold=0.6):
        self.det = cv2.FaceDetectorYN_create(DETECTOR_PATH, "", input_size)
        self.det.setScoreThreshold(conf_threshold)

    def detect_raw(self, frame):
        h, w = frame.shape[:2]
        self.det.setInputSize((w, h))
        ok, faces = self.det.detect(frame)
        if not ok or faces is None:
            return None
        return np.asarray(faces)


def make_detector():
    return FaceDetector(input_size=DETECTOR_SIZE, conf_threshold=DETECTOR_CONF)


# HOG descriptor ===========================================================
class HOG:

    def __init__(self, cell_size=8, block_size=2, nbins=9):
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


def make_hog(pp_art=None):
    if pp_art is None:
        return HOG(cell_size=8, block_size=2, nbins=12)
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


def detect_align_crop_relaxed(frame_bgr, detector):
    """Highest-confidence face, eye-aligned, margin crop, any size."""
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


def preprocess_gray(src_bgr, size=128):
    gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)
    gray = CLAHE.apply(gray)
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return gray


def transform_features(feat, pp_art):
    F = pp_art["scaler"].transform(feat)
    pca = pp_art.get("pca")
    if pca is not None:
        F = pca.transform(F)
    return F


def load_features(root, cfg, hog):
    """Extract HOG features for a folder tree using the pipeline crop style."""
    style = cfg["crop_style"]
    detector = make_detector() if style == "relaxed" else None
    classes = sorted(d for d in os.listdir(root)
                     if os.path.isdir(os.path.join(root, d)))
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
            if style == "relaxed":
                img, _ = detect_align_crop_relaxed(img, detector)
            gray = preprocess_gray(img, size=cfg["image_size"])
            feats.append(hog.compute(gray).reshape(-1))
            y.append(label)
    return np.vstack(feats).astype(np.float32), np.array(y)


def build_and_save_preprocess(cfg, train_root=os.path.join(DATA_DIR, "train")):
    hog = HOG(cell_size=cfg["hog_cell"], block_size=cfg["hog_block"],
              nbins=cfg["hog_bins"])
    print(f"Extracting HOG features "
          f"({cfg['image_size']}x{cfg['image_size']}, "
          f"bins={cfg['hog_bins']}, cell={cfg['hog_cell']})...")
    F_train, y_train_names = load_features(train_root, cfg, hog)
    print(f"  train feature shape: {F_train.shape}")

    classes = sorted(set(y_train_names))
    label_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_label = {i: c for c, i in label_to_idx.items()}
    y_train = np.array([label_to_idx[c] for c in y_train_names])
    print(f"Train images: {len(F_train)}, classes: {len(classes)}")

    scaler = StandardScaler()
    F_train = scaler.fit_transform(F_train).astype(np.float32)

    pca = PCA(n_components=cfg["pca_components"], svd_solver="randomized",
              random_state=SEED)
    F_train = pca.fit_transform(F_train).astype(np.float32)
    print(f"PCA applied: {F_train.shape[1]} components "
          f"(explained var {pca.explained_variance_ratio_.sum():.3f})")

    preprocess = {
        "crop_style": cfg["crop_style"],
        "classes": classes,
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "scaler": scaler,
        "pca": pca,
        "image_size": cfg["image_size"],
        "hog_cell": cfg["hog_cell"],
        "hog_block": cfg["hog_block"],
        "hog_bins": cfg["hog_bins"],
        "face_crop_margin": FACE_CROP_MARGIN,
    }
    os.makedirs(MODEL_DIR, exist_ok=True)
    preprocess_path = os.path.join(MODEL_DIR, cfg["preprocess"])
    with open(preprocess_path, "wb") as fh:
        pickle.dump(preprocess, fh)
    print(f"Preprocess saved to {preprocess_path}")
    return F_train, y_train, preprocess


def load_preprocess(mode="crop"):
    cfg = config(mode)
    path = os.path.join(MODEL_DIR, cfg["preprocess"])
    with open(path, "rb") as fh:
        return pickle.load(fh)