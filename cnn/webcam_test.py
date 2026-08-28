import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "outputs_cnn" / "cnn_face_model.pth"
CLASS_NAMES_PATH = BASE_DIR / "outputs_cnn" / "class_names.json"
DETECTOR_PATH = BASE_DIR / "models" / "face_detection_yunet_2023mar.onnx"
SNAPSHOT_DIR = BASE_DIR / "snapshots"
IMG_SIZE = (96, 96)
CAMERA_INDEX = 0
MAX_FACES = 5
SCORE_THRESHOLD = 0.70
MIN_FACE_SIZE = (80, 80)
MARGIN_RATIO = 0.20
UNKNOWN_THRESHOLD = 0.80
GREEN = (0, 255, 0)
RED = (0, 0, 255)
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)


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


def load_resources():
    if not MODEL_PATH.exists() or not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            "Model artifacts not found. Run cnn.py first to train the CNN."
        )
    if not DETECTOR_PATH.exists():
        raise FileNotFoundError(
            f"Face detector not found at {DETECTOR_PATH}. "
            "Download face_detection_yunet_2023mar.onnx into the models folder."
        )
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    with open(CLASS_NAMES_PATH) as f:
        class_names = json.load(f)
    model = FaceRecognitionCNN(len(class_names)).to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    detector = cv2.FaceDetectorYN_create(
        str(DETECTOR_PATH),
        "",
        (320, 240),
        score_threshold=SCORE_THRESHOLD,
    )
    return model, class_names, detector


def open_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open webcam index {CAMERA_INDEX}. "
            "Change CAMERA_INDEX or connect a camera."
        )
    return cap


def detect_faces(frame, detector):
    detector.setInputSize((frame.shape[1], frame.shape[0]))
    _, faces = detector.detect(frame)
    results = []
    if faces is not None:
        for face in faces:
            x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
            score = float(face[14])
            if w >= MIN_FACE_SIZE[0] and h >= MIN_FACE_SIZE[1]:
                results.append((face, x, y, w, h, score))
    results.sort(key=lambda r: r[5], reverse=True)
    return results[:MAX_FACES]


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


def crop_face(frame, face_data):
    face, fx, fy, fw, fh, _ = face_data
    h_frame, w_frame = frame.shape[:2]
    mh, mw = int(fh * MARGIN_RATIO), int(fw * MARGIN_RATIO)
    y1, y2 = max(0, fy - mh), min(h_frame, fy + fh + mh)
    x1, x2 = max(0, fx - mw), min(w_frame, fx + fw + mw)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_rgb = align_face(img_rgb, face)
    img_rgb = img_rgb[y1:y2, x1:x2]
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_gray = CLAHE.apply(img_gray)
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    if img_rgb.size == 0:
        return None
    return cv2.resize(img_rgb, IMG_SIZE)


@torch.no_grad()
def predict_faces(model, class_names, crops_rgb):
    batch = np.stack(crops_rgb).astype(np.float32)
    X = torch.from_numpy(batch).permute(0, 3, 1, 2).to(DEVICE) / 255.0
    X = (X - MEAN) / STD
    probs = torch.softmax(model(X), dim=1).cpu().numpy()
    results = []
    for p in probs:
        idx = int(np.argmax(p))
        results.append((class_names[idx], float(p[idx])))
    return results


def draw_label(frame, text, x, y, color=GREEN):
    ty = y - 10 if y - 10 > 15 else y + 25
    cv2.putText(
        frame, text, (x, ty),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA,
    )
    cv2.putText(
        frame, text, (x, ty),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA,
    )


def main():
    print(f"Using device: {DEVICE}")
    model, class_names, detector = load_resources()
    cap = open_camera()
    SNAPSHOT_DIR.mkdir(exist_ok=True)

    print("Webcam scanner running. [S] snapshot | [Q] quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera stream ended unexpectedly.")
            break
        frame = cv2.flip(frame, 1)
        boxes = detect_faces(frame, detector)

        display_names = []
        if len(boxes):
            crops = [c for c in (crop_face(frame, b) for b in boxes) if c is not None]
            results = predict_faces(model, class_names, crops)
            for face_data, (name, conf) in zip(boxes, results):
                _, x, y, w, h, _ = face_data
                if conf < UNKNOWN_THRESHOLD:
                    label = "Unknown"
                    color = RED
                else:
                    label = f"{name} {conf * 100:.0f}%"
                    color = GREEN
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                draw_label(frame, label, x, y, color)
                display_names.append(label.split(" ")[0])

        cv2.putText(
            frame, "[S] save  [Q] quit", (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )
        cv2.imshow("Face Recognition - Webcam Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        if key in (ord("s"), ord("S")):
            stamp = np.datetime64("now").astype(str).replace(":", "-")
            path = SNAPSHOT_DIR / f"snapshot_{stamp}.png"
            cv2.imwrite(str(path), frame)
            print(f"Snapshot saved: {path}")

    cap.release()
    cv2.destroyAllWindows()
    print("Scanner closed.")


if __name__ == "__main__":
    main()
