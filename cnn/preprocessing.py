"""CNN image preprocessing: YuNet face detection, eye alignment, 20%
margin crops, CLAHE enhancement and 96x96 RGB tensor preparation.

Everything that turns a raw image file into a model-ready image batch
lives here so cnn.py only contains the network, training and evaluation.
"""
from pathlib import Path

import cv2
import numpy as np
import torch

IMG_SIZE = (96, 96)
FACE_CROP_MARGIN = 0.20
DETECTOR_PATH = Path("models") / "face_detection_yunet_2023mar.onnx"
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# ImageNet channel statistics used to standardise inputs during training,
# validation and evaluation.
MODEL_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
MODEL_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


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