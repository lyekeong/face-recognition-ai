"""YOLO pipeline preprocessing: YOLOv8n-face detection, 20% margin crops and
ImageNet-normalised 224x224 tranforms for the ResNet-18/50 classifier.

Detection (detect_largest_face + crop_face) is shared by the evaluate
subcommand of yolo.py and the GUI's yolo backend; the train/eval transforms
are shared with yolo.py.
"""
import os

from torchvision import transforms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

DETECTOR_PATH = os.path.join(BASE_DIR, "yolov8n-face.pt")

MIN_FACE_SIZE = (80, 80)
MARGIN_RATIO = 0.20
DETECT_CONF = 0.70

TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


def build_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                               saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
        transforms.RandomErasing(p=0.2),
    ])
    return train_transform, TRANSFORM


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