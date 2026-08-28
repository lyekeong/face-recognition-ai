import json
import os

import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from ultralytics import YOLO

from train_classifier import make_model_for_inference

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "output")

MIN_FACE_SIZE = (10, 10)  # unused: size gate removed, every detection is classified
MARGIN_RATIO = 0.20
DETECT_CONF = 0.70
UNKNOWN_THRESHOLD = 0.80


def load_arch():
    with open(os.path.join(OUT_DIR, "arch.json"), encoding="utf-8") as fh:
        return json.load(fh)


arch = load_arch()
BACKBONE = arch["backbone"]
CLASS_NAMES = arch["class_names"]
NUM_CLASSES = arch["num_classes"]
MODEL_PATH = os.path.join(
    OUT_DIR, "resnet18_faces.pth" if BACKBONE == "resnet18"
    else "resnet50_faces.pth")

yolo_model = YOLO(os.path.join(BASE_DIR, "yolov8n-face.pt"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet_model = make_model_for_inference(BACKBONE, NUM_CLASSES)
resnet_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
resnet_model.to(device)
resnet_model.eval()
print(f"Loaded model: {BACKBONE} from {MODEL_PATH}")

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

cap = cv2.VideoCapture(0)

print("Starting webcam... (press 'q' to quit)")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Could not read camera frame")
        break

    results = yolo_model.predict(source=frame, conf=DETECT_CONF, verbose=False)

    for result in results:
        for box in result.boxes:
            if int(box.cls[0]) != 0:
                continue
            xmin, ymin, xmax, ymax = map(int, box.xyxy[0].cpu().numpy())

            h, w = frame.shape[:2]
            pad_x = int((xmax - xmin) * MARGIN_RATIO)
            pad_y = int((ymax - ymin) * MARGIN_RATIO)
            xmin, ymin = max(0, xmin - pad_x), max(0, ymin - pad_y)
            xmax, ymax = min(w, xmax + pad_x), min(h, ymax + pad_y)

            face_crop = frame[ymin:ymax, xmin:xmax]
            if face_crop.size == 0:
                continue

            face_tensor = transform(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)) \
                .unsqueeze(0).to(device)
            with torch.no_grad():
                proba = torch.softmax(resnet_model(face_tensor), dim=1)[0]
                max_prob, predicted = torch.max(proba, 0)
                prob_value = max_prob.item()
                predicted_name = CLASS_NAMES[predicted.item()]

            if prob_value < UNKNOWN_THRESHOLD:
                label = "Unknown"
                color = (0, 0, 255)
            else:
                label = f"{predicted_name} ({prob_value:.2f})"
                color = (0, 255, 0)

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(frame, label, (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("TARUMT AI Prototype - Live Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()