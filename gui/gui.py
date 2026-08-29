import json
import os
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageTk
from tkinter import font as tkfont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CNN_DIR = os.path.join(ROOT, "cnn")
YOLO_DIR = os.path.join(ROOT, "yolo")
ML_DIR = os.path.join(ROOT, "machine learning", "src")
for _d in (YOLO_DIR, ML_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

MODEL_NAMES = ["CNN", "YOLO (ResNet18)", "HOG + SVM"]

DISPLAY_W, DISPLAY_H = 680, 470
UNKNOWN_COLOR = (0, 0, 255)
KNOWN_COLOR = (0, 255, 0)


def draw_label(frame, text, x, y, color):
    ty = y - 10 if y - 10 > 15 else y + 25
    cv2.putText(frame, text, (x, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (x, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                color, 1, cv2.LINE_AA)


def format_box(box):
    x, y, w, h = box
    return (int(x), int(y), int(w), int(h))


# CNN (custom 4-block CNN + YuNet detector) ==============================
CNN_MODEL_PATH = Path(CNN_DIR) / "outputs_cnn" / "cnn_face_model.pth"
CNN_CLASS_NAMES_PATH = Path(CNN_DIR) / "outputs_cnn" / "class_names.json"
CNN_DETECTOR_PATH = Path(CNN_DIR) / "models" / "face_detection_yunet_2023mar.onnx"
CNN_IMG_SIZE = (96, 96)
CNN_MAX_FACES = 5
CNN_SCORE_THRESHOLD = 0.70
CNN_MIN_FACE_SIZE = (10, 10)
CNN_MARGIN_RATIO = 0.20
CNN_UNKNOWN_THRESHOLD = 0.60
CNN_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
_CNN_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_CNN_MEAN = torch.tensor([0.485, 0.456, 0.406], device=_CNN_DEVICE).view(1, 3, 1, 1)
_CNN_STD = torch.tensor([0.229, 0.224, 0.225], device=_CNN_DEVICE).view(1, 3, 1, 1)


class CNNConvBlock(nn.Module):
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


class CNNFaceRecognizer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.stem = nn.Sequential(
            CNNConvBlock(3, 32),
            CNNConvBlock(32, 64),
            CNNConvBlock(64, 128),
            CNNConvBlock(128, 256),
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


def _cnn_align_face(img_rgb, face):
    left_eye = (face[4], face[5])
    right_eye = (face[6], face[7])
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))
    center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img_rgb, M, (img_rgb.shape[1], img_rgb.shape[0]),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _cnn_crop_face(frame, face_data):
    face, fx, fy, fw, fh, _ = face_data
    h_frame, w_frame = frame.shape[:2]
    mh, mw = int(fh * CNN_MARGIN_RATIO), int(fw * CNN_MARGIN_RATIO)
    y1, y2 = max(0, fy - mh), min(h_frame, fy + fh + mh)
    x1, x2 = max(0, fx - mw), min(w_frame, fx + fw + mw)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_rgb = _cnn_align_face(img_rgb, face)
    img_rgb = img_rgb[y1:y2, x1:x2]
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_gray = CNN_CLAHE.apply(img_gray)
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    if img_rgb.size == 0:
        return None
    return cv2.resize(img_rgb, CNN_IMG_SIZE)


@torch.no_grad()
def _cnn_predict_faces(model, class_names, crops_rgb):
    batch = np.stack(crops_rgb).astype(np.float32)
    X = torch.from_numpy(batch).permute(0, 3, 1, 2).to(_CNN_DEVICE) / 255.0
    X = (X - _CNN_MEAN) / _CNN_STD
    probs = torch.softmax(model(X), dim=1).cpu().numpy()
    results = []
    for p in probs:
        idx = int(np.argmax(p))
        results.append((class_names[idx], float(p[idx])))
    return results


class CNNBackend:
    name = "CNN"
    icon = "CNN"

    def load(self):
        if not CNN_MODEL_PATH.exists() or not CNN_CLASS_NAMES_PATH.exists():
            raise FileNotFoundError(
                "CNN model artifacts not found. Run cnn.py first to train the CNN."
            )
        if not CNN_DETECTOR_PATH.exists():
            raise FileNotFoundError(
                f"Face detector not found at {CNN_DETECTOR_PATH}."
            )
        checkpoint = torch.load(CNN_MODEL_PATH, map_location=_CNN_DEVICE)
        with open(CNN_CLASS_NAMES_PATH) as f:
            self.class_names = json.load(f)
        self.model = CNNFaceRecognizer(len(self.class_names)).to(_CNN_DEVICE)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.detector = cv2.FaceDetectorYN_create(
            str(CNN_DETECTOR_PATH), "",
            (320, 240),
            score_threshold=CNN_SCORE_THRESHOLD,
        )

    def recognize(self, frame):
        ann = frame.copy()
        results = []
        self.detector.setInputSize((frame.shape[1], frame.shape[0]))
        _, faces = self.detector.detect(frame)
        boxes = []
        if faces is not None:
            for face in faces:
                x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                score = float(face[14])
                if w >= CNN_MIN_FACE_SIZE[0] and h >= CNN_MIN_FACE_SIZE[1]:
                    boxes.append((face, x, y, w, h, score))
            boxes.sort(key=lambda r: r[5], reverse=True)
            boxes = boxes[:CNN_MAX_FACES]
        if not boxes:
            return ann, results
        crops = [c for c in (_cnn_crop_face(frame, b) for b in boxes)
                 if c is not None]
        if not crops:
            return ann, results
        preds = _cnn_predict_faces(self.model, self.class_names, crops)
        for face_data, (name, conf) in zip(boxes, preds):
            _, x, y, w, h, _ = face_data
            box = format_box((x, y, w, h))
            if conf < CNN_UNKNOWN_THRESHOLD:
                label, color = "Unknown", UNKNOWN_COLOR
            else:
                label, color = f"{name} {conf * 100:.0f}%", KNOWN_COLOR
            cv2.rectangle(ann, (x, y), (x + w, y + h), color, 2)
            draw_label(ann, label, x, y, color)
            results.append({"label": label, "name": name,
                            "confidence": conf, "box": box})
        return ann, results


class YOLOBackend:
    name = "YOLO (ResNet18)"
    icon = "YOLO"
    unknown_threshold = 0.60

    def load(self):
        from torchvision import transforms
        import train_classifier
        from ultralytics import YOLO

        arch_path = os.path.join(YOLO_DIR, "output", "arch.json")
        with open(arch_path, encoding="utf-8") as fh:
            arch = json.load(fh)
        backbone = arch["backbone"]
        self.class_names = arch["class_names"]
        num_classes = arch["num_classes"]

        weights = os.path.join(YOLO_DIR, "output", f"{backbone}_faces.pth")
        yolo_path = os.path.join(YOLO_DIR, "yolov8n-face.pt")
        if not os.path.exists(weights) or not os.path.exists(yolo_path):
            raise FileNotFoundError(
                f"YOLO artifacts missing: {weights} or {yolo_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resnet = train_classifier.make_model_for_inference(
            backbone, num_classes)
        self.resnet.load_state_dict(torch.load(weights, map_location=self.device))
        self.resnet.to(self.device)
        self.resnet.eval()
        self.yolo = YOLO(yolo_path)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def recognize(self, frame):
        ann = frame.copy()
        h, w = frame.shape[:2]
        results = []
        dets = self.yolo.predict(source=frame, conf=0.70, verbose=False)
        for result in dets:
            for box in result.boxes:
                if int(box.cls[0]) != 0:
                    continue
                xmin, ymin, xmax, ymax = map(int, box.xyxy[0].cpu().numpy())
                pad_x = int((xmax - xmin) * 0.20)
                pad_y = int((ymax - ymin) * 0.20)
                xmin, ymin = max(0, xmin - pad_x), max(0, ymin - pad_y)
                xmax, ymax = min(w, xmax + pad_x), min(h, ymax + pad_y)
                crop = frame[ymin:ymax, xmin:xmax]
                if crop.size == 0:
                    continue
                tensor = self.transform(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)) \
                    .unsqueeze(0).to(self.device)
                with torch.no_grad():
                    proba = torch.softmax(self.resnet(tensor), dim=1)[0]
                    max_prob, predicted = torch.max(proba, 0)
                conf = float(max_prob.item())
                name = self.class_names[predicted.item()]
                if conf < self.unknown_threshold:
                    label, color = "Unknown", UNKNOWN_COLOR
                else:
                    label, color = f"{name} ({conf:.2f})", KNOWN_COLOR
                cv2.rectangle(ann, (xmin, ymin), (xmax, ymax), color, 2)
                draw_label(ann, label, xmin, ymin, color)
                results.append({"label": label, "name": name,
                                "confidence": conf,
                                "box": (xmin, ymin, xmax - xmin, ymax - ymin)})
        return ann, results


class MLBackend:
    name = "HOG + SVM"
    icon = "ML"

    def load(self):
        import ml
        self.mod = ml
        self.model, self.hog, self.pp_art = ml.load_model_and_preprocess()
        self.detector = ml.make_detector()

    def recognize(self, frame):
        m = self.mod
        ann = frame.copy()
        name, conf, box = m.predict_face(frame, self.model, self.hog,
                                         self.pp_art, self.detector)
        if box is None:
            return ann, []
        x0, y0, x1, y1 = (int(v) for v in box)
        w, h = x1 - x0, y1 - y0
        if conf < m.CONF_THRESHOLD:
            label, color = "Unknown", UNKNOWN_COLOR
        else:
            label, color = f"{name} ({conf:.2f})", KNOWN_COLOR
        cv2.rectangle(ann, (x0, y0), (x1, y1), color, 2)
        draw_label(ann, label, x0, y0, color)
        return ann, [{"label": label, "name": name,
                      "confidence": conf, "box": (x0, y0, w, h)}]


BACKENDS = {"CNN": CNNBackend, "YOLO (ResNet18)": YOLOBackend,
            "HOG + SVM": MLBackend}

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition GUI (CNN / YOLO / HOG+SVM)")
        self.root.geometry("820x700")
        self.root.minsize(760, 620)

        self.backends = {}
        self.locks = {}
        self.current = "CNN"
        self._cam = None
        self._webcam_on = False
        self._last_frame = None

        self._build_ui()

    def _build_ui(self):
        style = ttk.Style(self.root)
        style.configure("TButton", font=("Segoe UI", 12))
        style.configure("TCombobox", font=("Segoe UI", 12))
        style.configure("TLabel", font=("Segoe UI", 12))

        base_font = tkfont.nametofont("TkDefaultFont")
        base_font.configure(size=12)

        top = ttk.Frame(self.root, padding=(10, 8))
        top.pack(fill="x")

        ttk.Label(top, text="Model:").pack(side="left")
        self.model_var = tk.StringVar(value=self.current)
        self.model_cb = ttk.Combobox(top, textvariable=self.model_var,
                                     values=MODEL_NAMES, state="readonly",
                                     width=18)
        self.model_cb.pack(side="left", padx=(4, 14))
        self.model_cb.bind("<<ComboboxSelected>>", self.on_model_change)

        self.btn_open = ttk.Button(top, text="Open Image...",
                                   command=self.on_open_image)
        self.btn_open.pack(side="left", padx=(0, 8))

        self.btn_webcam = ttk.Button(top, text="Live Webcam",
                                     command=self.on_webcam_toggle)
        self.btn_webcam.pack(side="left")

        self.status = tk.StringVar(value="Select a model and open an image, "
                                         "or start the webcam.")
        self._status_label = ttk.Label(self.root, textvariable=self.status,
                                       anchor="w", padding=(10, 4),
                                       font=("Segoe UI", 13, "bold"),
                                       foreground="#0b5394")
        self._status_label.pack(fill="x")

        frame = ttk.Frame(self.root)
        frame.pack(fill="both", expand=True, padx=10, pady=(4, 10))
        self.canvas = tk.Label(frame, background="#1c1c1c",
                               width=DISPLAY_W, height=DISPLAY_H)
        self.canvas.pack(fill="both", expand=True)
        self._show_placeholder()

    def _show_placeholder(self):
        img = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
        self._show_frame(img)

    def _show_frame(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        scale = min(DISPLAY_W / w, DISPLAY_H / h, 1.0)
        nw, nh = int(w * scale), int(h * scale)
        if scale < 1.0:
            disp = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        else:
            disp = frame_bgr
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        self._tkimg = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.configure(image=self._tkimg)

    def on_model_change(self, event=None):
        name = self.model_var.get()
        if name == self.current:
            return
        self.current = name
        self._set_status_color(
            f"Loading {name} model - first load may take a moment...")
        threading.Thread(target=self._load_backend,
                         args=(name,), daemon=True).start()

    def _load_backend(self, name):
        try:
            self._ensure_backend(name)
        except Exception as exc:  # noqa: BLE001
            self.root.after(0, lambda e=exc: self._report_error(name, e))
            return
        self.root.after(0, lambda: self._set_status_color(
            f"{name} model loaded. Open an image or start the webcam."))

    def _ensure_backend(self, name):
        if name in self.backends:
            return self.backends[name]
        lock = self.locks.setdefault(name, threading.Lock())
        with lock:
            if name not in self.backends:
                backend = BACKENDS[name]()
                backend.load()
                self.backends[name] = backend
        return self.backends[name]

    def _report_error(self, name, exc):
        self._set_status_color(f"{name} failed to load: {exc}", error=True)
        messagebox.showerror("Model load error", f"{name}:\n{exc}")

    def on_open_image(self):
        if self._webcam_on:
            self._stop_webcam()
        path = filedialog.askopenfilename(
            title="Choose an image to recognize",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp"),
                       ("All files", "*.*")])
        if not path:
            return
        self._set_status_color(f"Running {self.current} on "
                               f"{os.path.basename(path)}...")
        self.model_cb.state(["disabled"])
        threading.Thread(target=self._run_image, args=(path,),
                         daemon=True).start()

    def _run_image(self, path):
        try:
            backend = self._ensure_backend(self.current)
            frame = cv2.imread(path)
            if frame is None:
                raise ValueError(f"Cannot read image: {path}")
            ann, results = backend.recognize(frame)
        except Exception as exc:  # noqa: BLE001
            self.root.after(0, lambda e=exc: self._report_error(self.current, e))
            self.root.after(0, lambda: self.model_cb.state(["!disabled"]))
            return
        self.root.after(0, lambda: self._finish_image(ann, results, path))

    def _finish_image(self, ann, results, path):
        self.model_cb.state(["!disabled"])
        self._show_frame(ann)
        if not results:
            self._set_status_color(
                f"No face detected in {os.path.basename(path)} "
                f"(model: {self.current}).")
            return
        parts = [r["label"] for r in results]
        self._set_status_color(
            f"{self.current} | {os.path.basename(path)} | "
            f"{len(results)} face(s): " + ", ".join(parts))

    def on_webcam_toggle(self):
        if self._webcam_on:
            self._stop_webcam()
            return
        self._set_status_color(f"Starting webcam with {self.current}...")
        self.btn_open.state(["disabled"])
        self.model_cb.state(["disabled"])
        threading.Thread(target=self._start_webcam, daemon=True).start()

    def _start_webcam(self):
        try:
            backend = self._ensure_backend(self.current)
            self._cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self._cam.isOpened():
                self._cam = cv2.VideoCapture(0)
            if not self._cam.isOpened():
                raise RuntimeError("Unable to open the webcam.")
            self._backend = backend
            self._webcam_on = True
            self.root.after(0, lambda: self.btn_webcam.configure(text="Stop Webcam"))
            self.root.after(0, lambda: self._set_status_color(
                f"Live recognition with {backend.icon}. Click 'Stop Webcam' to end."))
            self._poll_webcam()
        except Exception as exc:  # noqa: BLE001
            self._webcam_on = False
            if self._cam is not None:
                self._cam.release()
            self.root.after(0, lambda e=exc: self._report_error(self.current, e))
            self.root.after(0, self._reset_controls)

    def _poll_webcam(self):
        if not self._webcam_on or self._cam is None:
            return
        ok, frame = self._cam.read()
        if ok:
            frame = cv2.flip(frame, 1)
            ann, results = self._backend.recognize(frame)
            self._last_frame = ann
            self.root.after(0, lambda: self._show_frame(ann))
            if results:
                parts = [r["label"] for r in results]
                self.root.after(0, lambda: self._set_status_color(
                    f"{self.current} live | " + ", ".join(parts)))
            else:
                self.root.after(0, lambda: self._set_status_color(
                    f"{self.current} live | No face detected"))
        self.root.after(30, self._poll_webcam)

    def _stop_webcam(self):
        self._webcam_on = False
        if self._cam is not None:
            self._cam.release()
            self._cam = None
        self.btn_webcam.configure(text="Live Webcam")
        self._reset_controls()
        self._set_status_color("Webcam stopped.")

    def _reset_controls(self):
        self.btn_open.state(["!disabled"])
        self.model_cb.state(["!disabled"])

    def _set_status_color(self, text, error=False):
        self.status.set(text)
        self._status_label.configure(
            foreground="#c0392b" if error else "#0b5394")

    def _on_close(self):
        self._stop_webcam()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", app._on_close)
    root.mainloop()


if __name__ == "__main__":
    main()