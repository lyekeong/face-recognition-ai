import os
import sys
import time
import argparse
import cv2
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ml as pp

# ML-specific unknown gate. Calibrated SVM confidence saturates low
# (isotonic calibration in 17-class ovr; max achievable confidence is
# ~0.716), so 0.50 is the best precision/recall balance for the ML model
# while CNN/YOLO keep 0.80.
CONF_THRESHOLD = 0.50


def load_model():
    pp_art = pp.load_preprocess()
    model = joblib.load(os.path.join(pp.MODEL_DIR, "svm_model.joblib"))
    hog = pp.make_hog(pp_art)
    return model, hog, pp_art


def recognize_frame(frame_bgr, model, hog, pp_art, detector):
    feat, box = pp.frame_to_features(frame_bgr, detector, hog,
                                     image_size=pp_art["image_size"])
    if box is None:
        return frame_bgr, None, 0.0

    x0, y0, x1, y1 = box
    feat = pp.transform_features(feat, pp_art)
    proba = model.predict_proba(feat)[0]
    idx = int(proba.argmax())
    name = pp_art["idx_to_label"][idx]
    confidence = float(proba[idx])

    if confidence >= CONF_THRESHOLD:
        label = f"{name} {confidence:.2f}"
        color = (0, 255, 0)
    else:
        label = "Unknown"
        name = "Unknown"
        color = (0, 0, 255)

    cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), color, 2)
    cv2.putText(frame_bgr, label, (x0, max(0, y0 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame_bgr, name, confidence


def run_webcam(camera_index=0):
    model, hog, pp_art = load_model()
    detector = pp.make_detector()
    print("Models loaded. Starting webcam... (q=quit, p=pause)")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: could not open webcam.", file=sys.stderr)
        return

    paused = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if not paused:
            frame, latest, _ = recognize_frame(frame, model, hog, pp_art,
                                               detector)
        elif latest:
            cv2.putText(frame, "PAUSED", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow("Face Recognition (HOG+SVM) - q to quit", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p"):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Webcam face recognition")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index (default 0)")
    args = parser.parse_args()
    run_webcam(camera_index=args.camera)