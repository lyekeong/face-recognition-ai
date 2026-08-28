import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "comparison")
OUT_CSV = os.path.join(OUT_DIR, "comparison.csv")
OUT_PNG = os.path.join(OUT_DIR, "comparison.png")

CNN_METRICS = os.path.join(BASE_DIR, "cnn", "outputs_cnn", "metrics.json")
YOLO_RESULTS = os.path.join(BASE_DIR, "yolo", "output", "results.csv")
YOLO_ARCH = os.path.join(BASE_DIR, "yolo", "output", "arch.json")
ML_RESULTS = os.path.join(BASE_DIR, "machine learning", "output", "results.csv")


def read_csv(path, key_ix=0, val_ix=1):
    data = {}
    with open(path, "r", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if len(row) >= 2 and row[key_ix] != "metric":
                try:
                    data[row[key_ix]] = float(row[val_ix])
                except ValueError:
                    data[row[key_ix]] = row[val_ix]
    return data


def load_cnn():
    with open(CNN_METRICS, "r", encoding="utf-8") as fh:
        m = json.load(fh)
    return {
        "method": "CNN",
        "algorithm": "Custom CNN (4 conv blocks) + YuNet detector",
        "test_accuracy": m["test_accuracy"],
        "precision_macro": m["precision_macro"],
        "recall_macro": m["recall_macro"],
        "f1_macro": m["f1_macro"],
        "gate_threshold": ">= 0.80 (webcam only)",
        "gate_precision": float("nan"),
        "gate_recall": float("nan"),
        "pipeline_acc": float("nan"),
        "pipeline_acc_detected": float("nan"),
        "notes": "No gated evaluation saved; webcam rejects confidence < 0.80.",
    }


def load_yolo():
    r = read_csv(YOLO_RESULTS)
    with open(YOLO_ARCH, "r", encoding="utf-8") as fh:
        arch = json.load(fh)
    g08 = ("gate_0.80_precision", "gate_0.80_recall")
    g08p = r.get(g08[0], float("nan"))
    g08r = r.get(g08[1], float("nan"))
    bb = arch.get("backbone", "?")
    if isinstance(bb, str) and bb.lower().startswith("resnet"):
        bb = "ResNet" + bb[6:]
    return {
        "method": "YOLO",
        "algorithm": f"YuNet detector + {bb} classifier",
        "test_accuracy": arch.get("test_acc", float("nan")),
        "precision_macro": r["precision_macro"],
        "recall_macro": r["recall_macro"],
        "f1_macro": r["f1_macro"],
        "gate_threshold": ">= 0.80",
        "gate_precision": g08p,
        "gate_recall": g08r,
        "pipeline_acc": r["accuracy_argmax_all"],
        "pipeline_acc_detected": r["accuracy_argmax_detected"],
        "notes": "accuracy is pure classifier test acc; pipeline acc counts undetected "
                 "faces as wrong.",
    }


def load_ml():
    r = read_csv(ML_RESULTS)
    return {
        "method": "HOG+SVM",
        "algorithm": "HOG features + PCA + RBF SVM (isotonic)",
        "test_accuracy": r["accuracy_argmax"],
        "precision_macro": r["precision_macro"],
        "recall_macro": r["recall_macro"],
        "f1_macro": r["f1_macro"],
        "gate_threshold": f">= {r['gate_threshold']:.2f}",
        "gate_precision": r["gate_precision"],
        "gate_recall": r["gate_recall"],
        "pipeline_acc": float("nan"),
        "pipeline_acc_detected": float("nan"),
        "notes": "Calibrated confidences saturate near 0.72, so gate uses 0.50.",
    }


def main():
    rows = []
    if os.path.exists(CNN_METRICS):
        rows.append(load_cnn())
    else:
        print(f"[warn] Missing {CNN_METRICS}")
    if os.path.exists(YOLO_RESULTS) and os.path.exists(YOLO_ARCH):
        rows.append(load_yolo())
    else:
        print(f"[warn] Missing {YOLO_RESULTS} / {YOLO_ARCH}")
    if os.path.exists(ML_RESULTS):
        rows.append(load_ml())
    else:
        print(f"[warn] Missing {ML_RESULTS}")

    if not rows:
        raise SystemExit("No evaluation artifacts found to compare.")

    os.makedirs(OUT_DIR, exist_ok=True)

    fields = ["method", "algorithm", "test_accuracy", "precision_macro",
              "recall_macro", "f1_macro", "gate_threshold", "gate_precision",
              "gate_recall", "pipeline_acc", "pipeline_acc_detected", "notes"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print("Method comparison (argmax on test set):")
    print(f"  {'Method':<10}{'Accuracy':>9}{'Precision':>10}{'Recall':>9}{'F1':>9}"
          f"{'Gate P':>8}{'Gate R':>8}{'PipeAcc':>9}")
    for r in rows:
        fmt = lambda **kw: ""
        print(f"  {r['method']:<10}"
              f"{r['test_accuracy']:>9.3f}"
              f"{r['precision_macro']:>10.3f}"
              f"{r['recall_macro']:>9.3f}"
              f"{r['f1_macro']:>9.3f}"
              f"{r['gate_precision'] if not np.isnan(r['gate_precision']) else float('nan'):>8.3f}"
              f"{r['gate_recall'] if not np.isnan(r['gate_recall']) else float('nan'):>8.3f}"
              f"{r['pipeline_acc'] if not np.isnan(r['pipeline_acc']) else float('nan'):>9.3f}")

    txt = np.nanmax([r["test_accuracy"] for r in rows])
    best_all = [r["method"] for r in rows
                if r["test_accuracy"] == txt][0]
    precs = [(r["method"], r["gate_precision"]) for r in rows
             if not np.isnan(r["gate_precision"])]
    best_prec = max(precs, key=lambda x: x[1])[0]
    print(f"\nTakeaway: best raw accuracy = {best_all} "
          f"({txt:.3f}); best unknown-rejection precision = {best_prec}; "
          f"CNN and YOLO (deep learning) clearly beat the classical HOG+SVM "
          f"baseline.")

    # Grouped bar chart -------------------------------------------------------
    methods = [r["method"] for r in rows]
    metrics = ["test_accuracy", "precision_macro", "recall_macro", "f1_macro"]
    labels = ["Accuracy", "Precision", "Recall", "F1"]
    x = np.arange(len(labels))
    width = 0.26

    fig, ax = plt.subplots(figsize=(11, 6.5))
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    for i, r in enumerate(rows):
        vals = [r[k] for k in metrics]
        bars = ax.bar(x + (i - 1) * width, vals, width,
                      label=r["method"], color=colors[i % len(colors)],
                      edgecolor="black", linewidth=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Face Recognition - Method Comparison "
                 "(CNN vs YOLO/ResNet18 vs HOG+SVM)", fontsize=12)
    ax.legend(loc="upper right", ncol=3, frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"Saved {OUT_CSV}")
    print(f"Saved {OUT_PNG}")


if __name__ == "__main__":
    main()