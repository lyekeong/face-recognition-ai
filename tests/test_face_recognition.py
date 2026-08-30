"""Unit tests for the face recognition project.

Run with:  python -m pytest tests/ -v
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]


def _add_src(subpath):
    p = str(PROJECT_DIR / subpath)
    if p not in sys.path:
        sys.path.insert(0, p)


_add_src("cnn")
_add_src("gui")
_add_src(os.path.join("machine learning", "src"))
_add_src("yolo")

import cnn       # noqa: E402
import gui       # noqa: E402
import ml        # noqa: E402
import evaluate  # noqa: E402


# ---------------------------------------------------------------------------
# cnn.py
# ---------------------------------------------------------------------------
class TestCNN:
    def test_collect_images(self, tmp_path):
        (tmp_path / "A").mkdir()
        (tmp_path / "B").mkdir()
        (tmp_path / "A" / "a.jpg").write_bytes(b"x")
        (tmp_path / "A" / "b.png").write_bytes(b"x")
        (tmp_path / "B" / "c.jpeg").write_bytes(b"x")
        (tmp_path / "B" / "ignore.txt").write_text("not an image")
        fps, labels, class_names = cnn.collect_images(tmp_path)
        assert class_names == ["A", "B"]
        assert len(fps) == 3
        assert set(labels) == {0, 1}

    def test_collect_images_filters_extensions(self, tmp_path):
        (tmp_path / "A").mkdir()
        (tmp_path / "A" / "a.webp").write_bytes(b"x")
        fps, labels, class_names = cnn.collect_images(tmp_path)
        # .webp is not in cnn.VALID_EXTS
        assert len(fps) == 0

    def test_align_face_identity_angle(self):
        # Horizontal eyes -> rotation angle of 0, image unchanged shape.
        face = np.zeros(15, dtype=np.float32)
        face[2], face[3] = 100, 100          # width, height
        face[4], face[5] = 10, 40            # left eye
        face[6], face[7] = 90, 40            # right eye
        face[14] = 0.9                        # score
        img = np.zeros((120, 120, 3), dtype=np.uint8)
        aligned = cnn.align_face(img, face)
        assert aligned.shape == img.shape

    def test_model_forward_shape(self):
        model = cnn.FaceRecognitionCNN(num_classes=5)
        model.eval()
        x = torch.zeros(2, 3, 96, 96)
        out = model(x)
        assert out.shape == (2, 5)

    def test_make_loaders_shapes(self):
        X = np.zeros((4, 96, 96, 3), dtype=np.uint8)
        y = np.array([0, 1, 0, 1])
        loader = cnn.make_loaders(X, y, shuffle=True)
        Xb, yb = next(iter(loader))
        assert Xb.shape[0] == 4
        assert Xb.dtype == torch.float32
        assert Xb.max() <= 1.0 and Xb.min() >= 0.0


# ---------------------------------------------------------------------------
# GUI helpers
# ---------------------------------------------------------------------------
class TestGUI:
    def test_format_box(self):
        assert gui.format_box((1.2, 2.7, 3.9, 4.1)) == (1, 2, 3, 4)


# ---------------------------------------------------------------------------
# ml.py
# ---------------------------------------------------------------------------
class TestML:
    def test_config_modes(self):
        crop = ml.config("crop")
        direct = ml.config("direct")
        assert crop["crop_style"] == "relaxed"
        assert direct["crop_style"] == "direct"
        assert crop["model"] != direct["model"]

    def test_config_invalid_mode(self):
        with pytest.raises(ValueError):
            ml.config("bogus")

    def test_hog_compute_shape(self):
        hog = ml.HOG(cell_size=8, block_size=2, nbins=9)
        img = np.zeros((64, 64), dtype=np.uint8)
        feat = hog.compute(img)
        # (64/8 - 1) x (64/8 - 1) blocks of 2*2*9
        assert feat.shape == (7 * 7 * 36,)

    def test_preprocess_gray_size(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        gray = ml.preprocess_gray(img, size=64)
        assert gray.shape == (64, 64)
        assert gray.ndim == 2

    def test_transform_features_without_pca(self):
        pp = {"scaler": _IdentityScaler(), "pca": None}
        feat = np.array([[1.0, 2.0, 3.0]])
        out = ml.transform_features(feat, pp)
        np.testing.assert_array_equal(out, feat)


# ---------------------------------------------------------------------------
# yolo/evaluate.py
# ---------------------------------------------------------------------------
class TestEvaluate:
    def test_align_index(self):
        assert evaluate.align_index("resnet18") == "resnet18"
        assert evaluate.align_index("resnet50") == "resnet50"

    def test_crop_face_clamps_to_frame(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # box extending beyond the frame on both top-left and bottom-right
        crop = evaluate.crop_face(frame, (-40, -40, 500, 500))
        assert crop is not None
        assert crop.shape[0] == 100 and crop.shape[1] == 100

    def test_crop_face_empty_box(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # degenerate box collapses to an empty crop
        crop = evaluate.crop_face(frame, (50, 50, 50, 50))
        assert crop is None


class _IdentityScaler:
    def transform(self, X):
        return X
