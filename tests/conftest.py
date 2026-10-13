"""Config compartida de pytest.

Mockea las dependencias pesadas/ausentes en el entorno local (ultralytics,
torch, torchvision) y expone helpers de frames sintéticos. Las dependencias
ligeras (numpy, opencv-python-headless, requests, ...) se instalan reales.
"""
import sys
import types

import numpy as np
import pytest


class StubYOLO:
    """YOLO falso: `predict` devuelve un result con `boxes=None`."""

    def __init__(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        return [StubPredictResult()]


class StubPredictResult:
    boxes = None


class StubBoxes:
    def __init__(self, xyxy=None):
        self.xyxy = xyxy

    def __len__(self):
        return 0 if self.xyxy is None else len(self.xyxy)

    def cpu(self):
        return self

    def numpy(self):
        return self.xyxy


def _install_stub(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


@pytest.fixture(scope="session", autouse=True)
def _stub_heavy_deps():
    """Inyecta stubs de ultralytics/torch/torchvision (no instalados en CI ni local)."""
    stubs = []
    stubs.append(_install_stub("ultralytics", YOLO=StubYOLO))
    stubs.append(_install_stub("torch"))
    stubs.append(_install_stub("torchvision"))
    yield
    for mod in stubs:
        sys.modules.pop(mod.__name__, None)


@pytest.fixture
def sample_frame():
    """Frame sintético 64x96 (h,w) en BGR para vision/mjpeg."""

    def _make(h=64, w=96, value=127):
        return np.full((h, w, 3), value, dtype=np.uint8)

    return _make
