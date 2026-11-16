"""Shared fixtures: a tiny dataset in the course layout, a stand-in pre-processing and a small random DINOv3."""
import csv
import json

import numpy as np
import pytest
from PIL import Image

from sarbench import backbones


class FakePseudoRGB:
    """PseudoRGB's interface without SAR-BM3D: the gray image repeated to 3 channels."""

    def cache_key(self):
        return 'fake'

    def __call__(self, img):
        return np.stack([np.asarray(img, dtype=np.float32)] * 3)


@pytest.fixture
def preprocess():
    return FakePseudoRGB()


@pytest.fixture
def course_data(tmp_path):
    """train/ (labels.csv, instances.json, images/) with 8 images of 2 classes, and val/images.json with 2 images.
    Every image is black with a bright rectangle exactly filling each of its boxes."""
    rng = np.random.default_rng(0)
    train = tmp_path / 'train'
    (train / 'images').mkdir(parents=True)
    rows, images, annotations = [], [], []
    for i in range(8):
        size = (256, 384) if i % 3 == 0 else (512, 512)  # (h, w): some non-square images
        label = i % 2  # aircraft (0) / airport (1)
        pixels = np.zeros(size, dtype=np.uint8)
        for k in range(2):
            x, y = int(rng.integers(0, size[1] - 40)), int(rng.integers(0, size[0] - 40))
            w, h = int(rng.integers(8, 40)), int(rng.integers(8, 40))
            pixels[y:y + h, x:x + w] = 255
            annotations.append({'id': len(annotations) + 1, 'image_id': i + 1, 'category_id': label + 1,
                                'bbox': [x, y, w, h], 'area': w * h, 'iscrowd': 0})
        name = f'{i + 1:05d}.png'
        Image.fromarray(pixels).save(train / 'images' / name)
        rows.append({'image_id': i + 1, 'file_name': f'images/{name}', 'label': label})
        images.append({'id': i + 1, 'file_name': f'images/{name}', 'height': size[0], 'width': size[1]})
    with open(train / 'labels.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, ['image_id', 'file_name', 'label'])
        writer.writeheader()
        writer.writerows(rows)
    categories = [{'id': c + 1, 'name': n} for c, n in enumerate(
        ['aircraft', 'airport', 'bridge', 'car', 'harbor', 'oil_tank', 'playground', 'ship', 'wind_turbine'])]
    (train / 'instances.json').write_text(json.dumps({'images': images, 'annotations': annotations,
                                                      'categories': categories}))
    val = tmp_path / 'val'
    (val / 'images').mkdir(parents=True)
    for i in range(2):
        Image.fromarray(np.zeros((300, 400), dtype=np.uint8)).save(val / 'images' / f'{i + 1:05d}.png')
    (val / 'images.json').write_text(json.dumps({'images': [
        {'id': i + 1, 'file_name': f'images/{i + 1:05d}.png', 'height': 300, 'width': 400} for i in range(2)]}))
    return tmp_path


@pytest.fixture
def tiny_dinov3(tmp_path, monkeypatch):
    """Make `DINOv3(pretrained=False)` build a 2-block, 128-wide ViT (same code path, CPU-sized)."""
    config = json.loads(backbones.CONFIG_JSON.read_text())
    config.update(hidden_size=128, intermediate_size=256, num_hidden_layers=2, num_attention_heads=2)
    path = tmp_path / 'tiny_dinov3.json'
    path.write_text(json.dumps(config))
    monkeypatch.setattr(backbones, 'CONFIG_JSON', path)
