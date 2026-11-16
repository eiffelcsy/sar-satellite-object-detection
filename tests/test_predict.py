import json
import zipfile

import torch
from torch.utils.data import DataLoader

from predict import predict
from sarbench.config import load_config
from sarbench.data import SARImages, collate_images
from sarbench.model import build_model
from sarbench.submission import write_submission
from train import ROOT


def test_write_submission_maps_boxes_to_original_pixels(tmp_path):
    images = [{'id': 7, 'file_name': 'a.png', 'width': 256, 'height': 128}]
    write_submission(images, {'a.png': (3, [[0., 0., 512., 256.]], [3], [0.9])}, out=tmp_path / 's.zip')
    with zipfile.ZipFile(tmp_path / 's.zip') as z:
        assert z.read('classification.csv').decode() == 'image_id,label\n7,3\n'
        assert json.loads(z.read('detection.json')) == [
            {'image_id': 7, 'category_id': 4, 'bbox': [0.0, 0.0, 256.0, 64.0], 'score': 0.9}]


def test_predict_writes_one_row_per_image(course_data, preprocess, tiny_dinov3, tmp_path):
    model, _ = build_model(load_config(ROOT / 'configs' / 'final.yaml'), pretrained=False)
    dataset = SARImages(course_data / 'val' / 'images.json', preprocess)
    out = tmp_path / 'submission_val.zip'
    write_submission(*predict(model, DataLoader(dataset, 2, collate_fn=collate_images), torch.device('cpu')), out=out)
    with zipfile.ZipFile(out) as z:
        rows = z.read('classification.csv').decode().split()
        detections = json.loads(z.read('detection.json'))
    assert rows[0] == 'image_id,label' and [r.split(',')[0] for r in rows[1:]] == ['1', '2']
    assert all(1 <= d['category_id'] <= 9 and len(d['bbox']) == 4 for d in detections)
