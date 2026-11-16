"""evaluate.py: a perfect submission scores 100 / 100, and Δm follows the leaderboard formula."""
import csv
import json
import zipfile

import pytest

from evaluate import delta_m, score


def test_delta_m_reference_is_zero():
    assert delta_m(93.99, 25.41, 'test') == pytest.approx(0.0)
    assert delta_m(94.86, 48.95, 'val') == pytest.approx(44.50, abs=0.01)  # the d16 val submission


def test_perfect_submission(tmp_path):
    images = [{'id': i + 1, 'file_name': f'images/{i:05d}.png', 'width': 256, 'height': 256} for i in range(9)]
    boxes = [{'id': i + 1, 'image_id': i + 1, 'category_id': i + 1, 'bbox': [10, 20, 30, 40], 'area': 1200,
              'iscrowd': 0} for i in range(9)]
    (tmp_path / 'inst.json').write_text(json.dumps(
        {'images': images, 'annotations': boxes, 'categories': [{'id': i + 1, 'name': str(i)} for i in range(9)]}))
    with open(tmp_path / 'labels.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['image_id', 'file_name', 'label'])
        w.writerows([im['id'], im['file_name'], i] for i, im in enumerate(images))
    with zipfile.ZipFile(tmp_path / 'sub.zip', 'w') as z:
        z.writestr('classification.csv', 'image_id,label\n' + ''.join(f'{i + 1},{i}\n' for i in range(9)))
        z.writestr('detection.json', json.dumps([{**{k: b[k] for k in ('image_id', 'category_id', 'bbox')},
                                                  'score': 0.9} for b in boxes]))
    s = score(tmp_path / 'sub.zip', tmp_path / 'labels.csv', tmp_path / 'inst.json', 'test')
    assert s['macro_f1'] == pytest.approx(100) and s['mAP'] == pytest.approx(100)
