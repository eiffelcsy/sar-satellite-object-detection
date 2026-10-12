"""make_submission.py: the zip has one classification.csv and one detection.json in the required format."""
import json
import zipfile

from PIL import Image
import torch

from make_submission import write_submission
from sarbench.data import SARImages, collate_images


def _write(tmp_path, images, preds):
    out = tmp_path / 'submission.zip'
    write_submission(images, preds, out=str(out))
    with zipfile.ZipFile(out) as z:
        return z.read('classification.csv').decode().splitlines(), json.loads(z.read('detection.json'))


def test_submission_files_and_scaling(tmp_path):
    # boxes are XYXY on the 512 px model input; 0-indexed classes (write_submission adds 1 for category_id)
    images = [{'id': 3, 'file_name': 'a', 'width': 256, 'height': 256},
              {'id': 44, 'file_name': 'b', 'width': 512, 'height': 256}]
    preds = {'a': (7, [[100.0, 20.0, 180.0, 60.0]], [3], [0.9]),
             'b': (4, [], [], [])}
    rows, dets = _write(tmp_path, images, preds)

    assert rows[0] == 'image_id,label'
    assert rows[1:] == ['3,7', '44,4']
    assert dets == [{'image_id': 3, 'category_id': 4, 'bbox': [50.0, 10.0, 40.0, 20.0], 'score': 0.9}]


def test_submission_record_types(tmp_path):
    images = [{'id': 1, 'file_name': 'a', 'width': 300, 'height': 200}]
    preds = {'a': (0, [[0.0, 0.0, 512.0, 512.0]], [8], [0.5])}
    rows, dets = _write(tmp_path, images, preds)

    (record,) = dets
    assert isinstance(record['image_id'], int)
    assert isinstance(record['category_id'], int) and 1 <= record['category_id'] <= 9
    assert len(record['bbox']) == 4 and all(isinstance(v, float) for v in record['bbox'])
    assert record['bbox'][2] > 0 and record['bbox'][3] > 0 and isinstance(record['score'], float)
    assert int(rows[1].split(',')[1]) == 0


def test_sar_images_reads_unlabelled_split(tmp_path):
    # an images.json with no annotations: one image, id/size/file_name only
    (tmp_path / 'imgs').mkdir()
    Image.new('L', (300, 200), color=128).save(tmp_path / 'imgs' / 'x.bmp')
    (tmp_path / 'images.json').write_text(json.dumps({'images': [
        {'id': 7, 'file_name': 'imgs/x.bmp', 'width': 300, 'height': 200}]}))

    dataset = SARImages(tmp_path / 'images.json')
    image, image_id, orig_size, file_name = dataset[0]
    assert image.shape == (1, 512, 512) and image.dtype == torch.float32
    assert image_id == 7 and orig_size == (200, 300) and file_name == 'imgs/x.bmp'

    images, ids, sizes, names = collate_images([dataset[0], dataset[0]])
    assert images.shape == (2, 1, 512, 512) and ids == [7, 7] and sizes == [(200, 300)] * 2
    assert names == ['imgs/x.bmp', 'imgs/x.bmp']
