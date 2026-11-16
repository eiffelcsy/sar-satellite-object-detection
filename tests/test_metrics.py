import pytest

from sarbench.data import SARMultiTask, to_original_xywh
from sarbench.metrics import classification_metrics, detection_metrics


def test_classification_metrics_by_hand():
    m = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], num_classes=2)
    assert m['accuracy'] == 0.75 and m['macro_f1'] == pytest.approx((2 / 3 + 0.8) / 2)
    assert m['confusion_matrix'] == [[1, 1], [0, 2]]


def test_ground_truth_scores_map_1(course_data, preprocess):
    pytest.importorskip('pycocotools')
    dataset = SARMultiTask(course_data, 'train', preprocess)
    predictions, image_ids = [], []
    for i in range(len(dataset)):
        _, _, t = dataset[i]
        image_ids.append(t['image_id'])
        predictions += [{'image_id': t['image_id'], 'category_id': c, 'bbox': b, 'score': 1.0}
                        for b, c in zip(to_original_xywh(t['boxes'], t['orig_size']).tolist(), t['labels'].tolist())]
    assert detection_metrics(dataset.instances_json, predictions, image_ids)['mAP'] == pytest.approx(1.0)
