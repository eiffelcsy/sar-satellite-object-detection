import torch

from sarbench.data import SARImages, SARMultiTask, copy_paste, drop_degenerate_boxes, holdout, to_original_xywh


def test_holdout_is_disjoint_complete_and_fixed_by_its_seed():
    rows = list(range(100))
    train, val = holdout(rows, 'train', 0.1, 0), holdout(rows, 'val', 0.1, 0)
    assert len(val) == 10 and sorted(train + val) == rows
    assert holdout(rows, 'val', 0.1, 0) == val != holdout(rows, 'val', 0.1, 1)
    assert holdout(rows, 'train', 0.0, 0) == rows


def test_boxes_follow_the_resize_and_flips(course_data, preprocess):
    dataset = SARMultiTask(course_data, 'train', preprocess, train=True)
    for seed in range(4):
        torch.manual_seed(seed)
        image, label, target = dataset[0]  # 256 x 384, stretched to 512 x 512
        assert image.shape == (3, 512, 512) and target['labels'].tolist() == [label + 1] * 2
        for x1, y1, x2, y2 in target['boxes'].round().long().tolist():
            assert image[0, y1 + 1:y2 - 1, x1 + 1:x2 - 1].min() > 0.9  # the bright rectangle moved with its box


def test_boxes_map_back_to_original_pixels(course_data, preprocess):
    _, _, target = SARMultiTask(course_data, 'train', preprocess)[0]
    original = [a['bbox'] for a in SARMultiTask(course_data, 'train', preprocess).annotations[target['image_id']]]
    assert torch.allclose(to_original_xywh(target['boxes'], target['orig_size']), torch.tensor(original).float(),
                          atol=1e-3)


def test_copy_paste_adds_objects_with_their_margin_and_never_overlaps():
    torch.manual_seed(0)
    image, donor = torch.zeros(3, 64, 64), torch.zeros(3, 64, 64)
    donor[:, 10:20, 10:20] = 1.0
    donor[:, 9, 9] = 0.5  # inside the 1 px margin
    boxes = torch.tensor([[0., 0., 8., 8.]])
    out, out_boxes, labels = copy_paste(image, boxes, torch.tensor([1]), donor, torch.tensor([[10., 10., 20., 20.]]),
                                        torch.tensor([1]), margin=1)
    assert len(out_boxes) == 2 and labels.tolist() == [1, 1]
    x1, y1, x2, y2 = out_boxes[1].long().tolist()
    assert out[0, y1:y2, x1:x2].min() == 1.0 and out[0, y1 - 1, x1 - 1] == 0.5


def test_copy_paste_draws_donors_of_the_same_class(course_data, preprocess):
    dataset = SARMultiTask(course_data, 'train', preprocess, train=True, copy_paste=1.0)
    torch.manual_seed(0)
    for i in range(len(dataset)):
        _, label, target = dataset[i]
        assert (target['labels'] == label + 1).all()


def test_drop_degenerate_boxes():
    boxes = torch.tensor([[0., 0., 5., 5.], [3., 3., 3., 9.]])
    kept, labels = drop_degenerate_boxes(boxes, torch.tensor([1, 2]))
    assert kept.tolist() == [[0., 0., 5., 5.]] and labels.tolist() == [1]


def test_unlabelled_split_and_preprocess_cache(course_data, preprocess, tmp_path):
    dataset = SARImages(course_data / 'val' / 'images.json', preprocess, cache_dir=tmp_path / 'cache')
    image, image_id, orig_size, file_name = dataset[1]
    assert image.shape == (3, 512, 512) and (image_id, orig_size, file_name) == (2, (300, 400), 'images/00002.png')
    assert len(list((tmp_path / 'cache').glob('*__fake.npy'))) == 1
    assert torch.equal(dataset[1][0], image)  # second read: from the cache
