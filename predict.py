"""Predict an unlabelled split with the trained model and write the Codabench submission zip.

    python predict.py --weights model.pt --images-json /path/to/cs701-sar-course-data/test/images.json
    # -> submission_test.zip (classification.csv + detection.json)

Needs no Hugging Face access: the DINOv3 architecture comes from sarbench/dinov3_vitb16_config.json and every weight
from model.pt. Inference runs in bf16 autocast on a GPU, as for the submitted results (on a CPU it runs in fp32, which
gives near-identical but not bit-identical boxes). The pseudo-RGB input runs SAR-BM3D on every image (~1.5 s each,
in parallel over --workers); --cache keeps the results for a second run.
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from sarbench.config import load_config
from sarbench.data import SARImages, collate_images
from sarbench.model import build_model
from sarbench.preprocess import PseudoRGB
from sarbench.submission import write_submission

ROOT = Path(__file__).resolve().parent


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--weights', type=Path, required=True, help='model.pt written by train.py')
    p.add_argument('--images-json', type=Path, required=True, help='images.json of the split (e.g. test/images.json)')
    p.add_argument('--image-dir', type=Path, help='where its images are (default: next to images.json)')
    p.add_argument('--out', type=Path, help='the zip to write (default: submission_<split dir name>.zip)')
    p.add_argument('--config', type=Path, default=ROOT / 'configs' / 'final.yaml')
    p.add_argument('--nms-thresh', type=float, help='override model.nms_thresh (0.85; the val submission used 0.8)')
    p.add_argument('--cache', type=Path, help='cache the pre-processed images here (default: no cache)')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args(argv)


@torch.no_grad()
def predict(model, loader, device):
    """(images, preds) for `write_submission`: per image its class and its boxes, classes and scores."""
    model.eval()
    sub_images, sub_preds = [], {}
    for images, image_ids, orig_sizes, file_names in loader:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            logits, outputs, _ = model(images.to(device))
        for image_id, (height, width), file_name, out, label in zip(image_ids, orig_sizes, file_names, outputs,
                                                                     logits.argmax(dim=1).tolist()):
            sub_images.append({'id': image_id, 'file_name': file_name, 'width': int(width), 'height': int(height)})
            sub_preds[file_name] = (label, out['boxes'].tolist(), (out['labels'] - 1).tolist(),
                                    out['scores'].tolist())
        print(f'{len(sub_images)} / {len(loader.dataset)} images', flush=True)
    return sub_images, sub_preds


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.nms_thresh is not None:
        config['model']['nms_thresh'] = args.nms_thresh
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True  # as in training and the submitted predictions
    torch.backends.cudnn.allow_tf32 = True
    model, _ = build_model(config, pretrained=False)
    model.load_state_dict(torch.load(args.weights, map_location='cpu'))
    model.to(device)
    dataset = SARImages(args.images_json, PseudoRGB(config['preprocess']['clip_percentile']), args.image_dir,
                        cache_dir=args.cache)
    loader = DataLoader(dataset, args.batch_size, num_workers=args.workers, collate_fn=collate_images)
    out = args.out or Path(f'submission_{args.images_json.resolve().parent.name}.zip')
    write_submission(*predict(model, loader, device), out=out)
    print(f'wrote {out} ({len(dataset)} images)')


if __name__ == '__main__':
    main()
