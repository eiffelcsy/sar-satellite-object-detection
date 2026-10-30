"""Reuse a pre-processing (SAR-BM3D) cache after the dataset moved to another path, e.g. another cluster account.

Cache files are named `{stem}_{md5(resolved image path)[:10]}__{settings}.npy` (`sarbench/data.py: _cache_name`), so
a cache copied next to a dataset at a different path is never hit and every image is despeckled again (~0.2 images/s
for the first epoch). This copies (or renames, `--in-place`) each file to the name the new path gives it; the
settings part of the name is kept, so only identically pre-processed files are reused.

    python rekey_preprocess_cache.py --cache /path/to/copied/preprocess_cache \\
        --old-root /common/scratch/users/e/eiffelchong.2023/cs701-sar-course-data \\
        --new-root /common/scratch/projectgrps/CS425/CS425G6/cs701-sar-course-data \\
        --out preprocess_cache

`--old-root` is the dataset root as the original account resolved it (the directory holding train/ and val/); it
does not need to be readable here. Images are matched by their path relative to the root (train/images/00001.jpg).
"""
import argparse
import hashlib
import shutil
from pathlib import Path

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def digest(path):
    return hashlib.md5(str(path).encode()).hexdigest()[:10]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cache', type=Path, required=True, help='the copied cache directory')
    p.add_argument('--old-root', type=Path, required=True, help='dataset root the cache was built from')
    p.add_argument('--new-root', type=Path, required=True, help='dataset root on this account')
    p.add_argument('--out', type=Path, default=None, help='where to write the re-keyed cache (default: --cache)')
    p.add_argument('--in-place', action='store_true', help='rename instead of copying (needs --out unset)')
    args = p.parse_args()
    out = args.out or args.cache
    out.mkdir(parents=True, exist_ok=True)

    new_root = args.new_root.resolve()
    rename = {}  # old cache prefix (stem_digest) -> new prefix
    for image in new_root.rglob('*'):
        if image.suffix.lower() in IMAGE_SUFFIXES:
            relative = image.relative_to(new_root)
            rename[f'{image.stem}_{digest(args.old_root / relative)}'] = f'{image.stem}_{digest(image.resolve())}'

    moved = skipped = 0
    for cached in args.cache.glob('*.npy'):
        prefix, sep, settings = cached.name.partition('__')
        if not sep or prefix not in rename:
            skipped += 1
            continue
        target = out / f'{rename[prefix]}__{settings}'
        if target.exists():
            moved += 1
            continue
        if args.in_place and out == args.cache:
            cached.rename(target)
        else:
            shutil.copy2(cached, target)
        moved += 1
    print(f'{moved} cache files re-keyed into {out}; {skipped} did not match an image under {new_root}')


if __name__ == '__main__':
    main()
