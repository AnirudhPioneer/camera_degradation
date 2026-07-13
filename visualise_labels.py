"""
visualise_labels.py — Scale gtLabel images so they are visible in any viewer.

Pixel values 0/1/2 are multiplied by 127:
  0 (clean)             -> 0   (black)
  1 (opaque soiling)    -> 127 (grey)
  2 (transparent soil)  -> 254 (white)

Output mirrors the dataset folder structure under a new root.
"""

import os
import numpy as np
from PIL import Image

DATASET_ROOT = r"d:\soiling_dataset-001"
OUT_ROOT     = r"d:\soiling_dataset-001_labels_visible"
SPLITS       = ("train", "test")


def process_split(split):
    in_dir  = os.path.join(DATASET_ROOT, split, "gtLabels")
    out_dir = os.path.join(OUT_ROOT,     split, "gtLabels")
    os.makedirs(out_dir, exist_ok=True)

    files = [f for f in sorted(os.listdir(in_dir)) if f.endswith(".png")]
    print(f"[{split}] {len(files)} label images -> {out_dir}")

    for i, fn in enumerate(files):
        src = os.path.join(in_dir, fn)
        dst = os.path.join(out_dir, fn)

        data = np.array(Image.open(src).convert("L"), dtype=np.uint8)
        scaled = (data * 127).clip(0, 255).astype(np.uint8)
        Image.fromarray(scaled, mode="L").save(dst)

        if (i + 1) % 100 == 0 or (i + 1) == len(files):
            print(f"  {i+1}/{len(files)}", flush=True)


for split in SPLITS:
    process_split(split)

print("Done.")
