"""
run_test.py  —  Validate the degradation detector against the soiling dataset.

Correct test methodology
------------------------
This detector learns a SPECIFIC camera's clean-scene baseline, then flags
deviations.  Feeding it completely different scenes as "clean reference" vs
"test image" would always look like a change — the detector would be right
but the test would be meaningless.

Instead, for each test image we:
  1. Use 80 copies of a MATCHED clean image (same camera angle suffix:
     FV, MVL, MVR, RV) as the warmup baseline.
  2. Immediately follow with the soiled test image.
  3. Record whether the detector flagged it.

This simulates the real use case: a camera that was clean and then became
soiled while looking at roughly similar outdoor scenes.

Label encoding (gtLabels)
--------------------------
  0 = clean pixels
  1 = opaque soiling  (dirt / mud)        — we want to DETECT this
  2 = transparent soiling (water smear)   — OUT OF SCOPE; expect no output
"""

import os, sys, subprocess, csv
from PIL import Image

# ── Configuration ────────────────────────────────────────────────────────────

DATASET_ROOT  = r"d:\soiling_dataset-001"
HARNESS_EXE   = r"d:\Work\camera_degradation\test_harness.exe"
WORK_DIR      = r"d:\Work\camera_degradation"
TARGET_W      = 1280
TARGET_H      = 960
WARMUP_COPIES = 80    # repeat clean reference (>= 60 needed; 80 gives extra EMA settling)
FRAME_STRIDE  = 1     # all frames are pre-selected; process every one
TESTS_PER_CAT = 20    # soiled images to evaluate per camera angle

OUT_BIN = os.path.join(WORK_DIR, "test_frames.bin")
OUT_CSV = os.path.join(WORK_DIR, "test_output.csv")

CAMERA_ANGLES = ("FV", "MVL", "MVR", "RV")

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_gray(path):
    img = Image.open(path).convert("L")
    if img.size != (TARGET_W, TARGET_H):
        img = img.resize((TARGET_W, TARGET_H), Image.BILINEAR)
    return img.tobytes()


def soiling_fractions(label_path):
    """Return (frac_clean, frac_opaque, frac_transparent)."""
    data = Image.open(label_path).convert("L").tobytes()
    n = len(data)
    c0 = sum(1 for b in data if b == 0)
    c1 = sum(1 for b in data if b == 1)
    c2 = sum(1 for b in data if b == 2)
    return c0 / n, c1 / n, c2 / n


def camera_angle(filename):
    """Extract angle suffix from filename like 0253_FV.png -> 'FV'."""
    stem = os.path.splitext(filename)[0]   # "0253_FV"
    parts = stem.split("_")
    return parts[-1] if len(parts) > 1 else "UNK"


def scan_by_angle(split):
    """
    Returns two dicts keyed by camera angle:
      clean[angle]  = list of rgb image paths (>95% clean pixels)
      opaque[angle] = list of rgb image paths (>40% opaque soiling pixels)
    """
    rgb_dir = os.path.join(DATASET_ROOT, split, "rgbImages")
    lbl_dir = os.path.join(DATASET_ROOT, split, "gtLabels")

    clean  = {a: [] for a in CAMERA_ANGLES}
    opaque = {a: [] for a in CAMERA_ANGLES}

    for fn in sorted(os.listdir(lbl_dir)):
        if not fn.endswith(".png"):
            continue
        angle = camera_angle(fn)
        if angle not in CAMERA_ANGLES:
            continue
        f_clean, f_op, _ = soiling_fractions(os.path.join(lbl_dir, fn))
        rgb = os.path.join(rgb_dir, fn)
        if f_clean > 0.95:
            clean[angle].append(rgb)
        elif f_op > 0.40:
            opaque[angle].append(rgb)

    return clean, opaque


def run_pair_test(clean_ref_path, soiled_path):
    """
    Write a binary of [80 × clean ref] + [1 × soiled], run the harness,
    return the classification of the last (soiled) frame.
    """
    clean_frame  = load_gray(clean_ref_path)
    soiled_frame = load_gray(soiled_path)

    with open(OUT_BIN, "wb") as f:
        for _ in range(WARMUP_COPIES):
            f.write(clean_frame)
        f.write(soiled_frame)

    cmd = [HARNESS_EXE, OUT_BIN, str(TARGET_W), str(TARGET_H), OUT_CSV, str(FRAME_STRIDE)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return None, None

    # Read CSV — last row is the soiled frame (frame index == WARMUP_COPIES)
    last_cls, last_conf, last_features = "?", 0, {}
    with open(OUT_CSV, newline="") as cf:
        reader = csv.DictReader(cf)
        for row in reader:
            if int(row["frame"]) == WARMUP_COPIES and row["warmup"] == "0":
                last_cls   = row["classification"]
                last_conf  = int(row["confidence"])
                last_features = {k: row[k] for k in row if k not in
                                 ("frame","warmup","classification","confidence")}
    return last_cls, last_conf


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Scanning dataset labels by camera angle ...")
    train_clean, train_opaque = scan_by_angle("train")
    test_clean,  test_opaque  = scan_by_angle("test")

    for angle in CAMERA_ANGLES:
        print(f"  {angle}: train clean={len(train_clean[angle])}"
              f"  train opaque={len(train_opaque[angle])}"
              f"  test clean={len(test_clean[angle])}"
              f"  test opaque={len(test_opaque[angle])}")

    print()

    all_results = {}

    for angle in CAMERA_ANGLES:
        # Need at least one clean reference and some soiled test images
        clean_refs  = train_clean[angle] or test_clean[angle]
        soiled_imgs = (test_opaque[angle] or train_opaque[angle])[:TESTS_PER_CAT]

        if not clean_refs or not soiled_imgs:
            print(f"[{angle}] Skipping — not enough images.")
            continue

        clean_ref = clean_refs[0]
        n = len(soiled_imgs)
        print(f"[{angle}] Testing {n} soiled images vs ref {os.path.basename(clean_ref)}")

        detected = 0
        for i, soiled_path in enumerate(soiled_imgs):
            cls, conf = run_pair_test(clean_ref, soiled_path)
            triggered = cls != "CLEAN" and cls is not None
            if triggered:
                detected += 1
            status = f"{cls}(conf={conf})" if cls else "ERR"
            print(f"  [{i+1:2d}/{n}] {os.path.basename(soiled_path):20s} -> {status}")

        rate = detected / n if n else 0
        all_results[angle] = {"total": n, "detected": detected, "rate": rate}
        print(f"  Detection rate: {detected}/{n} = {rate*100:.0f}%\n")

    # ── Clean false-positive check ─────────────────────────────────────────
    print("False-positive check (clean images should always return CLEAN):")
    fp_results = {}
    for angle in CAMERA_ANGLES:
        clean_imgs = (test_clean[angle] or train_clean[angle])[:TESTS_PER_CAT]
        if not clean_imgs or len(clean_imgs) < 2:
            continue
        clean_ref  = clean_imgs[0]
        test_imgs  = clean_imgs[1:TESTS_PER_CAT+1]   # skip ref, test the rest
        n = len(test_imgs)
        fp = 0
        for soiled_path in test_imgs:
            cls, _ = run_pair_test(clean_ref, soiled_path)
            if cls != "CLEAN":
                fp += 1
        fp_results[angle] = {"total": n, "fp": fp, "fp_rate": fp/n if n else 0}
        print(f"  {angle}: {fp}/{n} false positives ({fp/n*100:.0f}%)" if n else f"  {angle}: no data")

    # ── Summary ───────────────────────────────────────────────────────────
    print()
    print("=" * 55)
    print("DETECTION SUMMARY (opaque soiling)")
    print("=" * 55)
    print(f"{'Angle':<8} {'Tested':>7} {'Detected':>9} {'Rate':>7}")
    print("-" * 55)
    for angle, r in all_results.items():
        print(f"{angle:<8} {r['total']:>7} {r['detected']:>9} {r['rate']*100:>6.0f}%")

    print()
    print("FALSE POSITIVE SUMMARY (clean images)")
    print("=" * 55)
    print(f"{'Angle':<8} {'Tested':>7} {'FP':>9} {'FP Rate':>9}")
    print("-" * 55)
    for angle, r in fp_results.items():
        print(f"{angle:<8} {r['total']:>7} {r['fp']:>9} {r['fp_rate']*100:>8.0f}%")

    if os.path.exists(OUT_BIN):
        os.remove(OUT_BIN)


if __name__ == "__main__":
    main()
