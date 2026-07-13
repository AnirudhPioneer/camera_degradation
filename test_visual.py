"""
test_visual.py — Visual validation of the Python degradation detector.

For each camera angle:
  1. Run 80-frame warmup once against the clean reference.
  2. Clone the warmed-up state for each test image (one test = one extra frame).
  3. Render a colour-coded image grid: green border = CLEAN, others = detected class.

Saves:
  results_soiled.png  — soiled-image detection grid
  results_fp.png      — false-positive check (clean images)
"""

import os
import copy
import numpy as np
import matplotlib
matplotlib.use("Agg")          # write PNG without a display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from detector import (
    DegradationState, degradation_update,
    CLASS_NAMES, CLASS_COLORS, CLEAN,
    DEGR_WARMUP_FRAMES,
)

# ── Config ─────────────────────────────────────────────────────────────────────
DATASET_ROOT  = r"d:\soiling_dataset-001"
CAMERA_ANGLES = ("FV", "MVL", "MVR", "RV")
WARMUP_COPIES = 80
TESTS_PER_CAT = 20
THUMB_W, THUMB_H = 200, 150    # thumbnail display size (pixels)
OUT_SOILED = r"d:\Work\camera_degradation\results_soiled.png"
OUT_FP     = r"d:\Work\camera_degradation\results_fp.png"


# ── Dataset helpers ────────────────────────────────────────────────────────────
def camera_angle(filename):
    stem = os.path.splitext(filename)[0]
    parts = stem.split("_")
    return parts[-1] if len(parts) > 1 else "UNK"


def soiling_fractions(label_path):
    data = np.array(Image.open(label_path).convert("L"))
    n = data.size
    return (data == 0).sum() / n, (data == 1).sum() / n, (data == 2).sum() / n


def scan_by_angle(split):
    rgb_dir = os.path.join(DATASET_ROOT, split, "rgbImages")
    lbl_dir = os.path.join(DATASET_ROOT, split, "gtLabels")
    clean  = {a: [] for a in CAMERA_ANGLES}
    opaque = {a: [] for a in CAMERA_ANGLES}
    for fn in sorted(os.listdir(lbl_dir)):
        if not fn.endswith(".png"):
            continue
        ang = camera_angle(fn)
        if ang not in CAMERA_ANGLES:
            continue
        f_clean, f_op, f_transp = soiling_fractions(os.path.join(lbl_dir, fn))
        rgb = os.path.join(rgb_dir, fn)
        # Require >98% clean AND no significant transparent soiling
        if f_clean > 0.98 and f_transp < 0.01:
            clean[ang].append(rgb)
        elif f_op > 0.40:
            opaque[ang].append(rgb)
    return clean, opaque


# ── Image loading ──────────────────────────────────────────────────────────────
def load_gray(path):
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)


def load_rgb_thumb(path):
    return np.array(
        Image.open(path).convert("RGB").resize((THUMB_W, THUMB_H), Image.BILINEAR)
    )


# ── Detection ─────────────────────────────────────────────────────────────────
def warmup_state(ref_path):
    """Run WARMUP_COPIES frames of the clean reference and return the state."""
    state = DegradationState()
    frame = load_gray(ref_path)
    for _ in range(WARMUP_COPIES):
        degradation_update(state, frame)
    return state


def test_one(warmed_state, test_path):
    """Clone the warmed state and run one test frame. Returns result dict."""
    s = warmed_state.clone()
    return degradation_update(s, load_gray(test_path))


def run_angle(ref_path, test_paths):
    """Warm up once, test each path. Returns list of (path, result)."""
    print(f"    warming up on {os.path.basename(ref_path)} ...", flush=True)
    ws = warmup_state(ref_path)
    results = []
    for i, p in enumerate(test_paths):
        res = test_one(ws, p)
        results.append((p, res))
        cls = res["class_name"]
        print(f"    [{i+1:2d}/{len(test_paths)}] {os.path.basename(p):<20s} -> "
              f"{cls}  (conf={res['confidence']})", flush=True)
    return results


# ── Visualisation ──────────────────────────────────────────────────────────────
BORDER_W = 8   # border line width in points

def _cell(ax, img_rgb, title, color):
    """Render one image cell: thumbnail + coloured border + title."""
    ax.imshow(img_rgb, aspect="auto")
    rect = mpatches.FancyBboxPatch(
        (0, 0), 1, 1,
        transform=ax.transAxes,
        boxstyle="square,pad=0",
        linewidth=BORDER_W,
        edgecolor=color,
        facecolor="none",
        clip_on=False,
    )
    ax.add_patch(rect)
    ax.set_title(title, fontsize=6.5, color=color, fontweight="bold",
                 pad=3, wrap=False)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def make_grid(title, angle_data, out_path):
    """
    angle_data: dict  angle -> {"ref": path, "tests": [(path, result), ...]}
    Renders one PNG with CAMERA_ANGLES rows and (1 + TESTS_PER_CAT) columns.
    """
    n_rows = len(CAMERA_ANGLES)
    n_cols = 1 + TESTS_PER_CAT

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * (THUMB_W / 72), n_rows * (THUMB_H / 72 + 0.55)),
        dpi=100,
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    for row, angle in enumerate(CAMERA_ANGLES):
        data  = angle_data.get(angle)
        tests = data["tests"] if data else []
        ref   = data["ref"]   if data else None

        detected = sum(1 for _, r in tests if r["class_name"] != "CLEAN")
        rate     = detected / len(tests) * 100 if tests else 0

        # Reference cell
        ax0 = axes[row, 0]
        if ref:
            _cell(ax0,
                  load_rgb_thumb(ref),
                  f"REF · {angle}\n{detected}/{len(tests)} det  {rate:.0f}%",
                  "#3498db")
        else:
            ax0.axis("off")

        # Test cells
        for col in range(1, n_cols):
            ax = axes[row, col]
            if col - 1 < len(tests):
                path, res = tests[col - 1]
                cname = res["class_name"]
                _cell(ax,
                      load_rgb_thumb(path),
                      f"{cname}\n{os.path.basename(path)[:14]}",
                      CLASS_COLORS[cname])
            else:
                ax.axis("off")

    # Legend strip
    legend_patches = [
        mpatches.Patch(color=CLASS_COLORS[n], label=n)
        for n in CLASS_NAMES
    ]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=len(CLASS_NAMES), fontsize=8,
               bbox_to_anchor=(0.5, -0.03), frameon=False)

    fig.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig(out_path, bbox_inches="tight", dpi=100)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Scanning dataset labels by camera angle ...")
    train_clean, train_opaque = scan_by_angle("train")
    test_clean,  test_opaque  = scan_by_angle("test")

    for ang in CAMERA_ANGLES:
        print(f"  {ang}: train clean={len(train_clean[ang])}  "
              f"train opaque={len(train_opaque[ang])}  "
              f"test clean={len(test_clean[ang])}  "
              f"test opaque={len(test_opaque[ang])}")
    print()

    soiled_data = {}
    fp_data     = {}

    for angle in CAMERA_ANGLES:
        clean_refs  = train_clean[angle] or test_clean[angle]
        soiled_imgs = (test_opaque[angle] or train_opaque[angle])[:TESTS_PER_CAT]

        if not clean_refs or not soiled_imgs:
            print(f"[{angle}] Skipping — not enough images.\n")
            continue

        ref = clean_refs[0]
        fc, fo, ft = soiling_fractions(
            os.path.join(DATASET_ROOT, "train", "gtLabels",
                         os.path.basename(ref))
            if os.path.exists(os.path.join(DATASET_ROOT, "train", "gtLabels", os.path.basename(ref)))
            else os.path.join(DATASET_ROOT, "test",  "gtLabels", os.path.basename(ref))
        )
        print(f"[{angle}] Soiled detection — {len(soiled_imgs)} images vs ref {os.path.basename(ref)}"
              f"  (clean={fc*100:.1f}% opaque={fo*100:.1f}% transp={ft*100:.1f}%)")
        tests = run_angle(ref, soiled_imgs)
        detected = sum(1 for _, r in tests if r["class_name"] != "CLEAN")
        print(f"  Detection rate: {detected}/{len(soiled_imgs)} = "
              f"{detected/len(soiled_imgs)*100:.0f}%\n")
        soiled_data[angle] = {"ref": ref, "tests": tests}

        # False-positive check with a different clean ref image
        clean_pool = test_clean[angle] or train_clean[angle]
        if len(clean_pool) >= 2:
            fp_ref   = clean_pool[0]
            fp_imgs  = clean_pool[1:TESTS_PER_CAT + 1]
            print(f"[{angle}] FP check — {len(fp_imgs)} clean images vs ref {os.path.basename(fp_ref)}")
            fp_tests = run_angle(fp_ref, fp_imgs)
            fp = sum(1 for _, r in fp_tests if r["class_name"] != "CLEAN")
            print(f"  FP rate: {fp}/{len(fp_imgs)} = {fp/len(fp_imgs)*100:.0f}%\n")
            fp_data[angle] = {"ref": fp_ref, "tests": fp_tests}

    # ── Summary table ─────────────────────────────────────────────────────────
    print("=" * 55)
    print("DETECTION SUMMARY  (opaque soiling)")
    print("=" * 55)
    print(f"{'Angle':<8} {'Tested':>7} {'Detected':>9} {'Rate':>7}")
    print("-" * 55)
    for angle in CAMERA_ANGLES:
        d = soiled_data.get(angle)
        if d:
            n = len(d["tests"])
            det = sum(1 for _, r in d["tests"] if r["class_name"] != "CLEAN")
            print(f"{angle:<8} {n:>7} {det:>9} {det/n*100:>6.0f}%")

    print()
    print("FALSE POSITIVE SUMMARY  (clean images)")
    print("=" * 55)
    print(f"{'Angle':<8} {'Tested':>7} {'FP':>9} {'FP Rate':>9}")
    print("-" * 55)
    for angle in CAMERA_ANGLES:
        d = fp_data.get(angle)
        if d:
            n  = len(d["tests"])
            fp = sum(1 for _, r in d["tests"] if r["class_name"] != "CLEAN")
            print(f"{angle:<8} {n:>7} {fp:>9} {fp/n*100:>8.0f}%")

    print()
    print("Rendering grids ...")
    make_grid("Soiled image detection",               soiled_data, OUT_SOILED)
    make_grid("False-positive check (clean images)",  fp_data,     OUT_FP)

    # Try to open in default viewer
    try:
        os.startfile(OUT_SOILED)
        os.startfile(OUT_FP)
    except Exception:
        pass


if __name__ == "__main__":
    main()
