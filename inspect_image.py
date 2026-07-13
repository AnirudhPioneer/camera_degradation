"""
inspect_image.py — Classify a single image and show its features.

Usage:
    py inspect_image.py <test_image> [reference_image]

    test_image       : image to classify (any PIL-readable format)
    reference_image  : clean reference for warmup (optional)
                       If omitted, raw features are shown but no classification.

Examples:
    py inspect_image.py 0000_FV.png 0253_FV.png
    py inspect_image.py "d:/soiling_dataset-001/test/rgbImages/0000_FV.png" ^
                        "d:/soiling_dataset-001/train/rgbImages/0253_FV.png"
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from detector import (DegradationState, degradation_update,
                      CLASS_NAMES, CLASS_COLORS,
                      DEGR_WARMUP_FRAMES,
                      DEGR_THR_LAP_LOW, DEGR_THR_CONTRAST_LOW,
                      DEGR_THR_SPREAD_VLOW,
                      DEGR_THR_MEAN_HIGH, DEGR_THR_MEAN_NEAR_LO, DEGR_THR_MEAN_NEAR_HI,
                      DEGR_THR_OCC_HIGH, DEGR_THR_OCC_LOW,
                      DEGR_THR_OBS_HIGH,
                      DEGR_THR_CELLVAR_HIGH, DEGR_THR_CELLVAR_LOW)

WARMUP_COPIES = 80
TEST_COPIES   = 8     # feed test image this many times; hysteresis needs 5 to enter
NEUTRAL       = 128   # ratio value meaning "no change from baseline"


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_gray(path):
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)

def load_rgb(path, max_w=640):
    img = Image.open(path).convert("RGB")
    if img.width > max_w:
        scale = max_w / img.width
        img = img.resize((max_w, int(img.height * scale)), Image.BILINEAR)
    return np.array(img)

def run_detection(test_path, ref_path):
    """Warmup on ref × WARMUP_COPIES, then feed test × TEST_COPIES.
    Feeding the test image multiple times lets hysteresis (needs 5 frames) fire.
    Returns the result from the final test frame."""
    state = DegradationState()
    ref   = load_gray(ref_path)
    for _ in range(WARMUP_COPIES):
        degradation_update(state, ref)
    test  = load_gray(test_path)
    result = None
    for _ in range(TEST_COPIES):
        result = degradation_update(state, test)
    return result

def features_only(test_path):
    """Single pass on test image with no baseline — returns features, no classification."""
    state = DegradationState()
    # Feed same image 61 times so warmup completes and we get a result
    frame = load_gray(test_path)
    result = None
    for _ in range(DEGR_WARMUP_FRAMES + 1):
        result = degradation_update(state, frame)
    return result


# ── Visualisation ─────────────────────────────────────────────────────────────
def plot_ratio_bar(ax, label, value, low_thr=None, high_thr=None, low_bad=True):
    """
    Draw a single horizontal ratio bar.
    value    : 0–255 (128 = neutral)
    low_thr  : threshold below which something fires (low_bad=True) or above (False)
    high_thr : threshold above which something fires
    low_bad  : True if a DROP is the danger direction (lap, contrast, spread)
               False if a RISE is the danger direction (mean, occ, obs, cellvar)
    """
    # Background track
    ax.barh(0, 255, left=0, height=0.6, color="#2c3e50", zorder=1)

    # Value bar — colour encodes deviation from neutral
    if value < NEUTRAL:
        bar_color = "#e74c3c" if low_bad else "#27ae60"
    elif value > NEUTRAL:
        bar_color = "#e74c3c" if not low_bad else "#27ae60"
    else:
        bar_color = "#27ae60"

    ax.barh(0, value, left=0, height=0.6, color=bar_color, zorder=2)

    # Neutral line
    ax.axvline(NEUTRAL, color="#ecf0f1", linewidth=1.2, linestyle="--", zorder=3)

    # Threshold lines
    if low_thr is not None:
        ax.axvline(low_thr,  color="#f39c12", linewidth=1.5, zorder=4)
    if high_thr is not None:
        ax.axvline(high_thr, color="#f39c12", linewidth=1.5, zorder=4)

    # Value label
    ax.text(value + 4, 0, str(value), va="center", ha="left",
            fontsize=8, color="white", zorder=5)

    ax.set_xlim(0, 270)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xticks([0, 64, 128, 192, 255])
    ax.tick_params(labelsize=7, colors="#95a5a6")
    ax.set_ylabel(label, fontsize=8, color="#ecf0f1", rotation=0,
                  labelpad=90, va="center")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("#1a252f")


def render(test_path, ref_path, result, out_path):
    has_ref = ref_path is not None
    ratios  = result.get("ratios", {})
    feats   = result["features"]
    cls     = result["class_name"]
    conf    = result["confidence"]
    color   = CLASS_COLORS.get(cls, "#ffffff")

    fig = plt.figure(figsize=(14, 9), facecolor="#1a252f")
    fig.suptitle(f"Camera Degradation Inspector", fontsize=13,
                 color="#ecf0f1", fontweight="bold", y=0.98)

    # ── Layout: left = images, right = feature bars ───────────────────────
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.4], wspace=0.05,
                          left=0.02, right=0.98, top=0.93, bottom=0.04)

    # ── Left panel: images ────────────────────────────────────────────────
    img_rows = 2 if has_ref else 1
    gs_left  = gs[0].subgridspec(img_rows, 1, hspace=0.08)

    ax_test = fig.add_subplot(gs_left[0])
    ax_test.imshow(load_rgb(test_path))
    ax_test.set_title(f"TEST:  {os.path.basename(test_path)}",
                      fontsize=8, color="#ecf0f1", pad=3)
    raw_cls = result.get("raw_class_name", cls)
    badge   = f"{cls}  (raw: {raw_cls})  conf={conf}"
    ax_test.text(0.01, 0.97, badge,
                 transform=ax_test.transAxes,
                 fontsize=10, fontweight="bold", color=color,
                 va="top", bbox=dict(boxstyle="round,pad=0.3",
                                     facecolor="#1a252f", alpha=0.8,
                                     edgecolor=color, linewidth=2))
    rect = mpatches.FancyBboxPatch((0, 0), 1, 1,
                                    transform=ax_test.transAxes,
                                    boxstyle="square,pad=0",
                                    linewidth=6, edgecolor=color,
                                    facecolor="none", clip_on=False)
    ax_test.add_patch(rect)
    ax_test.axis("off")

    if has_ref:
        ax_ref = fig.add_subplot(gs_left[1])
        ax_ref.imshow(load_rgb(ref_path))
        ax_ref.set_title(f"REF:  {os.path.basename(ref_path)}",
                         fontsize=8, color="#3498db", pad=3)
        ax_ref.axis("off")

    # ── Right panel: ratio bars (or raw features if no ref) ───────────────
    if ratios:
        bars = [
            ("r_lap\n(sharpness)",    ratios["r_lap"],     DEGR_THR_LAP_LOW,      None,                True),
            ("r_ctr\n(contrast)",     ratios["r_ctr"],     DEGR_THR_CONTRAST_LOW, None,                True),
            ("r_spr\n(hist spread)",  ratios["r_spr"],     DEGR_THR_SPREAD_VLOW,  None,                True),
            ("r_mean\n(brightness)",  ratios["r_mean"],    None,                  DEGR_THR_MEAN_HIGH,  False),
            ("r_occ\n(occlusion)",    ratios["r_occ"],     None,                  DEGR_THR_OCC_HIGH,   False),
            ("r_obs\n(obstruction)",  ratios["r_obs"],     None,                  DEGR_THR_OBS_HIGH,   False),
            ("r_cellvar\n(uniformity)",ratios["r_cellvar"],DEGR_THR_CELLVAR_LOW,  DEGR_THR_CELLVAR_HIGH,False),
        ]
        title = f"Ratios  (128 = baseline, orange lines = thresholds)"
    else:
        bars = [
            ("lap_var\n(sharpness)",   feats["laplacian_var"],      None, None, True),
            ("contrast",               feats["global_contrast"],     None, None, True),
            ("hist spread",            feats["histogram_spread"],    None, None, True),
            ("mean brightness",        feats["global_mean"],         None, None, False),
            ("occlusion",              feats["occlusion_score"],     None, None, False),
            ("obstruction",            feats["obstruction_score"],   None, None, False),
            ("cell variance",          feats["cell_mean_variance"],  None, None, False),
        ]
        title = "Raw features  (no reference — comparison not possible)"

    gs_right = gs[1].subgridspec(len(bars) + 1, 1, hspace=0.5)
    ax_title = fig.add_subplot(gs_right[0])
    ax_title.text(0.5, 0.5, title, ha="center", va="center",
                  fontsize=9, color="#bdc3c7")
    ax_title.axis("off")

    for i, (lbl, val, lo, hi, low_bad) in enumerate(bars):
        ax = fig.add_subplot(gs_right[i + 1])
        plot_ratio_bar(ax, lbl, val, lo, hi, low_bad)

    # ── Decision trail ────────────────────────────────────────────────────
    if ratios:
        r = ratios
        checks = [
            ("OBSTRUCTION", r["r_obs"] > DEGR_THR_OBS_HIGH
                            and r["r_cellvar"] > DEGR_THR_CELLVAR_HIGH),
            ("FROST",       r["r_lap"] < DEGR_THR_LAP_LOW
                            and r["r_mean"] > DEGR_THR_MEAN_HIGH),
            ("FOG",         r["r_ctr"] < DEGR_THR_CONTRAST_LOW
                            and r["r_lap"] < DEGR_THR_LAP_LOW),
        ]
        trail = "  |  ".join(
            f"{'✓' if fired else '✗'} {name}"
            for name, fired in checks
        )
        fig.text(0.5, 0.01, trail, ha="center", va="bottom",
                 fontsize=8, color="#95a5a6")

    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {out_path}")
    try:
        os.startfile(out_path)
    except Exception:
        pass


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    test_path = sys.argv[1]
    ref_path  = sys.argv[2] if len(sys.argv) >= 3 else None

    if not os.path.exists(test_path):
        print(f"Error: test image not found: {test_path}")
        sys.exit(1)
    if ref_path and not os.path.exists(ref_path):
        print(f"Error: reference image not found: {ref_path}")
        sys.exit(1)

    print(f"Test image : {test_path}")
    if ref_path:
        print(f"Reference  : {ref_path} (warmup x{WARMUP_COPIES})")
        result = run_detection(test_path, ref_path)
        print(f"\nResult (hysteresis): {result['class_name']}  "
              f"(raw: {result['raw_class_name']})  conf={result['confidence']}")
        print(f"Ratios: { {k: v for k, v in result['ratios'].items()} }")
    else:
        print("No reference — showing raw features only (no classification).")
        result = features_only(test_path)
        print(f"\nFeatures: { {k: v for k, v in result['features'].items()} }")

    stem    = os.path.splitext(os.path.basename(test_path))[0]
    out_png = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           f"inspect_{stem}.png")
    render(test_path, ref_path, result, out_png)


if __name__ == "__main__":
    main()
