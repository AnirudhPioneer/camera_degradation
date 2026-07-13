"""
equiv_check.py — prove the C core matches detector.py frame-for-frame.

Generates a deterministic synthetic frame stream (clean / fog / frost /
obstruction / dark), writes it as raw uint8, runs BOTH:
  * the compiled C harness (test_harness.exe) at stride 1
  * detector.py in Python
and compares every feature + classification column on every frame.

Only needs numpy (no cv2).  Run:  py equiv_check.py
"""
import csv
import subprocess
import sys
import numpy as np

import detector as D

W, H = 320, 240
RAW   = "equiv_frames.raw"
C_CSV = "equiv_c.csv"

xv, yv = np.meshgrid(np.arange(W), np.arange(H))


def clean_frame():
    """Sharp, wide-histogram, textured scene → high laplacian + spread."""
    base    = (xv * 200 // W)                 # horizontal gradient 0..200
    vert    = (yv * 40 // H)                   # vertical gradient
    texture = (((xv ^ yv) & 1) * 50)           # fine checker → high frequency
    return np.clip(base + vert + texture, 0, 255).astype(np.uint8)


def fog_frame():
    """Low contrast, MILDLY blurry (not zero), mid-brightness, uniform → FOG.
    A tiny checker keeps laplacian in the 5..15 'moderate blur' band so this
    reads as fog, not as the lap<5 'on-lens coating' that means frost."""
    smooth = 118 + (xv * 12 // W) + (yv * 8 // H)
    texture = (((xv ^ yv) & 1) * 3)                 # amp 3 → lap ≈ 9
    return np.clip(smooth + texture, 0, 255).astype(np.uint8)


def frost_frame():
    """Very blurry, bright, uniform → FROST."""
    smooth = 205 + (xv * 6 // W) + (yv * 4 // H)
    return np.clip(smooth, 0, 255).astype(np.uint8)


def obstruction_frame():
    """Dark sharp scene with a flat BRIGHT slab over the top-left cells.
    Big brightness gap between slab and scene → high cell_mean_variance;
    the flat slab cells → obstruction_score ≥ 3."""
    base    = (xv * 60 // W)                    # dark gradient, mean ~30
    texture = (((xv ^ yv) & 1) * 40)            # sharp → high laplacian
    f = np.clip(base + texture, 0, 255).astype(np.int32)
    f[0:H*2//3, 0:W//2] = 240                   # flat bright slab over ~4 cells
    return np.clip(f, 0, 255).astype(np.uint8)


def dark_frame():
    """Near-black scene → must stay CLEAN (mean below fog floor)."""
    return np.full((H, W), 5, dtype=np.uint8)


def build_stream():
    frames = []
    frames += [clean_frame()       for _ in range(70)]   # warmup + clean
    frames += [fog_frame()         for _ in range(20)]
    frames += [frost_frame()       for _ in range(20)]
    frames += [obstruction_frame() for _ in range(20)]
    frames += [dark_frame()        for _ in range(15)]
    frames += [clean_frame()       for _ in range(20)]   # recover
    return frames


def run_python(frames):
    state = D.DegradationState()
    rows  = []
    for fr in frames:
        in_warmup = 0 if state.warmup_done else 1   # capture BEFORE update (matches C harness)
        r = D.degradation_update(state, fr)
        f = r["features"]
        rows.append({
            "warmup":            in_warmup,
            "classification":    r["class_name"],
            "confidence":        r["confidence"],
            "laplacian_var":     f["laplacian_var"],
            "global_contrast":   f["global_contrast"],
            "histogram_spread":  f["histogram_spread"],
            "global_mean":       f["global_mean"],
            "occlusion_score":   f["occlusion_score"],
            "obstruction_score": f["obstruction_score"],
            "cell_mean_variance":f["cell_mean_variance"],
            "temporal_mad":      f["temporal_mad"],
            "temporal_mad_var":  f["temporal_mad_var"],
        })
    return rows


def run_c(frames):
    with open(RAW, "wb") as fh:
        for fr in frames:
            fh.write(fr.tobytes())
    proc = subprocess.run(["./test_harness.exe", RAW, str(W), str(H), C_CSV, "1"],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        print("C harness failed:\n", proc.stderr)
        sys.exit(1)
    rows = []
    with open(C_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    return rows


def main():
    frames = build_stream()
    py = run_python(frames)
    c  = run_c(frames)

    if len(py) != len(c):
        print(f"FRAME COUNT MISMATCH: python={len(py)} c={len(c)}")
        sys.exit(1)

    cols = ["warmup", "classification", "confidence", "laplacian_var",
            "global_contrast", "histogram_spread", "global_mean",
            "occlusion_score", "obstruction_score", "cell_mean_variance",
            "temporal_mad", "temporal_mad_var"]

    mismatches = 0
    for i, (p, q) in enumerate(zip(py, c)):
        for col in cols:
            pv = str(p[col])
            qv = str(q[col])
            if pv != qv:
                mismatches += 1
                if mismatches <= 30:
                    print(f"frame {i:3d}  {col:18s}  python={pv:12s}  c={qv}")

    # Classification timeline (sanity — did each class actually fire?)
    seen = {}
    for q in c:
        seen[q["classification"]] = seen.get(q["classification"], 0) + 1

    print("-" * 56)
    print("class distribution (C harness):", dict(seen))
    if mismatches == 0:
        print(f"PASS — all {len(py)} frames × {len(cols)} columns identical.")
        sys.exit(0)
    else:
        print(f"FAIL — {mismatches} mismatched cells.")
        sys.exit(1)


if __name__ == "__main__":
    main()
