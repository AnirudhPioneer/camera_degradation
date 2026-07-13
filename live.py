"""
live.py — Real-time camera degradation detection on a live video feed.

Usage:
    py live.py              # webcam 0
    py live.py 1            # webcam 1
    py live.py video.mp4    # video file path

Controls:
    r   : reset baseline (restarts warmup)
    q   : quit
"""

import sys
import cv2
import numpy as np

try:
    from detector import (
        DegradationState, degradation_update,
        CLASS_NAMES, CLASS_COLORS,
        DEGR_WARMUP_FRAMES, DEGR_GRID_COLS, DEGR_GRID_ROWS, DEGR_GRID_CELLS,
        THR_LAP_BLUR, THR_CONTRAST_LOW, THR_SPREAD_FOG, THR_SPREAD_FROST,
        THR_MEAN_BRIGHT, THR_CELLVAR_UNIFORM, THR_CELLVAR_LOCAL,
        THR_OBS_CELLS,
    )
except ImportError:
    print("ERROR: detector.py not found. Run this script from the project directory.")
    sys.exit(1)


# ── Config ────────────────────────────────────────────────────────────────────
DETECTION_STRIDE = 3     # run detector every Nth frame (3 → 10 Hz on 30 fps)
PANEL_W          = 290   # width of the right-side dashboard panel
NEUTRAL          = 128   # ratio value meaning "no change from baseline"

# OpenCV is BGR, not RGB.
# Each entry is (B, G, R).  Alpha overlays tinted in draw_grid_overlay().
CLASS_BGR = {
    "CLEAN":       ( 71, 178,  39),   # green
    "FOG":         (141, 128, 127),   # grey
    "FROST":       (255, 185, 116),   # light blue
    "OBSTRUCTION": ( 60,  76, 231),   # red  — covers dark blobs and bright blobs
    "UNKNOWN":     (182,  89, 155),   # purple
}


# ── Drawing helpers ───────────────────────────────────────────────────────────
def _text(img, txt, x, y, scale=0.50, color=(255, 255, 255), thick=1):
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thick, cv2.LINE_AA)


def draw_threshold_row(panel, y, label, value, threshold, bad_if_below=True, scale=255):
    """
    Draw one feature row: label | value bar | threshold line | pass/fail dot.

    bad_if_below=True  → value below threshold is bad (red)   e.g. sharpness, contrast
    bad_if_below=False → value above threshold is bad (red)   e.g. occlusion count
    scale              → the maximum value the bar represents (255 or 12 for cell counts)
    """
    BAR_X, BAR_W, BAR_H = 116, 130, 9
    bar_y = y + 2

    value     = max(0, min(scale, value))
    threshold = max(0, min(scale, threshold))

    # Determine pass/fail
    if bad_if_below:
        failing = value < threshold
    else:
        failing = value >= threshold

    bar_color  = (50, 60, 210) if failing else (50, 180, 60)   # red-ish or green
    dot_color  = (40,  40, 220) if failing else (40, 200, 40)

    # Track
    cv2.rectangle(panel, (BAR_X, bar_y), (BAR_X + BAR_W, bar_y + BAR_H), (45, 45, 45), -1)

    # Filled bar proportional to value
    filled = int(BAR_W * value / scale)
    if filled > 0:
        cv2.rectangle(panel, (BAR_X, bar_y), (BAR_X + filled, bar_y + BAR_H), bar_color, -1)

    # Threshold line (white)
    tx = BAR_X + int(BAR_W * threshold / scale)
    cv2.line(panel, (tx, bar_y - 2), (tx, bar_y + BAR_H + 2), (230, 230, 230), 1)

    # Pass/fail dot
    cv2.circle(panel, (BAR_X + BAR_W + 10, bar_y + BAR_H // 2), 4, dot_color, -1)

    # Labels
    _text(panel, label,       4,               y + 10, 0.37, (200, 200, 200))
    _text(panel, str(value),  BAR_X - 26,      y + 10, 0.37, (220, 220, 220))
    _text(panel, f">{threshold}" if not bad_if_below else f"<{threshold}",
                              BAR_X + BAR_W + 18, y + 10, 0.34, (160, 160, 160))


# ── Grid cell overlay ────────────────────────────────────────────────────────
def draw_grid_overlay(frame, features, frame_h, frame_w):
    """
    Draw the 4×3 detection grid on *frame* (in-place).

    Cell shading:
      Dark red  → occlusion (dark flat blob — DIRT signature)
      Dark blue → obstruction (any-brightness flat blob)
      No fill   → normal cell

    The cell mean brightness is printed in each cell as a small number.
    """
    cell_occ  = features.get("_cell_occ",   [False] * DEGR_GRID_CELLS)
    cell_obs  = features.get("_cell_obs",   [False] * DEGR_GRID_CELLS)
    cell_mean = features.get("_cell_means", [0]     * DEGR_GRID_CELLS)

    overlay = frame.copy()

    for r in range(DEGR_GRID_ROWS):
        y0 = r       * frame_h // DEGR_GRID_ROWS
        y1 = (r + 1) * frame_h // DEGR_GRID_ROWS
        for c in range(DEGR_GRID_COLS):
            x0 = c       * frame_w // DEGR_GRID_COLS
            x1 = (c + 1) * frame_w // DEGR_GRID_COLS
            ci = r * DEGR_GRID_COLS + c

            # Fill flagged cells with a translucent colour
            if cell_occ[ci]:
                # Occlusion (dark blob) — warm red tint
                cv2.rectangle(overlay, (x0, y0), (x1, y1), (40, 40, 200), -1)
            elif cell_obs[ci]:
                # Obstruction (any brightness) — blue tint
                cv2.rectangle(overlay, (x0, y0), (x1, y1), (200, 80, 40), -1)

            # Print the cell mean brightness in the cell centre
            label = str(cell_mean[ci])
            cx = (x0 + x1) // 2 - 8
            cy = (y0 + y1) // 2 + 4
            _text(frame, label, cx, cy, 0.36, (200, 200, 200), 1)

    # Blend the filled overlay at 30 % opacity so the video still shows through
    cv2.addWeighted(overlay, 0.30, frame, 0.70, 0, frame)

    # Draw grid lines on top of the blend
    for r in range(DEGR_GRID_ROWS + 1):
        y = r * frame_h // DEGR_GRID_ROWS
        cv2.line(frame, (0, y), (frame_w, y), (90, 90, 90), 1)
    for c in range(DEGR_GRID_COLS + 1):
        x = c * frame_w // DEGR_GRID_COLS
        cv2.line(frame, (x, 0), (x, frame_h), (90, 90, 90), 1)


# ── Dashboard panel ───────────────────────────────────────────────────────────
def draw_dashboard(panel, result, frame_no):
    """
    Render classification label + per-feature threshold rows onto the panel.

    Each row shows:
      raw value | bar (filled to value, white line at threshold) | pass/fail dot | threshold label
    Green dot  = feature is on the "clean" side of its threshold.
    Red dot    = feature is on the "bad" side — contributing to a detection.
    """
    panel[:] = 30

    cls  = result["class_name"]
    conf = result["confidence"]
    raw  = result.get("raw_class_name", cls)
    bgr  = CLASS_BGR.get(cls, (255, 255, 255))
    feat = result.get("features", {})
    wu   = result.get("warmup", False)

    y = 22

    # ── Classification header ────────────────────────────────────────────────
    _text(panel, cls,  8, y, 0.72, bgr, 2)
    y += 26
    conf_txt = f"raw:{raw}  conf:{conf}"
    if wu:
        done = feat.get("_frame_count", 0)
        pct  = min(1.0, done / DEGR_WARMUP_FRAMES)
        conf_txt = f"warmup {int(pct*100)}%  frame {done}/{DEGR_WARMUP_FRAMES}"
    _text(panel, conf_txt, 8, y, 0.36, (140, 140, 140))
    y += 16

    cv2.line(panel, (4, y), (PANEL_W - 4, y), (70, 70, 70), 1)
    y += 10

    # ── Feature rows with absolute thresholds ───────────────────────────────
    # Each tuple: (label, feature_key, threshold, bad_if_below, scale)
    #   bad_if_below=True  → low value = bad  (sharpness, contrast, spread)
    #   bad_if_below=False → high value = bad (brightness, cell counts, cellvar)
    rows = [
        # label          key                   thr                 bad_below  scale
        ("sharpness",   "laplacian_var",       THR_LAP_BLUR,       True,      255),
        ("contrast",    "global_contrast",     THR_CONTRAST_LOW,   True,      255),
        ("spread/fog",  "histogram_spread",    THR_SPREAD_FOG,     True,      224),
        ("spread/frost","histogram_spread",    THR_SPREAD_FROST,   True,      224),
        ("brightness",  "global_mean",         THR_MEAN_BRIGHT,    False,     255),
        ("cellvar/unif","cell_mean_variance",  THR_CELLVAR_UNIFORM,False,      80),
        ("cellvar/loc", "cell_mean_variance",  THR_CELLVAR_LOCAL,  False,      80),
        ("blocked cells","obstruction_score",  THR_OBS_CELLS,      False,      12),
    ]

    for label, key, thr, bad_below, scale in rows:
        val = feat.get(key, 0)
        draw_threshold_row(panel, y, label, val, thr, bad_below, scale)
        y += 18

    cv2.line(panel, (4, y), (PANEL_W - 4, y), (70, 70, 70), 1)
    y += 8

    # ── Hysteresis counters ──────────────────────────────────────────────────
    _text(panel, f"frame {frame_no}", 8, y, 0.37, (100, 100, 100))

    # ── Grid legend ──────────────────────────────────────────────────────────
    y += 20
    _text(panel, "grid:", 8, y, 0.37, (160, 160, 80))
    y += 14
    cv2.rectangle(panel, (8, y - 8), (18, y + 2), (40, 40, 200), -1)
    _text(panel, "flat + dark  (occlusion)",   22, y, 0.34, (180, 180, 180))
    y += 14
    cv2.rectangle(panel, (8, y - 8), (18, y + 2), (200, 80, 40), -1)
    _text(panel, "flat + bright (obstruction)", 22, y, 0.34, (180, 180, 180))
    y += 14
    _text(panel, "both types -> OBSTRUCTION",   8, y, 0.34, (140, 140, 140))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    source = sys.argv[1] if len(sys.argv) > 1 else 0
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: cannot open video source: {source}")
        sys.exit(1)

    ok, frame = cap.read()
    if not ok:
        print("ERROR: cannot read from video source.")
        sys.exit(1)

    h, w  = frame.shape[:2]
    panel = np.zeros((h, PANEL_W, 3), dtype=np.uint8)

    print(f"Source: {source}  |  Resolution: {w}×{h}")
    print(f"Detector runs every {DETECTION_STRIDE} frames  ({DEGR_WARMUP_FRAMES}-frame warmup)")
    print("Controls: [r] reset baseline   [q] quit")

    state    = DegradationState()
    result   = {
        "class_name":     "CLEAN",
        "raw_class_name": "CLEAN",
        "confidence":     0,
        "features":       {"_cell_occ": [False]*DEGR_GRID_CELLS,
                           "_cell_obs": [False]*DEGR_GRID_CELLS,
                           "_cell_means": [0]*DEGR_GRID_CELLS},
        "ratios":         {},
        "warmup":         True,
    }
    frame_no = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            if isinstance(source, str):          # loop video files
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                state  = DegradationState()
                ok, frame = cap.read()
                if not ok:
                    break
            else:
                break

        # ── Run detector every DETECTION_STRIDE frames ────────────────────
        if frame_no % DETECTION_STRIDE == 0:
            gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            result = degradation_update(state, gray)
            result["features"]["_frame_count"] = state.frame_count

        # ── Grid overlay ──────────────────────────────────────────────────
        # Always drawn — shows cell means + any flagged cells
        draw_grid_overlay(frame, result["features"], h, w)

        # ── Coloured border when a degradation is active ──────────────────
        cls = result["class_name"]
        if not result.get("warmup", True):
            bgr = CLASS_BGR.get(cls, (255, 255, 255))
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), bgr, 6)

        # ── Dashboard panel ───────────────────────────────────────────────
        draw_dashboard(panel, result, frame_no)

        # ── Show ──────────────────────────────────────────────────────────
        canvas = np.hstack([frame, panel])
        cv2.imshow("Camera Degradation Detector  [r=reset  q=quit]", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            state    = DegradationState()
            frame_no = 0
            print(f"[frame {frame_no}] Baseline reset — warmup restarted.")

        frame_no += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
