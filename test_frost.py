"""
test_frost.py — Isolated frost detection tester.

Shows the 4 classifier conditions plus three additional diagnostic metrics
(edge density, temporal stability, scatter/glare ratio) so you can see
exactly why a material passes or fails as a frost proxy.

Usage:
    py test_frost.py          # webcam 0
    py test_frost.py 1        # webcam index
    py test_frost.py file.mp4 # video file

Controls:
    q : quit
    r : reset state
"""

import sys
from collections import deque

import cv2
import numpy as np

from detector import (
    DegradationState, degradation_update,
    THR_LAP_FROST, THR_MEAN_BRIGHT, THR_SPREAD_FROST, THR_CELLVAR_UNIFORM,
    FROST, DEGR_STRIDE,
)

PANEL_W         = 320
STABILITY_WINDOW = 20    # frames used for temporal stability score
EDGE_LAP_THR    = 20     # |Laplacian| above this counts as an edge pixel

# Soft thresholds used only for the indicator display — not the classifier.
# Each represents the value at which the metric enters the frost-friendly zone.
SOFT_EDGE_DENSITY_THR    = 40    # frost wants BELOW this (few sharp edges)
SOFT_STABILITY_THR       = 160   # frost wants ABOVE this (static scene)
SOFT_SCATTER_THR         = 100   # frost wants ABOVE this (bright + flat)
SOFT_BRIGHT_FRAC_THR     = 80    # frost wants ABOVE this (lots of bright pixels)


# ── Drawing helpers ───────────────────────────────────────────────────────────
def _text(img, txt, x, y, scale=0.50, color=(255, 255, 255), thick=1):
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thick, cv2.LINE_AA)


def draw_metric_row(panel, y, label, value, threshold, need_below, scale=255, soft=False):
    """
    One metric row with value bar, threshold line, and pass/fail dot.

    soft=True  → grey dot instead of red when failing (diagnostic, not a classifier gate)
    need_below → value should be below threshold to be in the frost-friendly zone
    scale      → maximum the bar represents
    """
    BAR_X, BAR_W, BAR_H = 158, 110, 10
    bar_y = y + 1

    value     = max(0, min(scale, value))
    threshold = max(0, min(scale, threshold))
    passing   = (value < threshold) if need_below else (value > threshold)

    bar_color  = (50, 200, 50)  if passing else ((80, 80, 80) if soft else (50, 50, 220))
    dot_color  = (40, 200, 40)  if passing else ((90, 90, 90) if soft else (40, 40, 220))

    # Track
    cv2.rectangle(panel, (BAR_X, bar_y), (BAR_X + BAR_W, bar_y + BAR_H), (40, 40, 40), -1)

    # Value bar
    filled = int(BAR_W * value / max(scale, 1))
    if filled > 0:
        cv2.rectangle(panel, (BAR_X, bar_y), (BAR_X + filled, bar_y + BAR_H), bar_color, -1)

    # Threshold line
    tx = BAR_X + int(BAR_W * threshold / max(scale, 1))
    cv2.line(panel, (tx, bar_y - 2), (tx, bar_y + BAR_H + 2), (220, 220, 220), 1)

    # Pass/fail dot
    cv2.circle(panel, (BAR_X + BAR_W + 11, bar_y + BAR_H // 2), 4, dot_color, -1)

    # Status label
    status       = "OK" if passing else ("  " if soft else "--")
    status_color = (50, 220, 50) if passing else ((90, 90, 90) if soft else (80, 80, 220))
    _text(panel, status, BAR_X + BAR_W + 20, y + 10, 0.38, status_color, 1)

    direction = f"<{threshold}" if need_below else f">{threshold}"
    _text(panel, label,          4,           y + 10, 0.38, (190, 190, 190))
    _text(panel, f"{value:3d}",  BAR_X - 30,  y + 10, 0.38, (220, 220, 220))
    _text(panel, direction,      BAR_X - 96,  y + 10, 0.32, (120, 120, 120))


# ── Extra metric computation ──────────────────────────────────────────────────
def compute_extra_metrics(gray):
    """
    Compute edge density, scatter ratio, and bright pixel fraction from the
    raw grayscale frame using the same stride-4 sample grid as the detector.
    """
    h, w = gray.shape
    ys = np.arange(1, h - 1, DEGR_STRIDE)
    xs = np.arange(1, w - 1, DEGR_STRIDE)
    Y, X = np.meshgrid(ys, xs, indexing='ij')
    p = gray[Y, X].astype(np.int32)

    lap = (4 * p
           - gray[Y - 1, X].astype(np.int32)
           - gray[Y + 1, X].astype(np.int32)
           - gray[Y,     X - 1].astype(np.int32)
           - gray[Y,     X + 1].astype(np.int32))
    lap_abs = np.abs(lap)

    # Edge density: fraction of sampled pixels with a sharp Laplacian response.
    # Frost scatters light and destroys edges; bubble wrap preserves them.
    n = lap_abs.size
    edge_density = int(255 * int(np.sum(lap_abs > EDGE_LAP_THR)) // n)

    p_flat = p.ravel()
    pmax   = int(p_flat.max())
    pmin   = int(p_flat.min())
    pmean  = int(p_flat.sum()) // n

    # Scatter ratio: high mean relative to contrast signals a diffuse bright veil.
    # Formula: mean * 128 / (contrast + 1)  — same integer form as the C code.
    contrast     = (pmax - pmin) * 256 // (pmax + pmin + 1)
    scatter_ratio = min(255, pmean * 128 // (contrast + 1))

    # Bright pixel fraction: pixels in the top 2 histogram bins (≥ 192).
    # Frost washes the frame toward saturation; a clear or dark scene stays low.
    bright_fraction = int(255 * int(np.sum(p_flat >= 192)) // n)

    return {
        "edge_density":    edge_density,
        "scatter_ratio":   scatter_ratio,
        "bright_fraction": bright_fraction,
    }


# ── Temporal stability tracker ────────────────────────────────────────────────
class TemporalTracker:
    """
    Tracks key features over a rolling window and produces a stability score
    (0–255).  255 = perfectly stable across all features (frost is static);
    near 0 = high frame-to-frame variation (motion, changing scene).
    """
    def __init__(self, n=STABILITY_WINDOW):
        keys = ("laplacian_var", "global_mean", "histogram_spread", "cell_mean_variance")
        self._bufs = {k: deque(maxlen=n) for k in keys}

    def push(self, features):
        for k, buf in self._bufs.items():
            buf.append(features.get(k, 0))

    def stability(self):
        min_len = min(len(b) for b in self._bufs.values())
        if min_len < 3:
            return 0
        total_cv = 0.0
        for buf in self._bufs.values():
            arr  = np.array(buf, dtype=np.float64)
            mean = arr.mean()
            # coefficient of variation; +1 avoids div-by-zero on zero-mean features
            total_cv += arr.std() / (mean + 1.0)
        avg_cv = total_cv / len(self._bufs)
        # cv = 0 → 255 (perfectly stable); cv = 0.25 → 0 (25 % variation = unstable)
        return int(max(0, min(255, 255.0 * (1.0 - avg_cv * 4.0))))


# ── Panel renderer ────────────────────────────────────────────────────────────
def draw_panel(panel, features, extra, stability, is_frost, frame_no, raw_class_name=""):
    panel[:] = 28

    lap  = features.get("laplacian_var",      0)
    mean = features.get("global_mean",         0)
    spr  = features.get("histogram_spread",    0)
    cvar = features.get("cell_mean_variance",  0)

    conditions_met = sum([
        lap  < THR_LAP_FROST,
        mean > THR_MEAN_BRIGHT,
        cvar < THR_CELLVAR_UNIFORM,
    ])

    # ── Header ────────────────────────────────────────────────────────────────
    y     = 22
    label = "FROST" if is_frost else "not frost"
    color = (255, 185, 116) if is_frost else (120, 120, 120)
    _text(panel, label, 8, y, 0.80, color, 2)
    y += 28

    _text(panel, f"{conditions_met}/3 classifier conditions met", 8, y, 0.40, (160, 160, 160))
    if raw_class_name and not is_frost:
        _text(panel, f"detector sees: {raw_class_name}", 8, y + 12, 0.36, (180, 120, 60))
    y += 30

    cv2.line(panel, (4, y), (PANEL_W - 4, y), (60, 60, 60), 1)
    y += 12

    # ── Classifier conditions (hard gates) ───────────────────────────────────
    _text(panel, "-- classifier gates --", 4, y, 0.33, (80, 80, 80))
    y += 13

    draw_metric_row(panel, y, "sharpness",   lap,  THR_LAP_FROST,        need_below=True)
    y += 20
    draw_metric_row(panel, y, "brightness",  mean, THR_MEAN_BRIGHT,     need_below=False)
    y += 20
    draw_metric_row(panel, y, "spatial unif",cvar, THR_CELLVAR_UNIFORM, need_below=True)
    y += 24

    cv2.line(panel, (4, y), (PANEL_W - 4, y), (60, 60, 60), 1)
    y += 12

    # ── Additional diagnostic indicators ─────────────────────────────────────
    _text(panel, "-- diagnostic indicators --", 4, y, 0.33, (80, 80, 80))
    y += 13

    # Hist spread: diagnostic only — real ice frost compresses to <64,
    # vaseline stays ~190. Not a classifier gate.
    draw_metric_row(panel, y, "hist spread",    spr,
                    THR_SPREAD_FROST,      need_below=True,  scale=224, soft=True)
    y += 20

    # Edge density: frost destroys edges, other materials don't
    draw_metric_row(panel, y, "edge density",   extra["edge_density"],
                    SOFT_EDGE_DENSITY_THR, need_below=True,  scale=255, soft=True)
    y += 20

    # Temporal stability: frost is physically static, live scenes aren't
    draw_metric_row(panel, y, "temporal stab.", stability,
                    SOFT_STABILITY_THR,    need_below=False, scale=255, soft=True)
    y += 20

    # Scatter ratio: high mean + low contrast = diffuse bright veil
    draw_metric_row(panel, y, "scatter ratio",  extra["scatter_ratio"],
                    SOFT_SCATTER_THR,      need_below=False, scale=255, soft=True)
    y += 20

    # Bright pixel fraction: frost pushes pixels toward saturation
    draw_metric_row(panel, y, "bright pixels",  extra["bright_fraction"],
                    SOFT_BRIGHT_FRAC_THR,  need_below=False, scale=255, soft=True)
    y += 24

    cv2.line(panel, (4, y), (PANEL_W - 4, y), (60, 60, 60), 1)
    y += 10

    # ── Hints for failing classifier gates ───────────────────────────────────
    hints = []
    if lap  >= THR_LAP_FROST:
        hints.append(f"sharpness {lap} >= {THR_LAP_FROST}: material has visible")
        hints.append("  edges. use petrol jelly / frosted glass.")
    if mean <= THR_MEAN_BRIGHT:
        hints.append(f"brightness {mean} <= {THR_MEAN_BRIGHT}: no bright veil.")
        hints.append("  face a light source through the material.")
    if cvar >= THR_CELLVAR_UNIFORM:
        hints.append(f"cvar {cvar} >= {THR_CELLVAR_UNIFORM}: uneven coverage.")
        hints.append("  spread material across the full lens surface.")

    for hint in hints[:6]:
        _text(panel, hint, 4, y, 0.32, (160, 155, 90))
        y += 13

    # ── Frame counter ─────────────────────────────────────────────────────────
    _text(panel, f"frame {frame_no}", 8, panel.shape[0] - 8, 0.34, (70, 70, 70))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    source = sys.argv[1] if len(sys.argv) > 1 else 0
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: cannot open source: {source}")
        sys.exit(1)

    ok, frame = cap.read()
    if not ok:
        print("ERROR: cannot read from source.")
        sys.exit(1)

    h, w  = frame.shape[:2]
    panel = np.zeros((h, PANEL_W, 3), dtype=np.uint8)

    print(f"Source {source}  {w}x{h}  |  [q] quit  [r] reset")

    state   = DegradationState()
    tracker = TemporalTracker()
    extra   = {"edge_density": 0, "scatter_ratio": 0, "bright_fraction": 0}
    features = {}
    is_frost = False
    frame_no = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            if isinstance(source, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                state   = DegradationState()
                tracker = TemporalTracker()
                ok, frame = cap.read()
                if not ok:
                    break
            else:
                break

        gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result   = degradation_update(state, gray)
        features = result["features"]
        is_frost = result["classification"] == FROST

        extra = compute_extra_metrics(gray)
        tracker.push(features)
        stability = tracker.stability()

        if is_frost:
            tint = np.full_like(frame, (220, 210, 180))
            cv2.addWeighted(tint, 0.25, frame, 0.75, 0, frame)
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (255, 185, 116), 5)

        draw_panel(panel, features, extra, stability, is_frost, frame_no,
                   raw_class_name=result["raw_class_name"])

        canvas = np.hstack([frame, panel])
        cv2.imshow("Frost Detection Test  [r=reset  q=quit]", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            state    = DegradationState()
            tracker  = TemporalTracker()
            frame_no = 0
            print("Reset.")

        frame_no += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
