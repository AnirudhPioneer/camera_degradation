"""
detector.py — Python port of degradation.c / degradation.h.

Exact integer arithmetic (all ops match the C line-by-line).
Uses numpy for vectorised sampling; no float in the algorithm itself.
"""
import numpy as np

# ── Constants (mirrors degradation.h exactly) ─────────────────────────────────
DEGR_STRIDE        = 4
DEGR_GRID_COLS     = 4
DEGR_GRID_ROWS     = 3
DEGR_GRID_CELLS    = 12
DEGR_THUMB_W       = 16
DEGR_THUMB_H       = 12
DEGR_THUMB_PIXELS  = 192
DEGR_HIST_BINS     = 8

DEGR_EMA_FAST_SHIFT = 4
DEGR_EMA_SLOW_SHIFT = 8
DEGR_WARMUP_FRAMES  = 60
DEGR_ENTER_FRAMES   = 5
DEGR_CLEAR_FRAMES   = 15
DEGR_RATIO_SCALE    = 128
DEGR_CELL_LVAR_THR  = 8

# ── Absolute classifier thresholds ───────────────────────────────────────────
# Applied directly to raw feature values — no baseline, no ratios.
# Tune these by watching the raw feature readout in the live display
# while exposing the lens to each condition.
#
# Quick reference — feature ranges:
#   laplacian_var     0–255   0 = completely flat/blurry, 255 = very sharp
#   global_contrast   0–255   0 = solid grey, 255 = full black-to-white range
#   histogram_spread  0–224   steps of 32; 0 = all pixels one brightness band
#   global_mean       0–255   average pixel brightness across the frame
#   occlusion_score   0–12    cells that are dark AND flat (dirt signature)
#   obstruction_score 0–12    cells that are flat at any brightness
#   cell_mean_variance 0–255  0 = every cell same brightness, 255 = very uneven

# Sharpness thresholds — two levels:
#   below FROST = severe optical-path blur (vaseline, ice directly on lens).
#   Lap hits near-0 because the coating sits in the optical path.
#   below BLUR  = moderate blur (atmospheric fog, slight defocus).
#   Atmospheric fog rarely drops lap below 5; on-lens coatings do.
THR_LAP_FROST      = 5
THR_LAP_BLUR       = 15

# Michelson contrast — below this global contrast is too low (fog)
THR_CONTRAST_LOW   = 40

# Histogram spread — two levels:
#   below FOG   = moderate compression (fog)
#   below FROST = noticeable compression toward the bright end (frost).
#   Real ice frost pushes almost everything white (2 bins); vaseline and
#   similar diffusers lift the floor without fully collapsing the ceiling,
#   landing around 3–4 bins (96–128).  128 captures both.
THR_SPREAD_FOG     = 96
THR_SPREAD_FROST   = 128

# Mean brightness — minimum illumination for frost; just rules out a
# pitch-dark covered lens.  Vaseline in any normally-lit environment
# produces mean well above this.  Not used as the frost discriminator
# (that role now belongs to THR_LAP_FROST).
THR_MEAN_BRIGHT    = 60

# Mean brightness floor — below this the scene is simply dark, not foggy.
# Fog scatters ambient light and keeps mid-tones elevated.
# Pure darkness (no light source) produces mean near 0 and must not be
# classified as fog even though all other fog features are satisfied.
THR_MEAN_FLOOR     = 40

# Spatial uniformity — two levels:
#   below UNIFORM = frame degraded evenly across all cells (fog / frost).
#   Set to 50: vaseline is never perfectly uniform and underlying scene
#   structure (bright window, dark wall) persists through the smear —
#   observed cvar ~40 with full vaseline coverage.
#   above LOCAL   = degradation confined to some cells (dirt / obstruction).
#   Set to 60: genuine obstructions (tape, mud blob on 3-6 cells) produce
#   cvar 80-200; vaseline covering all 12 cells produces cvar ~40.
THR_CELLVAR_UNIFORM = 50
THR_CELLVAR_LOCAL   = 60

# Cell counts (out of 12 grid cells)
THR_OCC_CELLS      = 2    # dark flat cells needed to call DIRT
THR_OBS_CELLS      = 3    # flat cells (any brightness) needed to call OBSTRUCTION

# NOTE: an experimental "dark_occlusion" fine-grid feature was trialled here to
# catch broad soft-edged mud, but it FAILED validation on the WoodScape soiling
# dataset (fires on the fisheye vignette / vehicle body / any dim scene → ~22%
# false positives on clean frames) and was removed so detector.py stays in exact
# lock-step with degradation.c.  It survives ONLY as a labelled demo in
# pipeline_visualiser.html.  The real fix for broad occlusion is relative/baseline
# mode (dual-EMA deviation), which needs video and is not built yet.

# Aliases kept for API compatibility
DEGR_THR_LAP_LOW       = THR_LAP_FROST
DEGR_THR_CONTRAST_LOW  = THR_CONTRAST_LOW
DEGR_THR_SPREAD_VLOW   = THR_SPREAD_FROST
DEGR_THR_MEAN_HIGH     = THR_MEAN_BRIGHT
DEGR_THR_OBS_HIGH      = THR_OBS_CELLS
DEGR_THR_CELLVAR_HIGH  = THR_CELLVAR_LOCAL
DEGR_THR_CELLVAR_LOW   = THR_CELLVAR_UNIFORM

CLEAN       = 0
FOG         = 1
FROST       = 2
OBSTRUCTION = 3   # any localised physical block — dark blob, tape, ice, mud
UNKNOWN     = 4
CLASS_NAMES  = ["CLEAN", "FOG", "FROST", "OBSTRUCTION", "UNKNOWN"]
CLASS_COLORS = {
    "CLEAN":       "#27ae60",
    "FOG":         "#7f8c8d",
    "FROST":       "#74b9ff",
    "OBSTRUCTION": "#e74c3c",
    "UNKNOWN":     "#9b59b6",
}


# ── State ─────────────────────────────────────────────────────────────────────
class DegradationState:
    """Mirrors DegradationState in degradation.h. Zero-initialised = valid start."""
    __slots__ = (
        "lap_fast",     "lap_slow",
        "contrast_fast","contrast_slow",
        "spread_fast",  "spread_slow",
        "mean_fast",    "mean_slow",
        "occ_fast",     "occ_slow",
        "obs_fast",     "obs_slow",
        "cellvar_fast", "cellvar_slow",
        "mad_fast",     "mad_slow",
        "thumbnail",
        "frame_count", "warmup_done",
        "hyst_class", "confirm_count", "clear_count",
    )

    def __init__(self):
        self.lap_fast = self.lap_slow = 0
        self.contrast_fast = self.contrast_slow = 0
        self.spread_fast   = self.spread_slow   = 0
        self.mean_fast     = self.mean_slow     = 0
        self.occ_fast      = self.occ_slow      = 0
        self.obs_fast      = self.obs_slow      = 0
        self.cellvar_fast  = self.cellvar_slow  = 0
        self.mad_fast      = self.mad_slow      = 0
        self.thumbnail     = np.zeros((DEGR_THUMB_H, DEGR_THUMB_W), dtype=np.uint8)
        self.frame_count   = 0
        self.warmup_done   = False
        self.hyst_class    = CLEAN
        self.confirm_count = 0
        self.clear_count   = 0

    def clone(self):
        import copy
        return copy.deepcopy(self)


# ── Arithmetic helpers (match C helpers exactly) ──────────────────────────────
def _u8(v):
    return min(255, max(0, int(v)))

def _ema(ema, raw, shift):
    return (ema - (ema >> shift) + raw) & 0xFFFF

def _raw(ema, shift):
    return (ema >> shift) & 0xFF

def _ratio(fast_ema, slow_ema):
    f = _raw(fast_ema, DEGR_EMA_FAST_SHIFT)
    s = _raw(slow_ema, DEGR_EMA_SLOW_SHIFT)
    return min(255, f * DEGR_RATIO_SCALE // (s + 1))


# ── Main update ───────────────────────────────────────────────────────────────
def _classify_absolute(features: dict) -> tuple:
    """
    Classify degradation using fixed absolute thresholds on raw feature values.
    No baseline, no EMA ratios — what the numbers are is what fires.

    Decision priority (same order as C decision tree):
      1. DIRT        — dark flat cells + spatially non-uniform frame
      2. OBSTRUCTION — flat cells at any brightness + spatially non-uniform
      3. FROST       — blurry + bright mean + severely compressed histogram + uniform
      4. FOG         — low contrast + blurry + moderately compressed histogram + uniform
      5. CLEAN       — none of the above

    Returns (class_int, confidence 0-255).
    """
    lap  = features["laplacian_var"]
    ctr  = features["global_contrast"]
    spr  = features["histogram_spread"]
    mean = features["global_mean"]
    occ  = features["occlusion_score"]
    obs  = features["obstruction_score"]
    cvar = features["cell_mean_variance"]

    # 1. OBSTRUCTION: flat cells at any brightness AND spatially non-uniform.
    #    Covers dark blobs (mud, dirt) and bright/neutral blobs (tape, ice)
    #    equally — the operational response is the same: clean the lens.
    if obs >= THR_OBS_CELLS and cvar >= THR_CELLVAR_LOCAL:
        conf = min(255, (obs - THR_OBS_CELLS + 1) * 50 + (cvar - THR_CELLVAR_LOCAL) * 2)
        return OBSTRUCTION, conf

    # 2. FROST: severe optical-path blur + scene has light + spatially uniform.
    #    THR_LAP_FROST (5) distinguishes on-lens coatings (lap → 0) from
    #    atmospheric fog (lap 5–15).  Mean > THR_MEAN_BRIGHT (60) only rules
    #    out a completely dark covered lens — it is not a scene-brightness gate.
    if (lap  <  THR_LAP_FROST
            and mean >  THR_MEAN_BRIGHT
            and cvar <  THR_CELLVAR_UNIFORM):
        conf = min(255, (THR_LAP_BLUR - lap) * 10)
        return FROST, conf

    # 3. FOG: low contrast + blurry + moderately compressed histogram + spatially uniform
    #    AND mean above the floor — pure darkness satisfies every other fog
    #    condition but has mean ≈ 0, so we exclude it explicitly.
    if (ctr  <  THR_CONTRAST_LOW
            and lap  <  THR_LAP_BLUR
            and spr  <  THR_SPREAD_FOG
            and cvar <  THR_CELLVAR_UNIFORM
            and mean >  THR_MEAN_FLOOR):
        conf = min(255, (THR_CONTRAST_LOW - ctr) * 6)
        return FOG, conf

    return CLEAN, 0


def degradation_update(state: DegradationState, frame: np.ndarray) -> dict:
    """
    frame : H×W uint8 numpy array (grayscale).
    Returns a result dict with keys:
        classification (int), class_name (str), confidence (int 0-255),
        features (dict of raw values), ratios (dict), warmup (bool).
    """
    h, w = frame.shape

    # ── Sampled coordinates — start at 1, stop before h-1/w-1 (same as C) ──
    ys = np.arange(1, h - 1, DEGR_STRIDE)
    xs = np.arange(1, w - 1, DEGR_STRIDE)
    Y, X = np.meshgrid(ys, xs, indexing='ij')   # shape (Ny, Nx)

    p = frame[Y, X].astype(np.int32)            # sampled pixels

    # ── Laplacian — immediate ±1 neighbours, same as C ──────────────────────
    lap = (4 * p
           - frame[Y - 1, X].astype(np.int32)
           - frame[Y + 1, X].astype(np.int32)
           - frame[Y,     X - 1].astype(np.int32)
           - frame[Y,     X + 1].astype(np.int32))
    lap_abs   = np.abs(lap).clip(0, 255)
    lap_sumsq = int((lap_abs * lap_abs).sum())
    lap_count = lap_abs.size
    laplacian_var = _u8((lap_sumsq // lap_count) >> 4)

    # ── Global min / max / sum ───────────────────────────────────────────────
    p_flat    = p.ravel()
    gmin      = int(p_flat.min())
    gmax      = int(p_flat.max())
    gsum      = int(p_flat.sum())
    gcount    = p_flat.size

    # ── Michelson contrast ───────────────────────────────────────────────────
    global_contrast = _u8((gmax - gmin) * 256 // (gmax + gmin + 1))

    # ── Global mean ──────────────────────────────────────────────────────────
    global_mean = _u8(gsum // gcount)

    # ── 8-bin histogram ──────────────────────────────────────────────────────
    bins = (p_flat >> 5).clip(0, 7).astype(np.int64)
    hist = np.bincount(bins, minlength=DEGR_HIST_BINS)

    # ── Histogram spread: 5th–95th percentile (exact C port) ────────────────
    target = gcount // 20
    cumul, bin_lo, found = 0, 0, False
    for b in range(DEGR_HIST_BINS):
        cumul += int(hist[b])
        if not found and cumul > target:
            bin_lo = b; found = True
    cumul, bin_hi, found = 0, DEGR_HIST_BINS - 1, False
    for b in range(DEGR_HIST_BINS - 1, -1, -1):
        cumul += int(hist[b])
        if not found and cumul > target:
            bin_hi = b; found = True
    histogram_spread = (bin_hi - bin_lo) * 32 if bin_hi > bin_lo else 0

    # ── Grid cell analysis ────────────────────────────────────────────────────
    cx = (X.astype(np.int64) * DEGR_GRID_COLS // w).clip(0, DEGR_GRID_COLS - 1)
    cy = (Y.astype(np.int64) * DEGR_GRID_ROWS // h).clip(0, DEGR_GRID_ROWS - 1)
    ci = (cy * DEGR_GRID_COLS + cx).ravel()
    pf = p_flat.astype(np.float64)

    cell_count_arr = np.bincount(ci, minlength=DEGR_GRID_CELLS).astype(np.int64)
    cell_sum_arr   = np.bincount(ci, weights=pf,    minlength=DEGR_GRID_CELLS).astype(np.int64)
    cell_sumsq_arr = np.bincount(ci, weights=pf*pf, minlength=DEGR_GRID_CELLS).astype(np.int64)

    occ_score = obs_score = 0
    cm_sum = cm_sumsq = 0
    # Per-cell data stored for the live overlay (not used by the algorithm)
    cell_means_arr = [0] * DEGR_GRID_CELLS
    cell_occ_mask  = [False] * DEGR_GRID_CELLS   # dark flat blob
    cell_obs_mask  = [False] * DEGR_GRID_CELLS   # any-brightness flat blob

    for c in range(DEGR_GRID_CELLS):
        cnt = int(cell_count_arr[c])
        if cnt == 0:
            continue
        mean_c = int(cell_sum_arr[c]) // cnt

        # local variance: (sumsq/n - mean²) >> 8  — matches C exactly
        var_big = int(cell_sumsq_arr[c]) // cnt - mean_c * mean_c
        lvar8   = min(255, max(0, var_big >> 8))

        if lvar8 < DEGR_CELL_LVAR_THR:
            obs_score += 1
            cell_obs_mask[c] = True
            if mean_c < (global_mean >> 1):
                occ_score += 1
                cell_occ_mask[c] = True

        cell_means_arr[c] = mean_c
        cm_sum   += mean_c
        cm_sumsq += mean_c * mean_c

    cm_var_num        = DEGR_GRID_CELLS * cm_sumsq - cm_sum * cm_sum
    cm_var            = cm_var_num // (DEGR_GRID_CELLS * DEGR_GRID_CELLS)
    cell_mean_variance = min(255, max(0, cm_var >> 6))

    # ── Thumbnail MAD ─────────────────────────────────────────────────────────
    fy = (np.arange(DEGR_THUMB_H) * h // DEGR_THUMB_H).astype(np.int32)
    fx = (np.arange(DEGR_THUMB_W) * w // DEGR_THUMB_W).astype(np.int32)
    curr_thumb = frame[np.ix_(fy, fx)].copy()

    mad_sum = int(np.abs(curr_thumb.astype(np.int32)
                         - state.thumbnail.astype(np.int32)).sum())
    raw_mad  = mad_sum // DEGR_THUMB_PIXELS

    prev_fast_raw  = _raw(state.mad_fast, DEGR_EMA_FAST_SHIFT)
    state.mad_fast = _ema(state.mad_fast, raw_mad, DEGR_EMA_FAST_SHIFT)
    if state.hyst_class == CLEAN:                                  # freeze with rest of slow EMAs
        state.mad_slow = _ema(state.mad_slow, raw_mad, DEGR_EMA_SLOW_SHIFT)
    state.thumbnail = curr_thumb

    temporal_mad     = state.mad_fast
    temporal_mad_var = abs(raw_mad - prev_fast_raw)

    features = {
        "laplacian_var":      laplacian_var,
        "global_contrast":    global_contrast,
        "histogram_spread":   histogram_spread,
        "global_mean":        global_mean,
        "occlusion_score":    occ_score,
        "obstruction_score":  obs_score,
        "cell_mean_variance": cell_mean_variance,
        "temporal_mad":       temporal_mad,
        "temporal_mad_var":   temporal_mad_var,
        # Per-cell data for the live overlay — not used by the algorithm itself
        "_cell_means": cell_means_arr,   # list[int], one per grid cell
        "_cell_occ":   cell_occ_mask,    # list[bool] dark flat cells (dirt)
        "_cell_obs":   cell_obs_mask,    # list[bool] any-brightness flat cells
    }

    # ── Update all baselines ──────────────────────────────────────────────────
    # Fast EMA: always updates. It tracks the current scene so the ratio
    # numerator stays meaningful even during prolonged degradation.
    state.lap_fast      = _ema(state.lap_fast,      laplacian_var,      DEGR_EMA_FAST_SHIFT)
    state.contrast_fast = _ema(state.contrast_fast, global_contrast,    DEGR_EMA_FAST_SHIFT)
    state.spread_fast   = _ema(state.spread_fast,   histogram_spread,   DEGR_EMA_FAST_SHIFT)
    state.mean_fast     = _ema(state.mean_fast,     global_mean,        DEGR_EMA_FAST_SHIFT)
    state.occ_fast      = _ema(state.occ_fast,      occ_score,          DEGR_EMA_FAST_SHIFT)
    state.obs_fast      = _ema(state.obs_fast,      obs_score,          DEGR_EMA_FAST_SHIFT)
    state.cellvar_fast  = _ema(state.cellvar_fast,  cell_mean_variance, DEGR_EMA_FAST_SHIFT)

    # Slow EMA: frozen while any degradation is confirmed.
    # This locks the clean-scene reference in place so the ratio stays large
    # for as long as the degradation persists, no matter how many frames pass.
    # During warmup hyst_class == CLEAN, so both EMAs build freely — correct.
    if state.hyst_class == CLEAN:
        state.lap_slow      = _ema(state.lap_slow,      laplacian_var,      DEGR_EMA_SLOW_SHIFT)
        state.contrast_slow = _ema(state.contrast_slow, global_contrast,    DEGR_EMA_SLOW_SHIFT)
        state.spread_slow   = _ema(state.spread_slow,   histogram_spread,   DEGR_EMA_SLOW_SHIFT)
        state.mean_slow     = _ema(state.mean_slow,     global_mean,        DEGR_EMA_SLOW_SHIFT)
        state.occ_slow      = _ema(state.occ_slow,      occ_score,          DEGR_EMA_SLOW_SHIFT)
        state.obs_slow      = _ema(state.obs_slow,      obs_score,          DEGR_EMA_SLOW_SHIFT)
        state.cellvar_slow  = _ema(state.cellvar_slow,  cell_mean_variance, DEGR_EMA_SLOW_SHIFT)

    # ── Warmup ────────────────────────────────────────────────────────────────
    state.frame_count = min(state.frame_count + 1, 65535)
    if not state.warmup_done and state.frame_count >= DEGR_WARMUP_FRAMES:
        state.warmup_done = True

    # ── Absolute-threshold classifier ─────────────────────────────────────────
    # Works directly on raw feature values. No baseline, no ratios.
    # Active from frame 1 — warmup flag is kept for display only.
    raw_class, confidence = _classify_absolute(features)

    # ── Hysteresis (exact C port) ─────────────────────────────────────────────
    if state.hyst_class == CLEAN:
        if raw_class != CLEAN:
            state.confirm_count += 1
            state.clear_count    = 0
            if state.confirm_count >= DEGR_ENTER_FRAMES:
                state.hyst_class    = raw_class
                state.confirm_count = 0
        else:
            state.confirm_count = 0
    else:
        if raw_class == CLEAN:
            state.clear_count   += 1
            state.confirm_count  = 0
            if state.clear_count >= DEGR_CLEAR_FRAMES:
                state.hyst_class  = CLEAN
                state.clear_count = 0
        else:
            state.clear_count = 0
            state.hyst_class  = raw_class

    cls = state.hyst_class
    return {
        "classification":      cls,
        "class_name":          CLASS_NAMES[cls],
        "raw_classification":  raw_class,
        "raw_class_name":      CLASS_NAMES[raw_class],
        "confidence":          confidence,
        "features":            features,
        "ratios":              {},        # unused in absolute mode; kept for API compat
        "warmup":              not state.warmup_done,
    }
