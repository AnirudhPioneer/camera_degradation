#include "degradation.h"
#include <string.h>   /* memset, memcpy */

/*==========================================================================
  INTERNAL HELPERS
==========================================================================*/

/* Clamp an unsigned 16-bit value to the uint8 range. */
static uint8_t u8_clamp(uint16_t v)
{
    return (v > 255u) ? 255u : (uint8_t)v;
}

/* Update one uint16 EMA.
   Formula:  ema = ema - (ema >> shift) + raw
   At steady state this converges to raw * (1 << shift).
   Overflow is impossible at steady state (see degradation.h for proof).
   During transients: ema can temporarily exceed steady-state, but the
   subtraction term (ema >> shift) grows proportionally, preventing runaway. */
static uint16_t ema_update(uint16_t ema, uint8_t raw, uint8_t shift)
{
    return (uint16_t)(ema - (ema >> shift) + (uint16_t)raw);
}

/* Recover the approximate raw value from a uint16 EMA.
   Inverse of the scaling applied by ema_update. */
static uint8_t ema_raw(uint16_t ema, uint8_t shift)
{
    return (uint8_t)(ema >> shift);
}

/*==========================================================================
  ABSOLUTE-THRESHOLD CLASSIFIER
  ─────────────────────────────────────────────────────────────────────────
  Works directly on raw feature values — no baseline, no ratios.  Priority
  order (most specific / strongest first), identical to detector.py:

    1. OBSTRUCTION — flat cells at any brightness + spatially non-uniform.
                     Covers dark blobs (mud, dirt) and bright/neutral blobs
                     (tape, ice) alike; the operational response is the same.
    2. FROST       — severe optical-path blur + scene has light + uniform.
    3. FOG         — low contrast + blurry + compressed histogram + uniform,
                     and mean above the darkness floor.
    4. CLEAN       — none of the above.

  *out_conf receives a 0–255 firing strength.  Returns the class.
==========================================================================*/
static DegradationClass classify_absolute(const DegradationFeatures *f,
                                          uint8_t *out_conf)
{
    uint8_t lap  = f->laplacian_var;
    uint8_t ctr  = f->global_contrast;
    uint8_t spr  = f->histogram_spread;
    uint8_t mean = f->global_mean;
    uint8_t obs  = f->obstruction_score;
    uint8_t cvar = f->cell_mean_variance;

    /* 1. OBSTRUCTION */
    if (obs >= DEGR_THR_OBS_CELLS && cvar >= DEGR_THR_CELLVAR_LOCAL) {
        uint16_t c = (uint16_t)(obs - DEGR_THR_OBS_CELLS + 1u) * 50u
                   + (uint16_t)(cvar - DEGR_THR_CELLVAR_LOCAL) * 2u;
        *out_conf = u8_clamp(c);
        return DEGRADATION_OBSTRUCTION;
    }

    /* 2. FROST */
    if (lap < DEGR_THR_LAP_FROST &&
        mean > DEGR_THR_MEAN_BRIGHT &&
        cvar < DEGR_THR_CELLVAR_UNIFORM) {
        *out_conf = u8_clamp((uint16_t)(DEGR_THR_LAP_BLUR - lap) * 10u);
        return DEGRADATION_FROST;
    }

    /* 3. FOG */
    if (ctr < DEGR_THR_CONTRAST_LOW &&
        lap < DEGR_THR_LAP_BLUR &&
        spr < DEGR_THR_SPREAD_FOG &&
        cvar < DEGR_THR_CELLVAR_UNIFORM &&
        mean > DEGR_THR_MEAN_FLOOR) {
        *out_conf = u8_clamp((uint16_t)(DEGR_THR_CONTRAST_LOW - ctr) * 6u);
        return DEGRADATION_FOG;
    }

    *out_conf = 0u;
    return DEGRADATION_CLEAN;
}

/*==========================================================================
  MAIN UPDATE — single-pass feature extraction + classification
==========================================================================*/

uint8_t degradation_update(DegradationState  *state,
                           const uint8_t     *frame,
                           uint16_t           width,
                           uint16_t           height,
                           DegradationResult *result)
{
    /*----------------------------------------------------------------------
      Stack working storage.  All sizes are compile-time constants.
      Rough total: cell arrays 120 B + hist 16 B + scalars ~30 B = ~166 B.
    ----------------------------------------------------------------------*/

    /* Grid cell accumulators: one entry per cell (12 total).
       sum holds the pixel sum; sumsq holds sum of squares.
       Overflow analysis (for a 320×240 frame at stride 4):
         samples per cell ≈ (320/4/4) × (240/4/3) = 20 × 20 = 400
         max sum   = 400 × 255        =   102 000  < 2^32  ✓
         max sumsq = 400 × 255²       = 26 010 000 < 2^32  ✓  */
    uint32_t cell_sum  [DEGR_GRID_CELLS];
    uint32_t cell_sumsq[DEGR_GRID_CELLS];
    uint16_t cell_count[DEGR_GRID_CELLS];

    /* Laplacian accumulator.
       We clamp |lap| to uint8 before squaring (see main loop).
       max sumsq = (320×240 / 16 - edges) × 255² ≈ 4 560 × 65 025 ≈ 296 M < 2^32  ✓ */
    uint32_t lap_sumsq = 0u;
    uint32_t lap_count = 0u;

    /* 8-bin histogram.  Max count per bin ≤ total samples ≈ 4 560 < 65 535  ✓ */
    uint16_t hist[DEGR_HIST_BINS];

    /* Global statistics accumulated in the main loop. */
    uint32_t global_sum   = 0u;
    uint16_t global_count = 0u;
    uint8_t  global_min   = 255u;
    uint8_t  global_max   = 0u;

    /* Thumbnail inter-frame absolute difference accumulator.
       max = 192 × 255 = 48 960 < 65 535  — uint16_t would suffice, but
       uint32_t avoids any future widening concern if geometry changes.     */
    uint32_t mad_sum = 0u;

    memset(cell_sum,   0, sizeof(cell_sum));
    memset(cell_sumsq, 0, sizeof(cell_sumsq));
    memset(cell_count, 0, sizeof(cell_count));
    memset(hist,       0, sizeof(hist));

    /*----------------------------------------------------------------------
      SINGLE PASS over the frame at spatial stride DEGR_STRIDE (4).
      We start at y=1 and stop at height-2 so the Laplacian can safely
      read the row above and below without a bounds check inside the loop.
      The outermost 1-pixel border is never sampled; for a 320×240 frame
      that is <1% of pixels — negligible.
    ----------------------------------------------------------------------*/
    {
        uint16_t y, x;
        for (y = 1u; y < (uint16_t)(height - 1u); y += DEGR_STRIDE) {
            for (x = 1u; x < (uint16_t)(width - 1u); x += DEGR_STRIDE) {

                uint8_t p = frame[(uint32_t)y * width + x];

                /* ── Global min/max/sum ─────────────────────────────── */
                if (p < global_min) global_min = p;
                if (p > global_max) global_max = p;
                global_sum += p;
                global_count++;

                /* ── 8-bin histogram (bin index = pixel >> 5) ─────── */
                hist[p >> 5]++;

                /* ── Laplacian: 3×3 kernel [0,1,0; 1,-4,1; 0,1,0] ── */
                {
                    /* Range of lap: 4*255 - 0 = +1020  to  4*0 - 4*255 = -1020.
                       We take |lap| and clamp to 255 before squaring so that
                       lap_sumsq stays in uint32 (see accumulator comment above). */
                    int16_t lap = (int16_t)(4 * (int16_t)p)
                                - (int16_t)frame[((uint32_t)y - 1u) * width + x]
                                - (int16_t)frame[((uint32_t)y + 1u) * width + x]
                                - (int16_t)frame[(uint32_t)y * width + x - 1u]
                                - (int16_t)frame[(uint32_t)y * width + x + 1u];
                    if (lap < 0)   lap = -lap;
                    if (lap > 255) lap = 255;
                    uint8_t lu = (uint8_t)lap;
                    lap_sumsq += (uint32_t)lu * lu;
                    lap_count++;
                }

                /* ── Grid cell assignment ───────────────────────────── */
                {
                    /* Integer map: x → column, y → row, no division in inner loop.
                       Multiply then divide: (x * COLS / width) stays in uint32. */
                    uint8_t cx = (uint8_t)((uint32_t)x * DEGR_GRID_COLS / width);
                    uint8_t cy = (uint8_t)((uint32_t)y * DEGR_GRID_ROWS / height);
                    uint8_t ci = cy * DEGR_GRID_COLS + cx;

                    cell_sum  [ci] += p;
                    cell_sumsq[ci] += (uint32_t)p * p;
                    cell_count[ci]++;
                }
            }
        }
    }

    /*----------------------------------------------------------------------
      THUMBNAIL UPDATE  (192 pixel reads — negligible vs. main pass)
      Map each 16×12 thumbnail pixel to the nearest frame pixel.
      Accumulate |current - previous| for the MAD computation.
    ----------------------------------------------------------------------*/
    {
        uint8_t tx, ty;
        for (ty = 0u; ty < DEGR_THUMB_H; ty++) {
            for (tx = 0u; tx < DEGR_THUMB_W; tx++) {
                uint16_t fy = (uint16_t)((uint32_t)ty * height / DEGR_THUMB_H);
                uint16_t fx = (uint16_t)((uint32_t)tx * width  / DEGR_THUMB_W);
                uint8_t  curr = frame[(uint32_t)fy * width + fx];
                uint8_t  prev = state->thumbnail[ty][tx];

                mad_sum += (curr > prev) ? (uint32_t)(curr - prev)
                                         : (uint32_t)(prev - curr);
                state->thumbnail[ty][tx] = curr;
            }
        }
    }

    /*----------------------------------------------------------------------
      DERIVE FEATURES from the accumulated values.
    ----------------------------------------------------------------------*/
    DegradationFeatures *f = &result->features;

    /* ── Laplacian variance ─────────────────────────────────────────────
       Mean square of the (clamped) Laplacian response.  A sharp image
       produces large Laplacian values; a blurred one produces near zero.
       We use mean-square rather than true variance because in a
       degraded frame the mean Laplacian is also near zero, so
       mean-square ≈ variance.  Scaling >>4: max = 65025/16 = 4064, clamped
       to 255.                                                              */
    f->laplacian_var = (lap_count > 0u)
        ? u8_clamp((uint16_t)((lap_sumsq / lap_count) >> 4u))
        : 0u;

    /* ── Michelson contrast ─────────────────────────────────────────────
       (max-min)*256 / (max+min+1).
       Overflow: numerator max = 255*256 = 65280, fits in uint16_t.
       Result always ≤ 255 when max≤255 (verified: 255*256/256 = 255).   */
    if (global_count > 0u) {
        uint16_t num = (uint16_t)(global_max - global_min) * 256u;
        uint16_t den = (uint16_t)global_max + global_min + 1u;
        f->global_contrast = (uint8_t)(num / den);
    } else {
        f->global_contrast = 0u;
    }

    /* ── Global mean ────────────────────────────────────────────────────*/
    f->global_mean = (global_count > 0u)
        ? (uint8_t)(global_sum / global_count)
        : 0u;

    /* ── Histogram spread: 5th–95th percentile ──────────────────────────
       Walk from each end of the histogram accumulating counts until we
       reach 5% of total.  The spread between those two bin indices,
       multiplied by 32 (each bin covers 32 intensity values), gives a
       uint8 measure of dynamic range (max = 7*32 = 224).               */
    {
        uint8_t  bin_lo = 0u, bin_hi = DEGR_HIST_BINS - 1u;
        uint16_t target  = global_count / 20u;   /* 5% of samples */
        uint16_t cumul   = 0u;
        uint8_t  b;
        uint8_t  found   = 0u;

        for (b = 0u; b < DEGR_HIST_BINS; b++) {
            cumul += hist[b];
            if (!found && cumul > target) { bin_lo = b; found = 1u; }
        }
        cumul = 0u; found = 0u;
        for (b = DEGR_HIST_BINS; b > 0u; b--) {
            cumul += hist[b - 1u];
            if (!found && cumul > target) { bin_hi = b - 1u; found = 1u; }
        }

        f->histogram_spread = (bin_hi > bin_lo)
            ? (uint8_t)((bin_hi - bin_lo) * 32u)
            : 0u;
    }

    /* ── Grid cell analysis: occlusion, obstruction, spatial variance ───
       For each cell compute mean and internal variance.
       occlusion   : low internal variance AND dark relative to global mean
       obstruction : low internal variance regardless of brightness
       cell_mean_variance: variance of the 12 cell means — high when parts
                    of the frame look very different from each other.
       Per-cell mean and the two flat-cell masks are copied into result for
       the host viewer's grid overlay (diagnostic only).                  */
    {
        uint8_t  cell_mean[DEGR_GRID_CELLS];
        uint8_t  occ_score = 0u, obs_score = 0u;
        uint16_t obs_mask = 0u, occ_mask = 0u;
        uint16_t cm_sum   = 0u;
        uint32_t cm_sumsq = 0u;
        uint8_t  ci;

        for (ci = 0u; ci < DEGR_GRID_CELLS; ci++) {
            if (cell_count[ci] == 0u) {
                cell_mean[ci]          = 0u;
                result->cell_mean[ci]  = 0u;
                continue;
            }

            uint32_t cnt = cell_count[ci];
            cell_mean[ci] = (uint8_t)(cell_sum[ci] / cnt);

            /* Local variance = sumsq/n - mean^2.
               Computed signed: integer flooring of sumsq/n and mean can make
               this very slightly negative even though true variance is ≥ 0,
               so we clamp the negative case to 0 (matches detector.py's
               max(0, var_big >> 8)).  Scale to uint8 with >>8 (max ≈ 254).  */
            int32_t var_big = (int32_t)(cell_sumsq[ci] / cnt)
                            - (int32_t)cell_mean[ci] * (int32_t)cell_mean[ci];
            uint8_t lvar8;
            if (var_big <= 0) {
                lvar8 = 0u;
            } else {
                uint32_t v = (uint32_t)var_big >> 8u;
                lvar8 = (v > 255u) ? 255u : (uint8_t)v;
            }

            if (lvar8 < DEGR_CELL_LVAR_THR) {
                /* Obstruction: flat cell at any brightness */
                obs_score++;
                obs_mask = (uint16_t)(obs_mask | (1u << ci));

                /* Occlusion: flat cell AND significantly darker than global
                   mean.  "Half the global mean" is the threshold.          */
                if (cell_mean[ci] < (f->global_mean >> 1u)) {
                    occ_score++;
                    occ_mask = (uint16_t)(occ_mask | (1u << ci));
                }
            }

            result->cell_mean[ci] = cell_mean[ci];
            cm_sum   += cell_mean[ci];
            /* Overflow: cm_sumsq += mean² ≤ 255² = 65025.
               Over 12 cells: max = 12 × 65025 = 780 300 < 2^32.            */
            cm_sumsq += (uint32_t)cell_mean[ci] * cell_mean[ci];
        }

        f->occlusion_score    = occ_score;
        f->obstruction_score  = obs_score;
        result->cell_obs_mask = obs_mask;
        result->cell_occ_mask = occ_mask;

        /* Variance of the 12 cell means.
           var = (n * sumsq - sum^2) / n^2.  Non-negative by construction
           (Cauchy–Schwarz), so no underflow in uint32.
           Overflow check:
             n * sumsq = 12 × 780300 = 9 363 600 < 2^32  ✓
             sum^2 = (12×255)^2 = 3060^2 = 9 363 600 < 2^32  ✓
           Scale to uint8: >>6 gives max = 16256/64 = 254.                   */
        uint32_t cm_var_num = (uint32_t)DEGR_GRID_CELLS * cm_sumsq
                            - (uint32_t)cm_sum * cm_sum;
        uint32_t cm_var     = cm_var_num
                            / ((uint32_t)DEGR_GRID_CELLS * DEGR_GRID_CELLS);
        f->cell_mean_variance = (cm_var > (255u << 6u))
            ? 255u : (uint8_t)(cm_var >> 6u);
    }

    /* ── Temporal MAD ───────────────────────────────────────────────────
       raw_mad = mean absolute difference between this frame's thumbnail
       and the previous one, stored in state->thumbnail (already updated).
       We update mad_fast and mad_slow baselines with raw_mad, then:
         temporal_mad     = mad_fast (the smoothed MAD — in EMA units)
         temporal_mad_var = |raw_mad - ema_raw(mad_fast)| — instantaneous
                            deviation from the smooth EMA.                  */
    {
        uint8_t raw_mad = (uint8_t)(mad_sum / DEGR_THUMB_PIXELS);
        uint8_t prev_fast_raw = ema_raw(state->baseline.mad_fast, DEGR_EMA_FAST_SHIFT);

        state->baseline.mad_fast = ema_update(state->baseline.mad_fast,
                                              raw_mad, DEGR_EMA_FAST_SHIFT);
        /* Slow EMA frozen during confirmed degradation — see comment below. */
        if (state->hyst_class == (uint8_t)DEGRADATION_CLEAN) {
            state->baseline.mad_slow = ema_update(state->baseline.mad_slow,
                                                  raw_mad, DEGR_EMA_SLOW_SHIFT);
        }

        f->temporal_mad = state->baseline.mad_fast;
        f->temporal_mad_var = (raw_mad > prev_fast_raw)
            ? (uint16_t)(raw_mad - prev_fast_raw)
            : (uint16_t)(prev_fast_raw - raw_mad);
    }
    f->_pad = 0u;

    /*----------------------------------------------------------------------
      UPDATE ALL BASELINES
      Fast EMA: always updates.  Slow EMA: frozen while any degradation is
      confirmed in the hysteresis state machine, so the clean-scene reference
      stays put for as long as the degradation persists.  During warmup
      hyst_class == DEGRADATION_CLEAN (zero-initialised), so both EMAs update
      freely — correct.  These baselines are not read by the absolute
      classifier; they back temporal_mad and the NVM persistence API and are
      kept available for a future relative-mode classifier.
      Mad baselines were already updated in the temporal section above.
    ----------------------------------------------------------------------*/
    state->baseline.lap_fast      = ema_update(state->baseline.lap_fast,
                                               f->laplacian_var, DEGR_EMA_FAST_SHIFT);
    state->baseline.contrast_fast = ema_update(state->baseline.contrast_fast,
                                               f->global_contrast, DEGR_EMA_FAST_SHIFT);
    state->baseline.spread_fast   = ema_update(state->baseline.spread_fast,
                                               f->histogram_spread, DEGR_EMA_FAST_SHIFT);
    state->baseline.mean_fast     = ema_update(state->baseline.mean_fast,
                                               f->global_mean, DEGR_EMA_FAST_SHIFT);
    state->baseline.occ_fast      = ema_update(state->baseline.occ_fast,
                                               f->occlusion_score, DEGR_EMA_FAST_SHIFT);
    state->baseline.obs_fast      = ema_update(state->baseline.obs_fast,
                                               f->obstruction_score, DEGR_EMA_FAST_SHIFT);
    state->baseline.cellvar_fast  = ema_update(state->baseline.cellvar_fast,
                                               f->cell_mean_variance, DEGR_EMA_FAST_SHIFT);

    if (state->hyst_class == (uint8_t)DEGRADATION_CLEAN) {
        state->baseline.lap_slow      = ema_update(state->baseline.lap_slow,
                                                   f->laplacian_var, DEGR_EMA_SLOW_SHIFT);
        state->baseline.contrast_slow = ema_update(state->baseline.contrast_slow,
                                                   f->global_contrast, DEGR_EMA_SLOW_SHIFT);
        state->baseline.spread_slow   = ema_update(state->baseline.spread_slow,
                                                   f->histogram_spread, DEGR_EMA_SLOW_SHIFT);
        state->baseline.mean_slow     = ema_update(state->baseline.mean_slow,
                                                   f->global_mean, DEGR_EMA_SLOW_SHIFT);
        state->baseline.occ_slow      = ema_update(state->baseline.occ_slow,
                                                   f->occlusion_score, DEGR_EMA_SLOW_SHIFT);
        state->baseline.obs_slow      = ema_update(state->baseline.obs_slow,
                                                   f->obstruction_score, DEGR_EMA_SLOW_SHIFT);
        state->baseline.cellvar_slow  = ema_update(state->baseline.cellvar_slow,
                                                   f->cell_mean_variance, DEGR_EMA_SLOW_SHIFT);
    }

    /*----------------------------------------------------------------------
      WARMUP BOOKKEEPING
      The flag is informational only — classification runs every frame.
    ----------------------------------------------------------------------*/
    if (state->frame_count < 65535u) state->frame_count++;
    if (!state->warmup_done && state->frame_count >= DEGR_WARMUP_FRAMES) {
        state->warmup_done = 1u;
    }

    /*----------------------------------------------------------------------
      ABSOLUTE-THRESHOLD CLASSIFICATION (runs from frame 1)
    ----------------------------------------------------------------------*/
    uint8_t          confidence = 0u;
    DegradationClass raw_class  = classify_absolute(f, &confidence);

    /*----------------------------------------------------------------------
      HYSTERESIS STATE MACHINE
      Entering degraded: raw_class != CLEAN for DEGR_ENTER_FRAMES frames.
      Leaving degraded:  raw_class == CLEAN for DEGR_CLEAR_FRAMES frames.
      Between degraded types: switch immediately (no hysteresis on type).
    ----------------------------------------------------------------------*/
    {
        DegradationClass prev_hyst = (DegradationClass)state->hyst_class;

        if (prev_hyst == DEGRADATION_CLEAN) {
            if (raw_class != DEGRADATION_CLEAN) {
                state->confirm_count++;
                state->clear_count = 0u;
                if (state->confirm_count >= DEGR_ENTER_FRAMES) {
                    state->hyst_class    = (uint8_t)raw_class;
                    state->confirm_count = 0u;
                }
            } else {
                /* Clean frame resets the confirmation window. */
                state->confirm_count = 0u;
            }
        } else {
            /* Currently in a degraded state. */
            if (raw_class == DEGRADATION_CLEAN) {
                state->clear_count++;
                state->confirm_count = 0u;
                if (state->clear_count >= DEGR_CLEAR_FRAMES) {
                    state->hyst_class  = (uint8_t)DEGRADATION_CLEAN;
                    state->clear_count = 0u;
                }
            } else {
                /* Still degraded.  Switch type immediately so a transition
                   (e.g. obstruction → frost) is reported without delay.    */
                state->clear_count = 0u;
                state->hyst_class  = (uint8_t)raw_class;
            }
        }
    }

    result->classification     = (DegradationClass)state->hyst_class;
    result->raw_classification = (uint8_t)raw_class;
    result->confidence         = confidence;
    result->_pad[0] = 0u; result->_pad[1] = 0u;
    return DEGR_OK;
}

/*==========================================================================
  SERIALIZE / DESERIALIZE  — baseline persistence for NVM storage
==========================================================================*/

uint8_t degradation_serialize(const DegradationState *state,
                              uint8_t                *buf,
                              uint16_t                buf_size)
{
    if (buf_size < (uint16_t)sizeof(DegradationBaseline)) return DEGR_ERR;
    memcpy(buf, &state->baseline, sizeof(DegradationBaseline));
    return DEGR_OK;
}

uint8_t degradation_deserialize(DegradationState *state,
                                const uint8_t    *buf,
                                uint16_t          buf_size)
{
    if (buf_size < (uint16_t)sizeof(DegradationBaseline)) return DEGR_ERR;

    memcpy(&state->baseline, buf, sizeof(DegradationBaseline));

    /* Reset everything that is NOT the baseline so the system starts
       from a known-good state after a reboot.  The restored baseline
       already represents a converged clean-scene reference, so warmup
       is marked done immediately — no 60-frame suppression window.     */
    memset(state->thumbnail, 0, sizeof(state->thumbnail));
    state->frame_count   = DEGR_WARMUP_FRAMES;  /* acts as if warmup completed */
    state->warmup_done   = 1u;
    state->hyst_class    = (uint8_t)DEGRADATION_CLEAN;
    state->confirm_count = 0u;
    state->clear_count   = 0u;

    return DEGR_OK;
}
