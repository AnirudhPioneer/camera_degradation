#ifndef DEGRADATION_H
#define DEGRADATION_H

#include <stdint.h>

/*==========================================================================
  FRAME GEOMETRY
==========================================================================*/

/* How many pixels we skip between samples.  Stride 4 means we look at
   every 4th pixel in both x and y — 1/16th of all pixels.  This trades
   resolution for speed while keeping enough spatial coverage to detect
   all degradation types reliably on a wide-angle lens. */
#define DEGR_STRIDE         4

/* The frame is divided into a 4-column × 3-row grid for spatial analysis.
   12 cells total.  4×3 was chosen to give horizontal asymmetry detection
   (e.g. dirt on one side) without becoming too fine-grained. */
#define DEGR_GRID_COLS      4
#define DEGR_GRID_ROWS      3
#define DEGR_GRID_CELLS    12    /* DEGR_GRID_COLS * DEGR_GRID_ROWS */

/* Previous-frame thumbnail dimensions.  16×12 = 192 bytes.  Small enough
   to fit in budget, large enough to capture meaningful frame-to-frame
   motion.  Aspect ratio matches the grid so cells align naturally. */
#define DEGR_THUMB_W       16
#define DEGR_THUMB_H       12
#define DEGR_THUMB_PIXELS 192    /* DEGR_THUMB_W * DEGR_THUMB_H */

/* Number of histogram bins.  8 bins = 3-bit precision (pixel >> 5).
   Enough to distinguish fog (mid-range compression) from frost (high-end
   compression) without needing division or a large table. */
#define DEGR_HIST_BINS      8

/*==========================================================================
  TEMPORAL PARAMETERS
==========================================================================*/

/* EMA alpha values expressed as right-shift amounts.  Bit-shifts replace
   division completely — no division instruction needed on Cortex-M0.
   Shift 4  → alpha = 1/16  → time constant ≈ 16 frames  (fast EMA)
   Shift 8  → alpha = 1/256 → time constant ≈ 256 frames (slow EMA)         */
#define DEGR_EMA_FAST_SHIFT 4
#define DEGR_EMA_SLOW_SHIFT 8

/* How many frames the system consumes at startup before the warmup flag
   is cleared.  60 frames at 10 Hz = 6 seconds.  In the absolute-threshold
   classifier the warmup window does NOT gate classification — the decision
   tree runs from frame 1.  The flag exists so a host application can choose
   to suppress its own alarms/UI until the slow EMA baseline has converged
   (see degradation_update() doc comment). */
#define DEGR_WARMUP_FRAMES 60

/* Hysteresis frame counts prevent flickering.  A suspected degradation
   must persist for ENTER consecutive frames before we report it.  A
   recovery must persist for CLEAR consecutive frames to return to CLEAN. */
#define DEGR_ENTER_FRAMES   5
#define DEGR_CLEAR_FRAMES  15

/*==========================================================================
  CLASSIFIER THRESHOLDS  —  ABSOLUTE, applied to raw feature values
  ─────────────────────────────────────────────────────────────────────────
  The classifier does NOT compare against the EMA baseline.  It tests the
  raw per-frame feature values directly against fixed thresholds.  Tune
  these by watching the live raw-feature readout while exposing the lens to
  each condition.  Quick reference for feature ranges:

    laplacian_var      0–255   0 = flat/blurry, 255 = very sharp
    global_contrast    0–255   0 = solid grey, 255 = full black-to-white
    histogram_spread   0–224   steps of 32; 0 = one brightness band only
    global_mean        0–255   average pixel brightness across the frame
    occlusion_score    0–12    cells that are dark AND flat (dirt signature)
    obstruction_score  0–12    cells that are flat at any brightness
    cell_mean_variance 0–255   0 = every cell same brightness, 255 = uneven

  The dual-timescale EMA baseline is still maintained every frame (used for
  temporal_mad and reserved for future relative-mode work / NVM persistence)
  but it is not consulted by the absolute decision tree.
==========================================================================*/

/* Sharpness — two levels.
     below FROST = severe optical-path blur (vaseline, ice on the lens);
                   the Laplacian collapses to near zero.
     below BLUR  = moderate blur (atmospheric fog, slight defocus).        */
#define DEGR_THR_LAP_FROST      5
#define DEGR_THR_LAP_BLUR      15

/* Michelson contrast — below this, global contrast is too low (fog).      */
#define DEGR_THR_CONTRAST_LOW  40

/* Histogram spread — two levels.
     below FOG   = moderate compression (fog).
     below FROST = noticeable compression toward the bright end (frost).    */
#define DEGR_THR_SPREAD_FOG    96
#define DEGR_THR_SPREAD_FROST 128

/* Mean brightness — minimum illumination for frost; just rules out a
   pitch-dark covered lens.  Not the frost discriminator (that is the
   Laplacian).                                                              */
#define DEGR_THR_MEAN_BRIGHT   60

/* Mean brightness floor — below this the scene is simply dark, not foggy.
   Pure darkness satisfies every other fog feature but has mean near 0 and
   must not be classified as fog.                                          */
#define DEGR_THR_MEAN_FLOOR    40

/* Spatial uniformity — two levels.
     below UNIFORM = frame degraded evenly across all cells (fog / frost).
     at/above LOCAL = degradation confined to some cells (obstruction).     */
#define DEGR_THR_CELLVAR_UNIFORM 50
#define DEGR_THR_CELLVAR_LOCAL   60

/* Cell counts (out of 12 grid cells). */
#define DEGR_THR_OCC_CELLS      2   /* dark flat cells — reported via occlusion_score */
#define DEGR_THR_OBS_CELLS      3   /* flat cells (any brightness) needed to call OBSTRUCTION */

/* A grid cell whose internal variance (scaled, see degradation.c) is below
   this is considered "flat".  Used to build occlusion/obstruction scores. */
#define DEGR_CELL_LVAR_THR      8

/*==========================================================================
  RETURN CODES  — no assert(), no errno; errors are explicit uint8_t values
==========================================================================*/
#define DEGR_OK  ((uint8_t)0)
#define DEGR_ERR ((uint8_t)1)

/*==========================================================================
  TYPES
==========================================================================*/

/* The five possible outputs of the detector.
   DIRT is intentionally NOT a separate class: a dark flat blob (mud, dirt)
   and a bright/neutral flat blob (tape, ice) both mean "something is stuck
   on the lens, clean it", so they collapse into OBSTRUCTION.  The dark-blob
   signature is still surfaced separately as occlusion_score for diagnostics.
   These five are the ONLY classes; the deliberately-excluded smear class
   from the brief is not represented here and must never be added. */
typedef enum {
    DEGRADATION_CLEAN       = 0,
    DEGRADATION_FOG         = 1,
    DEGRADATION_FROST       = 2,
    DEGRADATION_OBSTRUCTION = 3,
    DEGRADATION_UNKNOWN     = 4
} DegradationClass;


/* All 8 metrics produced by the single-pass sweep.
   uint16_t fields come first so the compiler does not need to insert
   padding bytes before them; the six uint8_t fields follow, plus one
   explicit pad byte to keep the total an even number of bytes.
   Total: 4 + 8 + 1 (pad) = 13 → padded to 14? No: 4 + 6×1 + 1 pad = 11,
   rounded to even = 12 bytes.                                          */
typedef struct {
    uint16_t temporal_mad;       /* EMA of inter-frame mean-absolute-difference */
    uint16_t temporal_mad_var;   /* |raw MAD − fast-EMA MAD| (motion variability)*/
    uint8_t  laplacian_var;      /* 3×3 Laplacian variance at stride-4 samples   */
    uint8_t  global_contrast;    /* Michelson: (max-min)*256 / (max+min+1)       */
    uint8_t  histogram_spread;   /* 5th–95th percentile span of 8-bin histogram  */
    uint8_t  global_mean;        /* mean of all stride-4 samples                 */
    uint8_t  occlusion_score;    /* cells with low variance AND dark mean (0-12) */
    uint8_t  obstruction_score;  /* cells with low variance, any brightness(0-12)*/
    uint8_t  cell_mean_variance; /* variance of the 12 grid-cell means           */
    uint8_t  _pad;               /* keeps struct size a multiple of 2 bytes      */
} DegradationFeatures;           /* 12 bytes, 2-byte aligned                     */


/* EMA storage for each feature, fast and slow track independently.
   ─────────────────────────────────────────────────────────────────
   Update rule (same for both, only the shift differs):
       ema = ema - (ema >> shift) + raw_value

   At steady state this converges to:
       fast_ema → raw_value * 16    (shift 4, so ema>>4 recovers raw)
       slow_ema → raw_value * 256   (shift 8, so ema>>8 recovers raw)

   Using uint16_t keeps the scaled values in range:
       max fast_ema = 255 * 16  =  4 080  < 65535  ✓
       max slow_ema = 255 * 256 = 65 280  < 65535  ✓

   Maintained every frame for temporal_mad and for optional NVM persistence;
   the absolute classifier does not read these.  Zero-initializable.      */
typedef struct {
    uint16_t lap_fast,       lap_slow;
    uint16_t contrast_fast,  contrast_slow;
    uint16_t spread_fast,    spread_slow;
    uint16_t mean_fast,      mean_slow;
    uint16_t occ_fast,       occ_slow;
    uint16_t obs_fast,       obs_slow;
    uint16_t cellvar_fast,   cellvar_slow;
    uint16_t mad_fast,       mad_slow;
} DegradationBaseline;       /* 16 × uint16_t = 32 bytes                        */


/* Complete module state.  Declare exactly ONE of these as a static
   variable in the application and zero-initialise it once:
       static DegradationState cam_state;   (zero at link time — fine)
   No init function is required.  Zero state means "start of warmup".

   Memory layout (offsets verified to be aligned):
     [  0.. 31] DegradationBaseline  (32 bytes)
     [ 32..223] thumbnail            (192 bytes)
     [224..225] frame_count          (uint16_t, offset 224 is even ✓)
     [226]      warmup_done
     [227]      hyst_class
     [228]      confirm_count
     [229]      clear_count
   Total: 230 bytes                                                        */
typedef struct {
    DegradationBaseline baseline;
    uint8_t  thumbnail[DEGR_THUMB_H][DEGR_THUMB_W];  /* previous-frame 16×12 */
    uint16_t frame_count;    /* total frames processed; used for warmup logic  */
    uint8_t  warmup_done;    /* 0 = baseline still converging, 1 = converged    */
    uint8_t  hyst_class;     /* current hysteresis label (cast: DegradationClass)*/
    uint8_t  confirm_count;  /* consecutive frames matching suspected class     */
    uint8_t  clear_count;    /* consecutive CLEAN frames while in degraded state*/
} DegradationState;          /* 230 bytes                                       */


/* What degradation_update() returns on every call.
   ─────────────────────────────────────────────────────────────────────────
   The first 17 bytes are the embedded essentials (features + classification
   + confidence).  The fields after that are DIAGNOSTIC OVERLAY DATA: a host
   PC viewer uses them to shade the 4×3 grid and print per-cell means, but an
   embedded integrator that only needs the classification can ignore them.

   Layout (offsets verified aligned):
     [ 0..11] features            (12 bytes)
     [12..15] classification      (enum = int, 4-byte aligned ✓)
     [16]     confidence
     [17]     raw_classification  (pre-hysteresis class — what this frame alone says)
     [18..19] cell_obs_mask       (bit i = cell i is flat at any brightness)
     [20..21] cell_occ_mask       (bit i = cell i is flat AND dark)
     [22..33] cell_mean[12]       (per-cell mean brightness, 0 if empty)
     [34..35] _pad
   Total: 36 bytes, 4-byte aligned.                                        */
typedef struct {
    DegradationFeatures features;       /* raw per-frame metrics for logging    */
    DegradationClass    classification; /* hysteresis-filtered detection output  */
    uint8_t             confidence;     /* 0–255: classifier firing strength     */
    uint8_t             raw_classification; /* class before hysteresis (diag)    */
    uint16_t            cell_obs_mask;  /* diag: flat cells, any brightness      */
    uint16_t            cell_occ_mask;  /* diag: flat AND dark cells             */
    uint8_t             cell_mean[DEGR_GRID_CELLS]; /* diag: per-cell mean       */
    uint8_t             _pad[2];        /* keep struct 4-byte aligned            */
} DegradationResult;         /* 36 bytes                                        */


/*==========================================================================
  PUBLIC API
==========================================================================*/

/* Process one frame and return the current detection state.
   Call this for every frame you wish to analyse (e.g. every 3rd frame on
   a 30fps stream gives 10Hz detection).

   state  : persistent state; the caller owns one static instance.
   frame  : pointer to the start of a row-major uint8 grayscale buffer,
            width * height bytes, no stride padding.
   width, height : frame dimensions in pixels.
   result : written on every call.  classification is produced by the
            absolute-threshold decision tree on every frame, including
            during warmup (state->warmup_done == 0).  A host application
            that wishes to mimic the old "suppress alarms until baseline
            converges" behaviour can simply ignore the result while
            state->warmup_done is 0.
   Returns DEGR_OK always; DEGR_ERR reserved for future parameter checks. */
uint8_t degradation_update(DegradationState  *state,
                           const uint8_t     *frame,
                           uint16_t           width,
                           uint16_t           height,
                           DegradationResult *result);

/* Copy the baseline into buf for non-volatile storage.
   buf must be at least sizeof(DegradationBaseline) bytes.
   Only the baseline is serialised — hysteresis state is not saved
   (the system restarts detection from CLEAN on deserialise).
   Returns DEGR_ERR if buf_size < sizeof(DegradationBaseline).          */
uint8_t degradation_serialize(const DegradationState *state,
                              uint8_t                *buf,
                              uint16_t                buf_size);

/* Restore a previously serialised baseline into state.
   Resets hysteresis counters and clears the thumbnail (safe fresh start).
   warmup_done is set to 1: the restored baseline counts as converged.
   Returns DEGR_ERR if buf_size < sizeof(DegradationBaseline).          */
uint8_t degradation_deserialize(DegradationState *state,
                                const uint8_t    *buf,
                                uint16_t          buf_size);

#endif /* DEGRADATION_H */
