# Camera Degradation Detection — Project README

## What This Project Does

This module is a real-time watchdog for an externally mounted automotive or robotic camera. It runs ~10 times per second and raises an alarm when the lens becomes unreliable due to environmental conditions. It does **detection only** — no image correction, no dehazing, no filtering.

**Outputs one of:** `CLEAN` | `FOG` | `FROST` | `OBSTRUCTION` | `UNKNOWN`

> **Note on classes.** An earlier design had a separate `DIRT` class. Dark blobs (mud, dirt) and bright/neutral blobs (tape, ice) now collapse into a single **`OBSTRUCTION`** class — the operational response is identical ("something is on the lens, clean it"). The dark-blob signature is still surfaced separately as `occlusion_score` for diagnostics. `UNKNOWN` is reserved in the enum but not currently emitted by the classifier. The smear class called out in the project brief is **out of scope** and is deliberately absent from the code.

---

## Two Implementations, One Algorithm

This repo holds a **Python prototype** (the tuning/validation ground) and a **C core** (the firmware target). They implement the *same integer arithmetic*, line for line.

| Layer | Files | Role |
|-------|-------|------|
| **Python prototype** | `detector.py`, `live.py` | Fast iteration on thresholds; live webcam dashboard via OpenCV-Python |
| **C core (portable)** | `degradation.h`, `degradation.c` | The actual firmware code — pure C99, no float, no malloc, no stdio |
| **C test harness** | `test_harness.c` | PC-only: reads raw frames, writes a CSV of features + classifications |
| **C live viewer** | `live_view.cpp` | PC-only: webcam/video → the C core → on-screen dashboard (the C twin of `live.py`) |
| **Equivalence check** | `equiv_check.py` | Proves the C core matches `detector.py` frame-for-frame |

`degradation.h`/`degradation.c` are **pure portable C99** — no `#ifdef`, no platform code, compile identically for PC and ARM. Only `test_harness.c` (and the PC-only `live_view.cpp`) use the standard library / OpenCV.

---

## Running on a Live Video Feed (C core)

`live_view.cpp` is the C counterpart of `live.py`. It opens a webcam or video file, converts each frame to grayscale, feeds it to the **unmodified** `degradation_update()` from `degradation.c`, and draws the same dashboard + 4×3 grid overlay. What you see is exactly what the firmware would classify.

### Build (MSYS2 UCRT64 — OpenCV already installed)

The core must be compiled **as C** (so its symbols stay unmangled) and the viewer **as C++**, then linked. Two-step build (warning-free):

```bash
# 1) embedded core, compiled as C
gcc -O2 -Wall -Wextra -std=c99 -c degradation.c -o degradation.o

# 2) viewer, compiled as C++ and linked against OpenCV
g++ -O2 -Wall -Wextra -std=c++17 $(pkg-config --cflags opencv4) \
    -o live_view.exe live_view.cpp degradation.o \
    $(pkg-config --libs-only-L opencv4) \
    -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio
```

> Compiling `degradation.c` with `g++` directly will C++-mangle `degradation_update`, and the `extern "C"` viewer won't find it. Always build the core with `gcc`.

### Run

```bash
./live_view.exe            # webcam 0
./live_view.exe 1          # webcam 1
./live_view.exe clip.mp4   # video file (loops)
```

Controls: **`r`** reset baseline (restart warmup) · **`q`** / **Esc** quit.

The right-hand panel shows the live classification, a per-feature bar with its threshold tick and a pass/fail dot, and a grid legend. The colored border (when not in warmup) marks the active class.

---

## Build Commands (C core / firmware)

### PC build — test harness
```bash
gcc -O2 -Wall -Wextra -std=c99 -o test_harness.exe degradation.c test_harness.c
```

### ARM target build
```bash
arm-none-eabi-gcc -mcpu=cortex-m0 -mthumb -msoft-float -O2 -fno-builtin -Wall -Wextra -c degradation.c -o degradation.o
```

### Verify no floating-point in ARM binary (must produce no output)
```bash
arm-none-eabi-objdump -d degradation.o | grep -E "__aeabi_fmul|__aeabi_fadd|__aeabi_fdiv|__aeabi_dmul"
```

### Check static RAM usage
```bash
arm-none-eabi-nm --size-sort -td degradation.o
```

---

## Test Harness & Raw Frames

The harness reads raw binary grayscale frames (no header — just width × height bytes per frame, back-to-back). Extract them from a video with ffmpeg:

```bash
ffmpeg -i input.mp4 -vf "scale=320:240,format=gray" -f rawvideo raw_frames.bin
test_harness.exe raw_frames.bin 320 240 results.csv 3   # last arg = frame stride
```

---

## Verifying the C↔Python Port

`equiv_check.py` builds a deterministic synthetic stream (clean / fog / frost / obstruction / dark), runs **both** the compiled C harness and `detector.py` on identical bytes, and diffs every feature + classification column on every frame.

```bash
py equiv_check.py
# → PASS — all 165 frames × 12 columns identical.
```

It also prints the class distribution as a quick functional sanity check (all of CLEAN/FOG/FROST/OBSTRUCTION should fire). Only `numpy` is required (no cv2).

---

## Algorithm Overview

### Single-Pass Pipeline

All 8 features are computed in **one pass** over the image at stride 4 (every 4th pixel in x and y):

```
Raw frame (uint8, row-major, W×H)
    │
    ▼  Single pass, stride 4
    ├─ Laplacian variance     → sharpness proxy            (laplacian_var)
    ├─ Michelson contrast     → (max−min)×256/(max+min+1)  (global_contrast)
    ├─ 8-bin histogram        → 5th–95th pct spread + mean (histogram_spread, global_mean)
    ├─ 4×3 grid cell stats    → occlusion_score, obstruction_score, cell_mean_variance
    └─ 16×12 thumbnail MAD    → inter-frame motion EMA      (temporal_mad)
    │
    ▼  Absolute-threshold decision tree (priority order)
       1. OBSTRUCTION — flat cells (any brightness) + spatially non-uniform
       2. FROST       — severe blur (lap<5) + bright + uniform
       3. FOG         — low contrast + moderate blur + compressed histogram + uniform + not dark
       4. CLEAN       — none of the above
    │
    ▼  Hysteresis state machine
       Enter degraded: 5 consecutive frames
       Clear to clean: 15 consecutive frames
```

A dual-timescale EMA baseline (fast α=1/16, slow α=1/256) is still maintained every frame — it backs `temporal_mad` and the NVM-persistence API and is reserved for a future relative-mode classifier — **but the current decision tree does not consult it.** Classification is purely on absolute feature values.

### Classification Logic (absolute thresholds)

```
OBSTRUCTION : obstruction_score ≥ 3  AND  cell_mean_variance ≥ 60
FROST       : laplacian_var < 5  AND  global_mean > 60  AND  cell_mean_variance < 50
FOG         : global_contrast < 40  AND  laplacian_var < 15  AND  histogram_spread < 96
              AND  cell_mean_variance < 50  AND  global_mean > 40
```

Priority matters: a near-zero Laplacian with a bright mean reads as **frost** (on-lens coating) before fog is considered; the dark-mean floor (`global_mean > 40`) stops a pitch-black scene from being mistaken for fog.

### Key Discriminating Features

| Feature | Frost | Fog | Obstruction |
|---------|-------|-----|-------------|
| `laplacian_var` | **Very LOW** (<5) | LOW (<15) | Normal in clean areas |
| `global_contrast` | Low | **LOW** (<40) | Normal |
| `histogram_spread` | Low (high end) | **Low** (<96) | Normal |
| `global_mean` | **HIGH** (>60) | Mid (>40) | Normal |
| `obstruction_score` | Low | Low | **HIGH** (≥3) |
| `cell_mean_variance` | **LOW** (uniform) | **LOW** (uniform) | **HIGH** (≥60, local) |

### Thresholds (all in `degradation.h`)

| Macro | Value | Meaning |
|-------|-------|---------|
| `DEGR_THR_LAP_FROST` | 5 | below → severe optical-path blur (frost) |
| `DEGR_THR_LAP_BLUR` | 15 | below → moderate blur (fog) |
| `DEGR_THR_CONTRAST_LOW` | 40 | below → contrast too low (fog) |
| `DEGR_THR_SPREAD_FOG` | 96 | below → histogram compressed (fog) |
| `DEGR_THR_SPREAD_FROST` | 128 | below → compressed toward bright end (frost reference) |
| `DEGR_THR_MEAN_BRIGHT` | 60 | above → scene has light (frost gate) |
| `DEGR_THR_MEAN_FLOOR` | 40 | below → scene is dark, not foggy |
| `DEGR_THR_CELLVAR_UNIFORM` | 50 | below → frame degraded evenly (fog/frost) |
| `DEGR_THR_CELLVAR_LOCAL` | 60 | at/above → degradation localised (obstruction) |
| `DEGR_THR_OBS_CELLS` | 3 | flat cells needed to call obstruction |
| `DEGR_CELL_LVAR_THR` | 8 | a cell below this internal variance is "flat" |

### Startup Warmup (informational only)

The first 60 frames build the EMA baseline. **Classification now runs from frame 1** — the warmup flag (`state.warmup_done`) is informational. A host that wants the old "suppress alarms until baseline converges" behaviour can simply ignore the result while `warmup_done == 0` (which is exactly what `live.py` / `live_view.cpp` do: they hide the colored border during warmup).

---

## Memory Budget (firmware core)

| Component | Allocation |
|-----------|-----------|
| `DegradationState.baseline` (16 × uint16) | 32 bytes |
| Temporal thumbnail (16×12 uint8) | 192 bytes |
| Hysteresis + frame counter | ~6 bytes |
| Per-call stack: grid accumulators (12 × uint32 sum + uint32 sumsq + uint16 count) | ~120 bytes |
| Per-call stack: histogram (8 × uint16) + scalars | ~30 bytes |
| **Persistent state total** | **~230 bytes** |

Target is under 2048 bytes. `DegradationResult` carries ~16 bytes of optional diagnostic overlay data (per-cell means + flat-cell masks) the live viewer uses; firmware integrators that only need the classification can ignore those fields.

---

## Threshold Tuning Guide

Thresholds are absolute, so tuning is direct: **watch the live per-feature readout** in `live_view.exe` (or `live.py`) while exposing the lens to each condition, and set each threshold between the clean and degraded readings.

1. Point the camera at a clean scene; note the resting feature values.
2. Apply each degradation (fog box, petroleum jelly for frost, tape/mud for obstruction); note how each feature moves.
3. Set the threshold roughly halfway between the clean and degraded values.
4. Confirm no false positives over several minutes of clean footage (day + night).

Because the thresholds live in `degradation.h`, the Python prototype and the C core stay in lock-step — change a value there, rebuild, re-run `equiv_check.py`.

---

## Status / Remaining Work

- [x] `degradation.h` / `degradation.c` — absolute-threshold classifier, 5 classes, zero GCC `-Wall -Wextra` warnings (PC build).
- [x] `test_harness.c` — raw-frame runner + CSV.
- [x] `live_view.cpp` — standalone C++/OpenCV live viewer.
- [x] `equiv_check.py` — C↔Python equivalence proven (165 frames identical).
- [ ] **ARM verification** — build with `arm-none-eabi-gcc`, confirm no FPU instructions, confirm static RAM < 2KB. *(Not run yet — ARM toolchain not installed on this machine.)*
- [ ] Validate against real footage for each degradation type and finalise thresholds.
