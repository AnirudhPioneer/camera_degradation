# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Real-time camera degradation detection for an externally mounted automotive/robotic camera. Pure C99 embedded systems project targeting ARM Cortex-M0. Detection only — no image restoration. Classifies: CLEAN, FOG, FROST, DIRT, OBSTRUCTION, UNKNOWN. WATER_SMEAR is explicitly out of scope and must never appear anywhere in the code.

## Build Commands

**PC build (testing, Windows MinGW):**
```
gcc -O2 -Wall -Wextra -std=c99 -o test_harness.exe degradation.c test_harness.c
```

**ARM target build:**
```
arm-none-eabi-gcc -mcpu=cortex-m0 -mthumb -msoft-float -O2 -fno-builtin -Wall -Wextra -c degradation.c -o degradation.o
```

**Verify no floating-point in ARM binary (must produce no output):**
```
arm-none-eabi-objdump -d degradation.o | grep -E "__aeabi_fmul|__aeabi_fadd|__aeabi_fdiv|__aeabi_dmul"
```

**Check static RAM usage:**
```
arm-none-eabi-nm --size-sort -td degradation.o
```

## File Structure

| File | Role |
|------|------|
| `project_brief.md` | Full specification — source of truth |
| `degradation.h` | All public types, enums, structs, function declarations |
| `degradation.c` | Full implementation (no stubs, no TODOs) |
| `test_harness.c` | PC-only test runner: reads raw binary frames, writes CSV |
| `README.md` | Build instructions, threshold tuning, memory map, To Do |

`degradation.h` and `degradation.c` must compile for **both** targets with zero `#ifdef` — pure portable C99. Only `test_harness.c` may use `stdio.h`.

## Architecture

### Single-Pass Pipeline

All metrics computed in **one pass** over the image at spatial stride 4:

```
uint8 grayscale frame (row-major, W×H)
  └─ Single pass, stride 4
       ├─ Laplacian Variance     → laplacian_var      (uint8)
       ├─ Michelson Contrast     → global_contrast    (uint8) = (max-min)*256/(max+min+1)
       ├─ Histogram 8-bin + mean → histogram_spread   (uint8) 5th–95th percentile
       │                           global_mean        (uint8) sum/count
       ├─ Grid 4×3 = 12 cells    → occlusion_score    (uint8) dark-blob detector
       │   mean + variance/cell  → obstruction_score  (uint8) any-brightness blob
       │                           cell_mean_variance (uint8) spatial non-uniformity
       └─ 16×12 thumbnail MAD   → temporal_mad       (uint16) inter-frame EMA
                                   temporal_mad_var   (uint16) motion variability
  └─ Dual-timescale EMA baseline
       ├─ Fast EMA  α = 1/16  (right-shift 4)
       └─ Slow EMA  α = 1/256 (right-shift 8)
  └─ Ratio: ratio = (current * 128) / (baseline + 1)
  └─ Decision tree (priority order):
       1. DIRT        — occlusion_score HIGH + cell_mean_variance HIGH
       2. OBSTRUCTION — obstruction_score HIGH + cell_mean_variance HIGH + occlusion LOW
       3. FROST       — laplacian_var LOW + global_mean HIGH + histogram_spread VERY LOW + cell_mean_variance LOW
       4. FOG         — global_contrast LOW + laplacian_var LOW + global_mean NEAR BASELINE + cell_mean_variance LOW
       5. CLEAN       — none triggered
       6. UNKNOWN     — contradictory features
  └─ Hysteresis: 5 frames to enter degraded, 15 frames to clear
```

### Baseline Calibration

- 60-frame startup warmup; alarms suppressed
- After warmup: fast EMA locked, slow EMA continues
- Baseline struct must be zero-initializable (no explicit init needed)
- Serialize/deserialize functions for optional NVM persistence

## Absolute Constraints — Never Violate

- **No `float` or `double`** anywhere — not even in initialization
- **No `malloc`/`free`** — static allocation only
- **No stdlib** beyond `<stdint.h>` and `<string.h>` in `degradation.h`/`.c`
- **No `printf`/`fprintf`** in `degradation.h` or `degradation.c`
- **No `assert()`** — use explicit `uint8_t` return codes
- **No recursion**, no variable-length arrays, no `uint64_t` in hot path
- **No `#ifdef`** in `degradation.h`/`degradation.c`
- **No WATER_SMEAR** — not a classification output, not in code, not in comments
- All buffer sizes must be **compile-time constants**
- Integer overflow must be **explicitly documented and mitigated** at every risk point

## Memory Budget

Target: **< 2KB total static SRAM**

| Component | Size |
|-----------|------|
| Temporal thumbnail (16×12 uint8) | 192 bytes |
| Grid cell accumulators (12 cells) | ~120 bytes |
| Histogram (8 × uint16) | 16 bytes |
| Feature struct | ~12 bytes |
| Baseline struct | ~36 bytes |
| State machine | ~8 bytes |
| Working variables | ~40 bytes |

## Development Approach

Work **one file at a time** in this order: `degradation.h` → `degradation.c` → `test_harness.c` → `README.md`. Compile and review each file before starting the next. The user is learning C through this project — explain every design decision clearly when implementing. No stubs, no placeholders, no TODOs in delivered code.

## Validation Checklist

1. Zero GCC warnings: `-Wall -Wextra -std=c99`
2. Zero ARM warnings: `-mcpu=cortex-m0 -mthumb -msoft-float`
3. No FPU instructions in ARM disassembly
4. Static RAM < 2KB
5. Clean footage → zero false positives after 60-frame warmup
6. Fog simulation → FOG within 1 second (10 frames at 10Hz)
7. Dark covering → DIRT within 0.5 seconds (5 frames)
8. Petroleum jelly (frost proxy) → FROST within 1 second
9. Bright tape on corner → OBSTRUCTION within 0.5 seconds
