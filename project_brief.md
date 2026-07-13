I am developing a real-time camera degradation detection system for an externally mounted 
automotive/robotic vision camera operating in uncontrolled outdoor environments. This is 
an embedded systems project with extremely severe hardware constraints.

## Project Goal

Detect when a camera feed becomes unreliable due to environmental or optical degradation.
This is DETECTION ONLY — not image restoration or dehazing.
Optionally classify the degradation type at a coarse level.

## Degradation Types to Detect

Priority order — the system must be highly reliable on the first three:

1. [PRIMARY] Lens frosting and condensation — large blurred translucent regions, 
   bright veiling glare, halos around light sources, severe and spatially uneven blur, 
   elevated mean brightness, collapsed histogram toward high end
   
2. [PRIMARY] Mud and dirt accumulation — opaque dark blobs, localized occlusion, 
   partial blockage, spatially confined regions with near-zero local variance, 
   mean intensity significantly below neighboring cells
   
3. [PRIMARY] Atmospheric fog and haze — global contrast reduction, edge loss, 
   desaturation, visibility reduction, uniform low-frequency response across 
   the whole frame, moderate histogram compression toward mid-tones

4. [SECONDARY] Partial lens obstruction — similar signature to dirt but the 
   occluding material may not be dark (e.g. ice, sticker, tape). Detected via 
   local variance collapse without the dark-mean signature of mud.

5. [OUT OF SCOPE] Water droplets and dynamic smearing — explicitly excluded.
   Do not implement detection logic for this case. Do not add placeholder code 
   or comments suggesting future addition. The classifier should never output 
   WATER_SMEAR.

6. [OUT OF SCOPE] Dust film — explicitly excluded. Fine dust produces effects 
   too subtle and too similar to mild fog to distinguish reliably at this compute 
   budget. Treat any mild global degradation as fog.

## Distinguishing the Three Primary Types

The classifier must correctly separate frost, dirt, and fog from each other 
and from clean scenes. Key discriminating features:

Frost vs Fog:
- Frost elevates mean brightness (bright veiling), fog keeps mid-tones
- Frost collapses the histogram toward the HIGH end
- Fog collapses the histogram toward the MIDDLE
- Frost creates spatially uneven blur (worse at edges/corners where condensation 
  forms first), fog degrades uniformly across the frame
- Frost: histogram_spread very low AND mean high
- Fog: histogram_spread low AND mean mid-range

Frost vs Dirt:
- Frost is bright, dirt is dark — opposite mean intensity signatures
- Frost degrades the whole frame, dirt is spatially localized
- Frost: occlusion_score low (no single dark blob), global sharpness collapsed
- Dirt: occlusion_score high (localized dark cells), global sharpness may be 
  normal in unaffected regions

Fog vs Dirt:
- Fog is global and uniform, dirt is local and spatially bounded
- Fog: all grid cells degrade similarly, low cell-to-cell variance of means
- Dirt: high cell-to-cell variance of means (some cells dark, others normal)
- Dirt: affected cells have low internal variance (opaque blob, no texture)
- Fog: all cells have similarly reduced (but nonzero) internal variance

Clean vs All:
- All three degradations reduce laplacian_var relative to baseline
- Clean scenes may have low texture but temporal MAD will be variable 
  (scene content changes); degraded scenes have low temporal MAD variance 
  (the degradation locks the frame texture)

## Hardware Constraints — These Are Absolute

- Target MCU: Cortex-M0 class (e.g. STM32F0 series or equivalent)
- The program will run on a camera ISP that has a cortex M0
- No FPU — all arithmetic must be integer or fixed-point only
- No GPU, no NPU, no DSP extensions
- No dynamic memory allocation — all buffers must be statically allocated
- Very limited SRAM — target under 2KB total for the entire detection module
- MCU is handling other tasks simultaneously — compute budget is a fraction of one core
- Real-time operation required — must complete per-frame processing well within frame 
  budget (target 10Hz detection rate on 30fps stream, i.e. process every 3rd frame)
- No floating point anywhere — not even in initialization code
- No standard library dependencies beyond stdint.h and string.h

## Algorithm Design

The detection pipeline uses a single-pass approach over a spatially subsampled frame.
All of the following metrics are computed in ONE pass over the image:

1. Laplacian Variance (sharpness proxy)
   - 3x3 Laplacian kernel [0,1,0,1,-4,1,0,1,0]
   - Spatial stride of 4 (sample every 4th pixel in x and y)
   - Accumulate sum and sum-of-squares for variance computation
   - Needs row y-1 and y+1 — these are direct indexed reads, not a separate pass

2. Michelson Contrast
   - Track global min and max over subsampled pixels
   - Result = (max-min)*256 / (max+min+1), stored as uint8

3. Histogram Spread with Mean Tracking
   - 8-bin histogram of pixel intensities (shift pixel value right by 5)
   - Find 5th and 95th percentile bins after accumulation
   - Also track global mean intensity (sum / count, integer divide)
   - Mean is critical for frost vs fog discrimination
   - Result: spread as uint8 (0..255), mean as uint8

4. Grid Cell Statistics (occlusion and spatial uniformity detection)
   - Divide frame into 4x3 = 12 cells
   - Per cell: compute mean intensity and local variance
   - Occlusion score: count cells with low internal variance AND dark mean 
     relative to cell median (dirt/mud signature)
   - Obstruction score: count cells with low internal variance regardless of 
     brightness (catches bright obstructions like ice or tape)
   - Cell mean variance: variance of the 12 cell means — high value means 
     spatially non-uniform (dirt), low value means uniform degradation (fog/frost)
   - All three outputs used in classification

5. Temporal Thumbnail (inter-frame consistency)
   - Maintain a 16x12 = 192-byte thumbnail of previous frame
   - Compute Mean Absolute Difference (MAD) between current and previous thumbnail
   - Track EMA of MAD and variance of MAD using alpha=1/8 (shift-friendly)
   - Thumbnail downsampling folded into the main spatial pass (no second pass)
   - Low MAD + low MAD variance = scene texture locked = consistent with degradation
   - Note: a stationary camera on a static scene also produces low MAD — 
     the baseline and relative thresholding handle this (see below)

6. Dual-timescale Baseline
   - Fast EMA (alpha=1/16) tracks recent scene statistics
   - Slow EMA (alpha=1/256) tracks long-term reference
   - Ratio of fast/slow detects gradual degradation (slow frosting, accumulating dirt)
   - All baselines stored as uint16 to preserve resolution through EMA shifts
   - Startup warmup: 60 frames before alarms are enabled

## Feature Vector

After the single pass, 8 features are available:
- laplacian_var      : uint8  (sharpness — drops for fog and frost, 
                                normal in unaffected regions for dirt)
- global_contrast    : uint8  (Michelson contrast — drops for fog and frost)
- histogram_spread   : uint8  (dynamic range — drops for all degradations)
- global_mean        : uint8  (mean brightness — HIGH for frost, MID for fog, 
                                LOW/normal for dirt)
- occlusion_score    : uint8  (dark-blob grid detector — HIGH for dirt only)
- obstruction_score  : uint8  (any-brightness blob detector — HIGH for dirt 
                                and bright obstructions)
- cell_mean_variance : uint8  (spatial non-uniformity — HIGH for dirt, 
                                LOW for fog and frost)
- temporal_mad       : uint16 (inter-frame motion EMA)
- temporal_mad_var   : uint16 (motion variability — used as clean-scene 
                                confirmation, not primary degradation signal)

## Classification Logic

Hard-coded decision tree using relative thresholds against the bootstrapped baseline.
Classification outputs: CLEAN, FOG, FROST, DIRT, OBSTRUCTION, UNKNOWN.
WATER_SMEAR is not an output — do not include it anywhere in the code.
Confidence score output as uint8 (0..255).
Hysteresis state machine: 5 frames to enter degraded state, 15 frames to clear.

Decision order (test in this sequence to resolve ambiguities):

1. DIRT first — strongest and most localized signal
   Condition: occlusion_score high AND cell_mean_variance high
   (localized dark regions + spatial non-uniformity)

2. OBSTRUCTION second — localized but not necessarily dark
   Condition: obstruction_score high AND cell_mean_variance high 
   AND occlusion_score low (bright or neutral obstruction, not dark dirt)

3. FROST third — global blur + high mean + collapsed histogram high end
   Condition: laplacian_var low relative to baseline 
   AND global_mean high relative to baseline
   AND histogram_spread very low
   AND cell_mean_variance low (uniform degradation)

4. FOG fourth — global blur + mid-range mean + moderate histogram compression
   Condition: global_contrast low relative to baseline
   AND laplacian_var low relative to baseline
   AND global_mean near baseline mean (not elevated)
   AND cell_mean_variance low (uniform degradation)

5. CLEAN — none of the above triggered
6. UNKNOWN — reserved for contradictory feature combinations

All thresholds are expressed as ratios relative to the per-feature baseline 
(not absolute values) to handle scene-dependent variation.

## Relative Threshold Implementation

Do not use absolute threshold values in the classifier.
Instead compute a ratio for each feature:
  ratio = (current_value * 128) / (baseline_value + 1)
A ratio of 128 means no change. Below 128 means decrease. Above 128 means increase.
All comparisons are then against fixed ratio thresholds (e.g. ratio < 64 
means the feature dropped to less than 50% of baseline).
This makes the detector robust to lighting changes, scene content, 
and time-of-day variation without any per-deployment manual tuning.

## Baseline Calibration

- At startup: 60-frame warmup window, alarms suppressed during warmup
- After warmup: baseline locked for fast EMA, slow EMA continues updating
- Dual-timescale EMA detects gradual degradation by comparing fast vs slow
- Baseline struct must be fully initializable from zero (no runtime init 
  function required beyond feeding the first 60 frames)
- Optional: expose a function to serialize/deserialize baseline to a 
  byte array for non-volatile storage — caller handles actual NVM write

## Memory Budget Target

Total RAM for the entire module: under 2KB
Key allocations:
- Temporal thumbnail buffer: 192 bytes (16x12 uint8)
- Grid cell accumulators: ~96 bytes (12 cells x 2 x uint32 for sum/sq + uint16 count)
- Histogram: 16 bytes (8 x uint16)
- Feature struct: 12 bytes
- Baseline struct: ~24 bytes
- State machine: a few bytes
- Working variables: remainder

## Code Requirements

- Language: C99, pure C, no C++
- No dynamic allocation (no malloc/free anywhere)
- No floating point (no float, no double, not even a cast to float)
- No recursion
- Integer overflow must be explicitly managed — document every place where 
  overflow is possible and how it is handled
- All buffers statically allocated with explicit sizes
- Every function must have a clear contract: inputs, outputs, side effects, 
  assumptions about image format
- Image input format: uint8 grayscale, row-major, width x height
- All public types and functions in degradation.h
- Implementation in degradation.c
- A separate test harness in test_harness.c that:
  - Reads raw grayscale frames from binary files (simple fread)
  - Runs the detection pipeline
  - Writes per-frame feature vectors and classification results to a CSV file
  - Compiles and runs on Windows (MSVC or MinGW GCC)
  - Has no OpenCV or external dependencies

## Development Environment

- Development machine: Windows
- Target: ARM Cortex-M0 (arm-none-eabi-gcc)
- PC build for testing: MinGW GCC or MSVC
- The core files (degradation.h, degradation.c) must compile cleanly for BOTH targets
  without any #ifdef — they are pure portable C99
- The test harness (test_harness.c) is PC-only and may use stdio.h for file I/O
- Compile flags for ARM target:
    -mcpu=cortex-m0 -mthumb -msoft-float -O2 -fno-builtin -Wall -Wextra
- Compile flags for PC test:
    -O2 -Wall -Wextra -std=c99

## File Structure to Produce

degradation.h        — all public types, enums, structs, function declarations
degradation.c        — full implementation
test_harness.c       — PC-side test runner (reads raw frames, writes CSV)
README.md            — build instructions for both PC and ARM,
                       threshold tuning guide, memory map

## What I Will Test Against

1. Compile cleanly on PC with GCC -Wall -Wextra -std=c99 with zero warnings
2. Compile cleanly with arm-none-eabi-gcc -mcpu=cortex-m0 -msoft-float with 
   zero warnings
3. Run the test harness against raw grayscale video frames extracted from 
   real outdoor camera footage
4. Verify no floating point instructions appear in the ARM disassembly 
   (objdump grep for __aeabi_fmul, __aeabi_fadd, __aeabi_fdiv, __aeabi_dmul)
5. Verify total static RAM usage is under 2KB (map file check)
6. Verify correct detection on labeled test clips:
   - Clean outdoor footage (day and night): zero false positives
   - Spray-bottle fog simulation in front of lens: FOG within 1 second
   - Partial dark covering of lens: DIRT within 0.5 seconds
   - Lens coated with petroleum jelly or similar (frost proxy): FROST within 1 second
   - Bright tape covering corner of lens: OBSTRUCTION within 0.5 seconds

## Explicit Exclusions — Do Not Implement

- No WATER_SMEAR detection or classification output
- No dust detection — subsume mild global degradation into FOG
- No per-pixel operations that require the full unstrided image
- No lookup tables larger than 256 bytes
- No trigonometric or logarithmic functions
- No sqrt — use squared comparisons where distance metrics are needed
- No printf or fprintf inside degradation.h or degradation.c
- No assert() in production code — use explicit uint8 return codes
- No variable-length arrays — all sizes compile-time constants
- No uint64_t in the hot path — M0 has no 64-bit multiply, it will be emulated

## First Task

Produce all four files in full, but go one by one. We will have a file, compile it and see how it works and spend sometime to understand it before moving on to the next one. C is not my language of choice but I have to work with it here, so we will work slow, step by step and understand why everything is being done. Do not produce stubs or placeholders.
Do not add TODOs. Every function must be completely implemented.
Start with degradation.h, then degradation.c, then test_harness.c, then README.md.

After producing all files provide a summary of:
- Actual static RAM usage broken down by variable
- Every integer overflow risk identified and how it was mitigated
- Any deviations from this specification and the reason for each deviation
- The specific ratio threshold values chosen for each classifier branch 
  and the reasoning behind each value