/*  test_harness.c — PC-only test runner for the degradation detector.
 *
 *  Reads raw uint8 grayscale frames from a binary file (no header, no
 *  format — just width*height bytes per frame, back-to-back), runs the
 *  detector on every 3rd frame (10 Hz on a 30 fps stream), and writes
 *  per-frame results to a CSV file.
 *
 *  Build (Windows MinGW or Linux GCC):
 *    gcc -O2 -Wall -Wextra -std=c99 -o test_harness.exe degradation.c test_harness.c
 *
 *  Usage:
 *    test_harness.exe <raw_file> <width> <height> <output.csv> [stride]
 *
 *    raw_file   : binary file of raw uint8 grayscale frames
 *    width      : frame width  in pixels
 *    height     : frame height in pixels
 *    output.csv : path for the CSV results file
 *    stride     : optional; process every Nth frame (default 3 for 30fps→10Hz)
 *
 *  Extracting raw frames from a video with ffmpeg:
 *    ffmpeg -i input.mp4 -vf "scale=320:240,format=gray" -f rawvideo raw.bin
 */

#include <stdio.h>
#include <stdlib.h>   /* malloc, free, atoi, strtoul */
#include <string.h>

#include "degradation.h"

/*--------------------------------------------------------------------------
  Classification name table — index matches DegradationClass enum values.
--------------------------------------------------------------------------*/
static const char * const CLASS_NAMES[] = {
    "CLEAN",
    "FOG",
    "FROST",
    "OBSTRUCTION",
    "UNKNOWN"
};
#define CLASS_COUNT  ((int)(sizeof(CLASS_NAMES) / sizeof(CLASS_NAMES[0])))

static const char *class_name(DegradationClass c)
{
    int idx = (int)c;
    return (idx >= 0 && idx < CLASS_COUNT) ? CLASS_NAMES[idx] : "?";
}

/*--------------------------------------------------------------------------
  Persistent detector state.  File-scope static means the linker zeroes it
  automatically — no explicit memset needed.  One instance per run.
--------------------------------------------------------------------------*/
static DegradationState g_state;

/*--------------------------------------------------------------------------
  main
--------------------------------------------------------------------------*/
int main(int argc, char *argv[])
{
    /*----------------------------------------------------------------------
      Argument parsing
    ----------------------------------------------------------------------*/
    if (argc < 5) {
        fprintf(stderr,
            "Camera Degradation Test Harness\n"
            "Usage: %s <raw_file> <width> <height> <output.csv> [frame_stride]\n"
            "\n"
            "  raw_file     : binary stream of uint8 grayscale frames\n"
            "  width        : frame width in pixels\n"
            "  height       : frame height in pixels\n"
            "  output.csv   : CSV file to write results into\n"
            "  frame_stride : process every Nth frame (default: 3)\n"
            "\n"
            "Example (30fps video → 10Hz detection):\n"
            "  %s recording.bin 320 240 results.csv 3\n"
            "\n"
            "To extract raw frames from a video (requires ffmpeg):\n"
            "  ffmpeg -i input.mp4 -vf \"scale=320:240,format=gray\""
            " -f rawvideo recording.bin\n",
            argv[0], argv[0]);
        return 1;
    }

    const char *in_path  = argv[1];
    int         w_arg    = atoi(argv[2]);
    int         h_arg    = atoi(argv[3]);
    const char *out_path = argv[4];
    int         stride   = (argc >= 6) ? atoi(argv[5]) : 3;

    if (w_arg <= 0 || w_arg > 65535) {
        fprintf(stderr, "Error: width must be 1..65535 (got %d)\n", w_arg);
        return 1;
    }
    if (h_arg <= 0 || h_arg > 65535) {
        fprintf(stderr, "Error: height must be 1..65535 (got %d)\n", h_arg);
        return 1;
    }
    if (stride <= 0) {
        fprintf(stderr, "Error: frame_stride must be >= 1 (got %d)\n", stride);
        return 1;
    }

    uint16_t width  = (uint16_t)w_arg;
    uint16_t height = (uint16_t)h_arg;

    /*----------------------------------------------------------------------
      Open files
    ----------------------------------------------------------------------*/
    FILE *fin = fopen(in_path, "rb");
    if (!fin) {
        fprintf(stderr, "Error: cannot open input file: %s\n", in_path);
        return 1;
    }

    FILE *fout = fopen(out_path, "w");
    if (!fout) {
        fprintf(stderr, "Error: cannot create output file: %s\n", out_path);
        fclose(fin);
        return 1;
    }

    /*----------------------------------------------------------------------
      Frame buffer — malloc is allowed here (PC-only file).
      The core files (degradation.h / degradation.c) never call malloc.
    ----------------------------------------------------------------------*/
    size_t frame_bytes = (size_t)width * height;
    uint8_t *frame_buf = (uint8_t *)malloc(frame_bytes);
    if (!frame_buf) {
        fprintf(stderr, "Error: cannot allocate %zu bytes for frame buffer\n",
                frame_bytes);
        fclose(fin); fclose(fout);
        return 1;
    }

    /*----------------------------------------------------------------------
      CSV header — one column per feature plus metadata columns
    ----------------------------------------------------------------------*/
    fprintf(fout,
        "frame,"
        "warmup,"
        "classification,"
        "confidence,"
        "laplacian_var,"
        "global_contrast,"
        "histogram_spread,"
        "global_mean,"
        "occlusion_score,"
        "obstruction_score,"
        "cell_mean_variance,"
        "temporal_mad,"
        "temporal_mad_var\n");

    /*----------------------------------------------------------------------
      Console header for live monitoring
    ----------------------------------------------------------------------*/
    printf("%-8s %-6s %-12s %-6s %-4s %-4s %-4s %-4s %-4s %-4s\n",
           "frame", "warmup", "class", "conf",
           "lap", "ctr", "mean", "spr", "occ", "obs");
    printf("%-8s %-6s %-12s %-6s %-4s %-4s %-4s %-4s %-4s %-4s\n",
           "--------", "------", "------------", "------",
           "----", "----", "----", "----", "----", "----");

    /*----------------------------------------------------------------------
      Frame processing loop
    ----------------------------------------------------------------------*/
    DegradationResult result;
    unsigned long frame_idx  = 0;  /* index of every frame in the file */
    unsigned long processed  = 0;  /* count of frames passed to detector */
    unsigned long detections = 0;  /* count of non-CLEAN outputs */
    size_t        n;

    while ((n = fread(frame_buf, 1u, frame_bytes, fin)) == frame_bytes) {

        /* Only pass every Nth frame to the detector. */
        if ((int)(frame_idx % (unsigned long)stride) == 0) {

            /* Capture warmup state BEFORE update so we can log it correctly.
               The update call may flip warmup_done on the 60th call.         */
            uint8_t in_warmup = g_state.warmup_done ? 0u : 1u;

            degradation_update(&g_state, frame_buf, width, height, &result);

            const DegradationFeatures *f = &result.features;
            const char *cls = class_name(result.classification);

            /* ── CSV row ─────────────────────────────────────────────── */
            fprintf(fout,
                "%lu,"           /* frame              */
                "%u,"            /* warmup             */
                "%s,"            /* classification     */
                "%u,"            /* confidence         */
                "%u,"            /* laplacian_var      */
                "%u,"            /* global_contrast    */
                "%u,"            /* histogram_spread   */
                "%u,"            /* global_mean        */
                "%u,"            /* occlusion_score    */
                "%u,"            /* obstruction_score  */
                "%u,"            /* cell_mean_variance */
                "%u,"            /* temporal_mad       */
                "%u\n",          /* temporal_mad_var   */
                frame_idx,
                (unsigned)in_warmup,
                cls,
                (unsigned)result.confidence,
                (unsigned)f->laplacian_var,
                (unsigned)f->global_contrast,
                (unsigned)f->histogram_spread,
                (unsigned)f->global_mean,
                (unsigned)f->occlusion_score,
                (unsigned)f->obstruction_score,
                (unsigned)f->cell_mean_variance,
                (unsigned)f->temporal_mad,
                (unsigned)f->temporal_mad_var);

            /* ── Console row — print every detected frame during warmup,
                  then every 10 detections to avoid flooding the terminal ─ */
            if (in_warmup || (processed % 10u == 0u)) {
                printf("%-8lu %-6s %-12s %-6u %-4u %-4u %-4u %-4u %-4u %-4u\n",
                       frame_idx,
                       in_warmup ? "yes" : "no",
                       cls,
                       (unsigned)result.confidence,
                       (unsigned)f->laplacian_var,
                       (unsigned)f->global_contrast,
                       (unsigned)f->global_mean,
                       (unsigned)f->histogram_spread,
                       (unsigned)f->occlusion_score,
                       (unsigned)f->obstruction_score);
            }

            if (result.classification != DEGRADATION_CLEAN) detections++;
            processed++;
        }

        frame_idx++;
    }

    /*----------------------------------------------------------------------
      Check if we stopped due to EOF or a partial read (corrupt file / truncated)
    ----------------------------------------------------------------------*/
    if (n != 0 && n != frame_bytes) {
        fprintf(stderr,
            "Warning: trailing %zu bytes in input (expected %zu per frame)."
            " File may be truncated.\n",
            n, frame_bytes);
    }

    /*----------------------------------------------------------------------
      Summary
    ----------------------------------------------------------------------*/
    printf("\n");
    printf("Input:          %s\n",      in_path);
    printf("Resolution:     %u x %u\n", (unsigned)width, (unsigned)height);
    printf("Frame stride:   %d\n",      stride);
    printf("Total frames:   %lu\n",     frame_idx);
    printf("Processed:      %lu\n",     processed);
    printf("Detections:     %lu\n",     detections);
    if (processed > 0) {
        printf("Detection rate: %.1f%%\n",
               100.0 * (double)detections / (double)processed);
    }
    printf("Output CSV:     %s\n",      out_path);

    free(frame_buf);
    fclose(fin);
    fclose(fout);
    return 0;
}
