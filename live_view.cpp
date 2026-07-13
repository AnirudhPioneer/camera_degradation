/*  live_view.cpp — Standalone C++/OpenCV live viewer for the C detector.
 *
 *  This is the C counterpart of live.py.  It is a PC-only harness: it may
 *  use OpenCV and the C++ standard library freely.  It calls the *real*
 *  embedded core (degradation.c / degradation.h) unchanged — the same code
 *  that targets the Cortex-M0 — so what you see on screen is exactly what
 *  the firmware would classify.
 *
 *  Build (MSYS2 UCRT64, OpenCV from pkg-config):
 *    g++ -O2 -Wall -Wextra -std=c++17 -o live_view.exe \
 *        live_view.cpp degradation.c $(pkg-config --cflags --libs opencv4)
 *
 *  Run:
 *    ./live_view.exe            # webcam 0
 *    ./live_view.exe 1          # webcam 1
 *    ./live_view.exe clip.mp4   # video file (loops)
 *
 *  Controls:
 *    r : reset baseline (restart warmup)
 *    q / Esc : quit
 */

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <string>

/* degradation.h is a C header; pull it in with C linkage. */
extern "C" {
#include "degradation.h"
}

/* ── Config (mirrors live.py) ──────────────────────────────────────────── */
static const int DETECTION_STRIDE = 3;    /* run detector every Nth frame */
static const int PANEL_W          = 290;  /* right-side dashboard width    */

/* Class names indexed by DegradationClass (0..4). */
static const char *CLASS_NAMES[] = {
    "CLEAN", "FOG", "FROST", "OBSTRUCTION", "UNKNOWN"
};

/* Per-class colours, OpenCV BGR order — same values as live.py. */
static cv::Scalar class_bgr(int cls)
{
    switch (cls) {
        case DEGRADATION_CLEAN:       return cv::Scalar( 71, 178,  39); /* green  */
        case DEGRADATION_FOG:         return cv::Scalar(141, 128, 127); /* grey   */
        case DEGRADATION_FROST:       return cv::Scalar(255, 185, 116); /* l.blue */
        case DEGRADATION_OBSTRUCTION: return cv::Scalar( 60,  76, 231); /* red    */
        default:                      return cv::Scalar(182,  89, 155); /* purple */
    }
}

/* ── Drawing helpers ───────────────────────────────────────────────────── */
static void put_text(cv::Mat &img, const std::string &txt, int x, int y,
                     double scale = 0.50, cv::Scalar color = cv::Scalar(255,255,255),
                     int thick = 1)
{
    cv::putText(img, txt, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX,
                scale, color, thick, cv::LINE_AA);
}

/* One feature row: label | value bar | threshold line | pass/fail dot.
   bad_if_below=true  → value below threshold is bad   (sharpness, contrast)
   bad_if_below=false → value at/above threshold is bad (brightness, counts) */
static void draw_threshold_row(cv::Mat &panel, int y, const std::string &label,
                               int value, int threshold, bool bad_if_below, int scale)
{
    const int BAR_X = 116, BAR_W = 130, BAR_H = 9;
    int bar_y = y + 2;

    value     = std::max(0, std::min(scale, value));
    threshold = std::max(0, std::min(scale, threshold));

    bool failing = bad_if_below ? (value < threshold) : (value >= threshold);

    cv::Scalar bar_color = failing ? cv::Scalar(50, 60, 210) : cv::Scalar(50, 180, 60);
    cv::Scalar dot_color = failing ? cv::Scalar(40, 40, 220) : cv::Scalar(40, 200, 40);

    /* Track */
    cv::rectangle(panel, cv::Point(BAR_X, bar_y),
                  cv::Point(BAR_X + BAR_W, bar_y + BAR_H), cv::Scalar(45,45,45), -1);

    /* Filled bar proportional to value */
    int filled = BAR_W * value / scale;
    if (filled > 0)
        cv::rectangle(panel, cv::Point(BAR_X, bar_y),
                      cv::Point(BAR_X + filled, bar_y + BAR_H), bar_color, -1);

    /* Threshold tick */
    int tx = BAR_X + BAR_W * threshold / scale;
    cv::line(panel, cv::Point(tx, bar_y - 2), cv::Point(tx, bar_y + BAR_H + 2),
             cv::Scalar(230,230,230), 1);

    /* Pass/fail dot */
    cv::circle(panel, cv::Point(BAR_X + BAR_W + 10, bar_y + BAR_H/2), 4, dot_color, -1);

    /* Labels */
    put_text(panel, label, 4, y + 10, 0.37, cv::Scalar(200,200,200));
    put_text(panel, std::to_string(value), BAR_X - 26, y + 10, 0.37, cv::Scalar(220,220,220));
    char thr[16];
    std::snprintf(thr, sizeof(thr), bad_if_below ? "<%d" : ">%d", threshold);
    put_text(panel, thr, BAR_X + BAR_W + 18, y + 10, 0.34, cv::Scalar(160,160,160));
}

/* Draw the 4×3 detection grid on *frame* (in place), shading flagged cells
   and printing each cell's mean brightness.  Mirrors live.py. */
static void draw_grid_overlay(cv::Mat &frame, const DegradationResult &res, int H, int W)
{
    cv::Mat overlay = frame.clone();

    for (int r = 0; r < DEGR_GRID_ROWS; r++) {
        int y0 =  r      * H / DEGR_GRID_ROWS;
        int y1 = (r + 1) * H / DEGR_GRID_ROWS;
        for (int c = 0; c < DEGR_GRID_COLS; c++) {
            int x0 =  c      * W / DEGR_GRID_COLS;
            int x1 = (c + 1) * W / DEGR_GRID_COLS;
            int ci = r * DEGR_GRID_COLS + c;

            bool occ = (res.cell_occ_mask >> ci) & 1u;   /* flat + dark    */
            bool obs = (res.cell_obs_mask >> ci) & 1u;   /* flat (any)     */

            if (occ)
                cv::rectangle(overlay, cv::Point(x0,y0), cv::Point(x1,y1),
                              cv::Scalar(40,40,200), -1);   /* warm red */
            else if (obs)
                cv::rectangle(overlay, cv::Point(x0,y0), cv::Point(x1,y1),
                              cv::Scalar(200,80,40), -1);   /* blue     */

            int cx = (x0 + x1)/2 - 8, cy = (y0 + y1)/2 + 4;
            put_text(frame, std::to_string(res.cell_mean[ci]), cx, cy, 0.36,
                     cv::Scalar(200,200,200), 1);
        }
    }

    cv::addWeighted(overlay, 0.30, frame, 0.70, 0.0, frame);

    for (int r = 0; r <= DEGR_GRID_ROWS; r++) {
        int y = r * H / DEGR_GRID_ROWS;
        cv::line(frame, cv::Point(0,y), cv::Point(W,y), cv::Scalar(90,90,90), 1);
    }
    for (int c = 0; c <= DEGR_GRID_COLS; c++) {
        int x = c * W / DEGR_GRID_COLS;
        cv::line(frame, cv::Point(x,0), cv::Point(x,H), cv::Scalar(90,90,90), 1);
    }
}

/* Render classification + per-feature threshold rows onto the panel. */
static void draw_dashboard(cv::Mat &panel, const DegradationResult &res,
                           long frame_no, bool warmup, int frame_count)
{
    panel.setTo(cv::Scalar(30,30,30));
    const DegradationFeatures &f = res.features;

    int cls = (int)res.classification;
    cv::Scalar bgr = class_bgr(cls);
    int y = 22;

    put_text(panel, CLASS_NAMES[cls], 8, y, 0.72, bgr, 2);
    y += 26;

    char line[64];
    if (warmup) {
        int pct = frame_count * 100 / DEGR_WARMUP_FRAMES;
        if (pct > 100) pct = 100;
        std::snprintf(line, sizeof(line), "warmup %d%%  frame %d/%d",
                      pct, frame_count, DEGR_WARMUP_FRAMES);
    } else {
        std::snprintf(line, sizeof(line), "raw:%s  conf:%u",
                      CLASS_NAMES[res.raw_classification], (unsigned)res.confidence);
    }
    put_text(panel, line, 8, y, 0.36, cv::Scalar(140,140,140));
    y += 16;

    cv::line(panel, cv::Point(4,y), cv::Point(PANEL_W-4,y), cv::Scalar(70,70,70), 1);
    y += 10;

    /* (label, value, threshold, bad_if_below, scale) — same rows as live.py */
    draw_threshold_row(panel, y, "sharpness",   f.laplacian_var,      DEGR_THR_LAP_BLUR,       true,  255); y += 18;
    draw_threshold_row(panel, y, "contrast",    f.global_contrast,    DEGR_THR_CONTRAST_LOW,   true,  255); y += 18;
    draw_threshold_row(panel, y, "spread/fog",  f.histogram_spread,   DEGR_THR_SPREAD_FOG,     true,  224); y += 18;
    draw_threshold_row(panel, y, "spread/frost",f.histogram_spread,   DEGR_THR_SPREAD_FROST,   true,  224); y += 18;
    draw_threshold_row(panel, y, "brightness",  f.global_mean,        DEGR_THR_MEAN_BRIGHT,    false, 255); y += 18;
    draw_threshold_row(panel, y, "cellvar/unif",f.cell_mean_variance, DEGR_THR_CELLVAR_UNIFORM,false,  80); y += 18;
    draw_threshold_row(panel, y, "cellvar/loc", f.cell_mean_variance, DEGR_THR_CELLVAR_LOCAL,  false,  80); y += 18;
    draw_threshold_row(panel, y, "blocked",     f.obstruction_score,  DEGR_THR_OBS_CELLS,      false,  12); y += 18;

    cv::line(panel, cv::Point(4,y), cv::Point(PANEL_W-4,y), cv::Scalar(70,70,70), 1);
    y += 8;

    std::snprintf(line, sizeof(line), "frame %ld", frame_no);
    put_text(panel, line, 8, y, 0.37, cv::Scalar(100,100,100));

    /* Grid legend */
    y += 20;
    put_text(panel, "grid:", 8, y, 0.37, cv::Scalar(160,160,80));
    y += 14;
    cv::rectangle(panel, cv::Point(8,y-8), cv::Point(18,y+2), cv::Scalar(40,40,200), -1);
    put_text(panel, "flat + dark  (occlusion)", 22, y, 0.34, cv::Scalar(180,180,180));
    y += 14;
    cv::rectangle(panel, cv::Point(8,y-8), cv::Point(18,y+2), cv::Scalar(200,80,40), -1);
    put_text(panel, "flat + bright (obstruction)", 22, y, 0.34, cv::Scalar(180,180,180));
    y += 14;
    put_text(panel, "both types -> OBSTRUCTION", 8, y, 0.34, cv::Scalar(140,140,140));
}

/* ── main ──────────────────────────────────────────────────────────────── */
int main(int argc, char *argv[])
{
    /* Parse source: an all-digit argument is a webcam index, else a path. */
    std::string arg = (argc > 1) ? argv[1] : "0";
    bool is_index = !arg.empty() &&
                    arg.find_first_not_of("0123456789") == std::string::npos;

    cv::VideoCapture cap;
    if (is_index) cap.open(std::stoi(arg));
    else          cap.open(arg);

    if (!cap.isOpened()) {
        std::fprintf(stderr, "ERROR: cannot open video source: %s\n", arg.c_str());
        return 1;
    }

    cv::Mat frame;
    if (!cap.read(frame) || frame.empty()) {
        std::fprintf(stderr, "ERROR: cannot read from video source.\n");
        return 1;
    }

    int H = frame.rows, W = frame.cols;
    if (W > 65535 || H > 65535) {
        std::fprintf(stderr, "ERROR: frame too large (%dx%d).\n", W, H);
        return 1;
    }

    std::printf("Source: %s  |  Resolution: %dx%d\n", arg.c_str(), W, H);
    std::printf("Detector runs every %d frames  (%d-frame warmup)\n",
                DETECTION_STRIDE, DEGR_WARMUP_FRAMES);
    std::printf("Controls: [r] reset baseline   [q/Esc] quit\n");

    /* Zero-initialised state == start of warmup (no init function needed). */
    DegradationState state;  std::memset(&state, 0, sizeof(state));
    DegradationResult result; std::memset(&result, 0, sizeof(result));

    cv::Mat gray, panel(H, PANEL_W, CV_8UC3, cv::Scalar(0,0,0));
    const char *win = "Camera Degradation Detector  [r=reset  q=quit]";
    long frame_no = 0;

    while (true) {
        if (!cap.read(frame) || frame.empty()) {
            if (!is_index) {                       /* loop video files */
                cap.set(cv::CAP_PROP_POS_FRAMES, 0);
                std::memset(&state, 0, sizeof(state));
                frame_no = 0;
                if (!cap.read(frame) || frame.empty()) break;
            } else {
                break;
            }
        }

        /* Run the detector every DETECTION_STRIDE frames. */
        if (frame_no % DETECTION_STRIDE == 0) {
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            if (!gray.isContinuous()) gray = gray.clone();
            degradation_update(&state, gray.data,
                               (uint16_t)W, (uint16_t)H, &result);
        }

        bool warmup = (state.warmup_done == 0);

        draw_grid_overlay(frame, result, H, W);

        if (!warmup) {
            cv::rectangle(frame, cv::Point(0,0), cv::Point(W-1, H-1),
                          class_bgr((int)result.classification), 6);
        }

        draw_dashboard(panel, result, frame_no, warmup, state.frame_count);

        cv::Mat canvas;
        cv::hconcat(frame, panel, canvas);
        cv::imshow(win, canvas);

        int key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 27 /*Esc*/) break;
        if (key == 'r') {
            std::memset(&state, 0, sizeof(state));
            frame_no = 0;
            std::printf("[reset] Baseline reset — warmup restarted.\n");
            continue;
        }
        frame_no++;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
