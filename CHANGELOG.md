# Changelog

## 0.1.0 (2026-06-20)

Initial release.

- `SAM2Service` wrapper around Meta SAM2 video predictor
- Unified prompt API: mix masks, positive/negative points, and bounding boxes in one call
- Batch processing: configurable `batch_size` keeps GPU memory usage constant
- Incremental propagation: process a range of frames, inspect, and continue
- Automatic frame extraction from video files via OpenCV
- Centralized storage layout under `simple_sam2_storage/<video_name>/`
- Carry-over mask mechanism to maintain object identity across batch boundaries
