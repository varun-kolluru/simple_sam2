"""
simple-sam2
===========

A lightweight wrapper around Meta's SAM2 video predictor that makes
long-video segmentation practical:

- **Unified prompt API** — mix points, boxes, and masks in one call.
- **Batch processing** — only ``batch_size`` frames in GPU memory at once.
- **Incremental propagation** — segment, inspect, correct, and continue.

"""

from .service import SAM2Service

__all__ = ["SAM2Service"]
__version__ = "0.1.0"
