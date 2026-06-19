import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
import shutil
import cv2
from typing import Optional, Dict, List, Tuple, Callable


# Default storage root in cwd.
_DEFAULT_STORAGE = str(Path.cwd() / "simple_sam2_storage")


def _sanitize_label(label: str) -> str:
    """Sanitize label for safe use in filenames."""
    safe = "".join(c if c.isalnum() or c == "-" else "-" for c in label)
    return safe[:50] if safe else "object"


class SAM2Service:
    """
    A wrapper around Meta's SAM2 video predictor that adds:

    1. **Unified prompt handler** — accepts points, boxes, and masks together
       in a single call (SAM2 natively only accepts masks OR points+box).
    2. **Incremental frame segmentation** — segment N frames at a time so the
       user can inspect and correct before continuing (useful at occlusions).
    3. **Batch processing for long videos** — loads only ``batch_size`` frames
       at a time, keeping GPU/CPU memory usage constant regardless of video
       length.

    Storage layout
    --------------
    All files are written under ``storage_dir`` (default:
    ``<cwd>/simple_sam2_storage``)::

        simple_sam2_storage/
        └── <video_name>/
            ├── frames/        # extracted JPEG frames  (00000.jpg, …)
            ├── masks/         # output PNGs            (00042_1_person.png, …)
            └── tmp_batches/   # transient working dirs (auto-cleaned)

    Quick start
    -----------
    ::

        from simple_sam2 import SAM2Service

        svc = SAM2Service(cfg="sam2.1_hiera_t.yaml", ckpt="sam2.1_hiera_tiny.pt")

        info = svc.init_video("demo", video_path="clip.mp4")
        print(info["total_frames"], info["frame_dir"], info["masks_dir"])

        svc.add_prompts("demo", frame_idx=0, obj_id=1, pos_points=[[320, 240]])

        n = svc.propagate_and_save(
            "demo",
            obj_labels={"1": "person"},
            progress_callback=lambda p: print(f"{p:.1f}%"),
        )
        svc.clear_video("demo")
    """

    def __init__(
        self,
        cfg: str,
        ckpt: str,
        batch_size: int = 10,
        storage_dir: str = _DEFAULT_STORAGE,
    ):
        """
        Parameters
        ----------
        cfg : str
            Path to a SAM2 YAML config file.
            Configs live in the SAM2 repo under
            ``sam2/configs/sam2.1/``.
        ckpt : str
            Path to a SAM2 checkpoint ``.pt`` file.
            Download from https://github.com/facebookresearch/sam2#model-description
        batch_size : int
            Number of frames loaded into memory at once.
            Reduce if you run out of GPU/CPU memory. Default: 10.
        storage_dir : str
            Root directory for all simple-sam2 data.
            Defaults to ``simple_sam2_storage/`` in the current working
            directory.
        """
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.cfg = cfg
        self.ckpt = ckpt
        self.batch_size = batch_size
        self.storage_dir = storage_dir
        self._predictor = None

        self.video_metadata: Dict[str, Dict] = {}
        self.current_batch_info: Dict[str, Dict] = {}
        self.prompt_info: Dict[str, Dict] = {}

    # ── Model loading ────────────────────────────────────────────────────────

    def _get_predictor(self):
        """Lazily build and cache the SAM2 predictor (weights loaded once)."""
        if self._predictor is None:
            if not os.path.exists(self.ckpt):
                raise FileNotFoundError(
                    f"Checkpoint not found: {self.ckpt}\n"
                    "Download SAM2 weights from "
                    "https://github.com/facebookresearch/sam2#model-description"
                )
            self._predictor = build_sam2_video_predictor(
                config_file=self.cfg,
                ckpt_path=self.ckpt,
                device=self.device,
                vos_optimized=False,
            )
        return self._predictor

    # ── Mask conversion helpers ──────────────────────────────────────────────

    @staticmethod
    def _logits_to_uint8(logit_tensor: torch.Tensor) -> np.ndarray:
        """Convert SAM2 logit tensor → uint8 mask (0 or 255)."""
        mask = (logit_tensor > 0).cpu().numpy().astype("uint8") * 255
        return np.squeeze(mask)

    @staticmethod
    def _logits_to_bool(logit_tensor: torch.Tensor) -> np.ndarray:
        """Convert SAM2 logit tensor → boolean mask."""
        mask = (logit_tensor > 0).cpu().numpy().astype(bool)
        return np.squeeze(mask)

    # ── Path helpers ─────────────────────────────────────────────────────────

    def _video_dir(self, video_name: str) -> str:
        return os.path.join(self.storage_dir, video_name)

    def _frames_dir(self, video_name: str) -> str:
        return os.path.join(self._video_dir(video_name), "frames")

    def _masks_dir(self, video_name: str) -> str:
        return os.path.join(self._video_dir(video_name), "masks")

    def _tmp_batches_dir(self, video_name: str) -> str:
        return os.path.join(self._video_dir(video_name), "tmp_batches")

    # ── Batch / frame helpers ────────────────────────────────────────────────

    def _get_batch_number(self, frame_idx: int) -> int:
        return frame_idx // self.batch_size

    def _get_batch_frame_range(self, batch_num: int, total_frames: int) -> Tuple[int, int]:
        start = batch_num * self.batch_size
        end = min(start + self.batch_size, total_frames)
        return start, end

    def _get_frame_files(self, frame_dir: str) -> List[str]:
        return sorted(
            f for f in os.listdir(frame_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )

    def _count_frames(self, frame_dir: str) -> int:
        return len(self._get_frame_files(frame_dir))

    def _create_batch_folder(
        self,
        video_name: str,
        batch_num: int,
        total_frames: int,
    ) -> str:
        """
        Copy the frames for *batch_num* into a temp directory,
        renumbered from 00000 so SAM2 can load them in order.
        """
        batch_dir = os.path.join(
            self._tmp_batches_dir(video_name), f"batch_{batch_num}"
        )
        if os.path.exists(batch_dir):
            shutil.rmtree(batch_dir)
        os.makedirs(batch_dir, exist_ok=True)

        source_dir = self._frames_dir(video_name)
        start_frame, end_frame = self._get_batch_frame_range(batch_num, total_frames)
        all_frames = self._get_frame_files(source_dir)

        for i, frame_idx in enumerate(range(start_frame, end_frame)):
            if frame_idx < len(all_frames):
                src = os.path.join(source_dir, all_frames[frame_idx])
                _, ext = os.path.splitext(all_frames[frame_idx])
                dst = os.path.join(batch_dir, f"{i:05d}{ext}")
                if os.path.exists(src):
                    shutil.copy(src, dst)
                else:
                    print(f"Warning: source frame missing: {src}")

        return batch_dir

    # ── Public API ────────────────────────────────────────────────────────────

    def _extract_frames(self, video_path: str, frame_dir: str) -> int:
        """Extract frames from a video file into *frame_dir* using cv2."""
        os.makedirs(frame_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out_path = os.path.join(frame_dir, f"{frame_idx:05d}.jpg")
            cv2.imwrite(out_path, frame)
            frame_idx += 1

        cap.release()
        if frame_idx == 0:
            raise RuntimeError(f"No frames could be extracted from: {video_path}")

        print(f"Extracted {frame_idx} frames to {frame_dir}")
        return frame_idx

    def init_video(self, video_name: str, video_path: str) -> Dict:
        """
        Register a video and set up its storage layout.

        Frames are extracted automatically from *video_path* using cv2.
        No model weights are loaded yet — GPU memory is only touched when
        :meth:`add_prompts` or :meth:`propagate_and_save` is called.

        Storage layout created::

            simple_sam2_storage/
            └── <video_name>/
                ├── frames/       ← extracted frames land here
                ├── masks/        ← propagate_and_save() writes here
                └── tmp_batches/  ← transient; auto-managed

        If frames have already been extracted (e.g. from a previous run),
        extraction is skipped automatically.

        Parameters
        ----------
        video_name : str
            Unique identifier for this video used in all subsequent calls.
        video_path : str
            Path to the video file (.mp4, .avi, .mov, …).

        Returns
        -------
        dict
            ``total_frames``, ``total_batches``, ``batch_size``,
            ``frame_dir``, ``masks_dir``.

        Example
        -------
        ::

            info = svc.init_video("clip1", video_path="/data/videos/clip1.mp4")
            print(info["total_frames"])
            print(info["masks_dir"])
        """
        video_path = str(video_path)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file does not exist: {video_path}")

        # Create the canonical folder layout.
        frame_dir  = self._frames_dir(video_name)
        masks_dir  = self._masks_dir(video_name)
        tmp_dir    = self._tmp_batches_dir(video_name)
        for d in (frame_dir, masks_dir, tmp_dir):
            os.makedirs(d, exist_ok=True)

        # Extract frames (skip if already done).
        if self._count_frames(frame_dir) > 0:
            print(f"Frames already extracted at {frame_dir}, reusing.")
            total_frames = self._count_frames(frame_dir)
        else:
            print(f"Extracting frames from {video_path} …")
            total_frames = self._extract_frames(video_path, frame_dir)

        total_batches = (total_frames + self.batch_size - 1) // self.batch_size

        self.video_metadata[video_name] = {
            "frame_dir":    frame_dir,
            "masks_dir":    masks_dir,
            "total_frames": total_frames,
            "total_batches": total_batches,
        }
        self.current_batch_info[video_name] = {
            "batch_num": None,
            "state":     None,
            "batch_dir": None,
        }
        self.prompt_info[video_name] = {}

        return {
            "total_frames":  total_frames,
            "total_batches": total_batches,
            "batch_size":    self.batch_size,
            "frame_dir":     frame_dir,
            "masks_dir":     masks_dir,
        }

    def add_prompts(
        self,
        video_name: str,
        frame_idx: int,
        obj_id: int,
        pos_points: Optional[List[List[int]]] = None,
        neg_points: Optional[List[List[int]]] = None,
        box: Optional[List[int]] = None,
        binary_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Add prompts for an object at a specific frame.

        Any combination of mask, positive/negative points, and bounding box
        is accepted in a single call. The mask is applied first (SAM2
        requirement), then points and box refine it.

        Parameters
        ----------
        video_name : str
            Video registered with :meth:`init_video`.
        frame_idx : int
            Global frame index where the object first appears.
        obj_id : int
            Integer ID for this object. Use different IDs for different objects.
        pos_points : list of [x, y], optional
            Positive click points (include the object).
        neg_points : list of [x, y], optional
            Negative click points (exclude these regions).
        box : list of [x1, y1, x2, y2], optional
            Bounding box around the object.
        binary_mask : np.ndarray (H, W), optional
            Boolean or uint8 initial mask.

        Returns
        -------
        np.ndarray
            uint8 mask (0 or 255) for the object at *frame_idx*.

        Example
        -------
        ::

            mask = svc.add_prompts(
                "clip1", frame_idx=0, obj_id=1,
                pos_points=[[320, 240]],
                box=[200, 150, 450, 380],
            )
        """
        if video_name not in self.video_metadata:
            raise RuntimeError(
                f"Video '{video_name}' not initialised. Call init_video() first."
            )

        state, batch_frame_idx = self._batch_relative_frame_idx(video_name, frame_idx)
        predictor = self._get_predictor()

        self.prompt_info[video_name][obj_id] = {
            "frame_idx":   frame_idx,
            "pos_points":  pos_points,
            "neg_points":  neg_points,
            "box":         box,
            "binary_mask": binary_mask.copy() if binary_mask is not None else None,
        }

        has_points = bool(
            (pos_points and len(pos_points) > 0)
            or (neg_points and len(neg_points) > 0)
        )
        has_box  = box is not None
        has_mask = binary_mask is not None

        if not has_mask and not has_points and not has_box:
            raise ValueError(
                "Provide at least one of: binary_mask, pos_points/neg_points, or box."
            )

        with torch.inference_mode():
            if has_mask:
                _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                    inference_state=state,
                    frame_idx=batch_frame_idx,
                    obj_id=obj_id,
                    mask=binary_mask.astype(bool),
                )

            if has_points or has_box:
                all_points, all_labels = [], []
                if pos_points:
                    all_points.extend(pos_points)
                    all_labels.extend([1] * len(pos_points))
                if neg_points:
                    all_points.extend(neg_points)
                    all_labels.extend([0] * len(neg_points))

                kwargs: Dict = dict(
                    inference_state=state,
                    frame_idx=batch_frame_idx,
                    obj_id=obj_id,
                )
                if all_points:
                    kwargs["points"] = np.array(all_points, dtype=np.float32)
                    kwargs["labels"] = np.array(all_labels, dtype=np.int32)
                if box:
                    kwargs["box"] = np.array(box, dtype=np.float32)

                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(**kwargs)

        obj_index = list(out_obj_ids).index(obj_id) if obj_id in out_obj_ids else 0
        return self._logits_to_uint8(out_mask_logits[obj_index])

    def propagate_and_save(
        self,
        video_name: str,
        out_dir: Optional[str] = None,
        start_frame_idx: int = 0,
        end_frame_idx: Optional[int] = None,
        obj_labels: Optional[Dict[str, str]] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> int:
        """
        Propagate all tracked objects through the video and save masks.

        Uses batch processing so memory usage stays proportional to
        ``batch_size`` regardless of video length. Between batches, the last
        frame's mask is carried over as the initialisation prompt for the next
        batch.

        Output filenames follow the pattern::

            <frame_index>_<obj_id>_<label>.png
            # e.g. 00042_1_person.png

        Parameters
        ----------
        video_name : str
            Video registered with :meth:`init_video`.
        out_dir : str, optional
            Directory where mask PNGs are written. Defaults to
            ``simple_sam2_storage/<video_name>/masks/``.
        start_frame_idx : int
            First global frame to process. Default: 0.
        end_frame_idx : int, optional
            Last global frame (exclusive). Defaults to end of video.
        obj_labels : dict, optional
            ``{str(obj_id): "label"}`` mapping used in output filenames.
            Example: ``{"1": "person", "2": "car"}``.
        progress_callback : callable, optional
            Called with a float in 0–100 after each frame is processed.

        Returns
        -------
        int
            Total number of mask files saved.

        Example
        -------
        ::

            n = svc.propagate_and_save(
                "clip1",
                obj_labels={"1": "person", "2": "car"},
                progress_callback=lambda p: print(f"{p:.1f}%"),
            )
            print(f"Saved {n} masks")
        """
        if video_name not in self.video_metadata:
            raise RuntimeError(
                f"Video '{video_name}' not initialised. Call init_video() first."
            )
        if not self.prompt_info.get(video_name):
            raise RuntimeError(
                "No prompts added. Call add_prompts() before propagate_and_save()."
            )

        metadata  = self.video_metadata[video_name]
        predictor = self._get_predictor()

        # Default output to the canonical masks/ folder.
        if out_dir is None:
            out_dir = metadata["masks_dir"]
        out_dir = str(out_dir)
        os.makedirs(out_dir, exist_ok=True)

        obj_labels = obj_labels or {}

        total_frames   = metadata["total_frames"]
        total_batches  = metadata["total_batches"]
        end_frame_idx  = end_frame_idx if end_frame_idx is not None else total_frames
        total_to_process = max(1, end_frame_idx - start_frame_idx)
        processed_so_far = 0

        start_batch = self._get_batch_number(start_frame_idx)
        end_batch   = self._get_batch_number(min(end_frame_idx - 1, total_frames - 1))

        saved: int = 0
        last_batch_masks: Dict[int, np.ndarray] = {}
        batch_prompt_frame: int = 0

        for batch_num in range(start_batch, end_batch + 1):
            print(f"\nProcessing batch {batch_num + 1}/{total_batches}")
            batch_start, batch_end = self._get_batch_frame_range(batch_num, total_frames)
            print(f"  Global frames: {batch_start} – {batch_end - 1}")

            state = self._load_batch(video_name, batch_num)

            for obj_id, prompt_data in self.prompt_info[video_name].items():
                original_frame = prompt_data["frame_idx"]
                prompt_batch   = self._get_batch_number(original_frame)

                if batch_num == prompt_batch:
                    batch_prompt_frame = original_frame - batch_start
                    print(f"  Obj {obj_id}: original prompts at batch-frame {batch_prompt_frame}")
                    self._apply_prompts_to_state(
                        state, batch_prompt_frame, obj_id,
                        prompt_data["pos_points"],
                        prompt_data["neg_points"],
                        prompt_data["box"],
                        prompt_data["binary_mask"],
                    )

                elif batch_num > prompt_batch:
                    if obj_id in last_batch_masks:
                        print(f"  Obj {obj_id}: carried mask from previous batch → frame 0")
                        self._apply_prompts_to_state(
                            state, 0, obj_id,
                            binary_mask=last_batch_masks[obj_id],
                        )
                    else:
                        print(f"  Warning: obj {obj_id} has no carried mask, skipping.")
                        continue

            with torch.inference_mode():
                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                    state,
                    start_frame_idx=batch_prompt_frame if batch_num == start_batch else 0,
                ):
                    global_frame_idx = batch_start + out_frame_idx

                    if global_frame_idx < start_frame_idx or global_frame_idx >= end_frame_idx:
                        continue

                    processed_so_far += 1
                    if progress_callback:
                        progress_callback(
                            min(100.0, processed_so_far / total_to_process * 100.0)
                        )

                    for i, out_obj_id in enumerate(out_obj_ids):
                        mask_uint8 = self._logits_to_uint8(out_mask_logits[i])
                        label = obj_labels.get(str(out_obj_id), f"Object{out_obj_id}")
                        fname = (
                            f"{global_frame_idx:05d}"
                            f"_{out_obj_id}"
                            f"_{_sanitize_label(label)}.png"
                        )
                        Image.fromarray(mask_uint8).save(os.path.join(out_dir, fname))
                        saved += 1

                        # Store carry-over mask for the next batch.
                        if out_frame_idx == (batch_end - batch_start - 1):
                            last_batch_masks[out_obj_id] = self._logits_to_bool(
                                out_mask_logits[i]
                            )
                            print(f"  Stored carry-over mask for obj {out_obj_id}")

            print(f"  Running total: {saved} masks saved")

            if batch_num < end_batch:
                torch.cuda.empty_cache()

        print(f"\nDone. Saved {saved} masks to: {out_dir}")
        return saved

    def clear_video(self, video_name: str, delete_storage: bool = False) -> None:
        """
        Free all memory and temp files associated with a video.

        Parameters
        ----------
        video_name : str
            Video to clear.
        delete_storage : bool
            If ``True``, delete the entire video directory
            (``simple_sam2_storage/<video_name>/``) including frames, masks,
            and tmp_batches. Default: ``False``.
        """
        self.current_batch_info.pop(video_name, {})

        if delete_storage:
            video_dir = self._video_dir(video_name)
            if os.path.exists(video_dir):
                try:
                    shutil.rmtree(video_dir)
                    print(f"Deleted storage directory: {video_dir}")
                except Exception as e:
                    print(f"Warning: could not remove storage dir {video_dir}: {e}")
        else:
            # Only clean up the transient tmp_batches folder.
            tmp_dir = self._tmp_batches_dir(video_name)
            if os.path.exists(tmp_dir):
                try:
                    shutil.rmtree(tmp_dir)
                except Exception as e:
                    print(f"Warning: could not remove tmp_batches dir {tmp_dir}: {e}")

        self.video_metadata.pop(video_name, None)
        self.prompt_info.pop(video_name, None)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"Cleared '{video_name}' from memory.")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load_batch(self, video_name: str, batch_num: int):
        """Load a batch into a fresh SAM2 inference state."""
        metadata  = self.video_metadata[video_name]
        predictor = self._get_predictor()

        batch_dir   = self._create_batch_folder(
            video_name, batch_num, metadata["total_frames"]
        )
        batch_frames = self._get_frame_files(batch_dir)
        if not batch_frames:
            raise RuntimeError(f"Batch directory is empty: {batch_dir}")

        print(f"  Loading {len(batch_frames)} frames from {batch_dir}")

        with torch.inference_mode():
            state = predictor.init_state(video_path=batch_dir)
            predictor.reset_state(state)

        self.current_batch_info[video_name] = {
            "batch_num": batch_num,
            "state":     state,
            "batch_dir": batch_dir,
        }
        return state

    def _batch_relative_frame_idx(
        self, video_name: str, frame_idx: int
    ) -> Tuple[object, int]:
        """Return (state, batch_local_frame_idx), loading a new batch if needed."""
        batch_num    = self._get_batch_number(frame_idx)
        current_info = self.current_batch_info.get(video_name, {})

        if current_info.get("batch_num") != batch_num or current_info.get("state") is None:
            state = self._load_batch(video_name, batch_num)
        else:
            state = current_info["state"]

        start_frame, _ = self._get_batch_frame_range(
            batch_num, self.video_metadata[video_name]["total_frames"]
        )
        return state, frame_idx - start_frame

    def _apply_prompts_to_state(
        self,
        state,
        batch_frame_idx: int,
        obj_id: int,
        pos_points: Optional[List[List[int]]] = None,
        neg_points: Optional[List[List[int]]] = None,
        box: Optional[List[int]] = None,
        binary_mask: Optional[np.ndarray] = None,
    ) -> None:
        """Apply prompts to a batch state (used internally between batches)."""
        predictor = self._get_predictor()
        has_points = bool(
            (pos_points and len(pos_points) > 0)
            or (neg_points and len(neg_points) > 0)
        )
        has_box  = box is not None
        has_mask = binary_mask is not None

        with torch.inference_mode():
            if has_mask:
                predictor.add_new_mask(
                    inference_state=state,
                    frame_idx=batch_frame_idx,
                    obj_id=obj_id,
                    mask=binary_mask.astype(bool),
                )

            if has_points or has_box:
                all_points, all_labels = [], []
                if pos_points:
                    all_points.extend(pos_points)
                    all_labels.extend([1] * len(pos_points))
                if neg_points:
                    all_points.extend(neg_points)
                    all_labels.extend([0] * len(neg_points))

                kwargs: Dict = dict(
                    inference_state=state,
                    frame_idx=batch_frame_idx,
                    obj_id=obj_id,
                )
                if all_points:
                    kwargs["points"] = np.array(all_points, dtype=np.float32)
                    kwargs["labels"] = np.array(all_labels, dtype=np.int32)
                if box:
                    kwargs["box"] = np.array(box, dtype=np.float32)

                predictor.add_new_points_or_box(**kwargs)
