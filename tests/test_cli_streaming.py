"""CPU regressions for segment scheduling and the CLI's actual video I/O.

Run: python -m unittest discover -s tests -v
Video tests additionally need torch, numpy, OpenCV and ffmpeg. The model inference
function alone is replaced by an identity operation; no model downloads occur.
"""

import argparse
import ast
import contextlib
from functools import lru_cache
import importlib.util
import json
import io
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

from src.cli.multi_gpu import (
    plan_segments, plan_gpu_segments, process_multi_gpu_video, run_segment_workers, segment_output_path,
)


def fake_segment(segment, config, state):
    time.sleep(0.1 if segment.index == 0 else 0.01)
    state["count"] = state.get("count", 0) + 1
    Path(config["work_dir"], f"{segment.index}.json").write_text(json.dumps({
        "pid": os.getpid(), "count": state["count"],
        "device": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }))
    return segment.frame_count


def synchronized_segment(segment, config, state):
    """Require every assigned GPU to start, and the previous window to be saved."""
    root = Path(config["work_dir"])
    window = segment.index // 8
    if window and not (root / f"window_{window - 1}.saved").exists():
        raise RuntimeError("Next minute started before previous minute was saved")
    (root / f"started_{segment.index}").touch()
    deadline = time.monotonic() + 10
    while not all((root / f"started_{window * 8 + i}").exists() for i in range(8)):
        if time.monotonic() > deadline:
            raise RuntimeError("Not all eight GPU workers received their task")
        time.sleep(0.01)
    return fake_segment(segment, config, state)


def failing_segment(segment, config, state):
    raise RuntimeError("test encoder failure")


def crashed_segment(segment, config, state):
    os._exit(7)


def short_segment(segment, config, state):
    return segment.frame_count - 1


class SilentDebug:
    enabled = False

    def __init__(self, **kwargs):
        pass

    def log(self, *args, **kwargs):
        pass


@lru_cache(maxsize=1)
def cli_io():
    """Load real CLI functions without importing CUDA/model/Comfy dependencies.

    Parsing only the named definitions also avoids the CLI's import-time device
    argument handling in CPU tests. Bodies are compiled unchanged from source.
    """
    import typing
    import torch
    import numpy as np
    import cv2
    path = Path(__file__).resolve().parents[1] / "inference_cli.py"
    names = {
        "FFMPEGVideoWriter", "_read_frames_from_cap", "_stream_video_chunks",
        "_save_image_bgr", "save_frames_to_video", "save_frames_to_image",
        "_process_video_segment", "parse_arguments",
    }
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if getattr(node, "name", None) in names]
    namespace = dict(vars(typing))
    import math
    import platform
    import sys
    namespace.update(
        torch=torch, np=np, cv2=cv2, os=os, Path=Path, subprocess=subprocess,
        argparse=argparse, Debug=SilentDebug, debug=SilentDebug(),
        clear_memory=lambda **kwargs: None, segment_output_path=segment_output_path,
        math=math, platform=platform, sys=sys, DEFAULT_DIT="test",
        get_available_dit_models=lambda: ["test"], SEEDVR2_FOLDER_NAME="SEEDVR2",
    )

    def identity_core(frames_tensor, args, **kwargs):
        # Synthetic prefix is deliberately distinguishable from real frames.
        prefix = torch.zeros_like(frames_tensor[:1]).repeat(args.prepend_frames, 1, 1, 1)
        return torch.cat([prefix, frames_tensor], dim=0)

    namespace["_process_frames_core"] = identity_core
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


def identity_video_segment(segment, config, state):
    return cli_io()["_process_video_segment"](segment, config, state)


class PlanningTests(unittest.TestCase):
    def test_ranges_cover_selected_frames_exactly_once(self):
        for fps in (24, 30, 29.97, 60):
            for count in (1, 7, 65, 18001):
                segments = plan_segments(17, count, fps, 1, 4)
                self.assertEqual(segments[0].start, 17)
                self.assertEqual(segments[-1].end, 17 + count)
                self.assertEqual(sum(s.frame_count for s in segments), count)
                for previous, following in zip(segments, segments[1:]):
                    self.assertEqual(previous.end, following.start)
                for s in segments:
                    self.assertLessEqual(17, s.read_start)
                    self.assertLessEqual(s.read_start, s.start)
                    self.assertLessEqual(s.end, s.read_end)
                    self.assertLessEqual(s.read_end, 17 + count)

    def test_each_minute_is_split_across_eight_gpus(self):
        windows = plan_segments(0, 9000, 30, 60, 16)
        groups = plan_gpu_segments(windows, 8, 16)
        self.assertEqual(len(groups), 5)
        self.assertTrue(all(len(group) == 8 for group in groups))
        self.assertTrue(all(s.frame_count == 225 for group in groups for s in group))
        self.assertEqual(groups[0][1].read_start, 225 - 16)
        self.assertEqual(groups[1][0].read_start, 1800 - 16)
        self.assertEqual(groups[-1][-1].read_end, 9000)
        self.assertEqual([s.index for g in groups for s in g], list(range(40)))

    def test_short_window_and_uneven_gpu_shares_preserve_frame_ranges(self):
        for count in (1, 7, 8, 19, 43):
            windows = plan_segments(17, count, 10, 2, 4)
            groups = plan_gpu_segments(windows, 8, 4)
            flattened = [s for group in groups for s in group]
            self.assertEqual([f for s in flattened for f in range(s.start, s.end)],
                             list(range(17, 17 + count)))
            for window, group in zip(windows, groups):
                self.assertEqual(len(group), min(8, window.frame_count))
                self.assertLessEqual(max(s.frame_count for s in group) -
                                     min(s.frame_count for s in group), 1)
                for shard in group:
                    self.assertGreater(shard.frame_count, 0)
                    self.assertGreaterEqual(shard.read_start, 17)
                    self.assertLessEqual(shard.read_end, 17 + count)

    def test_empty_and_invalid_parameters(self):
        self.assertEqual(plan_segments(0, 0, 30), [])
        for duration in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                plan_segments(0, 10, 30, duration)
        for fps in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                plan_segments(0, 10, fps)
        with self.assertRaises(ValueError):
            plan_segments(0, 10, 30, overlap=-1)


class WorkerTests(unittest.TestCase):
    def test_rounds_reuse_workers_and_restore_environment(self):
        segments = plan_segments(0, 70, 10, 1)
        with tempfile.TemporaryDirectory() as directory:
            progress = []
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "original"}):
                count = run_segment_workers(
                    segments, ["2", "4"], fake_segment, {"work_dir": directory},
                    lambda done, total, index: progress.append(index),
                )
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "original")
            self.assertEqual(count, 70)
            self.assertEqual(sorted(progress), list(range(7)))
            records = [json.loads(p.read_text()) for p in Path(directory).glob("*.json")]
            self.assertEqual(len(records), 14)
            self.assertEqual(len({r["pid"] for r in records}), 2)
            self.assertTrue(any(r["count"] > 1 for r in records))
            self.assertEqual({r["device"] for r in records}, {"2", "4"})

    def test_all_eight_workers_participate_in_every_round_and_reuse_state(self):
        windows = plan_segments(0, 40, 10, 2, 4)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            finalized = []

            def save_window(window, group):
                self.assertEqual(len(group), 8)
                self.assertTrue(all((root / f"{s.index}.json").exists() for s in group))
                self.assertFalse((root / f"started_{group[-1].index + 1}").exists())
                (root / f"window_{window.index}.saved").touch()
                finalized.append(window.index)

            result = run_segment_workers(
                windows, [str(i) for i in range(8)], synchronized_segment,
                {"work_dir": directory}, on_window_complete=save_window,
            )
            self.assertEqual(result, 40)
            self.assertEqual(finalized, [0, 1])
            for slot in range(8):
                first = json.loads((root / f"{slot}.json").read_text())
                second = json.loads((root / f"{slot + 8}.json").read_text())
                self.assertEqual(first["device"], str(slot))
                self.assertEqual(first["pid"], second["pid"])
                self.assertEqual((first["count"], second["count"]), (1, 2))

    def test_window_merge_failure_does_not_start_next_window(self):
        windows = plan_segments(0, 40, 10, 2)
        with tempfile.TemporaryDirectory() as directory:
            def fail_merge(window, group):
                raise RuntimeError("merge stopped")

            with self.assertRaisesRegex(RuntimeError, "merge stopped"):
                run_segment_workers(windows, ["0", "1"], fake_segment,
                                    {"work_dir": directory}, on_window_complete=fail_merge)
            self.assertEqual(sorted(p.name for p in Path(directory).glob("*.json")),
                             ["0.json", "1.json"])
            self.assertEqual(mp.active_children(), [])

    def test_worker_errors_and_hard_crashes_do_not_hang(self):
        segments = plan_segments(0, 10, 10, 1)
        for callback in (failing_segment, crashed_segment, short_segment):
            with self.subTest(callback=callback.__name__):
                started = time.monotonic()
                with self.assertRaises(RuntimeError):
                    run_segment_workers(segments, ["0", "1"], callback, {})
                self.assertLess(time.monotonic() - started, 15)
                self.assertEqual(mp.active_children(), [])

    def test_failure_retains_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(segment_duration=1, segment_overlap=2, output_format="png")
            with self.assertRaisesRegex(RuntimeError, "Segment files retained"):
                process_multi_gpu_video(
                    "input.mp4", Path(directory) / "output", 10, 0, 10,
                    ["0", "1"], args, failing_segment,
                )
            work = next(Path(directory).glob(".seedvr2-segments-*"))
            self.assertIn("test encoder failure", (work / "FAILED.txt").read_text())
            self.assertTrue((work / "segments.json").exists())


HAS_VIDEO_DEPS = all(importlib.util.find_spec(name) for name in ("cv2", "torch", "numpy"))


@unittest.skipUnless(HAS_VIDEO_DEPS, "requires torch, numpy and OpenCV")
class VideoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="seedvr2 test ' ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.io = cli_io()
        self.cv2 = self.io["cv2"]
        self.np = self.io["np"]

    def make_input(self, fps=10):
        path = self.root / "source.mkv"
        writer = self.cv2.VideoWriter(str(path), self.cv2.VideoWriter_fourcc(*"FFV1"), fps, (128, 96))
        self.assertTrue(writer.isOpened())
        for i in range(31):
            writer.write(self.np.full((96, 128, 3), 10 + i * 6, dtype=self.np.uint8))
        writer.release()
        return path

    def args(self, output_format="png", backend="opencv"):
        return argparse.Namespace(
            segment_duration=0.7, segment_overlap=2, output_format=output_format,
            chunk_size=3, batch_size=5, temporal_overlap=2, prepend_frames=4,
            cache_dit=True, cache_vae=True, debug=False, video_backend=backend, use_10bit=False,
        )

    def test_png_order_overlap_prepend_and_short_last_segment(self):
        source = self.make_input()
        target = self.root / "frames"
        count = process_multi_gpu_video(source, target, 10, 3, 23, ["0", "1"],
                                        self.args(), identity_video_segment)
        self.assertEqual(count, 23)
        paths = sorted(target.glob("*.png"))
        self.assertEqual(len(paths), 23)
        for i, path in enumerate(paths):
            self.assertEqual(path.name, f"source_{i:06d}.png")
            mean = self.cv2.imread(str(path)).mean()
            self.assertLessEqual(abs(mean - (10 + (i + 3) * 6)), 1)
        self.assertEqual(list(self.root.glob(".seedvr2-segments-*")), [])

    def test_decoder_short_read_fails_instead_of_publishing_partial_video(self):
        source = self.make_input()
        with self.assertRaisesRegex(RuntimeError, "Incomplete segment"):
            process_multi_gpu_video(source, self.root / "frames", 10, 0, 40, ["0", "1"],
                                    self.args(), identity_video_segment)
        self.assertFalse((self.root / "frames").exists())

    def test_conversion_never_materializes_a_whole_chunk_numpy_array(self):
        torch = self.io["torch"]
        frames = torch.ones((7, 32, 48, 3), dtype=torch.float16) * 0.5
        original = torch.Tensor.numpy
        shapes = []

        def frame_only(tensor):
            shapes.append(tuple(tensor.shape))
            self.assertEqual(tensor.ndim, 3)
            return original(tensor)

        with patch.object(torch.Tensor, "numpy", frame_only):
            self.io["save_frames_to_image"](frames, str(self.root / "png"), "test")
            writer = self.io["save_frames_to_video"](frames, str(self.root / "video.mp4"), 10)
            writer.release()
        self.assertEqual(len(shapes), 14)

    def test_cli_defaults_and_validation(self):
        import sys
        with patch.object(sys, "argv", ["inference_cli.py", "input.mp4"]):
            args = self.io["parse_arguments"]()
            self.assertEqual(args.segment_duration, 60)
            self.assertEqual(args.segment_overlap, 4)
        for option, value in (("--segment_duration", "nan"), ("--chunk_size", "-1")):
            with patch.object(sys, "argv", ["inference_cli.py", "input.mp4", option, value]):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    self.io["parse_arguments"]()

    def test_chunk_output_is_released_before_processing_the_next_chunk(self):
        import weakref
        cap = self.cv2.VideoCapture(str(self.make_input()))
        previous = []

        def checked_core(frames_tensor, args, **kwargs):
            self.assertTrue(all(ref() is None for ref in previous))
            result = frames_tensor.clone()
            previous.append(weakref.ref(result))
            return result

        args = self.args()
        args.prepend_frames = 0
        with patch.dict(self.io, {"_process_frames_core": checked_core}):
            stream = self.io["_stream_video_chunks"](
                cap, 31, 3, 2, args, "0", SilentDebug(), None,
            )
            count = 0
            for result in stream:
                count += result.shape[0]
                del result
            self.assertEqual(count, 31)
        cap.release()

    def test_failed_ffmpeg_encoder_is_reported(self):
        from unittest.mock import Mock
        proc = Mock()
        proc.returncode = 1
        writer = self.io["FFMPEGVideoWriter"].__new__(self.io["FFMPEGVideoWriter"])
        writer.proc = proc
        with self.assertRaisesRegex(RuntimeError, "encoder exited"):
            writer.release()
        self.assertIsNone(writer.proc)

    @unittest.skipUnless(shutil.which("ffmpeg"), "requires ffmpeg")
    def test_real_mp4_concat_with_both_encoders_and_fractional_fps(self):
        for backend in ("opencv", "ffmpeg"):
            with self.subTest(backend=backend):
                fps = 30000 / 1001
                source = self.make_input(fps)
                target = self.root / f"{backend}.mp4"
                args = self.args("mp4", backend)
                args.segment_duration = 0.2  # multiple segments, uneven tail
                count = process_multi_gpu_video(source, target, fps, 3, 23, ["0", "1"],
                                                args, identity_video_segment)
                self.assertEqual(count, 23)
                cap = self.cv2.VideoCapture(str(target))
                values = []
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    values.append(frame.mean())
                cap.release()
                self.assertEqual(len(values), 23)
                for i, value in enumerate(values):
                    self.assertLess(abs(value - (10 + (i + 3) * 6)), 6)

    @unittest.skipUnless(shutil.which("ffmpeg"), "requires ffmpeg")
    def test_selected_audio_range_and_10bit_output(self):
        source = self.make_input()
        with_audio = self.root / "with audio.mkv"
        subprocess.run([
            "ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(source),
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=3.1",
            "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "pcm_s16le", str(with_audio),
        ], check=True)
        target = self.root / "audio_10bit.mp4"
        args = self.args("mp4", "ffmpeg")
        args.use_10bit = True
        count = process_multi_gpu_video(with_audio, target, 10, 5, 13, ["0", "1"],
                                        args, identity_video_segment)
        self.assertEqual(count, 13)
        decoded = subprocess.run([
            "ffmpeg", "-nostdin", "-v", "error", "-i", str(target),
            "-map", "0:a:0", "-f", "s16le", "-ac", "1", "-ar", "48000", "-",
        ], check=True, stdout=subprocess.PIPE).stdout
        self.assertAlmostEqual(len(decoded) / (48000 * 2), 1.3, delta=0.05)
        probe = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(target)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertIn(b"yuv420p10le", probe.stderr)

    @unittest.skipUnless(shutil.which("ffmpeg"), "requires ffmpeg")
    def test_merge_failure_preserves_existing_destination(self):
        target = self.root / "existing.mp4"
        target.write_bytes(b"previous output")
        with self.assertRaisesRegex(RuntimeError, "merge failed"):
            process_multi_gpu_video(self.make_input(), target, 10, 0, 10, ["0", "1"],
                                    self.args("mp4"), fake_segment)
        self.assertEqual(target.read_bytes(), b"previous output")
        self.assertTrue(list(self.root.glob(".seedvr2-segments-*/merge.stderr.log")))


if __name__ == "__main__":
    unittest.main()
