"""Bounded multi-GPU video scheduling.

Queues carry frame ranges and completion metadata only. Each persistent worker
encodes its segments to disk; no video tensors cross process boundaries.
This module deliberately has no torch/ComfyUI imports so scheduling and failure
handling can be exercised without GPUs or model weights.
"""

from dataclasses import asdict, dataclass
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty
import shutil
import subprocess
import tempfile
import traceback


@dataclass(frozen=True)
class VideoSegment:
    index: int
    start: int
    end: int
    read_start: int
    read_end: int

    @property
    def frame_count(self):
        return self.end - self.start


def plan_segments(start_frame, frame_count, fps, duration=60.0, overlap=4):
    """Plan disjoint output ranges with extra input context on both sides."""
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Video FPS must be finite and positive")
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("--segment_duration must be finite and positive")
    if start_frame < 0 or frame_count < 0 or overlap < 0:
        raise ValueError("Frame counts and --segment_overlap must be non-negative")
    segment_frames = max(1, round(fps * duration))
    end_frame = start_frame + frame_count
    return [
        VideoSegment(index, start, min(start + segment_frames, end_frame),
                     max(start_frame, start - overlap),
                     min(end_frame, start + segment_frames + overlap))
        for index, start in enumerate(range(start_frame, end_frame, segment_frames))
    ]


def segment_output_path(work_dir, index, output_format):
    suffix = ".mp4" if output_format == "mp4" else ""
    return Path(work_dir) / f"segment_{index:06d}{suffix}"


def _worker_loop(task_queue, result_queue, process_segment, config):
    state = {}  # Model cache lives for this worker's lifetime.
    while True:
        segment = task_queue.get()
        if segment is None:
            return
        try:
            count = process_segment(segment, config, state)
            if count != segment.frame_count:
                raise RuntimeError(
                    f"Segment {segment.index}: expected {segment.frame_count} frames, got {count}"
                )
            result_queue.put((segment.index, count, None))
        except BaseException:
            result_queue.put((segment.index, 0, traceback.format_exc()))
            return


def run_segment_workers(segments, devices, process_segment, config, progress=None):
    """Keep at most one task per worker in flight and detect crashed workers.

    process_segment must be a top-level, spawn-picklable callable. It receives
    (VideoSegment, config, persistent_state) and returns the written frame count.
    """
    if not segments:
        return 0
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("Specify at least one GPU, without duplicate device IDs")
    worker_count = min(len(devices), len(segments))
    context = mp.get_context("spawn")
    task_queue = context.Queue(maxsize=worker_count)
    result_queue = context.Queue(maxsize=worker_count)
    workers = []
    old_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    completed = set()
    next_task = 0
    succeeded = False
    try:
        try:
            for device in devices[:worker_count]:
                # Set before spawn: torch imports in the child must see one GPU.
                os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
                worker = context.Process(
                    target=_worker_loop,
                    args=(task_queue, result_queue, process_segment, config),
                )
                worker.start()
                workers.append(worker)
        finally:
            if old_visible_devices is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = old_visible_devices

        for segment in segments[:worker_count]:
            task_queue.put(segment)
            next_task += 1

        while len(completed) < len(segments):
            try:
                index, count, error = result_queue.get(timeout=0.5)
            except Empty:
                crashed = [p for p in workers if p.exitcode is not None]
                if crashed:
                    raise RuntimeError(
                        "GPU worker exited before completing all segments: "
                        + ", ".join(f"pid={p.pid}, exitcode={p.exitcode}" for p in crashed)
                    )
                continue
            if error:
                raise RuntimeError(f"GPU segment {index} failed:\n{error}")
            if index in completed or not 0 <= index < len(segments):
                raise RuntimeError(f"Unexpected segment completion: {index}")
            if count != segments[index].frame_count:
                raise RuntimeError(f"Incorrect frame count for segment {index}: {count}")
            completed.add(index)
            if progress:
                progress(len(completed), len(segments), index)
            if next_task < len(segments):
                task_queue.put(segments[next_task])
                next_task += 1

        for _ in workers:
            task_queue.put(None)
        for worker in workers:
            worker.join()
            if worker.exitcode != 0:
                raise RuntimeError(f"GPU worker exited with code {worker.exitcode}")
        succeeded = True
        return sum(segment.frame_count for segment in segments)
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
        for worker in workers:
            worker.join(timeout=5)
            if worker.is_alive():
                worker.kill()
                worker.join(timeout=5)
            worker.close()
        for queue in (task_queue, result_queue):
            if not succeeded:
                queue.cancel_join_thread()
            queue.close()
            if succeeded:
                queue.join_thread()


def merge_video_segments(segments, work_dir, output_path, input_path, fps, start_frame, frame_count):
    """Concatenate encoded video without re-encoding it; include selected audio."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("Multi-GPU MP4 output requires ffmpeg in PATH for segment merging")
    work_dir = Path(work_dir)
    manifest = work_dir / "concat.txt"
    # Only generated, relative filenames enter the concat file, so source/output
    # paths containing quotes, spaces, or non-ASCII characters need no escaping.
    manifest.write_text("".join(
        f"file '{segment_output_path(work_dir, s.index, 'mp4').name}'\n"
        for s in segments
    ), encoding="utf-8")
    merged = work_dir / ("merged" + (Path(output_path).suffix or ".mp4"))
    duration = frame_count / fps
    command = [
        ffmpeg, "-nostdin", "-y", "-v", "error",
        "-f", "concat", "-safe", "0", "-i", str(manifest),
        "-ss", str(start_frame / fps), "-t", str(duration), "-i", str(Path(input_path).resolve()),
        "-map", "0:v:0", "-map", "1:a?", "-c:v", "copy", "-c:a", "aac",
        "-t", str(duration), str(merged),
    ]
    # stderr goes to disk, not a pipe or an unbounded in-memory buffer.
    error_path = work_dir / "merge.stderr.log"
    with error_path.open("wb") as errors:
        result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=errors)
    if result.returncode:
        with error_path.open("rb") as errors:
            errors.seek(max(0, error_path.stat().st_size - 8192))
            detail = errors.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"FFmpeg segment merge failed ({result.returncode}): {detail}")
    os.replace(merged, output_path)


def process_multi_gpu_video(input_path, output_path, fps, start_frame, frame_count,
                            devices, args, process_segment, progress=None):
    """Plan, process, then publish disk-backed video segments or a PNG sequence."""
    segments = plan_segments(start_frame, frame_count, fps,
                             args.segment_duration, args.segment_overlap)
    if not segments:
        return 0
    output_path = Path(output_path).resolve()
    if args.output_format == "mp4" and shutil.which("ffmpeg") is None:
        raise RuntimeError("Multi-GPU MP4 output requires ffmpeg in PATH for segment merging")
    if args.output_format == "mp4" and output_path == Path(input_path).resolve():
        raise ValueError("Output video must differ from the input video")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix=".seedvr2-segments-", dir=output_path.parent))
    config = {
        "input_path": str(Path(input_path).resolve()),
        "work_dir": str(work_dir), "fps": fps, "start_frame": start_frame,
        "args": vars(args).copy(),
    }
    (work_dir / "segments.json").write_text(json.dumps({
        **config, "segments": [asdict(s) for s in segments],
    }, indent=2), encoding="utf-8")
    try:
        count = run_segment_workers(segments, devices, process_segment, config, progress)
        if args.output_format == "mp4":
            merge_video_segments(segments, work_dir, output_path, input_path,
                                 fps, start_frame, frame_count)
        else:
            output_path.mkdir(parents=True, exist_ok=True)
            for segment in segments:
                for frame in segment_output_path(work_dir, segment.index, "png").iterdir():
                    os.replace(frame, output_path / frame.name)
    except BaseException as exc:
        (work_dir / "FAILED.txt").write_text(str(exc), encoding="utf-8")
        if isinstance(exc, Exception):
            raise RuntimeError(f"{exc}\nSegment files retained at: {work_dir}") from exc
        raise
    else:
        shutil.rmtree(work_dir)
        return count
