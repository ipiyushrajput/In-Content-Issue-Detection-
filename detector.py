#!/usr/bin/env python3
"""
In-Content Issue Detection
Monitors audio/video segment directories and detects silence, black screen, and freeze issues.

Usage:
    python detector.py [path/to/input.json]

Defaults to input.json in the current directory.
"""

import itertools
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from queue import PriorityQueue


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEGMENT_EXTENSIONS = {".ts", ".mp4"}

# Seconds the file size must remain unchanged before we consider it fully written
FILE_STABLE_DURATION = 2.0
FILE_STABLE_INTERVAL = 0.5

# Seconds between directory scans
SCAN_INTERVAL = 1.0


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path) as fh:
        cfg = json.load(fh)

    required = ["audio_directory", "video_directory", "log_directory", "error_segment_directory"]
    for key in required:
        if key not in cfg:
            raise KeyError(f"Missing required key in config: {key}")
    return cfg


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # all.log — every INFO/WARNING/ERROR/CRITICAL message
    all_log = logging.getLogger("all")
    all_log.setLevel(logging.DEBUG)

    all_fh = logging.FileHandler(os.path.join(log_dir, f"all_{timestamp}.log"))
    all_fh.setFormatter(fmt)
    all_log.addHandler(all_fh)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    all_log.addHandler(console_handler)

    # errors.log — only segments where an issue was found
    err_log = logging.getLogger("error")
    err_log.setLevel(logging.ERROR)

    err_fh = logging.FileHandler(os.path.join(log_dir, f"errors_{timestamp}.log"))
    err_fh.setFormatter(fmt)
    err_log.addHandler(err_fh)

    return all_log, err_log


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def is_segment_file(path: str) -> bool:
    return Path(path).suffix.lower() in SEGMENT_EXTENSIONS


def wait_until_file_stable(filepath: str) -> None:
    """Block until the file size has not changed for FILE_STABLE_DURATION seconds."""
    prev_size = -1
    stable_ticks = 0
    ticks_needed = int(FILE_STABLE_DURATION / FILE_STABLE_INTERVAL)

    while True:
        try:
            current_size = os.path.getsize(filepath)
        except OSError:
            time.sleep(FILE_STABLE_INTERVAL)
            continue

        if current_size == prev_size:
            stable_ticks += 1
            if stable_ticks >= ticks_needed:
                return
        else:
            stable_ticks = 0
            prev_size = current_size

        time.sleep(FILE_STABLE_INTERVAL)


def save_evidence(filepath: str, error_segment_dir: str, issue_label: str) -> str:
    """Copy the problematic segment to the evidence directory and return the destination path."""
    os.makedirs(error_segment_dir, exist_ok=True)
    stem = Path(filepath).stem
    suffix = Path(filepath).suffix
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
    dest = os.path.join(error_segment_dir, f"{stem}__{issue_label}__{timestamp}{suffix}")
    shutil.copy2(filepath, dest)
    return dest


# ---------------------------------------------------------------------------
# FFmpeg runners
# ---------------------------------------------------------------------------

def _run_ffmpeg(command: list) -> str:
    """Run an ffmpeg command and return merged stderr+stdout."""
    try:
        result = subprocess.run(
            command,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        return result.stderr + result.stdout
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found. Please install ffmpeg and ensure it is on your PATH."
        )


def detect_silence(filepath: str) -> list:
    """Return list of dicts with 'start' and 'end' (seconds) for silence intervals."""
    command = [
        "ffmpeg", "-hide_banner",
        "-i", filepath,
        "-af", "silencedetect=n=-60dB:d=10",
        "-f", "null", "-",
    ]
    output = _run_ffmpeg(command)

    starts = re.findall(r"silence_start:\s*([\d.]+)", output)
    ends   = re.findall(r"silence_end:\s*([\d.]+)", output)

    issues = []
    for i, start in enumerate(starts):
        end = float(ends[i]) if i < len(ends) else None
        issues.append({"start": float(start), "end": end})
    return issues


def detect_black_screen(filepath: str) -> list:
    """Return list of dicts with 'start' and 'end' for black-screen intervals."""
    command = [
        "ffmpeg", "-hide_banner",
        "-i", filepath,
        "-vf", "blackdetect=d=10:pix_th=.001",
        "-vsync", "2",
        "-f", "null", "-",
    ]
    output = _run_ffmpeg(command)

    starts = re.findall(r"black_start:([\d.]+)", output)
    ends   = re.findall(r"black_end:([\d.]+)", output)

    issues = []
    for i, start in enumerate(starts):
        end = float(ends[i]) if i < len(ends) else None
        issues.append({"start": float(start), "end": end})
    return issues


def detect_freeze(filepath: str) -> list:
    """Return list of dicts with 'start' and 'end' for screen-freeze intervals."""
    command = [
        "ffmpeg", "-hide_banner",
        "-i", filepath,
        "-vf", "freezedetect=n=0.01:d=10",
        "-map", "0:v:0",
        "-f", "null", "-",
    ]
    output = _run_ffmpeg(command)

    starts = re.findall(r"freeze_start:\s*([\d.]+)", output)
    ends   = re.findall(r"freeze_end:\s*([\d.]+)", output)

    issues = []
    for i, start in enumerate(starts):
        end = float(ends[i]) if i < len(ends) else None
        issues.append({"start": float(start), "end": end})
    return issues


# ---------------------------------------------------------------------------
# Segment processor
# ---------------------------------------------------------------------------

def _fmt_time(seconds) -> str:
    return f"{seconds:.3f}s" if seconds is not None else "EOF"


def process_audio_segment(filepath: str, all_log, err_log, error_segment_dir: str) -> None:
    all_log.info(f"[AUDIO] Running silence detection  -> {filepath}")
    silence_issues = detect_silence(filepath)

    if silence_issues:
        for issue in silence_issues:
            msg = (
                f"[SILENCE] {filepath} | "
                f"start={_fmt_time(issue['start'])} | end={_fmt_time(issue['end'])}"
            )
            all_log.error(msg)
            err_log.error(msg)

        dest = save_evidence(filepath, error_segment_dir, "silence")
        all_log.info(f"[AUDIO] Evidence saved -> {dest}")
    else:
        all_log.info(f"[AUDIO] No silence issues found in {filepath}")


def process_video_segment(filepath: str, all_log, err_log, error_segment_dir: str) -> None:
    all_log.info(f"[VIDEO] Running black-screen detection -> {filepath}")
    black_issues = detect_black_screen(filepath)

    all_log.info(f"[VIDEO] Running freeze detection       -> {filepath}")
    freeze_issues = detect_freeze(filepath)

    issue_labels = []

    for issue in black_issues:
        msg = (
            f"[BLACK SCREEN] {filepath} | "
            f"start={_fmt_time(issue['start'])} | end={_fmt_time(issue['end'])}"
        )
        all_log.error(msg)
        err_log.error(msg)
        issue_labels.append("blackscreen")

    for issue in freeze_issues:
        msg = (
            f"[SCREEN FREEZE] {filepath} | "
            f"start={_fmt_time(issue['start'])} | end={_fmt_time(issue['end'])}"
        )
        all_log.error(msg)
        err_log.error(msg)
        issue_labels.append("freeze")

    if not black_issues:
        all_log.info(f"[VIDEO] No black-screen issues found in {filepath}")
    if not freeze_issues:
        all_log.info(f"[VIDEO] No freeze issues found in {filepath}")

    if issue_labels:
        # Save one evidence copy labelled with all issue types found
        combined_label = "_".join(sorted(set(issue_labels)))
        dest = save_evidence(filepath, error_segment_dir, combined_label)
        all_log.info(f"[VIDEO] Evidence saved -> {dest}")


def process_segment(
    filepath: str,
    file_type: str,
    all_log,
    err_log,
    error_segment_dir: str,
) -> None:
    all_log.info(f"Processing segment [{file_type.upper()}]: {filepath}")

    if not os.path.isfile(filepath):
        all_log.warning(f"File no longer exists, skipping: {filepath}")
        return

    try:
        if file_type == "audio":
            process_audio_segment(filepath, all_log, err_log, error_segment_dir)
        elif file_type == "video":
            process_video_segment(filepath, all_log, err_log, error_segment_dir)
        else:
            all_log.warning(f"Unknown file type '{file_type}', skipping: {filepath}")
    except RuntimeError as exc:
        all_log.critical(str(exc))
        raise
    except Exception as exc:
        all_log.exception(f"Unexpected error while processing {filepath}: {exc}")

    all_log.info(f"Finished segment: {filepath}")


# ---------------------------------------------------------------------------
# Directory watcher
# ---------------------------------------------------------------------------

class DirectoryWatcher:
    """
    Polls audio and video directories for new segment files and pushes them
    onto a PriorityQueue ordered by file modification time (arrival order).
    """

    def __init__(
        self,
        audio_dir: str,
        video_dir: str,
        file_queue: PriorityQueue,
        all_log,
    ):
        self.audio_dir = audio_dir
        self.video_dir = video_dir
        self.file_queue = file_queue
        self.all_log = all_log
        self.seen: set = set()
        self._counter = itertools.count()   # tie-breaker so tuples are always comparable
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def _scan(self, directory: str, file_type: str) -> None:
        try:
            entries = sorted(
                (e for e in os.scandir(directory) if e.is_file()),
                key=lambda e: e.stat().st_mtime,
            )
        except FileNotFoundError:
            self.all_log.warning(f"Directory not found (will retry): {directory}")
            return

        for entry in entries:
            if not is_segment_file(entry.path):
                continue
            if entry.path in self.seen:
                continue

            self.seen.add(entry.path)
            mtime = entry.stat().st_mtime
            seq = next(self._counter)
            self.file_queue.put((mtime, seq, entry.path, file_type))
            self.all_log.info(
                f"Queued [{file_type.upper()}] segment "
                f"(mtime={datetime.fromtimestamp(mtime).strftime('%H:%M:%S.%f')}): "
                f"{entry.path}"
            )

    def run(self) -> None:
        while not self._stop.is_set():
            self._scan(self.audio_dir, "audio")
            self._scan(self.video_dir, "video")
            time.sleep(SCAN_INTERVAL)


# ---------------------------------------------------------------------------
# Processor thread
# ---------------------------------------------------------------------------

def processor_thread(
    file_queue: PriorityQueue,
    all_log,
    err_log,
    error_segment_dir: str,
    stop_event: threading.Event,
) -> None:
    while not stop_event.is_set() or not file_queue.empty():
        try:
            mtime, _seq, filepath, file_type = file_queue.get(timeout=1.0)
        except Exception:
            continue

        all_log.info(f"Waiting for file to finish writing: {filepath}")
        wait_until_file_stable(filepath)

        try:
            process_segment(filepath, file_type, all_log, err_log, error_segment_dir)
        except RuntimeError:
            # ffmpeg not found — fatal, stop processing
            stop_event.set()
        except Exception as exc:
            all_log.exception(f"Failed to process {filepath}: {exc}")
        finally:
            file_queue.task_done()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "input.json"

    try:
        config = load_config(config_path)
    except (FileNotFoundError, KeyError) as exc:
        print(f"[ERROR] Could not load config: {exc}", file=sys.stderr)
        sys.exit(1)

    audio_dir         = config["audio_directory"]
    video_dir         = config["video_directory"]
    log_dir           = config["log_directory"]
    error_segment_dir = config["error_segment_directory"]

    for directory in (audio_dir, video_dir, log_dir, error_segment_dir):
        os.makedirs(directory, exist_ok=True)

    all_log, err_log = setup_logging(log_dir)

    all_log.info("=" * 60)
    all_log.info("In-Content Issue Detection — starting")
    all_log.info(f"  Audio directory   : {os.path.abspath(audio_dir)}")
    all_log.info(f"  Video directory   : {os.path.abspath(video_dir)}")
    all_log.info(f"  Log directory     : {os.path.abspath(log_dir)}")
    all_log.info(f"  Error segments dir: {os.path.abspath(error_segment_dir)}")
    all_log.info("=" * 60)

    file_queue  = PriorityQueue()
    stop_event  = threading.Event()

    watcher = DirectoryWatcher(audio_dir, video_dir, file_queue, all_log)
    watcher_t = threading.Thread(target=watcher.run, name="watcher", daemon=True)
    watcher_t.start()

    proc_t = threading.Thread(
        target=processor_thread,
        args=(file_queue, all_log, err_log, error_segment_dir, stop_event),
        name="processor",
        daemon=True,
    )
    proc_t.start()

    all_log.info("Monitoring active. Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        all_log.info("Shutdown requested — draining queue…")
        watcher.stop()
        stop_event.set()
        proc_t.join(timeout=60)
        all_log.info("Shutdown complete.")


if __name__ == "__main__":
    main()
