#!/usr/bin/env python3
"""
Read REI-Bench split (data/rei_bench/splits/rei_bench.json) and write REI-Bench fields
(memory*, task_dec, reference, etc.) into ALFRED annotation files under data/raw/alfred/json_2.1.0.

Each valid_seen entry is merged into the corresponding ann file:
  <alfred_root>/<entry["task"]>/pp/ann_<entry["repeat_idx"]>.json

Usage (from project root):
    python data/rei_bench/splits/prepare_rei_bench_splits.py
    python data/rei_bench/splits/prepare_rei_bench_splits.py --input path/to/rei_bench.json --alfred-root data/raw/alfred/json_2.1.0
    python data/rei_bench/splits/prepare_rei_bench_splits.py --dry-run   # only report what would be written
"""

import argparse
import json
import logging
import os
import sys

# Default paths (relative to project root)
DEFAULT_PROJECT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
DEFAULT_INPUT_SPLIT = os.path.join(
    DEFAULT_PROJECT_ROOT, "data", "rei_bench", "splits", "rei_bench.json"
)
DEFAULT_ALFRED_ROOT = os.path.join(
    DEFAULT_PROJECT_ROOT, "data", "raw", "alfred", "json_2.1.0"
)

# Keys from split entry to merge into ann (exclude identifiers used for path resolution)
SKIP_KEYS = {"task", "repeat_idx"}
# Do not bulk-copy memory* keys; we write robot_human_memory{dt} and task_desc{dt} per dt from entry["memory{dt}"]
MEMORY_KEY_PREFIX = "memory"

# All internal data_type suffixes (RE-context: 1-1..3-3) so ann has task_desc{dt} for evaluator
INTERNAL_DATA_TYPES = ["1-1", "1-2", "1-3", "2-1", "2-2", "2-3", "3-1", "3-2", "3-3"]


def parse_memory_string_to_dialogue_and_instruction(memory_str: str):
    """
    Parse one long memory string (Alice/Robot dialogue) into:
    - human_lines: list of "Human: ..." lines only (Robot lines and empty lines removed), for robot_human_memory{dt}
    - last_sentence: final Human instruction, for task_desc{dt}
    """
    if not memory_str or not memory_str.strip():
        return [], ""
    memory_lines = memory_str.splitlines()
    human_lines = [line.replace("Alice:", "Human:") for line in memory_lines]
    human_lines = [line for line in human_lines if line.strip().startswith("Human:")]
    if not human_lines:
        return [], ""
    last_sentence = human_lines[-1].strip()
    human_lines = human_lines[:-1]
    if len(human_lines) > 5:
        print(human_lines)
    return human_lines, last_sentence


def setup_logging(verbose: bool = False) -> None:
    """Configure logging level and format."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
        stream=sys.stdout,
    )


def load_split(path: str) -> dict:
    """Load split JSON file."""
    path = os.path.normpath(os.path.abspath(path))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Split file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "valid_seen" not in data:
        raise ValueError("Split file must contain 'valid_seen' key")
    return data


def get_ann_from_traj(data: dict, repeat_idx: int):
    """
    Get the annotation dict for this repeat from loaded ALFRED JSON.
    - If file has turk_annotations.anns list, return anns[repeat_idx].
    - Else treat whole file as single ann (some ALFRED pp layouts use one ann per file).
    """
    if "turk_annotations" in data and "anns" in data["turk_annotations"]:
        anns = data["turk_annotations"]["anns"]
        if isinstance(anns, list) and len(anns) > 0:
            idx = repeat_idx if repeat_idx < len(anns) else 0
            return anns[idx]
    # Single-ann file: root is the ann
    return data


def merge_entry_into_ann(ann: dict, entry: dict) -> None:
    """
    Merge REI-Bench fields from split entry into ann (in-place).
    Mirrors legacy script: copy non-memory keys (reference, task_dec, confuse_name, etc.);
    for each data_type dt, parse entry["memory{dt}"] only and write robot_human_memory{dt} + task_desc{dt}.
    """
    # Copy only non-memory keys so each dt is written from its own entry["memory{dt}"] below
    for k, v in entry.items():
        if k in SKIP_KEYS:
            continue
        if k.startswith(MEMORY_KEY_PREFIX) and k != MEMORY_KEY_PREFIX:
            continue
        ann[k] = v
    task_dec = entry.get("task_dec") or ann.get("task_desc", "")
    if task_dec:
        ann["task_desc"] = task_dec
    # Per data_type: read only entry["memory{dt}"], parse, write robot_human_memory{dt} and task_desc{dt}
    for dt in INTERNAL_DATA_TYPES:
        key = f"memory{dt}"
        memory_str = entry.get(key, "")
        if not memory_str:
            if task_dec:
                ann[f"task_desc{dt}"] = entry.get(f"task_desc{dt}", task_dec)
            continue
        human_lines, last_sentence = parse_memory_string_to_dialogue_and_instruction(memory_str)
        ann[f"robot_human_memory{dt}"] = human_lines
        ann[f"task_desc{dt}"] = last_sentence if last_sentence else task_dec


def main() -> int:
    """Read rei_bench.json and write REI-Bench fields into ALFRED pp/ann_*.json files."""
    parser = argparse.ArgumentParser(
        description="Read REI-Bench split and write memory/task_desc etc. into ALFRED json_2.1.0 pp/ann_*.json"
    )
    parser.add_argument(
        "--input",
        "-i",
        default=DEFAULT_INPUT_SPLIT,
        help="Path to REI-Bench split JSON (valid_seen with memory*, task_dec, etc.)",
    )
    parser.add_argument(
        "--alfred-root",
        "-a",
        default=DEFAULT_ALFRED_ROOT,
        help="Root of ALFRED dataset (e.g. data/raw/alfred/json_2.1.0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only log what would be written, do not modify files",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)
    alfred_root = os.path.normpath(os.path.abspath(args.alfred_root))

    try:
        data = load_split(args.input)
        entries = data.get("valid_seen", [])
        n = len(entries)
        logging.info("Loaded %d valid_seen entries from %s", n, args.input)
        if not entries:
            logging.warning("No valid_seen entries to process.")
            return 0

        written = 0
        skipped = 0
        errors = 0

        for i, entry in enumerate(entries):
            task_path = entry.get("task")
            repeat_idx = entry.get("repeat_idx", 0)
            if not task_path:
                logging.warning("Entry %d: missing 'task', skip", i + 1)
                skipped += 1
                continue

            ann_path = os.path.join(alfred_root, task_path, "pp", f"ann_{repeat_idx}.json")
            if not os.path.isfile(ann_path):
                logging.debug("Entry %d: ann file not found %s", i + 1, ann_path)
                skipped += 1
                continue

            if args.dry_run:
                logging.info("[dry-run] would merge into %s", ann_path)
                written += 1
                continue

            try:
                with open(ann_path, "r", encoding="utf-8") as f:
                    traj = json.load(f)
                ann = get_ann_from_traj(traj, repeat_idx)
                if ann is None:
                    logging.warning("Entry %d: no turk_annotations.anns in %s", i + 1, ann_path)
                    errors += 1
                    continue
                merge_entry_into_ann(ann, entry)
                with open(ann_path, "w", encoding="utf-8") as f:
                    json.dump(traj, f, ensure_ascii=False, indent=2)
                written += 1
                if args.verbose:
                    logging.debug("Written: %s", ann_path)
            except (json.JSONDecodeError, OSError) as e:
                logging.warning("Entry %d: %s -> %s", i + 1, ann_path, e)
                errors += 1

        logging.info("Done: %d written, %d skipped (no file), %d errors", written, skipped, errors)
        return 1 if errors else 0

    except FileNotFoundError as e:
        logging.error("%s", e)
        return 1
    except ValueError as e:
        logging.error("%s", e)
        return 1
    except json.JSONDecodeError as e:
        logging.error("JSON decode error: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
