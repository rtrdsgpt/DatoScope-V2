"""
train.py
DatoScope — Pipeline Runner

Imports and executes each pipeline step from the scripts/ package.
Every step is a registered PipelineStep that exposes a main() function.

Usage
-----
    # Run the full pipeline
    python train.py

    # Run only specific steps by number (1-indexed, space-separated)
    python train.py --steps 1 2

    # Skip specific steps
    python train.py --skip 4

    # Dry-run: print steps that would run without executing them
    python train.py --dry-run

    # Show help
    python train.py --help
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train")


# ── Make sure the project root is on sys.path so `scripts` can be imported ──
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── Pipeline step definition ─────────────────────────────────────────────────
@dataclass
class PipelineStep:
    """A single, self-contained pipeline step."""

    number: int                   # 1-indexed position in the pipeline
    module_name: str              # importable dotted name, e.g. "scripts.01_generate_data"
    description: str              # human-readable label shown in logs
    main_fn: Optional[Callable] = field(default=None, repr=False)

    # ------------------------------------------------------------------
    def load(self) -> "PipelineStep":
        """Import the module and resolve its main() function."""
        try:
            mod = importlib.import_module(self.module_name)
        except ModuleNotFoundError as exc:
            raise ImportError(
                f"Cannot import '{self.module_name}'. "
                f"Make sure the scripts/ directory exists and contains __init__.py.\n"
                f"Original error: {exc}"
            ) from exc

        if not hasattr(mod, "main"):
            raise AttributeError(
                f"Module '{self.module_name}' has no main() function. "
                "Each script must define `def main(): ...`."
            )
        self.main_fn = mod.main
        return self

    # ------------------------------------------------------------------
    def run(self) -> float:
        """Execute main() and return wall-clock seconds elapsed."""
        if self.main_fn is None:
            self.load()
        start = time.perf_counter()
        self.main_fn()
        return time.perf_counter() - start


# ── Pipeline registry ─────────────────────────────────────────────────────────
# Add or reorder steps here — the rest of train.py adapts automatically.
PIPELINE: list[PipelineStep] = [
    PipelineStep(1, "scripts.01_generate_data", "Generate synthetic datasets"),
    PipelineStep(2, "scripts.02_clean_data",    "Clean & preprocess datasets"),
    PipelineStep(3, "scripts.03_eda",           "Exploratory data analysis"),
    PipelineStep(4, "scripts.04_visualization", "Render plots & charts"),
]


# ── CLI argument parser ───────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train.py",
        description="DatoScope pipeline runner — executes scripts/ modules in order.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--steps",
        nargs="+",
        type=int,
        metavar="N",
        help="Only run these step numbers (1-indexed). Example: --steps 1 2",
    )
    p.add_argument(
        "--skip",
        nargs="+",
        type=int,
        metavar="N",
        help="Skip these step numbers. Example: --skip 4",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print which steps would run without executing any code.",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List all registered pipeline steps and exit.",
    )
    return p


# ── Step selector ─────────────────────────────────────────────────────────────
def select_steps(
    pipeline: list[PipelineStep],
    only: Optional[list[int]],
    skip: Optional[list[int]],
) -> list[PipelineStep]:
    """Return the ordered subset of steps that should run."""
    valid_numbers = {s.number for s in pipeline}

    for n in only or []:
        if n not in valid_numbers:
            log.error("--steps %d is not a valid step number. Valid: %s", n, sorted(valid_numbers))
            sys.exit(1)

    for n in skip or []:
        if n not in valid_numbers:
            log.error("--skip %d is not a valid step number. Valid: %s", n, sorted(valid_numbers))
            sys.exit(1)

    selected = [
        s for s in pipeline
        if (only is None or s.number in only)
        and (skip is None or s.number not in skip)
    ]

    if not selected:
        log.warning("No steps selected — nothing to run.")

    return selected


# ── Pretty helpers ────────────────────────────────────────────────────────────
def _bar(char: str = "─", width: int = 60) -> str:
    return char * width


def _fmt_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"


# ── Main pipeline runner ──────────────────────────────────────────────────────
def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # ── --list ───────────────────────────────────────────────────────────────
    if args.list:
        print(_bar())
        print("  DatoScope Pipeline Steps")
        print(_bar())
        for step in PIPELINE:
            print(f"  Step {step.number}: {step.description}")
            print(f"           module → {step.module_name}")
        print(_bar())
        return

    # ── select steps ─────────────────────────────────────────────────────────
    steps = select_steps(PIPELINE, getattr(args, "steps", None), getattr(args, "skip", None))

    if not steps:
        return

    # ── --dry-run ─────────────────────────────────────────────────────────────
    if args.dry_run:
        print(_bar())
        print("  DRY RUN — steps that would execute:")
        print(_bar())
        for step in steps:
            print(f"  [{step.number}] {step.description}  ({step.module_name})")
        print(_bar())
        return

    # ── pre-load all modules before running anything ──────────────────────────
    log.info("Loading %d pipeline step(s)...", len(steps))
    load_errors: list[tuple[PipelineStep, Exception]] = []
    for step in steps:
        try:
            step.load()
            log.info("  ✓ Step %d: %s", step.number, step.description)
        except (ImportError, AttributeError) as exc:
            log.error("  ✗ Step %d: %s — %s", step.number, step.description, exc)
            load_errors.append((step, exc))

    if load_errors:
        log.error(
            "%d step(s) failed to load. Fix the errors above before running the pipeline.",
            len(load_errors),
        )
        sys.exit(1)

    # ── execute ───────────────────────────────────────────────────────────────
    print()
    print(_bar("═"))
    print("  DatoScope Pipeline")
    print(_bar("═"))

    total_start = time.perf_counter()
    results: list[tuple[PipelineStep, float, Optional[Exception]]] = []

    for step in steps:
        print()
        print(_bar())
        log.info("Starting Step %d / %d — %s", step.number, len(steps), step.description)
        print(_bar())

        try:
            elapsed = step.run()
            results.append((step, elapsed, None))
            log.info("✓ Step %d complete  (%s)", step.number, _fmt_elapsed(elapsed))
        except Exception as exc:  # noqa: BLE001
            results.append((step, 0.0, exc))
            log.error("✗ Step %d FAILED — %s", step.number, exc)
            log.error("Aborting pipeline.")
            sys.exit(1)

    total_elapsed = time.perf_counter() - total_start

    # ── summary ───────────────────────────────────────────────────────────────
    print()
    print(_bar("═"))
    print("  Pipeline Summary")
    print(_bar("═"))
    for step, elapsed, err in results:
        status = "✓" if err is None else "✗"
        print(f"  {status} Step {step.number}: {step.description:<40} {_fmt_elapsed(elapsed):>7}")
    print(_bar())
    print(f"  Total time: {_fmt_elapsed(total_elapsed)}")
    print(_bar("═"))
    print()
    log.info("All %d step(s) completed successfully.", len(steps))


if __name__ == "__main__":
    main()