"""Command-line entry point for generation jobs."""

import argparse
import sys
from pathlib import Path

from turkish_tts_generation.config import ConfigError, load_config
from turkish_tts_generation.engine import create_default_registry
from turkish_tts_generation.pipeline import GenerationRunner


def get_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="tts-generate", description="Generate TTS artifacts from a Hugging Face dataset."
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to the generation YAML file.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Plan artifacts without loading an engine or writing audio."
    )
    parser.add_argument(
        "--force", action="store_true", help="Ignore successful manifest records and plan every sample."
    )
    return parser


def main(args: list[str] | None = None) -> int:
    """Load, validate, and execute one generation job."""
    options = get_parser().parse_args(args)
    try:
        config = load_config(options.config)
        runner = GenerationRunner(config, create_default_registry())
        summary = runner.run(dry_run=options.dry_run, force=options.force)
    except (ConfigError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(f"planned={summary.planned} succeeded={summary.succeeded} failed={summary.failed} skipped={summary.skipped}")
    return 1 if summary.failed else 0
