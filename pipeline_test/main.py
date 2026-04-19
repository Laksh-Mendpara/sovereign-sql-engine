"""CLI entrypoint for the end-to-end Text2SQL pipeline checker."""

from __future__ import annotations

import argparse
import json

from pipeline_test.config import load_settings
from pipeline_test.logging_utils import configure_logging
from pipeline_test.pipeline import PipelineChecker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full Text2SQL pipeline check for one NL query.")
    parser.add_argument("query", help="Natural-language query to evaluate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = configure_logging()
    settings = load_settings()
    checker = PipelineChecker(settings, logger)
    try:
        result = checker.run(args.query)
    finally:
        checker.close()

    print(json.dumps(result.as_dict(), indent=2))


if __name__ == "__main__":
    main()
