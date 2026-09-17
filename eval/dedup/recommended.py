"""Deprecated one-release compatibility entry point."""

from __future__ import annotations

import sys

from eval.dedup.cli import main as canonical_main


def main(argv: list[str] | None = None) -> int:
    print(
        "DEPRECATED: use `python -m eval.dedup`; this alias will be removed after v0.7.1.",
        file=sys.stderr,
    )
    return canonical_main(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    raise SystemExit(main())
