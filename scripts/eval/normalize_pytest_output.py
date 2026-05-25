"""v3.7.248: pytest output normalization for bit-identical reproducibility.

Used by both ``scripts/validate-patch.sh`` (via pipe) and the
``tests/test_validate_patch.py`` regression tests.

Normalizations applied:
  * ``passed in N.Ns`` / ``failed in N.Ns`` → ``passed in <NORMALIZED>s``
    (pytest run-duration timing is non-deterministic).
  * Absolute repo path → ``<REPO>``.
  * Arbitrary ``/Users/<x>/.pytest_cache`` paths → ``<HOME>/.pytest_cache``.

Lines containing PASSED / FAILED / test identifiers are preserved verbatim.
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def normalize_line(line: str, repo_root: str = str(REPO_ROOT)) -> str:
    line = re.sub(r"passed in \d+\.\d+s", "passed in <NORMALIZED>s", line)
    line = re.sub(r"failed in \d+\.\d+s", "failed in <NORMALIZED>s", line)
    line = line.replace(repo_root, "<REPO>")
    line = re.sub(r"/Users/[^ /]*/\.pytest_cache", "<HOME>/.pytest_cache", line)
    return line


def normalize_text(text: str, repo_root: str = str(REPO_ROOT)) -> str:
    return "\n".join(normalize_line(ln, repo_root) for ln in text.splitlines())


def main():
    text = sys.stdin.read()
    sys.stdout.write(normalize_text(text))


if __name__ == "__main__":
    main()
