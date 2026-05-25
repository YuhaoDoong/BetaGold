"""Regression tests for v3.7.248: pytest output normalization.

Locks the normalization rules used by ``scripts/validate-patch.sh`` so that
repeated runs against unchanged code produce byte-identical normalized
output (the bit-identical reproducibility contract).
"""
from __future__ import annotations

from scripts.eval.normalize_pytest_output import normalize_line, normalize_text


def test_strips_passed_duration():
    line = "============================== 9 passed in 0.13s =============================="
    out = normalize_line(line)
    assert "0.13s" not in out
    assert "<NORMALIZED>s" in out


def test_strips_failed_duration():
    line = "============================== 2 failed in 1.234s =============================="
    out = normalize_line(line)
    assert "1.234s" not in out
    assert "<NORMALIZED>s" in out


def test_strips_repo_root():
    repo = "/Users/yhdong/GoldDash"
    line = f"rootdir: {repo}"
    assert normalize_line(line, repo_root=repo) == "rootdir: <REPO>"


def test_preserves_test_identifiers_and_status():
    """PASSED/FAILED and test IDs must not be altered."""
    line = "tests/test_foo.py::test_bar PASSED                                       [ 50%]"
    out = normalize_line(line)
    assert "PASSED" in out
    assert "test_foo.py::test_bar" in out


def test_strips_arbitrary_pytest_cache_path():
    line = "cachedir: /Users/someone-else/.pytest_cache"
    out = normalize_line(line)
    assert "/Users/someone-else/.pytest_cache" not in out
    assert "<HOME>/.pytest_cache" in out


def test_idempotent_normalization():
    """Running normalize twice produces the same output as running once."""
    text = (
        "============================== 9 passed in 0.13s =============================="
        + "\n"
        + "rootdir: /Users/yhdong/GoldDash\n"
        + "cachedir: /Users/yhdong/.pytest_cache"
    )
    once = normalize_text(text, repo_root="/Users/yhdong/GoldDash")
    twice = normalize_text(once, repo_root="/Users/yhdong/GoldDash")
    assert once == twice


def test_bit_identical_across_two_runs_with_different_durations():
    """The whole point: two runs with different real durations produce the
    same normalized output."""
    run1 = (
        "============================== 9 passed in 0.13s =============================="
        + "\n"
        + "rootdir: /Users/yhdong/GoldDash"
    )
    run2 = (
        "============================== 9 passed in 0.87s =============================="
        + "\n"
        + "rootdir: /Users/yhdong/GoldDash"
    )
    n1 = normalize_text(run1, repo_root="/Users/yhdong/GoldDash")
    n2 = normalize_text(run2, repo_root="/Users/yhdong/GoldDash")
    assert n1 == n2
