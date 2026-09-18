"""Smoke tests for the pure helpers — no corpus on disk required."""

import pytest

from tx.cli import cut, field, parse_row, unwrap_tool


def test_field_flattens_whitespace():
    assert field("a\tb\nc") == "a b c"


def test_parse_row_roundtrip():
    row = "\t".join(["/p/s.jsonl", "412", "2026-09-17", "user", "text\twith tab"])
    assert parse_row(row) == ("/p/s.jsonl", 412, "2026-09-17", "user", "text\twith tab")


@pytest.mark.parametrize("bad", ["", "one\ttwo", "/p/s.jsonl\tNaN\tt\tr\tx"])
def test_parse_row_rejects_non_rows(bad):
    assert parse_row(bad) is None


def test_unwrap_tool_nests_input():
    rec = unwrap_tool({"text": '[Read] {"file_path": "/tmp/x"}'})
    assert rec["tool"] == "Read"
    assert rec["input"] == {"file_path": "/tmp/x"}


def test_unwrap_tool_leaves_prose_alone():
    rec = {"text": "just prose"}
    assert unwrap_tool(rec) is rec


def test_cut_respects_width():
    assert len(cut("x" * 500, 80)) <= 80
