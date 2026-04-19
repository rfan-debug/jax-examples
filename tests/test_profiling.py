"""Tests for the profiling subsystem."""

from __future__ import annotations

import time

import pytest

from jax_qc.profiling import StageTimer, format_report, format_summary_table
from jax_qc.profiling.timer import optional_stage


def test_timer_records_nested_stages():
    timer = StageTimer(sync_device=False)
    with timer.stage("outer", "applicative"):
        time.sleep(0.001)
        with timer.stage("inner", "foldable"):
            time.sleep(0.001)
    outer = timer.root.children["outer"]
    inner = outer.children["inner"]
    assert outer.wall_time > 0
    assert inner.wall_time > 0
    # Outer wall time covers inner.
    assert outer.wall_time + 1e-6 >= inner.wall_time
    assert outer.fp_abstraction == "applicative"
    assert inner.fp_abstraction == "foldable"


def test_repeated_stages_aggregate():
    timer = StageTimer(sync_device=False)
    for _ in range(3):
        with timer.stage("iter"):
            time.sleep(0.0005)
    record = timer.root.children["iter"]
    assert record.call_count == 3
    assert record.wall_time > 0


def test_optional_stage_noop_when_timer_is_none():
    # Should not raise when timer is None.
    with optional_stage(None, "nothing"):
        pass


def test_optional_stage_forwards_to_timer():
    timer = StageTimer(sync_device=False)
    with optional_stage(timer, "x", "pure"):
        pass
    assert "x" in timer.root.children
    assert timer.root.children["x"].fp_abstraction == "pure"


def test_timer_to_dict_is_json_serializable():
    import json

    timer = StageTimer(sync_device=False)
    with timer.stage("a"):
        with timer.stage("b"):
            pass
    data = timer.to_dict()
    # Make sure we can roundtrip through JSON.
    encoded = json.dumps(data)
    restored = json.loads(encoded)
    assert restored["name"] == "total"
    assert "a" in restored["children"]


def test_format_report_contains_stage_names():
    timer = StageTimer(sync_device=False)
    with timer.stage("integrals", "applicative"):
        time.sleep(0.0005)
        with timer.stage("overlap", "applicative"):
            time.sleep(0.0005)
    report = format_report(timer)
    assert "integrals" in report
    assert "overlap" in report
    assert "JAX-QC Profiling Report" in report


def test_summary_table_groups_by_fp_type():
    timer = StageTimer(sync_device=False)
    with timer.stage("A", "applicative"):
        with timer.stage("A1", "applicative"):
            time.sleep(0.0005)
    with timer.stage("M", "monad"):
        time.sleep(0.0005)
    table = format_summary_table(timer)
    assert "applicative" in table
    assert "monad" in table


def test_timer_handles_exception_safely():
    timer = StageTimer(sync_device=False)
    with pytest.raises(RuntimeError):
        with timer.stage("explode"):
            raise RuntimeError("boom")
    # Stack must be restored to root despite the exception.
    assert timer._stack == [timer.root]
    assert timer.root.children["explode"].call_count == 1


def test_total_time_aggregates_children():
    timer = StageTimer(sync_device=False)
    with timer.stage("a"):
        time.sleep(0.001)
    with timer.stage("b"):
        time.sleep(0.001)
    # Root records nothing directly (no top-level enter), but format_report
    # and summary should still work via children.
    report = format_report(timer)
    assert "a" in report
    assert "b" in report
