"""Pretty-print profiling reports.

The timing tree produced by ``StageTimer`` is rendered as a text table
with indentation conveying nesting. A flat summary grouped by FP
abstraction is also available.

FP: Pure rendering — takes a timer snapshot, returns a string.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Tuple

if TYPE_CHECKING:
    from jax_qc.profiling.timer import StageRecord, StageTimer


def _ms(seconds: float) -> float:
    return seconds * 1e3


def _collect_rows(
    record: "StageRecord",
    depth: int,
    rows: List[Tuple[int, "StageRecord"]],
    min_percent: float,
    total_seconds: float,
) -> None:
    if total_seconds > 0 and depth > 0:
        pct = 100.0 * record.wall_time / total_seconds
        if pct < min_percent:
            return
    rows.append((depth, record))
    for child in record.children.values():
        _collect_rows(child, depth + 1, rows, min_percent, total_seconds)


def format_report(timer: "StageTimer", min_percent: float = 0.0) -> str:
    """Render the timing tree as a text report.

    Args:
        timer:       the StageTimer to render.
        min_percent: hide stages whose share of total is below this value.
    """
    total = timer.root.wall_time
    if total <= 0:
        # Root never timed a top-level block; use the sum of its children.
        total = sum(child.wall_time for child in timer.root.children.values())

    rows: List[Tuple[int, "StageRecord"]] = []
    # Display children of the root instead of the root itself so "total"
    # isn't printed twice (header shows it already).
    for child in timer.root.children.values():
        _collect_rows(child, 0, rows, min_percent, total)

    name_col = max([len("Stage")] + [2 * depth + len(rec.name) for depth, rec in rows])
    name_col = max(name_col, 20)
    header_line = (
        f"| {'Stage'.ljust(name_col)} | "
        f"{'Wall (ms)':>10} | {'%':>6} | {'Calls':>6} | {'FP Type':<14} |"
    )
    sep = (
        "+-"
        + "-" * name_col
        + "-+-"
        + "-" * 10
        + "-+-"
        + "-" * 6
        + "-+-"
        + "-" * 6
        + "-+-"
        + "-" * 14
        + "-+"
    )
    out = [
        sep,
        f"| JAX-QC Profiling Report".ljust(len(sep) - 1) + "|",
        sep,
        header_line,
        sep,
        (
            f"| {'Total'.ljust(name_col)} | "
            f"{_ms(total):>10.3f} | {100.0:>6.1f} | {1:>6d} | {'':<14} |"
        ),
    ]
    for depth, rec in rows:
        indent = "  " * depth
        name = f"{indent}{rec.name}"
        pct = (100.0 * rec.wall_time / total) if total > 0 else 0.0
        out.append(
            f"| {name.ljust(name_col)} | "
            f"{_ms(rec.wall_time):>10.3f} | {pct:>6.1f} | "
            f"{rec.call_count:>6d} | {rec.fp_abstraction:<14} |"
        )
    out.append(sep)
    out.append(format_summary_table(timer))
    return "\n".join(out)


def format_summary_table(timer: "StageTimer") -> str:
    """Flat summary grouped by FP abstraction tag.

    Walks the full tree and sums wall time per tag. Children double-count
    their parents, so we use only leaf records (stages with no children)
    when aggregating.
    """
    totals: Dict[str, float] = {}

    def _walk(node: "StageRecord") -> None:
        if not node.children:
            tag = node.fp_abstraction or "(untagged)"
            totals[tag] = totals.get(tag, 0.0) + node.wall_time
        else:
            for child in node.children.values():
                _walk(child)

    _walk(timer.root)
    grand_total = sum(totals.values())
    lines = ["FP Abstraction Summary (leaf stages only):"]
    if grand_total == 0:
        lines.append("  (no timed work)")
        return "\n".join(lines)
    for tag in sorted(totals.keys()):
        t = totals[tag]
        pct = 100.0 * t / grand_total
        lines.append(f"  {tag:<14s} {_ms(t):>10.3f} ms  ({pct:5.1f}%)")
    return "\n".join(lines)
