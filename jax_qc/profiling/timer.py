"""Hierarchical timer with optional JAX device synchronization.

Example:

    timer = StageTimer()
    with timer.stage('integrals', 'applicative'):
        with timer.stage('overlap', 'applicative'):
            S = compute_overlap(basis)
        with timer.stage('eri', 'applicative'):
            ERI = compute_eri(basis)
    print(timer.report())

Design notes:

* Records are keyed by stage name within the current parent, so repeated
  stages (e.g. per-SCF-iteration ``fock_build``) aggregate wall time and
  call counts.
* When ``sync_device`` is True, we issue a JAX effects barrier at the
  start and end of each stage to ensure GPU work has actually completed
  before we read the wall clock. On CPU-only builds this is a no-op.
* The timer is **mutable** (it's a profiler, not part of any jit-compatible
  state), so we use plain dataclasses instead of chex.dataclass.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional

try:
    import jax

    _HAS_JAX = True
except ImportError:  # pragma: no cover - jax is a hard dep but be defensive
    _HAS_JAX = False


def _device_barrier() -> None:
    """Block until pending JAX device work completes, if possible."""
    if not _HAS_JAX:
        return
    barrier = getattr(jax, "effects_barrier", None)
    if barrier is not None:
        try:
            barrier()
        except Exception:
            pass


@dataclass
class StageRecord:
    """One node in the profiling tree.

    Attributes:
        name:           stage label.
        wall_time:      accumulated wall time in seconds across all calls.
        call_count:     number of times this stage was entered.
        children:       mapping from child stage name to StageRecord.
        parent:         reference to parent record (None for root).
        fp_abstraction: optional tag, e.g. 'applicative', 'monad', 'foldable'.
    """

    name: str
    wall_time: float = 0.0
    call_count: int = 0
    children: Dict[str, "StageRecord"] = field(default_factory=dict)
    parent: Optional["StageRecord"] = field(default=None, repr=False)
    fp_abstraction: str = ""

    def total_time(self) -> float:
        """Total wall time including children (used for percentages)."""
        return self.wall_time

    def to_dict(self) -> dict:
        """Serialize to a plain dict (JSON-friendly; drops parent ref)."""
        return {
            "name": self.name,
            "wall_time": self.wall_time,
            "call_count": self.call_count,
            "fp_abstraction": self.fp_abstraction,
            "children": {k: v.to_dict() for k, v in self.children.items()},
        }


class StageTimer:
    """Hierarchical profiling timer.

    Features:
    * Nested stages via the ``stage()`` context manager.
    * Optional JAX device synchronization (``effects_barrier``) for
      accurate GPU timing.
    * Aggregation of repeated stages (SCF iteration substages).
    * FP-abstraction annotation per stage.
    * Serialization to dict / JSON.
    """

    def __init__(self, sync_device: bool = True) -> None:
        self.root = StageRecord(name="total")
        self._stack: List[StageRecord] = [self.root]
        self._active_start: List[float] = []
        self._sync = sync_device

    @contextmanager
    def stage(self, name: str, fp_type: str = "") -> Iterator[StageRecord]:
        """Time a named computation stage.

        Args:
            name:    stage name; siblings with the same name aggregate.
            fp_type: FP abstraction tag (applicative, monad, ...). The first
                     value wins; subsequent enters leave the tag unchanged
                     unless they explicitly override.
        """
        parent = self._stack[-1]
        record = parent.children.get(name)
        if record is None:
            record = StageRecord(name=name, parent=parent, fp_abstraction=fp_type)
            parent.children[name] = record
        elif fp_type and not record.fp_abstraction:
            record.fp_abstraction = fp_type

        if self._sync:
            _device_barrier()
        t0 = time.perf_counter()
        self._stack.append(record)
        self._active_start.append(t0)
        try:
            yield record
        finally:
            if self._sync:
                _device_barrier()
            elapsed = time.perf_counter() - self._active_start.pop()
            record.wall_time += elapsed
            record.call_count += 1
            self._stack.pop()

    def to_dict(self) -> dict:
        """Export the timing tree as a plain dict."""
        return self.root.to_dict()

    def report(self, min_percent: float = 0.0) -> str:
        """Pretty-print the timing tree. See ``profiling.report``."""
        from jax_qc.profiling.report import format_report

        return format_report(self, min_percent=min_percent)

    def summary_table(self) -> str:
        """Flat summary grouped by FP abstraction type."""
        from jax_qc.profiling.report import format_summary_table

        return format_summary_table(self)

    def total_time(self) -> float:
        """Return the total accumulated wall time at the root."""
        return self.root.wall_time


@contextmanager
def optional_stage(
    timer: Optional[StageTimer], name: str, fp_type: str = ""
) -> Iterator[Optional[StageRecord]]:
    """No-op wrapper: enter ``timer.stage()`` only if ``timer`` is not None.

    Lets driver code write ``with optional_stage(timer, 'fock', 'applicative')``
    without branching every site.
    """
    if timer is None:
        yield None
    else:
        with timer.stage(name, fp_type) as record:
            yield record
