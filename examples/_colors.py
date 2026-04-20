"""Tiny ANSI color helper for the example scripts.

Respects ``NO_COLOR`` (https://no-color.org) and falls back to plain text
when the output is not a TTY. Keeps the examples dependency-free.
"""

from __future__ import annotations

import os
import sys

_RESET = "\033[0m"
_CODES = {
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
    "bright_green": "\033[92m",
    "bright_yellow": "\033[93m",
    "bright_cyan": "\033[96m",
}


def _enabled() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("JAX_QC_FORCE_COLOR"):
        return True
    return sys.stdout.isatty()


def c(style: str, text: str) -> str:
    """Wrap ``text`` in ANSI escape codes for ``style`` (space-separated tags)."""
    if not _enabled():
        return text
    codes = "".join(_CODES[s] for s in style.split() if s in _CODES)
    if not codes:
        return text
    return f"{codes}{text}{_RESET}"


def bold(text: str) -> str:
    return c("bold", text)


def header(text: str, color: str = "bright_cyan") -> str:
    """Bold colored banner for section headers."""
    return c(f"bold {color}", text)


def value(text: str) -> str:
    """Highlight a numerical value."""
    return c("bright_yellow", text)


def label(text: str) -> str:
    """Dim gray for labels / keys."""
    return c("cyan", text)


def ok(text: str = "OK") -> str:
    return c("bold bright_green", text)


def warn(text: str) -> str:
    return c("bold yellow", text)


def error(text: str) -> str:
    return c("bold red", text)


def banner(title: str, color: str = "bright_cyan", width: int = 72) -> str:
    """Return a framed banner line for section titles."""
    pad = max(1, (width - len(title) - 2) // 2)
    line = "=" * pad + " " + title + " " + "=" * pad
    return c(f"bold {color}", line[:width])


def compare(
    got: float,
    ref: float,
    *,
    label_got: str = "got",
    label_ref: str = "ref",
    tol: float = 1e-6,
) -> str:
    """Format a side-by-side comparison with a PASS/FAIL tag."""
    diff = abs(got - ref)
    passed = diff <= tol
    tag = ok("PASS") if passed else error("FAIL")
    return (
        f"{label(label_got)}={value(f'{got:.10f}')}  "
        f"{label(label_ref)}={value(f'{ref:.10f}')}  "
        f"|Δ|={value(f'{diff:.2e}')}  [{tag}]"
    )
