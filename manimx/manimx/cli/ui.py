"""How the CLI looks: one rich console, change lines, prompts and errors.

Prompts appear only when an argument is missing and the CLI runs in a
terminal; scripts and CI (no TTY, or ``--yes``) get the defaults or a
clear error instead, never a hanging prompt.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.traceback import Traceback

import manimx

from ..project import ProjectError, SceneError
from .scaffold import Change

console = Console(highlight=False)
errors = Console(stderr=True, highlight=False)

ACCENT = "#5ea3d0"
STYLE = questionary.Style([("qmark", f"fg:{ACCENT} bold"), ("pointer", f"fg:{ACCENT} bold"), ("highlighted", f"fg:{ACCENT} bold"), ("answer", "fg:#f2cc8f")])

_ACTIONS = {"create": ("CREATE", "green"), "update": ("UPDATE", "cyan"), "delete": ("DELETE", "red")}


class Interaction:
    """Whether the CLI may ask questions."""

    allowed = True

    @classmethod
    def possible(cls) -> bool:
        return cls.allowed and sys.stdin.isatty() and sys.stdout.isatty()


def ask_text(question: str, default: str = "", missing: str = "") -> str:
    """Ask for text in a terminal; otherwise the default (or fail if none)."""
    if Interaction.possible():
        answer = questionary.text(question, default=default, style=STYLE).ask()
        if answer is None:
            raise KeyboardInterrupt
        return answer.strip()
    if default:
        return default
    raise ProjectError(f"missing {missing or question.lower()}", hint="pass it as an argument")


def ask_choice(question: str, choices: Sequence[tuple[str, str]], default: str) -> str:
    """``choices`` are ``(value, description)``; non-interactive: ``default``."""
    if not Interaction.possible():
        return default
    width = max(len(value) for value, _ in choices)
    answer = questionary.select(
        question,
        choices=[questionary.Choice(f"{value:<{width}}   {description}", value=value) for value, description in choices],
        default=default,
        style=STYLE,
    ).ask()
    if answer is None:
        raise KeyboardInterrupt
    return answer


def confirm(question: str, default: bool = False) -> bool:
    if not Interaction.possible():
        return default
    answer = questionary.confirm(question, default=default, style=STYLE).ask()
    if answer is None:
        raise KeyboardInterrupt
    return answer


def changes(items: Sequence[Change], root: Path) -> None:
    """Angular-style ``CREATE path`` lines."""
    for change in items:
        label, color = _ACTIONS[change.action]
        line = Text.assemble((f"{label} ", f"bold {color}"), (_relative(change.path, root), ""))
        if change.note:
            line.append(f"  {change.note}", style="dim")
        console.print(line)


def success(message: str) -> None:
    console.print(Text.assemble(("✔ ", "bold green"), Text.from_markup(message)))


def header(*parts: str) -> None:
    console.print(Text(" · ").join(Text(part, style="bold" if i == 0 else "") for i, part in enumerate(parts)), style=ACCENT)


def error(problem: BaseException, traceback: bool = False) -> None:
    """Print an error as a panel: what went wrong, why, and what to do."""
    body = Text.assemble(("✖ ", "bold red"), (str(problem), "bold"))
    hint = getattr(problem, "hint", None)
    if hint:
        body.append("\n  help: ", style="bold cyan").append(hint)
    errors.print(Panel(body, border_style="red", title=_title(problem), title_align="left", expand=False))
    cause = problem.__cause__
    if isinstance(problem, SceneError) and cause is not None:
        # Scene code failed: show the user's frames (manimx's own frames
        # only with --traceback, they rarely help).
        suppress = [] if traceback else [manimx]
        errors.print(Traceback.from_exception(type(cause), cause, cause.__traceback__, suppress=suppress, max_frames=8))
    elif traceback and not isinstance(problem, ProjectError):
        errors.print(Traceback.from_exception(type(problem), problem, problem.__traceback__))


def _title(problem: BaseException) -> str:
    if isinstance(problem, SceneError):
        return "scene error"
    if isinstance(problem, ProjectError):
        return "project error"
    return type(problem).__name__


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)
