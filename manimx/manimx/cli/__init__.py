"""The ``manimx`` command line: projects, scenes and rendering.

    manimx new my-video          create a project (asks for what is missing)
    manimx g scene title-card    add a scene: module, import, timeline entry
    manimx render --preview      render the timeline quickly
    manimx still Intro -t 2      one frame as a PNG

Commands are Typer commands (cli/app.py), the file work lives in
cli/scaffold.py, and everything visible in cli/ui.py.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from typer.exceptions import Abort, Exit, TyperException

from ..project import ProjectError


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI; returns the exit code (errors are printed, not raised)."""
    from . import ui
    from .app import Options, app

    try:
        code = app(args=list(sys.argv[1:] if argv is None else argv), standalone_mode=False)
        if code == 130:  # Typer turns Ctrl+C into this exit code
            ui.errors.print("[dim]Cancelled.[/]")
        if isinstance(code, int):
            return code
    except Exit as exit:
        return exit.exit_code
    except (KeyboardInterrupt, Abort):
        ui.errors.print("[dim]Cancelled.[/]")
        return 130
    except TyperException as problem:  # usage errors: Typer prints them with the command's usage line
        problem.show()  # type: ignore[attr-defined]
        return getattr(problem, "exit_code", 2)
    except ProjectError as problem:
        ui.error(problem, traceback=Options.traceback)
        return 1
    except Exception as problem:  # noqa: BLE001 - shown as a panel, details with --traceback
        ui.error(problem, traceback=Options.traceback)
        return 1
    return 0


def run() -> None:
    """Console script entry point."""
    sys.exit(main())
