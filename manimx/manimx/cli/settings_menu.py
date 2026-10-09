"""``manimx settings``: every project setting in one collapsible tree.

    ▾ Effects                 Motion blur, bloom, shadows and tone mapping.
        ▾ Motion blur  [on]
            samples              8
            shutter angle        180°    0.008333 s exposure at 60 fps
        ▸ Bloom        [off]
    ▸ Output                  Video size, frame rate, background and files.

↑/↓ move, →/← open and close sections, space switches a section or a
yes/no value or steps through choices, enter opens a section or edits a
value, r resets a value to its default, s saves to manimx.json, q quits.
The description of the selected line is shown below the tree.

:class:`SettingsMenu` is the menu without a terminal: rows, cursor, edit
line and messages, changed by key names. The prompt_toolkit application
in :func:`run_menu` only draws it and forwards keys, so tests drive the
model directly. :func:`print_settings` is the same tree as plain text, for
scripts and as documentation.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field

from ..settings import Entry, Settings, SettingsError, entries, get, parse, set_value

INDENT = 4
LABEL_WIDTH = 26


@dataclass
class Row:
    entry: Entry
    depth: int

    @property
    def is_section(self) -> bool:
        return self.entry.is_section


@dataclass
class SettingsMenu:
    """The settings tree being edited, with cursor and messages."""

    settings: Settings
    saved: Settings = field(init=False)
    expanded: set[tuple[str, ...]] = field(default_factory=set)
    cursor: int = 0
    editing: str | None = None
    """The edit line's text while a value is being typed."""
    message: str = ""
    error: bool = False
    done: bool = False
    save: bool = False
    _quit_asked: bool = False

    def __post_init__(self) -> None:
        self.saved = copy.deepcopy(self.settings)

    # -- what is shown --------------------------------------------------------

    def rows(self) -> list[Row]:
        """The visible lines: sections, and the values of open sections
        (a section's ``enabled`` value is shown on the section's line)."""
        rows = []
        for entry in entries():
            parent = entry.path[:-1]
            if any(parent[:depth] not in self.expanded for depth in range(1, len(parent) + 1)):
                continue
            if entry.name == "enabled":
                continue
            rows.append(Row(entry, len(entry.path) - 1))
        return rows

    @property
    def current(self) -> Row:
        rows = self.rows()
        self.cursor = min(self.cursor, len(rows) - 1)
        return rows[self.cursor]

    @property
    def dirty(self) -> bool:
        return self.settings != self.saved

    def value_text(self, row: Row) -> str:
        """What the row shows after its label."""
        entry = row.entry
        if row.is_section:
            switch = self._switch(entry.path)
            return "" if switch is None else ("[on]" if switch else "[off]")
        value = get(self.settings, entry.path)
        if isinstance(value, bool):
            return "yes" if value else "no"
        text = f"{value:g}" if isinstance(value, float) else str(value)
        return text + entry.field.metadata.get("unit", "")

    def note(self, row: Row) -> str:
        if row.is_section:
            return "" if len(row.entry.path) > 1 and self._switch(row.entry.path) is not None else row.entry.description
        note = row.entry.field.metadata.get("note")
        return note(self.settings) if note is not None else ""

    def changed(self, row: Row) -> bool:
        """Whether the row (or anything in its section) differs from manimx.json."""
        return get(self.settings, row.entry.path) != get(self.saved, row.entry.path)

    def help_text(self) -> str:
        if self.editing is not None:
            return "enter: set   esc: cancel"
        return "↑↓ move   →← open/close   space switch   enter edit   r default   s save   q quit"

    # -- keys -----------------------------------------------------------------

    def key(self, name: str) -> None:
        """Handle one key: "up", "down", "left", "right", "space", "enter",
        "escape", "backspace", a letter, or a typed character while editing."""
        if self.editing is not None:
            self._edit_key(name)
            return
        if name != "q":
            self._quit_asked = False
        self.message, self.error = "", False
        handler = {
            "up": self._up,
            "down": self._down,
            "right": self._open,
            "left": self._close,
            "space": self._switch_current,
            "enter": self._enter,
            "r": self._reset,
            "s": self._save,
            "q": self._quit,
        }.get(name)
        if handler is not None:
            handler()

    def _up(self) -> None:
        self.cursor = max(0, self.cursor - 1)

    def _down(self) -> None:
        self.cursor = min(len(self.rows()) - 1, self.cursor + 1)

    def _open(self) -> None:
        row = self.current
        if row.is_section:
            self.expanded.add(row.entry.path)

    def _close(self) -> None:
        row = self.current
        if row.is_section and row.entry.path in self.expanded:
            self.expanded.discard(row.entry.path)
            return
        parent = row.entry.path[:-1]
        if parent:  # jump to the parent section and close it
            self.expanded.discard(parent)
            self.cursor = next(index for index, other in enumerate(self.rows()) if other.entry.path == parent)

    def _enter(self) -> None:
        row = self.current
        if row.is_section:
            path = row.entry.path
            self.expanded.symmetric_difference_update({path})
            return
        value = get(self.settings, row.entry.path)
        if isinstance(value, bool) or row.entry.field.metadata.get("choices") is not None:
            self._switch_current()
            return
        self.editing = f"{value:g}" if isinstance(value, float) else str(value)

    def _switch_current(self) -> None:
        row = self.current
        path = row.entry.path
        if row.is_section:
            switch = self._switch(path)
            if switch is None:
                self._enter()
            else:
                self._set((*path, "enabled"), not switch)
            return
        value = get(self.settings, path)
        choices = row.entry.field.metadata.get("choices")
        if isinstance(value, bool):
            self._set(path, not value)
        elif choices is not None:
            self._set(path, choices[(choices.index(value) + 1) % len(choices)])
        else:
            self._enter()

    def _reset(self) -> None:
        row = self.current
        if row.is_section:
            return
        self._set(row.entry.path, row.entry.field.default)
        self.message = f"{row.entry.label} back to its default"

    def _save(self) -> None:
        self.save = self.done = True

    def _quit(self) -> None:
        if self.dirty and not self._quit_asked:
            self._quit_asked = True
            self.message = "unsaved changes: s saves and quits, q again quits without saving"
            self.error = True
            return
        self.done = True

    def _edit_key(self, name: str) -> None:
        assert self.editing is not None
        if name == "escape":
            self.editing, self.message, self.error = None, "", False
        elif name == "enter":
            row = self.current
            try:
                value = parse(self.editing, row.entry.field, ".".join(row.entry.path))
            except SettingsError as problem:
                self.message, self.error = str(problem), True
                return
            self._set(row.entry.path, value)
            self.editing = None
        elif name == "backspace":
            self.editing = self.editing[:-1]
        elif len(name) == 1:
            self.editing += name
        elif name == "space":
            self.editing += " "

    def _set(self, path: tuple[str, ...], value: object) -> None:
        try:
            set_value(self.settings, path, value)
        except SettingsError as problem:
            self.message, self.error = str(problem), True

    def _switch(self, path: tuple[str, ...]) -> bool | None:
        """A section's ``enabled`` value, or None if it cannot be switched off."""
        section = get(self.settings, path)
        names = {item.name for item in dataclasses.fields(section)}
        return section.enabled if "enabled" in names else None


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------


def lines(menu: SettingsMenu) -> list[tuple[str, str]]:
    """The menu as prompt_toolkit formatted text: ``(style, text)`` pieces."""
    pieces: list[tuple[str, str]] = []
    selected = menu.current
    for index, row in enumerate(menu.rows()):
        chosen = index == menu.cursor
        indent = " " * (INDENT * row.depth)
        marker = ("▾ " if row.entry.path in menu.expanded else "▸ ") if row.is_section else "  "
        label = row.entry.label.capitalize() if row.is_section else row.entry.label
        head = f"{indent}{marker}{label}"
        value = menu.value_text(row)
        if chosen and menu.editing is not None:
            value = menu.editing + "█"
        style = "class:selected" if chosen else ("class:section" if row.is_section else "")
        pieces.append((style, f"{'›' if chosen else ' '} {head:<{LABEL_WIDTH}}"))
        value_style = "class:off" if value == "[off]" else ("class:changed" if menu.changed(row) else "class:value")
        pieces.append((value_style, f" {value:<12}"))
        pieces.append(("class:note", f" {menu.note(row)}\n"))
    pieces.append(("", "\n"))
    pieces.append(("class:description", f"  {selected.entry.description}\n"))
    if menu.message:
        pieces.append(("class:error" if menu.error else "class:note", f"  {menu.message}\n"))
    pieces.append(("class:help", f"\n  {menu.help_text()}{'   (unsaved changes)' if menu.dirty else ''}\n"))
    return pieces


STYLE = {
    "selected": "bold #5ea3d0",
    "section": "bold",
    "value": "#f2cc8f",
    "changed": "bold #81b29a",
    "off": "#888888",
    "note": "#888888",
    "description": "italic",
    "error": "#e07a5f",
    "help": "#888888",
}

_KEYS = {"up": "up", "down": "down", "left": "left", "right": "right", "enter": "enter", "c-m": "enter", "escape": "escape", "backspace": "backspace", "c-h": "backspace"}


def run_menu(menu: SettingsMenu, input=None, output=None) -> SettingsMenu:  # noqa: A002 - prompt_toolkit's names
    """Show ``menu`` until it is done (saved or quit); Ctrl+C quits."""
    from prompt_toolkit import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.styles import Style

    bindings = KeyBindings()

    def forward(name: str):
        def handler(event) -> None:
            menu.key(name)
            if menu.done:
                event.app.exit()

        return handler

    for key, name in _KEYS.items():
        bindings.add(key)(forward(name))
    bindings.add(" ")(forward("space"))

    @bindings.add("<any>")
    def typed(event) -> None:
        character = event.data
        if len(character) == 1 and character.isprintable():
            forward(character)(event)

    @bindings.add("c-c")
    def cancel(event) -> None:
        menu.done, menu.save = True, False
        event.app.exit()

    control = FormattedTextControl(lambda: lines(menu), show_cursor=False)
    app: Application = Application(
        layout=Layout(Window(control)), key_bindings=bindings, style=Style.from_dict(STYLE), full_screen=False, input=input, output=output
    )
    app.run()
    return menu


def print_settings(settings: Settings) -> str:
    """Every setting with its value and description, as indented text."""
    out = []
    for entry in entries():
        indent = "  " * (len(entry.path) - 1)
        if entry.is_section:
            out.append(f"{indent}{entry.name}: {entry.description}")
            continue
        value = get(settings, entry.path)
        meta = entry.field.metadata
        limits = []
        if meta.get("choices") is not None:
            limits.append("one of " + ", ".join(str(choice) for choice in meta["choices"]))
        if meta.get("minimum") is not None:
            limits.append(f"≥ {meta['minimum']:g}")
        if meta.get("maximum") is not None:
            limits.append(f"≤ {meta['maximum']:g}")
        shown = json_value(value)
        extra = f" ({'; '.join(limits)})" if limits else ""
        out.append(f"{indent}{entry.name} = {shown}{meta.get('unit', '')}  {entry.description}{extra}")
    return "\n".join(out)


def json_value(value: object) -> str:
    import json

    return json.dumps(value)
