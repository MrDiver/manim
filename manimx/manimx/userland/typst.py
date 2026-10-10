"""Typst text and formulas as Shape2D nodes.

:class:`Typst` compiles Typst markup to SVG with the ``typst`` package and
reads it with :mod:`manimx.userland.svg`: a Group with one Shape2D per
glyph, rule and shape, so Write, Create and Morph work on it like on any
other group. Sizes follow Typst's (a scene unit is one inch, see svg.py).
"""

from __future__ import annotations

import re
import unicodedata
import zlib
from functools import lru_cache
from typing import Any

from .svg import SVG, SVGPath

PAGE = "#set page(width: auto, height: auto, margin: 0pt, fill: none)\n"
"""Makes the page as large as its content, with no background."""


class Typst(SVG):
    """Typst markup as a Group of Shape2D: ``Typst("Hello $x^2$")``.

    ``preamble`` comes before the markup (``#set text(...)`` rules and
    imports). The group is named after the markup and every glyph's
    Shape2D after its character, so ``formula.find_all("x")`` gives the
    x's (letters as typed: a math italic 𝑥 is "x"); rules such as a
    fraction bar keep generated names. Colors come from the document (Typst's default is black);
    ``style`` (any Shape2D style argument, ``color`` for one) overrides
    them on every shape. ``width`` and ``height`` scale as in :class:`SVG`.
    """

    def __init__(
        self,
        markup: str,
        preamble: str = "",
        width: float | None = None,
        height: float | None = None,
        name: str | None = None,
        **style: Any,
    ) -> None:
        self.markup = markup
        source = PAGE + preamble + "\n" + markup
        svg = compile_typst(source)
        characters = glyph_characters(source)
        # Each glyph is one use of a symbol; a different count means the
        # two exports do not line up, and the glyphs stay unnamed.
        self.characters = characters if characters is not None and len(characters) == svg.count("<use") else None
        """The character of each glyph, in SVGPath.glyph order (None if unknown)."""
        super().__init__(svg, width=width, height=height, name=name or markup, **style)

    def _path_name(self, path: SVGPath) -> str | None:
        if self.characters is None or path.glyph is None or path.glyph >= len(self.characters):
            return None
        return self.characters[path.glyph] or None


class TypstMath(Typst):
    """A display formula: ``TypstMath("sum_(k=1)^n k")`` is ``$ ... $``."""

    def __init__(self, formula: str, preamble: str = "", width: float | None = None, height: float | None = None, name: str | None = None, **style: Any) -> None:
        super().__init__(f"$ {formula} $", preamble, width, height, name, **style)


@lru_cache(maxsize=256)
def compile_typst(source: str) -> str:
    """The SVG of the first page of the Typst document ``source``."""
    output = compiler().compile(input=source.encode(), format="svg")
    page = output[0] if isinstance(output, list) else output
    return page.decode()


@lru_cache(maxsize=1)
def compiler() -> Any:
    """One Typst compiler for the process: creating one loads the system
    fonts (about 0.2 s), compiling with it then takes milliseconds."""
    import typst  # imported here: compiling is the only use

    return typst.Compiler()




# --------------------------------------------------------------------------
# Which character each glyph is
# --------------------------------------------------------------------------
#
# The SVG export draws glyphs as uses of symbols and says nothing about
# the text. The PDF export of the same document draws the same glyphs in
# the same order, as text runs whose fonts map glyph codes to Unicode
# (ToUnicode CMaps). So the characters of the SVG's glyph uses are the
# characters of the PDF's glyph codes, in order.


@lru_cache(maxsize=256)
def glyph_characters(source: str) -> tuple[str, ...] | None:
    """The characters of the glyphs of the Typst document ``source``, in
    the order its SVG uses them (SVGPath.glyph), NFKC-normalized (a math
    italic 𝑥 is "x"). None if the PDF cannot be read that way."""
    try:
        return tuple(_pdf_glyph_text(compiler().compile(input=source.encode(), format="pdf")))
    except (ValueError, KeyError, IndexError, zlib.error):
        return None


_OBJECT = re.compile(rb"(\d+) 0 obj\s*(.*?)\s*endobj", re.S)
_STREAM = re.compile(rb"^(.*?)stream\r?\n(.*?)\r?\nendstream$", re.S)


def _pdf_glyph_text(pdf: bytes) -> list[str]:
    objects: dict[int, tuple[bytes, bytes]] = {}
    for match in _OBJECT.finditer(pdf):
        body = match.group(2)
        stream = _STREAM.match(body)
        if stream is None:
            objects[int(match.group(1))] = (body, b"")
        else:
            data = stream.group(2)
            objects[int(match.group(1))] = (stream.group(1), zlib.decompress(data) if b"FlateDecode" in stream.group(1) else data)
    fonts: dict[bytes, dict[int, str]] = {}
    for dictionary, _ in objects.values():
        for resources in re.findall(rb"/Font\s*<<(.*?)>>", dictionary, re.S):
            for name, number in re.findall(rb"/([\w.]+)\s+(\d+) 0 R", resources):
                to_unicode = re.search(rb"/ToUnicode\s+(\d+) 0 R", objects[int(number)][0])
                fonts[name] = _cmap(objects[int(to_unicode.group(1))][1]) if to_unicode else {}
    characters: list[str] = []
    for dictionary, _ in objects.values():
        if re.search(rb"/Type\s*/Page\b", dictionary):
            for number in re.findall(rb"(\d+) 0 R", re.search(rb"/Contents\s*(\[[^\]]*\]|\d+ 0 R)", dictionary).group(1)):
                characters += _content_text(objects[int(number)][1], fonts)
    return characters


def _cmap(data: bytes) -> dict[int, str]:
    """Glyph code to text of a ToUnicode CMap (bfchar and bfrange)."""
    mapping: dict[int, str] = {}
    for block in re.findall(rb"beginbfchar(.*?)endbfchar", data, re.S):
        for code, text in re.findall(rb"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", block):
            mapping[int(code, 16)] = bytes.fromhex(text.decode()).decode("utf-16-be")
    for block in re.findall(rb"beginbfrange(.*?)endbfrange", data, re.S):
        for low, high, start in re.findall(rb"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", block):
            first = int(start, 16)
            for offset in range(int(high, 16) - int(low, 16) + 1):
                mapping[int(low, 16) + offset] = chr(first + offset)
    return mapping


_CONTENT = re.compile(rb"/([\w.]+)\s+[-\d.]+\s+Tf|\((?:\\.|[^\\)])*\)|<[0-9A-Fa-f\s]*>(?!>)")


def _content_text(content: bytes, fonts: dict[bytes, dict[int, str]]) -> list[str]:
    """The text of every glyph a content stream shows, in order (two-byte
    codes, as Typst's Identity-H fonts use)."""
    characters: list[str] = []
    font: dict[int, str] = {}
    for match in _CONTENT.finditer(re.sub(rb"<<.*?>>", b"", content, flags=re.S)):
        if match.group(1) is not None:
            font = fonts[match.group(1)]
            continue
        token = match.group(0)
        raw = bytes.fromhex(token[1:-1].decode()) if token.startswith(b"<") else _literal(token[1:-1])
        codes = (int.from_bytes(raw[index : index + 2], "big") for index in range(0, len(raw) - 1, 2))
        characters.extend(unicodedata.normalize("NFKC", font.get(code, "")) for code in codes)
    return characters


def _literal(text: bytes) -> bytes:
    """A PDF literal string's bytes (backslash escapes, octal codes)."""
    out = bytearray()
    index = 0
    while index < len(text):
        byte = text[index]
        if byte != 0x5C:
            out.append(byte)
            index += 1
            continue
        octal = re.match(rb"[0-7]{1,3}", text[index + 1 : index + 4])
        if octal:
            out.append(int(octal.group(0), 8))
            index += 1 + len(octal.group(0))
        else:
            escaped = text[index + 1 : index + 2]
            out += {b"n": b"\n", b"r": b"\r", b"t": b"\t", b"b": b"\b", b"f": b"\f"}.get(escaped, escaped)
            index += 2
    return bytes(out)


__all__ = ["PAGE", "Typst", "TypstMath", "compile_typst", "glyph_characters"]
