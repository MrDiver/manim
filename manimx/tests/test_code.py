"""Code: glyphs on a grid of cells, focus, regions, windows and MorphCode."""

from __future__ import annotations

import numpy as np
import pytest
from manimx.animation import ClippingGroup, Scene
from manimx.userland import Code, MorphCode
from manimx.userland.code import _matching_cells, _tm_theme
from manimx.userland.svg import read_svg
from manimx.userland.typst import PAGE, compile_typst

SOURCE = "fib 0 = 0\nfib n = fib (n - 1)\n\tdone"


def prepare(construct):
    class Built(Scene):
        def construct(self):
            construct(self)

    return Built().prepare()


def test_svg_paths_of_glyphs_know_where_their_symbol_is_placed() -> None:
    paths = read_svg(compile_typst(PAGE + "ab"), centered=False)
    first, second = (path.origin for path in paths[:2])
    assert first[0] == pytest.approx(0.0, abs=1e-9) and second[0] > first[0] and first[1] == pytest.approx(second[1])
    assert read_svg('<svg xmlns="http://www.w3.org/2000/svg"><rect width="2" height="2"/></svg>')[0].origin is None


def test_every_character_gets_a_glyph_in_its_cell() -> None:
    code = Code(SOURCE, "haskell")
    assert code.source.endswith("    done")  # tabs become spaces
    assert "".join(glyph.name for glyph in code.glyphs) == "".join(SOURCE.split())
    assert [len(line.children) for line in code.lines] == [6, 13, 4]
    assert code.cells[code.line(3).children[0].uid] == (2, 4)
    grid = code.grid
    assert grid.advance > 0 and grid.pitch > grid.cap_height > 0
    first = code.glyphs[0].points.value
    assert first[:, 0].min() >= grid.left - 0.02 and first[:, 1].max() <= 0.0  # the origin is the top left corner
    with pytest.raises(IndexError, match="lines 1 to 3"):
        code.line(4)


def test_select_finds_text_by_occurrence_and_line() -> None:
    code = Code(SOURCE)
    assert [code.cells[glyph.uid] for glyph in code.select("fib", 1)] == [(1, 0), (1, 1), (1, 2)]
    assert [code.cells[glyph.uid] for glyph in code.select("fib", 1, line=2)] == [(1, 8), (1, 9), (1, 10)]
    assert [glyph.name for glyph in code.select("(n - 1)")] == list("(n-1)")
    with pytest.raises(LookupError, match="appears 3 times"):
        code.select("fib", 3)


def test_focus_dims_the_rest_and_unfocus_restores_it() -> None:
    code = Code(SOURCE)
    prepared = prepare(lambda s: (s.add(code), s.play(code.animate.focus(code.line(1), dim=0.2)), s.play(code.animate.unfocus())))
    prepared.evaluate(1.0)
    assert {glyph.alpha.value for glyph in code.line(1).children} == {1.0}
    assert [glyph.alpha.value for glyph in code.line(2).children] == pytest.approx([0.2] * 13)
    prepared.evaluate(2.0)
    assert {glyph.alpha.value for glyph in code.glyphs} == {1.0}


def test_region_frames_cells_where_the_code_is() -> None:
    code = Code(SOURCE).move_to((1.0, 2.0, 0.0))
    center, width, height = code.region(code.line(1), padding=0.0)
    assert width == pytest.approx(9 * code.grid.advance) and height == pytest.approx(code.grid.pitch)
    assert center[0] == pytest.approx(1.0 + code.grid.left + width / 2)
    code.scale_by(2.0)
    assert code.region(code.line(1), padding=0.0)[1] == pytest.approx(2 * width)
    with pytest.raises(ValueError, match="at least one glyph"):
        code.region([])


def test_a_window_shows_rows_and_scrolls_by_lines() -> None:
    code = Code(SOURCE)
    with pytest.raises(RuntimeError, match="clip"):
        code.scroll_to(2)
    window = code.clip(rows=2, padding=0.0)
    assert isinstance(window, ClippingGroup) and window.content is code
    assert window.mask.height.value == pytest.approx(2 * code.grid.pitch)
    prepared = prepare(lambda s: (s.add(window), s.play(code.animate.scroll_to(3))))
    prepared.evaluate(1.0)
    assert code.position.y.value == pytest.approx(2 * code.grid.pitch)


def test_colors_become_a_theme_and_color_the_scopes() -> None:
    assert "<string>keyword</string>" in _tm_theme({"keyword": "#ff0000"})
    code = Code("if x then 1 else 2", "haskell", colors={"keyword": "#ff0000"})
    np.testing.assert_allclose(code.select("if")[0].fill_color.value[0], (1.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="not both"):
        Code("x", colors={}, theme="theme.tmTheme")


def test_morph_code_moves_shared_tokens_and_fades_the_rest() -> None:
    assert _matching_cells("fib n = 1", "fib n =\n  go n") == {(0, 0): (0, 0), (0, 1): (0, 1), (0, 2): (0, 2), (0, 4): (0, 4), (0, 6): (0, 6)}
    before, after = Code("fib n = 1"), Code("fib n =\n  go n")
    morph = MorphCode(before, after)
    assert len(morph.inbetweens) == len(before.glyphs) + 3  # "1" fades out; "g", "o", "n" fade in
    prepared = prepare(lambda s: (s.add(before), s.play(morph)))
    prepared.evaluate(1.0)
    assert after.present.value and not before.present.value
    with pytest.raises(TypeError, match="Code into a Code"):
        MorphCode(before, Code("x").lines[0])


def test_flaps_count_through_the_alphabet_up_or_down() -> None:
    from manimx.userland.code import _flaps

    assert _flaps("C", "F", "ABCDEFG") == "CDEF"
    assert _flaps("F", "C", "ABCDEFG") == "FEDC"
    assert _flaps("x", "B", "ABC") == "xB"  # not on a flap: straight there
    assert _flaps("A", "A", "ABC") == "A"


def test_a_flipboard_flips_every_cell_to_the_target() -> None:
    from manimx.userland import Flipboard
    from manimx.userland.code import glyph_points

    before, after = Code("AB 1"), Code("AD 3\nZ")
    flip = Flipboard(before, after, lag_ratio=0.2)
    slots = flip.board.children
    assert len(slots) == 4  # A, B to D, 1 to 3 and the new row's Z; spaces get no cell
    prepared = prepare(lambda s: (s.add(before), s.play(flip, run_time=2.0)))
    prepared.evaluate(0.6)
    middle = [slot.points.value for slot in slots]
    prepared.evaluate(2.0)
    assert after.present.value and not before.present.value and not flip.board.present.value
    pen = after.grid.pen(0, 1)
    np.testing.assert_allclose(slots[1].points.value, glyph_points("D", after.setup) + pen)
    assert any(len(a) != len(b) or not np.allclose(a, b) for a, b in zip(middle, [slot.points.value for slot in slots], strict=True))
    assert len(glyph_points(" ", after.setup)) == 0
    with pytest.raises(TypeError, match="Code into a Code"):
        Flipboard(before, before.lines[0])  # type: ignore[arg-type]


def test_typing_shows_characters_key_by_key_and_untyping_takes_them_back() -> None:
    from manimx.userland import Blink, TypeCode, Untype

    code = Code("ab\ncd")
    cursor = code.cursor(fill_opacity=1.0)
    typing = TypeCode(code, cursor)
    assert typing.run_time == pytest.approx(5 / 20.0)  # four characters and a line break
    prepared = prepare(lambda s: (s.play(typing), s.play(Blink(cursor, times=2), run_time=1.0), s.play(Untype(code, cursor, run_time=1.0))))
    prepared.evaluate(0.1)
    assert [glyph.alpha.value for glyph in code.glyphs] == [1.0, 1.0, 0.0, 0.0]
    assert cursor.position.y.value == pytest.approx(code.in_parent(code.grid.caret(1, 0))[1])  # after the line break
    prepared.evaluate(0.25 + 0.3)
    assert cursor.alpha.value == 0.0  # blinking
    prepared.evaluate(1.25 + 0.5)
    assert [glyph.alpha.value for glyph in code.glyphs] == [1.0, 1.0, 0.0, 0.0]  # "cd" taken back
    prepared.evaluate(2.25)
    assert not code.present.value and cursor.present.value and cursor.alpha.value == 1.0
    assert cursor.position.x.value == pytest.approx(code.in_parent(code.grid.caret(0, 0))[0])
    with pytest.raises(TypeError, match="position"):
        TypeCode(code, code.lines[0].children[0].alpha)  # type: ignore[arg-type]


def test_jitter_varies_the_pauses_the_same_way_for_a_seed() -> None:
    from manimx.userland import TypeCode

    code = Code("abcdef")
    first, again, other = (TypeCode(code, jitter=0.8, seed=seed)._presses for seed in (1, 1, 2))
    np.testing.assert_allclose(first, again)
    assert not np.allclose(first, other)


def test_cycle_characters_swaps_whole_glyphs_without_folding() -> None:
    from manimx.userland import CycleCharacters
    from manimx.userland.code import glyph_points

    before, after = Code("A"), Code("E")
    cycle = CycleCharacters(before, after, alphabet="ABCDE")
    prepared = prepare(lambda s: (s.add(before), s.play(cycle, run_time=1.0)))
    pen = after.grid.pen(0, 0)
    for time, character in ((0.1, "A"), (0.3, "B"), (0.6, "C"), (0.9, "D"), (1.0, "E")):
        prepared.evaluate(time)
        np.testing.assert_allclose(cycle.board.children[0].points.value, glyph_points(character, after.setup) + pen)


def test_typing_and_cycling_play_a_random_sound_per_step() -> None:
    from manimx.animation import Sound
    from manimx.userland import CycleCharacters, Flipboard, TypeCode
    from manimx.userland.code import _ticks

    clicks = [Sound(np.zeros(10), 1000), Sound(np.ones(10) * 0.5, 1000)]
    code = Code("ab\ncd")
    typing = TypeCode(code, sounds=clicks, seed=3)
    audio = typing.audio()
    assert len(audio) == 5 and {sound for sound, _, _ in audio} <= set(clicks)
    assert [when for _, _, when in audio] == pytest.approx([0.0, 0.2, 0.4, 0.6, 0.8])
    assert TypeCode(code).audio() == []
    # A whole board stepping at once clicks once per step; folding flaps
    # click halfway into a step.
    ways = {(0, 0): "ABCD", (0, 1): "AB"}
    assert _ticks(ways, 3, {0: (0.0, 1.0), 1: (0.0, 1.0)}, 0.0) == pytest.approx([1 / 3, 2 / 3, 1.0])
    assert _ticks(ways, 3, {0: (0.0, 1.0), 1: (0.0, 1.0)}, 0.5) == pytest.approx([1 / 6, 1 / 2, 5 / 6])
    cycle = CycleCharacters(Code("A"), Code("C"), alphabet="ABC", sounds=clicks)
    assert [when for _, _, when in cycle.audio()] == pytest.approx([0.5, 1.0])
    assert len(Flipboard(Code("A"), Code("C"), alphabet="ABC", sounds=clicks).audio()) == 2

    class Built(Scene):
        def construct(self):
            self.play(TypeCode(Code("ab"), sounds=clicks), run_time=1.0)

    prepared = Built().prepare()
    assert len(prepared.segments[0].sounds) == 2
