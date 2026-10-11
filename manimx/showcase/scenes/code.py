"""Code: the Fibonacci numbers in Haskell, from the definition to a list
defined by itself.

Code sets source with Typst's highlighting, one Shape2D per character on
a grid of rows and columns. focus() dims all but some lines or tokens,
region() gives the box around them, MorphCode moves the tokens two
versions of a program share and fades the rest, and clip() shows a
window of rows that scroll_to() scrolls.
"""

from __future__ import annotations

from manimx.animation import Scene
from manimx.userland import (
    DOWN,
    LEFT,
    UP,
    Blink,
    Code,
    FadeIn,
    FadeOut,
    Morph,
    MorphCode,
    RoundedRectangle,
    TypeCode,
    Typst,
)

from .style import BLUE, GREEN, INK, MUTED, RED, SAND, VIOLET, caption

PREAMBLE = f'#set text(fill: rgb("{INK}"))\n#show raw: set text(size: 22pt)'
COLORS = {
    "keyword": RED,
    "keyword.operator": SAND,
    "constant.numeric": SAND,
    "storage.type": BLUE,
    "support": VIOLET,
    "entity.name.function": GREEN,
    "comment": MUTED,
}
NOTE = f'#set text(size: 20pt, fill: rgb("{MUTED}"))'
TOP_LEFT = LEFT * 6.3 + UP * 2.5

NAIVE = """\
fib :: Integer -> Integer
fib 0 = 0
fib 1 = 1
fib n = fib (n - 1) + fib (n - 2)"""

ACCUMULATING = """\
fib :: Integer -> Integer
fib n = go n 0 1
  where
    go 0 a _ = a
    go k a b = go (k - 1) b (a + b)"""

LAZY = """\
fib :: [Integer]
fib = 0 : 1 : zipWith (+) fib (tail fib)"""

MODULE = """\
module Main where

-- | The Fibonacci numbers, defined by themselves:
-- each one is the sum of the two before it.
fib :: [Integer]
fib = 0 : 1 : zipWith (+) fib (tail fib)

-- | The n-th Fibonacci number, counted from 0.
nth :: Int -> Integer
nth n = fib !! n

main :: IO ()
main = do
  print (take 10 fib)
  print (nth 100)"""


NUMBER = f'#set text(size: 22pt, fill: rgb("{INK}"), font: "DejaVu Sans Mono")'


def code(source: str) -> Code:
    return Code(source, "haskell", PREAMBLE, COLORS).move_to(TOP_LEFT)


def label_at(text: str, y: float) -> Code:
    """``text`` as code at the left edge, its row centered on ``y``."""
    label = Code(text, "haskell", PREAMBLE, COLORS).move_to((TOP_LEFT[0], y, 0.0))
    center, _, _ = label.region(label.glyphs)
    return label.shift((0.0, y - center[1], 0.0))


class Fibonacci(Scene):
    def construct(self) -> None:
        title = caption("Code", "Fibonacci in Haskell: focus, boxes, MorphCode and scrolling windows")
        self.play(FadeIn(title), run_time=0.6)

        naive = code(NAIVE)
        cursor = naive.cursor(fill_color=INK, fill_opacity=1.0, stroke_width=0.0)
        self.play(TypeCode(naive, cursor, speed=28.0, jitter=0.5))
        self.play(Blink(cursor, times=2), run_time=1.0)
        self.play(FadeOut(cursor), run_time=0.2)
        note = self.say("The definition, read literally", None)

        box = RoundedRectangle(*self.box_size(naive, naive.line(2), naive.line(3)), corner_radius=0.08, stroke_color=SAND, stroke_width=3.0)
        box.move_to(naive.region(naive.line(2), naive.line(3))[0])
        self.play(naive.animate.focus(naive.line(2), naive.line(3)), FadeIn(box), run_time=0.8)
        note = self.say("Two base cases", note)
        self.wait(0.6)
        self.play(naive.animate.focus(naive.line(4)), self.frame(box, naive, naive.line(4)), run_time=0.8)
        note = self.say("Two recursive calls each: exponential time", note)
        self.play(naive.animate.focus(naive.select("fib (n - 1)"), naive.select("fib (n - 2)")), self.frame(box, naive, naive.select("fib (n - 1)"), naive.select("fib (n - 2)")), run_time=0.8)
        self.wait(0.8)

        accumulating = code(ACCUMULATING)
        self.play(FadeOut(box), naive.animate.unfocus(), run_time=0.4)
        self.play(MorphCode(naive, accumulating), run_time=1.6)
        note = self.say("Carry two neighbours along: linear time", note)
        box = RoundedRectangle(*self.box_size(accumulating, accumulating.line(5)), corner_radius=0.08, stroke_color=SAND, stroke_width=3.0)
        box.move_to(accumulating.region(accumulating.line(5))[0])
        self.play(accumulating.animate.focus(accumulating.line(5)), FadeIn(box), run_time=0.8)
        self.wait(1.0)

        lazy = code(LAZY)
        self.play(FadeOut(box), accumulating.animate.unfocus(), run_time=0.4)
        self.play(MorphCode(accumulating, lazy), run_time=1.6)
        note = self.say("Or the whole list at once, defined by itself", note)
        self.wait(0.6)
        self.zip_with(lazy)
        note = self.say("Laziness computes each number once, when asked for", note)
        self.wait(1.0)

        module = code(MODULE)
        window = module.clip(rows=6)
        self.play(FadeOut(lazy), *(FadeOut(node) for node in self.table), FadeIn(window), run_time=0.8)
        note = self.say("A window on a longer program, scrolled", note)
        self.play(module.animate.scroll_to(5), run_time=1.2)
        self.play(module.animate.focus(module.line(6)), run_time=0.6)
        self.wait(0.6)
        self.play(module.animate.unfocus().scroll_to(10), run_time=1.2)
        self.play(module.animate.focus(module.line(14), module.line(15)), run_time=0.6)
        output = Typst("[0,1,1,2,3,5,8,13,21,34]\n\n354224848179261915075", NUMBER.replace(INK, GREEN))
        output.move_to(DOWN * 1.6)
        self.play(FadeIn(output, shift=UP * 0.3), run_time=0.8)
        self.wait(1.2)
        self.play(*(FadeOut(node) for node in (window, output, note, title)), run_time=0.6)

    # -- helpers ------------------------------------------------------------

    def say(self, text: str, old: Typst | None) -> Typst:
        """Replace the note at the bottom with ``text``."""
        new = Typst(text, NOTE).move_to(DOWN * 3.2)
        self.play(*([FadeOut(old)] if old is not None else []), FadeIn(new, shift=UP * 0.2), run_time=0.5)
        return new

    @staticmethod
    def box_size(code: Code, *parts) -> tuple[float, float]:
        _, width, height = code.region(*parts)
        return width, height

    @staticmethod
    def frame(box: RoundedRectangle, code: Code, *parts):
        center, width, height = code.region(*parts)
        return box.animate.move_to(center).set_size(width, height)

    def zip_with(self, lazy: Code) -> None:
        """Lay ``fib``, ``tail fib`` and their sums under each other and
        fill the list in, each sum becoming the next number of both rows."""
        values = [0, 1]
        while len(values) < 9:
            values.append(values[-1] + values[-2])
        rows = {"fib": -0.2, "tail fib": -0.9, "sums": -1.8}
        left, step = -3.4, 0.95

        def cell(value: int, row: str, column: int) -> Typst:
            return Typst(str(value), NUMBER).move_to((left + column * step, rows[row], 0.0))

        labels = [label_at(text, rows[row]) for text, row in (("fib", "fib"), ("tail fib", "tail fib"), ("zipWith (+)", "sums"))]
        rule = RoundedRectangle(8.6, 0.02, corner_radius=0.01, fill_color=MUTED, fill_opacity=1.0, stroke_width=0.0).move_to((0.0, -1.35, 0.0))
        fib = [cell(values[0], "fib", 0), cell(values[1], "fib", 1)]
        tail = [cell(values[1], "tail fib", 0)]
        self.play(*(FadeIn(label) for label in labels), FadeIn(rule), *(FadeIn(number) for number in (*fib, *tail)), run_time=0.8)
        self.table = [*labels, rule]

        line = lazy.line(2)
        parts = {"fib": lazy.select("fib", 1), "tail fib": lazy.select("tail fib"), "sums": lazy.select("zipWith (+)")}
        for row in ("fib", "tail fib", "sums"):
            self.play(lazy.animate.focus(parts[row]), run_time=0.35)
            self.wait(0.3)
        self.play(lazy.animate.focus(line), run_time=0.3)

        for column in range(7):
            total = values[column + 2]
            run_time = 0.9 if column < 2 else 0.45
            total_cell = cell(total, "sums", column)
            self.play(FadeIn(total_cell, shift=UP * 0.2), run_time=run_time * 0.6)
            moves = [Morph(cell(total, "sums", column), cell(total, "fib", column + 2))]
            if column + 1 < 8:
                moves.append(Morph(cell(total, "sums", column), cell(total, "tail fib", column + 1)))
            self.play(*moves, run_time=run_time)
            fib.append(moves[0].target)
            if len(moves) > 1:
                tail.append(moves[1].target)
            self.table.append(total_cell)
        self.table += [*fib, *tail]
        self.play(lazy.animate.unfocus(), run_time=0.4)
