import asyncio
import inspect
import operator
import os
import re
from contextlib import contextmanager, nullcontext
from itertools import chain, zip_longest
from typing import (
    Any,
    Awaitable,
    Callable,
    Collection,
    Coroutine,
    Dict,
    Iterable,
    List,
    Optional,
    TYPE_CHECKING,
    Union,
)


from asciimatics import screen
from asciimatics.event import KeyboardEvent
from asciimatics.exceptions import ResizeScreenError
from asciimatics.screen import Canvas, Screen
from asciimatics.widgets import Label, MultiColumnListBox
from terminaltables.width_and_alignment import RE_COLOR_ANSI
from wcwidth import wcwidth

from tui_progress.itertools import chunk
from tui_progress.terminal import original_stdstreams, visible_width

if TYPE_CHECKING:
    from tui_progress.table import RowDefn, SingleTable


__all__ = [
    'open_screen',
    'ManagedTui',
    'ColoredLabel',
    'ColoredMultiColumnListBox',
]


def open_screen(*args, **kwargs) -> Screen:
    """Open an asciimatics screen, ensuring size is set properly

    Sometimes ncurses doesn't report the correct terminal size,
    causing window resizes to leave a small TUI in a large window;
    or worse: cause wrapping to occur, absolutely mutilating the TUI.

    This method grabs the proper terminal size ourselves and reports it
    to the asciimatics screen.
    """
    size = os.get_terminal_size()

    kwargs['height'] = size.lines
    screen = Screen.open(*args, **kwargs)
    screen.height, screen.width = size.lines, size.columns

    return screen


class ManagedScreen(screen.ManagedScreen):
    """asciimatic's ManagedScreen, ensuring window resizes are properly handled"""

    def _open_screen(self) -> Screen:
        return open_screen()

    def __call__(self, *args, **kwargs):
        screen = self._open_screen()
        kwargs["screen"] = screen
        output = self.func(*args, **kwargs)
        screen.close()
        return output

    def __enter__(self):
        """
        Method used for with statement
        """
        self.screen = self._open_screen()
        return self.screen


class ManagedTui:
    """Renders and manages an async asciimatics Screen"""

    def __init__(
        self,
        init_screen: Callable[[Screen], Awaitable],
        tasks: Iterable[Union[asyncio.Task, Coroutine, Callable[[], Coroutine]]],
        *,
        tick: float = 0.05,
        enable_pdb: bool = False,
    ):
        self.init_screen = init_screen
        self.tasks = tasks
        self.tick = tick
        self.enable_pdb = enable_pdb

        self.screen: Optional[Screen] = None
        self._managed_tasks: Optional[List[asyncio.Task]]

    async def __aenter__(self):
        self._managed_tasks = []

        tasks = [
            self._managed_screen_loop,
            *self.tasks,
        ]
        for task in tasks:
            if inspect.iscoroutinefunction(task):
                task = task()
            if inspect.iscoroutine(task):
                task = asyncio.create_task(task)

            if isinstance(task, asyncio.Task):
                self._managed_tasks.append(task)
            else:
                raise TypeError(
                    f'Provided tasks must be asyncio.Tasks, '
                    f'coroutine functions (async def), or coroutines. '
                    f'Found {type(task)} for {task!r}')

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        for task in self._managed_tasks:
            task.cancel()

        if self.screen:
            self.screen.close()

    async def force_update(self):
        if self.screen:
            self.screen.force_update()

    async def _start_screen(self):
        self.screen = open_screen()
        await self.init_screen(self.screen)

    async def _managed_screen_loop(self):
        with original_stdstreams():
            while True:
                await self._start_screen()
                try:
                    await self._screen_render_loop()
                except ResizeScreenError:
                    # Close the screen and continue the loop, where a new Screen
                    # will be initialized with the new window size.
                    self.screen.close()

    async def _screen_render_loop(self):
        with (self.breakpoint_on_error() if self.enable_pdb else nullcontext()):
            while True:
                self.screen.draw_next_frame()
                if self.screen.has_resized():
                    raise ResizeScreenError('Screen resized')
                await asyncio.sleep(self.tick)

    @contextmanager
    def breakpoint_on_error(self):
        """Debugging helper — on Exception: closes screen, prints traceback, drops to PDB
        """
        import traceback

        try:
            yield
        except (ResizeScreenError, KeyboardInterrupt):
            raise
        except Exception:
            if self.screen:
                self.screen.close()
            traceback.print_exc()
            breakpoint()
            raise

    async def breakpoint(self, reopen_screen: bool = True):
        """Debugging helper — closes screen, prints traceback, drops to PDB
        """
        if self.screen:
            self.screen.close()
        else:
            reopen_screen = False

        breakpoint()

        if reopen_screen:
            self.screen = Screen.open()
            await self.init_screen(self.screen)


class ColoredMultiColumnListBox(MultiColumnListBox):
    ALIGNMENTS = {
        'left': '<',
        'center': '^',
        'right': '>',
    }

    def __init__(self,
                 *args,
                 key: Union[str, Collection[str], Callable[['RowDefn'], Any]] = None,
                 **kwargs):
        """A MultiColumnListBox supporting the output of colours (and other invisible chars)

        :param key:
            A key in each row that uniquely identifies it, or a function that accepts
            the row and returns a unique identifier. This will be used to ensure user
            selections are preserved across updates (if possible).

        """
        if not isinstance(key, str) and isinstance(key, Collection):
            keys = tuple(key)
            if len(keys) == 1:
                key, = keys
            else:
                key = lambda row: tuple(row[k] for k in keys)

        if isinstance(key, str):
            key = operator.itemgetter(key)

        self.key = key

        super().__init__(*args, **kwargs)

    @property
    def columns(self):
        return self._columns

    @columns.setter
    def columns(self, columns):
        self._columns = []
        self._align = []
        self._spacing = []
        for i, column in enumerate(columns):
            if isinstance(column, int):
                self._columns.append(column)
                self._align.append("<")
            else:
                match = re.match(r"([<>^]?)(\d+)([%]?)", column)
                self._columns.append(float(match.group(2)) / 100
                                     if match.group(3) else int(match.group(2)))
                self._align.append(match.group(1) if match.group(1) else "<")
            self._spacing.append(1 if (i > 0 and self._align[i] == "<" and self._align[i-1] == ">") else 0)

    @property
    def titles(self):
        return self._titles

    @titles.setter
    def titles(self, titles):
        self._titles = titles

    @property
    def height(self):
        return self._required_height

    @height.setter
    def height(self, height):
        self._required_height = height

    def parse_attrs_from_table(self, table: 'SingleTable') -> Dict[str, Any]:
        return dict(
            columns=[
                f'{self.ALIGNMENTS.get(justify, "")}{width + 2}'
                for justify, width in zip(table.justify_columns.values(), table.column_widths)
            ],
            options=[
                (row, self.key(table.source_dicts[i]) if self.key else i)
                for i, row in enumerate(table.table_data[1:])
            ],
            titles=table.table_data[0],
        )

    def update_from_attrs(self, attrs: Dict[str, Any], preserve_value: bool = True):
        with self.value_preserved() if preserve_value else nullcontext():
            for name, value in attrs.items():
                setattr(self, name, value)

    def update_from_table(self, table: 'SingleTable'):
        attrs = self.parse_attrs_from_table(table)
        self.update_from_attrs(attrs)

    @contextmanager
    def value_preserved(self):
        last_value = self.value
        yield
        self.value = last_value

    def process_event(self, event):
        if isinstance(event, KeyboardEvent):
            if len(self._options) > 0 and event.key_code == Screen.KEY_HOME:
                self._line = 0
                self.value = self._options[self._line][1]
            elif len(self._options) > 0 and event.key_code == Screen.KEY_END:
                self._line = len(self._options) - 1
                self.value = self._options[self._line][1]
            else:
                return super().process_event(event)
        else:
            return super().process_event(event)

    def update(self, frame_no):
        self._draw_label()

        # Calculate new visible limits if needed.
        height = self._h
        width = self._w
        dy = 0

        # Clear out the existing box content
        (colour, attr, bg) = self._frame.palette["field"]
        for i in range(height):
            self._frame.canvas.print_at(
                " " * width,
                self._x + self._offset,
                self._y + i + dy,
                colour, attr, bg)

        # Allow space for titles if needed.
        if self._titles:
            dy += 1
            height -= 1

        # Decide whether we need to show or hide the scroll bar and adjust width accordingly.
        if self._add_scroll_bar:
            self._add_or_remove_scrollbar(width, height, dy)
        if self._scroll_bar:
            width -= 1

        # Now draw the titles if needed.
        if self._titles:
            row_dx = 0
            colour, attr, bg = self._frame.palette["title"]
            for i, (title, align, space) in enumerate(zip(self._titles, self._align, self._spacing)):
                cell_width = self._get_width(self._columns[i], width)
                vis_width = visible_width(title)

                left_pad = 0
                right_pad = 0
                pad = cell_width - vis_width
                if align == '^':
                    left_pad = pad // 2
                    right_pad = pad - left_pad
                elif align == '>':
                    left_pad = pad
                else:
                    right_pad = pad

                _print_at_color_aware(
                    self._frame.canvas,
                    "{}{}{}".format(
                        " " * (space + left_pad),
                        title,
                        " " * right_pad,
                        ),
                    self._x + self._offset + row_dx,
                    self._y,
                    colour, attr, bg,
                    )
                row_dx += cell_width + space

        # Don't bother with anything else if there are no options to render.
        if len(self._options) <= 0:
            return

        # Render visible portion of the text.
        self._start_line = max(0, max(self._line - height + 1,
                                      min(self._start_line, self._line)))
        for i, [row, _] in enumerate(self._options):
            if not (self._start_line <= i < self._start_line + height):
                continue

            colour, attr, bg = self._pick_colours("field", i == self._line)
            row_dx = 0
            # Try to handle badly formatted data, where row lists don't
            # match the expected number of columns.
            for text, cell_width, align, space in zip_longest(row, self._columns, self._align, self._spacing, fillvalue=""):
                if cell_width == "":
                    break
                cell_width = self._get_width(cell_width, width)
                vis_width = visible_width(str(text))

                left_pad = 0
                right_pad = 0
                pad = cell_width - vis_width
                if align == '^':
                    left_pad = pad // 2
                    right_pad = pad - left_pad
                elif align == '>':
                    left_pad = pad
                else:
                    right_pad = pad

                _print_at_color_aware(
                    self._frame.canvas,
                    "{}{}{}".format(
                        " " * (space + left_pad),
                        text,
                        " " * right_pad,
                        ),
                    self._x + self._offset + row_dx,
                    self._y + i + dy - self._start_line,
                    colour, attr, bg,
                    )
                row_dx += cell_width + space

        # And finally draw any scroll bar.
        if self._scroll_bar:
            self._scroll_bar.update()


class ColoredLabel(Label):
    """Label widget supporting ANSI/xterm-256 colour printing
    """
    def update(self, frame_no):
        palette_key = self._pick_palette_key('label', selected=False, allow_input_state=False)
        color, attr, bg = self._frame.palette[palette_key]
        _print_at_color_aware(
            self._frame.canvas,
            self._text,
            self._x, self._y,
            color, attr, bg,
        )


def _print_at_color_aware(canvas: Canvas, text, x, y, color, attr, bg):
    RE_COLOR_ANSI = re.compile(r'(\033\[([\d;]+)m)', re.UNICODE)

    parts = RE_COLOR_ANSI.split(text)

    for escape, code, chars in chunk(chain(['\x1b[0m', str(color)], parts), 3):
        if not chars:
            continue

        ops = [int(op) for op in code.split(';')]
        if len(ops) == 1:
            color = ops[0]
        elif len(ops) == 3 and ops[:2] == [38, 5]:
            color = ops[2]

        canvas.print_at(chars, x, y, color, attr, bg)
        x += len(chars)


def _enforce_width_color_aware(text, width):
    # Double-width strings cannot be more than twice the string length, so no need to try
    # expensive truncation if this upper bound isn't an issue.
    if 2 * len(text) < width:
        return text

    final = ''
    parts = RE_COLOR_ANSI.split(text)

    size = 0
    for code, part in chunk(chain([''], parts), 2):
        final += code

        if size >= width:
            continue
        else:
            for i, c in enumerate(part):
                w = wcwidth(c) if ord(c) >= 256 else 1
                if size + w > width:
                    final += part[:i]
                    break
                size += w
            else:
                final += part

    return final
