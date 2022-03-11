import asyncio
import inspect
import sys
import time
import traceback
from contextlib import suppress
from datetime import datetime
from functools import cmp_to_key
from threading import Event, Thread
from typing import (
    Any, AsyncIterator, Awaitable, Callable, Collection, Dict, IO, Iterable,
    List, Literal, Optional, Tuple, TypedDict, Union,
)

import terminaltables
from asciimatics.exceptions import NextScene, ResizeScreenError, StopApplication
from asciimatics.scene import Scene
from asciimatics.screen import Screen
from asciimatics.widgets import Frame, Layout, Widget
from terminaltables.build import flatten
from terminaltables.width_and_alignment import RE_COLOR_ANSI

from tui_progress.terminal import original_stdstreams, visible_width
from tui_progress.tui import ColoredMultiColumnListBox, ManagedScreen

ColumnDefnSource = Union[str, 'FullColumnDefn', 'ColumnDefinition']
ValueRow = Dict[str, Any]
SeparatorRow = None
RowDefn = Union[ValueRow, SeparatorRow]

ValueTransformer = Callable[[Any], Any]
RowValueTransformer = Callable[[Any, Dict[str, Any], str, Dict[str, 'ColumnDefinition']], Any]
TransformerFunc = Union[ValueTransformer, RowValueTransformer]
TransformerSource = Union[str, TransformerFunc, Collection[TransformerFunc]]

InputsSource = Union[Iterable[RowDefn], Callable[[], Iterable[RowDefn]]]


class FullColumnDefn(TypedDict, total=False):
    title: str
    is_key: bool
    formatter: TransformerSource
    justify: Literal['left', 'right', 'center']
    sort: Optional[Literal['asc', 'desc']]
    sort_order: Union[float, int]
    sort_key: TransformerSource
    min_width: Optional[int]


class ColumnDefinition:
    # Key in row data to find the column value
    name: str

    # Title to display for the column
    title: str

    # Whether the column value should be used to uniquely identify the row
    #  (used in interactive tables, to ensure selection persists across updates)
    is_key: bool

    # Minimum width of the column
    min_width: Optional[int]

    # Transform a column value to a displayable format
    formatter: RowValueTransformer
    source_formatter: TransformerSource

    # Which direction to align column values to
    justify: Literal['left', 'right', 'center']

    # How to sort the column: asc or desc
    sort: Optional[Literal['asc', 'desc']]

    # If multiple columns are sorted, in which order is this column sorted
    sort_order: Union[float, int]

    # Transform a column into a sortable value
    sort_key: TransformerSource

    # Extra user-supplied keys to provide metadata on the column
    meta: Dict[str, Any]

    def __init__(
        self,
        name: str,
        title: str = None,
        is_key: bool = False,
        min_width: int = None,
        formatter: TransformerSource = lambda s: s,
        justify: Literal['left', 'right', 'center'] = 'left',
        sort: Optional[Literal['asc', 'desc']] = None,
        sort_order: Union[float, int] = float('Inf'),
        sort_key: TransformerSource = lambda s: s,
        **meta
    ):
        self.name = name
        self.title = title if title is not None else name.capitalize()
        self.is_key = is_key
        self.min_width = min_width

        self.source_formatter = formatter
        self.formatter = init_transformer(self.source_formatter)
        self.justify = justify

        self.sort = sort.lower() if sort else None
        self.sort_order = sort_order
        self.sort_key = init_transformer(sort_key)

        self.meta = meta

        self._cmp = CMP_STRATEGIES.get(self.sort)

    @property
    def is_sortable(self):
        return self._cmp is not None

    @property
    def cmp(self):
        return self._cmp

    def get_title(self) -> str:
        title = self.title

        if self.min_width:
            justify = {
                'left': str.ljust,
                'center': str.center,
                'right': str.rjust,
            }

            lines = title.splitlines()
            lines[0] = justify[self.justify](lines[0], self.min_width)
            title = '\n'.join(lines)

        return title

    def get_row_sort_key(self, data: Dict[str, Any], columns: Dict[str, 'ColumnDefinition']) -> Any:
        value = data[self.name]
        return self.cmp(self.sort_key(value, data, self.name, columns))

    def get_row_formatted_value(self, data: Dict[str, Any], columns: Dict[str, 'ColumnDefinition']) -> Any:
        value = data[self.name]
        return self.formatter(value, data, self.name, columns)


CMP_STRATEGIES = {
    'asc' : cmp_to_key(lambda x, y: (x > y) - (x < y)),
    'desc': cmp_to_key(lambda x, y: (x < y) - (x > y)),
}


def is_row_value_transformer(transformer: TransformerFunc) -> bool:
    """Determine whether a transformer func accepts row args

    >>> is_row_value_transformer(lambda v: v)
    False
    >>> is_row_value_transformer(lambda value, data, column, columns: value)
    True
    """
    sig = inspect.signature(transformer)

    try:
        sig.bind('value', 'data', 'column', 'columns')
    except TypeError:
        return False
    else:
        return True


def as_row_value_transformer(transformer: ValueTransformer) -> RowValueTransformer:
    """Wrap a value-transformer to be used as a row value-transformer
    """
    return lambda value, data, column, columns: transformer(value)


def init_transformer(transformer: TransformerSource) -> RowValueTransformer:
    """Process any type of transformer source into a row value-transformer
    """
    ###
    # A transformer may be the name of another column
    #
    if isinstance(transformer, str):
        column_name = transformer

        def get_column(value, data, column, columns):
            return data[column_name]

        get_column.__name__ = f'get_{column_name}'
        return get_column

    ###
    # A transformer may be a function that accepts either a single column value,
    # or the whole row's data + metadata (column value, row data, name of column,
    # and column definition)
    #
    elif callable(transformer):
        if not is_row_value_transformer(transformer):
            transformer = as_row_value_transformer(transformer)
        return transformer

    ###
    # A transformer may be a collection of the above types, which will be called
    # in sequence, using the previous transformer's return value as the next
    # transformer's column value (the only argument if a value-transformer, or
    # the first argument if a row value-transformer)
    #
    elif isinstance(transformer, Collection):
        transformers = transformer
        transformers = [
            init_transformer(transformer)
            for transformer in transformers
        ]

        def chained_transformer(value, data, column, columns):
            for transformer in transformers:
                value = transformer(value, data, column, columns)
            return value

        return chained_transformer

    else:
        raise ValueError(
            f'A transformer must either be a string, a single-argument method, '
            f'a four-argument method, or a list of all three. '
            f'Found: ({type(transformer)}) {transformer!r}')


def init_column(name: str,
                defn: ColumnDefnSource,
                **defaults,
                ) -> ColumnDefinition:
    """Fill a column's definition with defaults
    """
    if isinstance(defn, ColumnDefinition):
        return defn

    if isinstance(defn, str):
        defn = {
            'title': defn,
        }

    return ColumnDefinition(name, **defaults, **defn)


def init_columns(columns: Dict[str, ColumnDefnSource],
                 **defaults,
                 ) -> Dict[str, ColumnDefinition]:
    """Fill all column definitions with defaults
    """
    return {
        column: init_column(column, defn, **defaults)
        for column, defn in columns.items()
    }


def sort_by_columns(columns: Dict[str, ColumnDefinition],
                    inputs: Iterable[RowDefn],
                    ) -> Iterable[RowDefn]:
    """Sort input data as specified by column definitions
    """
    sorting = [
        defn
        for defn in sorted(columns.values(), key=lambda defn: defn.sort_order)
        if defn.is_sortable
    ]

    if sorting:
        inputs = list(inputs)
        if any(row is None for row in inputs):
            raise TypeError(
                'Table row separators (row values of None) are incompatible with sorting. '
                'Either remove the row separators, or disable sorting.')

        inputs = sorted(inputs, key=lambda data: [
            defn.get_row_sort_key(data, columns)
            for defn in sorting
        ])

    return inputs


def render_table(columns: Dict[str, ColumnDefnSource],
                 inputs: InputsSource,
                 title: str = None,
                 **kwargs,
                 ) -> 'SingleTable':
    """Format a list of dicts as a table
    """
    columns = init_columns(columns)

    if callable(inputs):
        get_inputs = inputs
        inputs = get_inputs()

    inputs = sort_by_columns(columns, inputs)

    def format_row(data, columns):
        if data is not None:
            return [
                defn.get_row_formatted_value(data, columns)
                for defn in columns.values()
            ]

    rows = [format_row(data, columns) for data in inputs]

    header = [defn.get_title() for defn in columns.values()]
    rows.insert(0, header)

    table = SingleTable(rows, source_dicts=inputs, title=title)
    for attr, value in kwargs.items():
        setattr(table, attr, value)

    table.justify_columns = {
        index: defn.justify
        for index, defn in enumerate(columns.values())
    }

    return table


def display_table(columns: Dict[str, ColumnDefnSource],
                  inputs: InputsSource,
                  title: str = None,
                  file: IO = None,
                  **kwargs,
                  ) -> None:
    """Display a list of dicts as a table in the terminal
    """
    table = render_table(columns=columns, inputs=inputs, title=title, **kwargs)
    print(table.table, file=file)


def watch_table(columns: Dict[str, ColumnDefnSource],
                get_inputs: Callable[[], Iterable[RowDefn]],
                title: str = None,
                refresh: float = 5,
                on_screen_change: Callable[['WatchTable'], Any] = None,
                **kwargs,
                ) -> None:
    """Format a list of dicts as an interactive, periodically-refreshing table
    """
    table = WatchTable(
        columns=columns,
        get_inputs=get_inputs,
        title=title,
        refresh=refresh,
        on_screen_change=on_screen_change,
        **kwargs,
    )
    table.display_forever()


class WatchTable:
    def __init__(
        self,
        columns: Dict[str, ColumnDefnSource],
        get_inputs: Callable[[], Iterable[RowDefn]],
        title: str = None,
        refresh: float = 5,
        on_screen_change: Callable[['WatchTable'], Any] = None,
        swallow_errors: bool = True,
        has_border: bool = True,
        **kwargs,
    ):
        # Pre-init column definitions, so they needn't be initialized on every render
        self.columns = init_columns(columns)
        self.key = [col.name for col in self.columns.values() if col.is_key]

        self.get_inputs = get_inputs
        self.title = title
        self.refresh = refresh
        self.on_screen_change = on_screen_change
        self.swallow_errors = swallow_errors
        self.has_border = has_border
        self.table_kwargs = kwargs

        self.is_initialized: bool = False
        self.list_box: Optional[ColoredMultiColumnListBox] = None
        self.screen: Optional[Screen] = None
        self.frame: Optional[TableFrame] = None

        #: Errors raised while calling get_inputs, as (raised_at, traceback_str)
        self.encountered_errors: List[Tuple[datetime, str]] = []

        self._halt = False

    def stop(self):
        self._halt = True

    def display_forever(self):
        """Render the interactive table, and refresh forever
        """
        with original_stdstreams():
            while not self._halt:
                try:
                    self.initialize()

                    with ManagedScreen() as screen:
                        update_thread = IntervalThread(self.refresh, self._update, screen)
                        update_thread.start()

                        real_refresh = screen.refresh

                        def haltable_refresh(*args, **kwargs):
                            if self._halt:
                                raise StopApplication('Halt requested')
                            return real_refresh(*args, **kwargs)

                        screen.refresh = haltable_refresh

                        try:
                            self._watch(screen)
                        finally:
                            update_thread.stop()
                            self.screen = None
                except ResizeScreenError:
                    pass
                except KeyboardInterrupt:
                    self.stop()
                    if self.screen:
                        self.screen.close()
                    self.print_encountered_errors()
                    raise

    def initialize(self):
        if self.is_initialized:
            return

        self.list_box = ColoredMultiColumnListBox(
            name=self.title,
            key=self.key,
            height=Widget.FILL_FRAME,
            columns=[],
            options=[],
        )

        while True:
            attrs = self.get_widget_attrs()
            if attrs:
                break
            else:
                time.sleep(1)
                continue

        self.list_box.update_from_attrs(attrs)
        self.is_initialized = True

    def _update(self, screen: Screen):
        updated_attrs = self.get_widget_attrs()
        if updated_attrs is None:
            return

        prev_value = self.list_box.value
        self.list_box.update_from_attrs(updated_attrs)
        self.list_box.value = prev_value
        self.list_box.update(None)
        screen.force_update()

    def _watch(self, screen: Screen):
        self.screen = screen
        self.frame = self.create_frame()
        scene = Scene([self.frame], 0)

        if self.on_screen_change:
            self.on_screen_change(self)

        self.screen.play([scene], start_scene=scene, stop_on_resize=True)

    def create_frame(self):
        return TableFrame(self.screen, self.list_box, title=self.title, has_border=self.has_border)

    def print_encountered_errors(self):
        if self.encountered_errors:
            print_err = lambda *a, **k: print(*a, **k, file=sys.stderr)

            print_err(f'Encountered {len(self.encountered_errors)} error(s) while refreshing\n')

            for exc_at, tb in self.encountered_errors:
                print_err(f'At {exc_at}\n')
                print_err(tb, '\n\n')

    def get_widget_attrs(self) -> Dict[str, Any]:
        try:
            return self._get_widget_attrs()
        except (StopApplication, NextScene):
            raise
        except KeyboardInterrupt:
            self.stop()
        except Exception:
            if self.swallow_errors:
                tb = traceback.format_exc()
                exc_at = datetime.now()
                self.encountered_errors.append((exc_at, tb))
            else:
                self.stop()
                raise

    def _get_widget_attrs(self) -> Dict[str, Any]:
        inputs = self.get_inputs()
        table = render_table(
            columns=self.columns, inputs=inputs, title=self.title, **self.table_kwargs)
        return self.list_box.parse_attrs_from_table(table)


class AsyncTable:
    """An interactive, periodically-refreshing table powered by asyncio

    This table can be used as an async context manager — useful when you're
    doing things and want to show progress info about them; or this table can
    be used as an awaitable — useful when you're merely displaying status info
    about things going on outside of the program.

    Usage as context manager:

        async def my_provider():
            while True:
                yield await get_rows()
                await asyncio.sleep(1)

        async with AsyncTable(columns=my_columns, provider=my_provider()):
            while things_to_do:
                await do_a_thing()

    Usage as awaitable:

        async def my_provider():
            while True:
                yield await get_infos()
                await asyncio.sleep(1)

        await AsyncTable(columns=my_columns, provider=my_provider())

    """

    def __init__(
        self,
        columns: Dict[str, ColumnDefnSource],
        provider: Union[AsyncIterator[Iterable[RowDefn]], Callable[[], Awaitable[Iterable[RowDefn]]]],
        title: str = None,
        refresh: float = None,
        on_screen_change: Callable[['AsyncTable'], Any] = None,
        swallow_errors: bool = True,
        has_border: bool = True,
        tick: float = 0.05,
        **kwargs,
    ):
        # Pre-init column definitions, so they needn't be initialized on every render
        self.columns = init_columns(columns)
        self.key = [col.name for col in self.columns.values() if col.is_key]

        self.provider = provider
        self.title = title
        self.refresh = refresh
        self.on_screen_change = on_screen_change
        self.swallow_errors = swallow_errors
        self.has_border = has_border
        self.tick = tick
        self.table_kwargs = kwargs

        self.list_box = ColoredMultiColumnListBox(
            name=self.title,
            key=self.key,
            height=Widget.FILL_FRAME,
            columns=[],
            options=[],
        )

        #: Errors raised while calling get_inputs, as (raised_at, traceback_str)
        self.encountered_errors: List[Tuple[datetime, str]] = []

        self._task: Optional[asyncio.Task] = None

    def __await__(self):
        return self._display().__await__()

    async def __aenter__(self):
        self._task = asyncio.create_task(self._display())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._task:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _inputs_iterator(self):
        if inspect.iscoroutinefunction(self.provider):
            while True:
                yield await self.provider()

        elif inspect.isasyncgen(self.provider):
            async for inputs in self.provider:
                yield inputs

    async def _display(self):
        screen: Optional[Screen] = None
        update_task: Optional[asyncio.Task] = None
        are_inputs_depleted = asyncio.Event()
        got_first_input = asyncio.Event()
        did_initial_refresh_screen_callback = False

        async def periodic_update():
            async for inputs in self._inputs_iterator():
                table = render_table(
                    columns=self.columns,
                    inputs=inputs,
                    title=self.title,
                    **self.table_kwargs,
                )
                self.list_box.update_from_table(table)

                self.list_box.update(None)
                if screen is not None:
                    screen.force_update()

                got_first_input.set()

                if self.refresh is not None:
                    await asyncio.sleep(self.refresh)

            # When inputs are depleted, end display
            are_inputs_depleted.set()

        try:
            with original_stdstreams():
                while not are_inputs_depleted.is_set():
                    with ManagedScreen() as screen:
                        frame = TableFrame(screen, self.list_box,
                                           title=self.title,
                                           has_border=self.has_border)
                        scene = Scene([frame], -1)
                        screen.set_scenes([scene], start_scene=scene)

                        if update_task is None:
                            update_task = asyncio.create_task(periodic_update())

                        if self.on_screen_change:
                            self.on_screen_change(self)

                        while not screen.has_resized() and not are_inputs_depleted.is_set():
                            screen.draw_next_frame()

                            if got_first_input.is_set() and not did_initial_refresh_screen_callback:
                                if self.on_screen_change:
                                    self.on_screen_change(self)
                                did_initial_refresh_screen_callback = True

                            await asyncio.sleep(self.tick)
        finally:
            if screen is not None:
                screen.close()

            if update_task:
                update_task.cancel()
                with suppress(asyncio.CancelledError):
                    await update_task


class IntervalThread(Thread):
    """Invokes a callback every N seconds, until stop() is called"""

    def __init__(self, interval: float, callback: Callable, *args, **kwargs):
        Thread.__init__(self)
        self.interval = interval
        self.callback = callback
        self.args = args
        self.kwargs = kwargs
        self._cancel = Event()

    def run(self):
        while not self._cancel.wait(self.interval):
            self.callback(*self.args, **self.kwargs)

    def stop(self):
        self._cancel.set()


class TableFrame(Frame):
    def __init__(self, screen: Screen, list_box: ColoredMultiColumnListBox, **kwargs):
        super().__init__(screen, screen.height, screen.width, **kwargs)

        layout = Layout([100], fill_frame=True)
        self.add_layout(layout)

        layout.add_widget(list_box)
        self.set_theme('monochrome')

        self.fix()


class TableSeparatorMixin:
    """Allow a row value of None to print a horizontal border

    By default, AsciiTables only allow a single footer row. This mixin also allows
    the inner_footing_row_border instance var to be set to an integer describing
    the number of footer rows there are (i.e. the number of rows to print after
    the footer border)

    XXX: this mixin now does more things — rename and redocument
    """

    show_header_row: bool

    table_data: List[Dict[str, Any]]

    inner_column_border: bool
    inner_footing_row_border: bool
    inner_heading_row_border: bool
    inner_row_border: bool
    outer_border: bool

    horizontal_border: Callable[[str, Collection[int]], Iterable[str]]
    gen_row_lines: Callable[[Dict, str, Collection[int], int], Iterable[str]]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.show_header_row = True

    def gen_table(self, inner_widths, inner_heights, outer_widths):
        """Combine everything and yield every line of the entire table with borders.

        :param iter inner_widths: List of widths (no padding) for each column.
        :param iter inner_heights: List of heights (no padding) for each row.
        :param iter outer_widths: List of widths (with padding) for each column.
        :return:
        """
        # Yield top border.
        if self.outer_border:
            yield self.horizontal_border('top', outer_widths)

        # Yield table body.
        row_count = len(self.table_data)
        last_row_index = row_count - 1
        before_footer_row_index = row_count - (self.inner_footing_row_border + 1)

        for i, row in enumerate(self.table_data):
            # Yield the row line by line (e.g. multi-line rows).
            if self.inner_heading_row_border and i == 0:
                if not self.show_header_row:
                    continue
                style = 'heading'
            elif self.inner_footing_row_border and i == last_row_index:
                style = 'footing'
            else:
                style = 'row'

            if row is None:
                yield self.horizontal_border('row', outer_widths)
                continue

            for line in self.gen_row_lines(row, style, inner_widths, inner_heights[i]):
                yield line

            # If this is the last row then break. No separator needed.
            if i == last_row_index:
                break

            # Yield heading separator.
            if self.inner_heading_row_border and i == 0:
                yield self.horizontal_border('heading', outer_widths)

            # Yield footing separator.
            elif self.inner_footing_row_border and i == before_footer_row_index:
                yield self.horizontal_border('footing', outer_widths)

            # Yield row separator.
            elif self.inner_row_border:
                yield self.horizontal_border('row', outer_widths)

        # Yield bottom border.
        if self.outer_border:
            yield self.horizontal_border('bottom', outer_widths)

    @property
    def table(self):
        """Return a large string of the entire table ready to be printed to the terminal."""
        dimensions = max_dimensions_ignore_none(self.table_data, self.padding_left, self.padding_right)[:3]
        return flatten(self.gen_table(*dimensions))


def max_dimensions_ignore_none(table_data, padding_left=0, padding_right=0, padding_top=0, padding_bottom=0):
    """Get maximum widths of each column and maximum height of each row.

    NOTE: this method skips None values, for use with TableSeparatorMixin

    :param iter table_data: List of list of strings (unmodified table data).
    :param int padding_left: Number of space chars on left side of cell.
    :param int padding_right: Number of space chars on right side of cell.
    :param int padding_top: Number of empty lines on top side of cell.
    :param int padding_bottom: Number of empty lines on bottom side of cell.

    :return: 4-item tuple of n-item lists. Inner column widths and row heights, outer column widths and row heights.
    :rtype: tuple
    """
    inner_widths = [0] * (max(len(r) if r is not None else 0 for r in table_data) if table_data else 0)
    inner_heights = [0] * len(table_data)

    # Find max width and heights.
    for j, row in enumerate(table_data):
        if row is None:
            continue

        for i, cell in enumerate(row):
            if not hasattr(cell, 'count') or not hasattr(cell, 'splitlines'):
                cell = str(cell)
            if not cell:
                continue
            inner_heights[j] = max(inner_heights[j], cell.count('\n') + 1)
            inner_widths[i] = max(inner_widths[i], *[visible_width(l) for l in cell.splitlines()])

    # Calculate with padding.
    outer_widths = [padding_left + i + padding_right for i in inner_widths]
    outer_heights = [padding_top + i + padding_bottom for i in inner_heights]

    return inner_widths, inner_heights, outer_widths, outer_heights


class SingleTable(TableSeparatorMixin, terminaltables.SingleTable):
    def __init__(self, table_data, source_dicts=None, title=None):
        """Constructor.

        :param iter table_data: List (empty or list of lists of strings) representing the table.
        :param title: Optional title to show within the top border of the table.
        """
        self.source_dicts = source_dicts
        super().__init__(table_data, title=title)


# Monkey-patch our fixed visible_width
import terminaltables.width_and_alignment

terminaltables.build.visible_width = visible_width
terminaltables.width_and_alignment.visible_width = visible_width
