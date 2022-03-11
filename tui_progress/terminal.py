import functools
import inspect
import os
import sys
import unicodedata
from contextlib import ContextDecorator, contextmanager
from datetime import datetime
from typing import Callable, ContextManager, Optional, TextIO, Union

import wrapt
from halo import Halo
from halo._utils import decode_utf_8_text
from humanize import naturaldelta
from log_symbols.symbols import LogSymbols
from terminaltables.width_and_alignment import RE_COLOR_ANSI
from tqdm import tqdm

from tui_progress.progress import ProgressIndicator

TERM_MOVE_UP = '\x1b[A'
TERM_CLEAR_LINE = '\x1b[2K'

NOTSET = object()


class SubtaskComplete(Exception):
    """Non-error signal exception to print subtask completion message

    If this is fired, the subtask knows not to print its own "Done"
    """
    def __init__(self, message: str = None):
        self.message = message


class Subtask(wrapt.ObjectProxy):
    """tqdm wrapper with extra subtask methods
    """
    step_prefix: str

    def __init__(self, wrapped, *, step_prefix: str):
        super().__init__(wrapped)
        self.step_prefix = step_prefix

    def done(self, msg: Optional[str] = 'Done!'):
        """Stop subtask execution and optionally print message"""
        raise SubtaskComplete(msg)

    def progress(self, *args, **kwargs) -> tqdm:
        default_bar_format = (
            '[{n_fmt} / {total_fmt}] '
            '{desc}{bar} '
            '[{elapsed}<{remaining}, {rate_fmt}{postfix}]'
        )
        bar_format = kwargs.pop('bar_format', None) or default_bar_format
        kwargs['bar_format'] = f'{self.step_prefix}{bar_format}'

        kwargs.setdefault('position', (self.pos or 0) + 1)

        pbar = tqdm(*args, **kwargs)
        return pbar


class NestedHalo(Halo):
    """Halo spinner which may be printed above/below other spinners
    """
    def __init__(self, text='', color='cyan', spinner=None,
                 animation=None, placement='left', interval=-1, enabled=True, stream=sys.stderr):
        super().__init__(text=text, color=color, spinner=spinner, animation=animation,
                         placement=placement, interval=interval, enabled=enabled, stream=stream,)

        self.indicator = ProgressIndicator(self._stream, name=self.__class__.__name__)

    def start(self, text=None):
        self.indicator.update_loop(self.frame, 0.001 * self._interval)
        return self

    def stop(self):
        self.indicator.stop()
        return self

    def stop_and_persist(self, symbol=' ', text=None):
        symbol = decode_utf_8_text(symbol)

        if text is not None:
            text = decode_utf_8_text(text)
        else:
            text = self._text['original']
        text = text.strip()

        if self._placement == 'right':
            prefix, suffix = text, symbol
        else:
            prefix, suffix = symbol, text

        output = f'{prefix} {suffix}'
        self.indicator.complete(output)

    def __call__(self, f):
        """Use Halo as decorator (may be used with async functions)"""

        if inspect.iscoroutinefunction(f):
            @functools.wraps(f)
            async def wrapped(*args, **kwargs):
                with self:
                    return await f(*args, **kwargs)

        else:
            @functools.wraps(f)
            def wrapped(*args, **kwargs):
                with self:
                    return f(*args, **kwargs)

        return wrapped


class AppendingHalo(NestedHalo):
    """Halo spinner whose message methods append to current text, not overwrite
    """
    def __init__(self, text='', color='cyan', spinner=None, animation=None,
                 placement='left', interval=-1, enabled=True, stream=sys.stderr, **kwargs):
        self.original_text = text

        super().__init__(text, color, spinner, animation, placement, interval,
                         enabled, stream, **kwargs)

    def __exit__(self, type, value, traceback):
        # Don't return a value, because the exception is silenced otherwise.
        # (The original Halo class does this)
        super().__exit__(type, value, traceback)

    def _append_message(self, method, text=None, overwrite=False):
        if not overwrite:
            text = f'{self.original_text} {text}'
        return method(text)

    def succeed(self, text=None, overwrite=False):
        return self._append_message(super().succeed, text=text, overwrite=overwrite)

    def fail(self, text=None, overwrite=False):
        return self._append_message(super().fail, text=text, overwrite=overwrite)

    def warn(self, text=None, overwrite=False):
        return self._append_message(super().warn, text=text, overwrite=overwrite)

    def info(self, text=None, overwrite=False):
        return self._append_message(super().info, text=text, overwrite=overwrite)

    def print(self, text: str = ''):
        """Print a message above the Halo line
        """
        self.clear()
        print(text)
        self._render_frame()

    def _print_with_symbol(self, symbol: str, text: str):
        self.print(f'{symbol} {text}')

    def print_succeed(self, text: str):
        self._print_with_symbol(LogSymbols.SUCCESS.value, text)

    def print_fail(self, text: str):
        self._print_with_symbol(LogSymbols.ERROR.value, text)

    def print_warn(self, text: str):
        self._print_with_symbol(LogSymbols.WARNING.value, text)

    def print_info(self, text: str):
        self._print_with_symbol(LogSymbols.INFO.value, text)


class TimedHalo(AppendingHalo):
    """Halo spinner showing elapsed time when succeed(), fail(), etc called
    """
    start_dt: Optional[datetime]
    end_dt: Optional[datetime]
    elapsed: Union[bool, str, Callable[[datetime, datetime], str]]

    def __init__(
        self,
        *args,
        elapsed: Union[bool, str, Callable[[datetime, datetime], str]] = True,
        **kwargs,
    ):
        """
        :param elapsed:
            Determines whether/how to show elapsed time with printed messages.
            If False, no elapsed times will be displayed with messages.
            If True, the elapsed time will be displayed as "[Δ 2 seconds]".
            If a string, it will be printed verbatim.
            If a callable, the passed method will be called with (start_dt, end_dt),
            and its return value will be printed.
        """
        super().__init__(*args, **kwargs)
        self.start_dt = None
        self.end_dt = None
        self.elapsed = elapsed

    def start_timer(self):
        self.start_dt = datetime.utcnow()

    def stop_timer(self):
        if not self.end_dt:
            self.end_dt = datetime.utcnow()

    def start(self, text=None):
        self.start_timer()
        return super().start(text)

    def stop(self):
        self.stop_timer()
        return super().stop()

    def stop_and_persist(self, symbol=' ', text=None):
        self.stop_timer()

        if text is not None:
            text = decode_utf_8_text(text)
        else:
            text = self._text['original']

        elapsed_text = self.format_elapsed()
        if elapsed_text:
            text = f'{text} {elapsed_text}'

        return super().stop_and_persist(symbol, text)

    def format_elapsed(self) -> str:
        """Return a string representation of the elapsed time
        """
        if not self.elapsed:
            return ''
        elif isinstance(self.elapsed, str):
            return self.elapsed
        elif callable(self.elapsed):
            return self.elapsed(self.start_dt, self.end_dt)
        else:
            elapsed = self.end_dt - self.start_dt
            return f' [Δ {naturaldelta(elapsed)}]'


devnull = open(os.devnull, 'w')


@contextmanager
def new_subtask(
    message: str,
    finished: Optional[str] = 'Done!',
    *,
    show_elapsed: bool = False,
    enabled: bool = True,
    **halo_kwargs,
) -> Union[ContextManager[TimedHalo], ContextDecorator]:
    """Show a spinner during part of a task, appending message upon completion

    NOTE: an ellipsis ("...") will be appended to the message.
    """
    stream = sys.stderr if enabled else devnull

    with TimedHalo(f'{message} ...', stream=stream, elapsed=show_elapsed, **halo_kwargs) as subtask:
        try:
            yield subtask
        except Exception:
            subtask.fail('ERROR')
            raise
        else:
            if finished is not None:
                subtask.succeed(finished)


def timed_subtask(*args, **kwargs) -> Union[ContextManager[TimedHalo], ContextDecorator]:
    """Shortcut for new_subtask(..., show_elapsed=True)"""
    return new_subtask(*args, **kwargs, show_elapsed=True)


class nested_tqdm(tqdm):
    """tqdm which may be nested/run concurrently with other nested_tqdm's/NestedHalos
    """
    def __init__(self, *args, file: TextIO = sys.stderr, **kwargs):
        self.indicator = ProgressIndicator(file, name=self.__class__.__name__)
        self.indicator.start()
        super().__init__(*args, file=file, **kwargs)

    def close(self):
        if self.leave:
            self.ncols = 0  # only show summary
            self.indicator.complete(self.__repr__())
        else:
            self.indicator.stop()

        self.disable = True

    def status_printer(self, file):
        return self.indicator.update

    def moveto(self, n):
        pass  # noop


class subtqdm(nested_tqdm):
    """tqdm with subtask convenience contextmanager
    """
    @contextmanager
    def subtask(self,
                title: str = None,
                default_done_msg: Optional[str] = 'Done!',
                print_spacer: bool = True,
                include_step_prefix: bool = True,
                ) -> ContextManager['Subtask']:
        """Declare a subtask, which updates the task pbar upon completion
        """
        step_prefix = ''
        if include_step_prefix:
            step_number = self.n + 1
            step_prefix = f'[STEP {step_number}] '

        if title is not None:
            self.write(f'{step_prefix}{title}')

        try:
            yield Subtask(self, step_prefix=step_prefix)
        except SubtaskComplete as done:
            message = done.message
        else:
            message = default_done_msg

        if message:
            self.write(f'{step_prefix}{message}')
        if print_spacer:
            self.write('')

        self.update()


class subtrange(subtqdm):
    """A shortcut for subtqdm(xrange(*args), **kwargs).
    """
    def __init__(self, *args, **kwargs):
        iterable = range(*args)
        super().__init__(iterable, **kwargs)


@contextmanager
def original_stdstreams():
    """Restore native stdout/stderr streams temporarily

    Upon import, the halo library automatically calls colorama.init(), which
    replaces stdout & stderr with wrapper streams, interfering with color output
    in asciimatics. Unfortunately, calling colorama.deinit() has *not* resolved
    the issue — stdout & stderr remain wrappers — but restoring stdout and stderr
    with their original values (provided by Python as sys.__stdout__ and
    sys.__stderr__) does resolve the issue.
    """
    prev_stdout = sys.stdout
    prev_stderr = sys.stderr

    try:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        yield
    finally:
        sys.stdout = prev_stdout
        sys.stderr = prev_stderr


def visible_length(unistr):
    """Returns the number of printed characters in a Unicode string.

    Source: https://stackoverflow.com/a/30888594/148585
    """
    return sum(1 for char in unistr if unicodedata.combining(char) == 0)


def visible_width(string):
    """Get the visible width of a unicode string.

    This version of terminaltables's visible_width also takes into account
    Unicode characters which provide ligatures or styling, which do not
    have any width.
    """
    if '\033' in string:
        string = RE_COLOR_ANSI.sub('', string)

    # Convert to unicode.
    try:
        string = string.decode('u8')
    except (AttributeError, UnicodeEncodeError):
        pass

    return visible_length(string)
