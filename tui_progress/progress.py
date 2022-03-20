import asyncio
import atexit
import threading
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from typing import Callable, Dict, List, Optional, TextIO

import cursor

from tui_progress.collections import OrderedSet

TERM_MOVE_UP = '\x1b[A'
TERM_CLEAR_LINE = '\x1b[2K'

NOTSET = object()


class RepeatingTimer(threading.Timer):
    """Timer which repeatedly calls a function until cancel() is called"""

    def run(self):
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)


class ProgressIndicator:
    """Represents a text-based progress/status indicator that prints to terminal
    """

    def __init__(self, stream: TextIO, name: str = None, **kwargs):
        super().__init__(**kwargs)
        self.stream: TextIO = stream
        self.name = name
        self.manager: ProgressIndicatorManager = _progress_indicators[self.stream]

        # Whether the indicator needs re-rendering
        self.is_dirty: bool = False
        # Text to render
        self.frame: str = ''
        # Optional frame to print after the indicator is finished
        self.summary_frame: Optional[str] = None

        # Backend used for update_loop
        self.loop_backend: Optional[_UpdateLoopBackend] = None

    def __str__(self):
        name = self.name or '<unnamed>'
        frame = self.summary_frame or self.frame
        return f'{name} {frame!r}'

    def __repr__(self):
        cls = self.__class__.__name__
        return f'<{cls}: {self}>'

    @property
    def is_managed(self):
        return self in self.manager

    def update(self, frame: str):
        """Set the next frame to render, and trigger a refresh"""
        self.frame = frame
        self.is_dirty = True

        if self.is_managed:
            self.manager.trigger_refresh()

    # For compatibility as a file-like object
    write = update

    def start(self):
        """Begin rendering the indicator"""
        self.manager.add(self)

    def stop(self):
        """Stop rendering the indicator"""
        if self.loop_backend:
            self.loop_backend.stop()
            self.loop_backend = None

        self.manager.remove(self)

    def complete(self, frame: str):
        """Stop rendering and print a final frame"""
        self.summary_frame = frame
        self.manager.trigger_refresh()
        self.stop()

    def update_loop(self, get_frame: Callable[[], str], interval: float):
        """Call a method to retrieve the next frame and update every interval"""
        if not self.is_managed:
            # Manager must be started to determine whether the render loop is async
            self.start()

        if self.manager.is_async():
            backend_cls = _AsyncUpdateLoopBackend
        else:
            backend_cls = _SyncUpdateLoopBackend

        self.loop_backend = backend_cls(self, get_frame, interval)
        self.loop_backend.start()


class ProgressIndicatorManager:
    """Handles the rendering of concurrent progress indicators
    """

    def __init__(self, stream: TextIO):
        self._stream = stream

        # Indicators we manage rendering of
        self.indicators: OrderedSet[ProgressIndicator] = OrderedSet()
        self.finished_indicators: OrderedSet[ProgressIndicator] = OrderedSet()
        # Temporary lists used to iterate the managed indicators during a
        # rendering iteration. It is stored as an instance variable to reduce
        # the number of allocations.
        self._rendering_indicators = []
        self._finished_indicators = []

        # Flag used to prevent multiple rendering loops from being started
        self._is_started: bool = False
        # Flag used to determine which rendering loop to use.
        # This value is set to True if a running event loop is found.
        self._async_mode: Optional[bool] = None

        # Handles the details of rendering indicators in a loop (sync vs async)
        self.backend: Optional[_IndicatorRenderLoopBackend] = None

        # Ensure a hidden TTY cursor is re-enabled at exit
        atexit.register(self.show_cursor)

    def __bool__(self):
        return bool(self.indicators)

    def __contains__(self, indicator: ProgressIndicator):
        return indicator in self.indicators

    def is_async(self) -> bool:
        """Return whether the rendering loop is async"""
        return self._async_mode

    def mutation_lock(self):
        if self.backend:
            return self.backend.mutation_lock()
        else:
            return nullcontext()

    def add(self, indicator: ProgressIndicator):
        """Begin management and rendering of an indicator"""
        with self.mutation_lock():
            if indicator not in self.indicators:
                self.indicators.add(indicator)
                self.start()

                # Any additions of indicators should trigger a refresh, as a
                # new line must be rendered.
                self.trigger_refresh()

    def remove(self, indicator: ProgressIndicator):
        """Stop managing and rendering the indicator"""
        with self.mutation_lock():
            if indicator in self.indicators:
                self.indicators.discard(indicator)
                self.finished_indicators.add(indicator)

        ###
        # NOTE: Any removal should trigger a refresh, as all the lines need to be
        #       re-rendered. However, if the last indicator has just been removed,
        #       the Main thread may immediately exit after this call — in that
        #       case, we manually render the final frame, and avoid triggering a
        #       refresh, because it may cause writes to a now-closed stdout/stderr
        #       in another thread
        #
        if not self.indicators:
            self.render_frame()
            self.stop()
        else:
            self.trigger_refresh()

    def trigger_refresh(self):
        """Trigger the rendering of indicators"""
        self.backend.trigger_refresh()

    def start(self):
        if self._is_started:
            return

        try:
            self._async_mode = asyncio.get_running_loop() is not None
        except RuntimeError:
            self._async_mode = False

        if self._async_mode:
            backend_cls = _AsyncIndicatorRenderLoopBackend
        else:
            backend_cls = _SyncIndicatorRenderLoopBackend

        self.backend = backend_cls(self)
        self.backend.start()

        self._is_started = True
        self.hide_cursor()

    def stop(self):
        if not self._is_started:
            return

        self.backend.stop()

        self._is_started = False
        self._async_mode = None
        self.show_cursor()

    def hide_cursor(self):
        if self._stream.isatty():
            cursor.hide(stream=self._stream)

    def show_cursor(self):
        if self._stream.isatty():
            cursor.show(stream=self._stream)

    def render_frame(self):
        indicators: List[ProgressIndicator] = self._rendering_indicators
        finished_indicators: List[ProgressIndicator] = self._finished_indicators

        with self.mutation_lock():
            indicators[:] = self.indicators
            finished_indicators[:] = self.finished_indicators
            self.finished_indicators.clear()

        # print finished summary lines first (don't include in moveup lines)
        # print regular indicators (forced if there were any finished indicators)
        # for every non-summary finished indicator, clear + newline
        # move up len(indicators) + len(non_summary)

        vanished_lines = 0
        for indicator in finished_indicators:
            if indicator.summary_frame is not None:
                self.clear_line()
                self._write(indicator.summary_frame + '\n')
                indicator.summary_frame = None
            else:
                vanished_lines += 1

        forced_render = bool(finished_indicators)
        for indicator in indicators:
            if forced_render or indicator.is_dirty:
                self.clear_line()
                self.render_indicator(indicator)
            else:
                self._write('\n')

        for _ in range(vanished_lines):
            self.clear_line()
            self._write('\n')

        if move_up := len(indicators) + vanished_lines:
            self.moveto(-move_up)

        self._flush()

    def render_indicator(self, indicator: ProgressIndicator, newline: bool = True):
        self._write(indicator.frame + '\n' * newline)
        indicator.is_dirty = False

    def render_summary(self, indicator: ProgressIndicator) -> bool:
        if indicator.summary_frame is not None:
            self._write(indicator.summary_frame)
            indicator.summary_frame = None
            return True
        else:
            return False

    def clear_line(self):
        self._write(TERM_CLEAR_LINE + '\r')

    def moveto(self, n):
        self._write('\n' * n + TERM_MOVE_UP * -n)

    def _write(self, s: str):
        self._stream.write(s)

    def _flush(self):
        self._stream.flush()


class _UpdateLoopBackend:
    def __init__(self, indicator: 'ProgressIndicator', get_frame: Callable[[], str], interval: float):
        self.indicator = indicator
        self.get_frame = get_frame
        self.interval = interval

    def update(self):
        frame = self.get_frame()
        self.indicator.update(frame)

    def start(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError


class _AsyncUpdateLoopBackend(_UpdateLoopBackend):
    def __init__(self, indicator: 'ProgressIndicator', get_frame: Callable[[], str], interval: float):
        super().__init__(indicator, get_frame, interval)
        self.task: Optional[asyncio.Task] = None

    def start(self):
        if self.task is None:
            self.task = asyncio.create_task(self.loop())

    def stop(self):
        if self.task:
            self.task.cancel()
            self.task = None

    async def loop(self):
        try:
            while True:
                self.update()
                await asyncio.sleep(self.interval)
        except asyncio.CancelledError:
            self.update()
            await asyncio.sleep(0)


class _SyncUpdateLoopBackend(_UpdateLoopBackend):
    def __init__(self, indicator: 'ProgressIndicator', get_frame: Callable[[], str], interval: float):
        super().__init__(indicator, get_frame, interval)
        self.timer: Optional[threading.Timer] = None

    def start(self):
        if self.timer is None:
            self.timer = RepeatingTimer(self.interval, self.update)
            self.timer.start()

    def stop(self):
        if self.timer:
            self.timer.cancel()
            self.timer = None


class _IndicatorRenderLoopBackend:
    def __init__(self, manager: 'ProgressIndicatorManager'):
        self.manager = manager

    def render_frame(self):
        self.manager.render_frame()

    @contextmanager
    def mutation_lock(self):
        raise NotImplementedError

    def start(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    def trigger_refresh(self):
        raise NotImplementedError


class _AsyncIndicatorRenderLoopBackend(_IndicatorRenderLoopBackend):
    def __init__(self, manager: 'ProgressIndicatorManager'):
        super().__init__(manager)
        self.task: Optional[asyncio.Task] = None
        self.refresh_needed: Optional[asyncio.Event] = None

    @contextmanager
    def mutation_lock(self):
        yield

    def start(self):
        if not self.task:
            self.refresh_needed = asyncio.Event()
            self.task = asyncio.create_task(self.render_loop())

    def stop(self):
        if self.task:
            self.task.cancel()
            self.task = None
            self.refresh_needed = None

    def trigger_refresh(self):
        self.refresh_needed.set()

    async def render_loop(self):
        while True:
            await self.refresh_needed.wait()
            self.render_frame()
            self.refresh_needed.clear()


class _SyncIndicatorRenderLoopBackend(_IndicatorRenderLoopBackend):
    def __init__(self, manager: 'ProgressIndicatorManager'):
        super().__init__(manager)
        self.thread: Optional[threading.Thread] = None

        self.halt: Optional[threading.Event] = None
        self.refresh_needed: Optional[threading.Event] = None
        self.has_event: Optional[threading.Condition] = None

        self.mutex: Optional[threading.Lock] = None

    @contextmanager
    def mutation_lock(self):
        with self.mutex:
            yield

    def start(self):
        if not self.thread:
            self.halt = threading.Event()
            self.refresh_needed = threading.Event()
            self.has_event = threading.Condition()
            self.mutex = threading.Lock()

            self.thread = threading.Thread(target=self.render_loop)
            self.thread.setDaemon(True)
            self.thread.start()

    def stop(self):
        if threading.current_thread() is self.thread:
            # Ensure we don't deadlock acquiring self.has_event by performing
            # stop in another thread.
            request_stop = threading.Thread(target=self.stop)
            request_stop.start()
            request_stop.join()
            return

        if self.thread:
            self.halt.set()
            with self.has_event:
                self.has_event.notify()

            self.thread.join()

            self.thread = None
            self.halt = None
            self.mutex = None

    def trigger_refresh(self):
        self.refresh_needed.set()
        with self.has_event:
            self.has_event.notify()

    def render_loop(self):
        with self.has_event:
            while not self.halt.is_set():
                if self.refresh_needed.is_set():
                    self.render_frame()
                    self.refresh_needed.clear()

                self.has_event.wait()


class ProgressIndicatorStreamManager(defaultdict):
    def __missing__(self, stream: TextIO):
        self[stream] = mgr = ProgressIndicatorManager(stream)
        return mgr


_progress_indicators: Dict[TextIO, ProgressIndicatorManager] = ProgressIndicatorStreamManager()
