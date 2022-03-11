from typing import Dict, Generic, Hashable, Iterable, Iterator, TypeVar

__all__ = ['OrderedSet']

V = TypeVar('V', bound=Hashable)


class OrderedSet(Generic[V]):
    """Bare implementation of set, preserving insertion order"""

    def __init__(self, items: Iterable[V] = ()):
        self._items: Dict[V, None] = {item: None for item in items}

    def __iter__(self) -> Iterator[V]:
        return iter(self._items)

    def __contains__(self, item: V):
        return item in self._items

    def add(self, item: V):
        self._items[item] = None

    def remove(self, item: V):
        del self._items[item]

    def discard(self, item: V):
        try:
            self.remove(item)
        except KeyError:
            pass

    def clear(self):
        self._items.clear()

    def update(self, items: Iterable[V]):
        self._items.update({item: None for item in items})
