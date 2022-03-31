from typing import Tuple, TypeVar

T = TypeVar('T')
def updateElement(tup: Tuple[T, ...], idx: int, val: T) -> Tuple[T, ...]:
    def _iter():
        for i, v in enumerate(tup):
            if i == idx: yield val
            else: yield v

    return tuple(_iter())
