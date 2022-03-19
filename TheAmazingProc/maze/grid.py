from typing import Tuple
from numba import njit
from utils.SamplableSet import SamplableSet

Coords = Tuple[int, int]

@njit(cache=True)
def clip(x: float, mi: float, ma: float):
    if x >= ma:
        return ma - 1

    if x < mi:
        return mi

    return x

@njit(cache=True)
def toBoundedState(state: Coords, shape: Coords) -> int:
    x, y = state
    mx, my = shape

    s = clip(y, 0, my) * shape[0] + clip(x, 0, mx)
    return int(s)

@njit(cache=True)
def toCoords(state: int, shape: Coords) -> Coords:
    y = int(state // shape[0])
    x = int(state % shape[0])
    return (x, y)

# TODO: for some weird reason this cannot be cached
@njit
def neighbors(state: int, shape: Coords):
    x, y = toCoords(state, shape)

    return SamplableSet([
        toBoundedState((x + 1, y), shape),
        toBoundedState((x - 1, y), shape),
        toBoundedState((x, y + 1), shape),
        toBoundedState((x, y - 1), shape),
    ])
