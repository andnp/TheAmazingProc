from typing import Tuple
from numba import njit
from TheAmazingProc.utils.SamplableSet import SamplableSet

Coords = Tuple[int, int]

RIGHT = 0
DOWN = 1
LEFT = 2
UP = 3

@njit(cache=True)
def intClip(x: int, mi: int, ma: int):
    if x >= ma:
        return ma - 1

    if x < mi:
        return mi

    return x

@njit(cache=True)
def toState(state: Coords, shape: Coords) -> int:
    x, y = state
    return int(y * shape[0] + x)

@njit(cache=True)
def toBoundedState(state: Coords, shape: Coords) -> int:
    x, y = state
    mx, my = shape

    return toState(
        (intClip(x, 0, mx), intClip(y, 0, my)),
        shape,
    )

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

@njit(cache=True)
def takeAction(state: Coords, action: int):
    x, y = state
    if action == RIGHT:
        x += 1
    elif action == DOWN:
        y -= 1
    elif action == LEFT:
        x -= 1
    elif action == UP:
        y += 1
    else:
        raise Exception(f'Unknown action: {action}')

    return x, y
