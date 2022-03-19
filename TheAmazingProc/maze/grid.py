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

@njit(cache=True)
def actions(state: int, nstate: int, shape: Coords):
    x, y = toCoords(state, shape)
    ret = []

    # this constructivist approach to building the action list accounts
    # for the possibility of multiple actions leading ot the same next state
    # for example, in a corner state multiple actions bump into a wall
    # and the state does not change
    up = toBoundedState((x, y + 1), shape)
    if up == nstate:
        ret.append(0)

    right = toBoundedState((x + 1, y - 1), shape)
    if right == nstate:
        ret.append(1)

    down = toBoundedState((x, y - 1), shape)
    if down == nstate:
        ret.append(2)

    left = toBoundedState((x - 1, y), shape)
    if left == nstate:
        ret.append(3)

    return ret
