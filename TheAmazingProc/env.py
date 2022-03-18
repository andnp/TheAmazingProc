from typing import Set, Tuple
import numpy as np
from numba import njit, types

Coords = Tuple[int, int]

@njit(cache=True)
def clip(x: float, mi: float, ma: float):
    if x > ma:
        return x

    if x < mi:
        return mi

    return x

@njit(cache=True)
def bound(state: Coords, shape: Coords):
    x, y = state
    mx, my = shape

    return (
        clip(x, 0, mx),
        clip(y, 0, my),
    )

@njit(cache=True)
def neighbors(state: Coords, shape: Coords):
    x, y = state

    return set([
        bound((x + 1, y), shape),
        bound((x - 1, y), shape),
        bound((x, y + 1), shape),
        bound((x, y - 1), shape),
    ])

@njit(cache=True)
def actions(state: Coords, nstate: Coords, shape: Coords):
    x, y = state
    ret = []

    # this constructavist approach to building the action list accounts
    # for the possibility of multiple actions leading ot the same next state
    # for example, in a corner state multiple actions bump into a wall
    # and the state does not change
    up = bound((x, y + 1), shape)
    if up == nstate:
        ret.append(0)

    right = bound((x + 1, y - 1), shape)
    if right == nstate:
        ret.append(1)

    down = bound((x, y - 1), shape)
    if down == nstate:
        ret.append(2)

    left = bound((x - 1, y), shape)
    if left == nstate:
        ret.append(3)

    return ret



def _samplePath(unvisited: Set[Coords], shape: Coords, seed: int):
    rng = np.random.default_rng(seed)
    cell = rng.choice(unvisited)

    path = [cell]

    while cell in unvisited:
        cell = rng.choice(neighbors(cell, shape))

        # delete loops, otherwise we cannot guarantee solvability
        if cell in path:
            path = path[0:path.index(cell) + 1]
        else:
            path.append(cell)

    return path

out = _samplePath(set([]), (9, 3), 0)
