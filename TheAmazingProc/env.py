import numpy as np
from typing import List, Tuple
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

@njit(cache=True)
def _samplePath(unvisited: SamplableSet, shape: Coords):
    cell = unvisited.sample()

    path = [cell]

    while unvisited.contains(cell):
        cell = neighbors(cell, shape).sample()

        # delete loops, otherwise we cannot guarantee solvability
        if cell in path:
            path = path[0:path.index(cell) + 1]
        else:
            path.append(cell)

    return path

@njit(cache=True)
def _drawLine(maze: np.ndarray, start: Coords, end: Coords):
    # moving left
    if start[0] > end[0]:
        y = start[1]
        for x in range(end[0], start[0] + 1):
            maze[x, y] = 0

    # moving down
    elif start[1] < end[1]:
        x = start[0]
        for y in range(start[1], end[1] + 1):
            maze[x, y] = 0

    # moving right
    elif start[0] < end[0]:
        y = start[1]
        for x in range(start[0], end[0] + 1):
            maze[x, y] = 0

    # moving up
    elif start[1] > end[1]:
        x = start[0]
        for y in range(end[1], start[1] + 1):
            maze[x, y] = 0

    return maze

@njit(cache=True)
def _project(state: int, shape: Coords):
    x, y = toCoords(state, shape)

    return ( 2 * x + 1, 2 * y + 1 )

@njit(cache=True)
def _carvePath(maze: np.ndarray, path: List[int], size: Coords):
    # walk through the path and mark visited states as non-wall states
    last = _project(path[0], size)
    for state in path:
        # project the x, y coord onto the full maze
        start = _project(state, size)
        maze = _drawLine(maze, start, last)

        last = start

    return maze

@njit(cache=True)
def sample(size: Coords, seed: int):
    # the random seed is set only for this function invocation
    # because of the njit compilation
    np.random.seed(seed)

    maze = np.ones((
        size[0] * 2 + 1,
        size[1] * 2 + 1,
    ), dtype=np.int_)

    unvisited = SamplableSet(range(size[0] * size[1]))
    start = unvisited.sample()
    unvisited.remove(start)

    while unvisited.length() > 0:
        path = _samplePath(unvisited, size)
        maze = _carvePath(maze, path, size)

        for cell in path[:-1]:
            unvisited.remove(cell)

    return maze


# out = neighbors(toBoundedState((4, 2), (5, 5)), (5, 5))
# print([toCoords(s, (5, 5)) for s in out])

# maze = np.ones((15, 15), dtype=np.int_)
# path = _samplePath(np.arange(1, 5 * 5), (5, 5))
# maze = _carvePath(maze, path)
# print([toCoords(s, (5,5)) for s in path])
# print(maze)

# out = sample((30, 30), 0)

# import matplotlib.pyplot as plt
# plt.imshow(out, cmap='gray_r')
# plt.show()
