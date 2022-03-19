from typing import List, Set, Tuple
import numpy as np
from numba import njit
import numba.typed.typedlist as NList

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

@njit(cache=True)
def neighbors(state: int, shape: Coords):
    x, y = toCoords(state, shape)

    return set([
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
def _samplePath(unvisited: List[int], shape: Coords):
    idx = np.random.randint(0, len(unvisited))
    cell = unvisited[idx]

    path = [cell]

    while cell in unvisited:
        possible_next = np.array(list(neighbors(cell, shape)))
        idx = np.random.randint(0, len(possible_next))
        cell = possible_next[idx]

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

    return ( 3 * x + 1, 3 * y + 1 )

@njit(cache=True)
def _carvePath(maze: np.ndarray, path: List[int]):
    # the inner shape defines the shape of the path generator
    # the maze itself will have walls which occupy space
    inner_shape = (
        int(maze.shape[0] // 3),
        int(maze.shape[1] // 3),
    )

    # walk through the path and mark visited states as non-wall states
    last = _project(path[0], inner_shape)
    for state in path:
        # project the x, y coord onto the full maze
        start = _project(state, inner_shape)
        maze = _drawLine(maze, start, last)

        last = start

    return maze

@njit(cache=True)
def sample(size: Coords, seed: int):
    # the random seed is set only for this function invocation
    # because of the njit compilation
    np.random.seed(seed)

    # the inner shape defines the shape of the path generator
    # the maze itself will have walls which occupy space
    # TODO: assumes even divisibility
    inner_shape = (
        int(size[0] // 3),
        int(size[1] // 3),
    )

    maze = np.ones(size, dtype=np.int_)

    # TODO: this _really_ should be a set instead of a list
    # However the default `Set` has O(n) sampling, so we just trade one
    # O(n) op for another. Can be solved with a smarter set type
    # for O(1) content checking and O(1) sampling
    unvisited = list(range(inner_shape[0] * inner_shape[1]))

    start = np.random.randint(0, len(unvisited))
    unvisited.remove(start)

    while len(unvisited) > 0:
        print(len(unvisited))
        path = _samplePath(unvisited, inner_shape)
        maze = _carvePath(maze, path)

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

# out = sample((90, 90), 0)

# import matplotlib.pyplot as plt
# plt.imshow(out)
# plt.show()
