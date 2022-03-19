import numpy as np
from numba import njit
from utils.SamplableSet import SamplableSet
from maze.grid import Coords, neighbors, toCoords

@njit(cache=True)
def _samplePath(unvisited: SamplableSet, shape: Coords):
    # grab a random unvisited cell that will start our path
    cell = unvisited.sample()
    path = [cell]

    # from that starting cell, randomly traverse neighboring cells
    # until we collide with any visited cell in the maze
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
def sample(size: Coords, seed: int):
    # the random seed is set only for this function invocation
    # because of the njit compilation
    np.random.seed(seed)

    # NOTE:
    # the maze will be twice the size of the requested coords
    # in order to accomodate the walls taking up a state cell
    # pay special attention to bottom and right walls
    maze = np.ones((
        size[0] * 2 + 1,
        size[1] * 2 + 1,
    ), dtype=np.int_)

    # mark every cell as unvisited within the "inner" grid
    # then pick a random starting point to seed the monte
    # carlo path sampling
    unvisited = SamplableSet(range(size[0] * size[1]))
    start = unvisited.sample()
    unvisited.remove(start)

    # while there are still unvisited states, keep sampling paths
    while unvisited.length() > 0:
        path = _samplePath(unvisited, size)

        # walk through the path and mark visited states as non-wall states
        last = _project(path[0], size)
        for cell in path:
            # project the x, y coord onto the full maze
            pcell = _project(cell, size)
            maze = _drawLine(maze, pcell, last)

            # technically only the last cell in the path won't be
            # in the unvisited set.. So this is just a little wasteful
            # but it's constant time compute, so not noticeable
            if unvisited.contains(cell):
                unvisited.remove(cell)
            last = pcell

    return maze
