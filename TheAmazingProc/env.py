import numpy as np
from typing import Dict, Tuple, TypedDict
from TheAmazingProc.maze.grid import Coords, takeAction
from TheAmazingProc.maze.maze import sample
from TheAmazingProc.utils.tuples import updateElement
from RlGlue import BaseEnvironment

class Element(TypedDict):
    last_visit: int
    type: str

class Patch:
    def __init__(self, size: Coords, seed: int):
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        self._size = size
        self._grid = sample(size, seed)
        self.elements: Dict[Coords, Element] = {}

    def _scatterFlowers(self):
        area = self._size[0] * self._size[1]
        n_flowers = self._rng.integers(int(area / 5), int(area / 3))
        xs = self._rng.integers(0, self._size[0], size=n_flowers)
        ys = self._rng.integers(0, self._size[1], size=n_flowers)

        for coord in zip(xs, ys):
            if self.isWall(coord): continue

            self.elements[coord] = {
                'last_visit': 0,
                'type': 'flower',
            }

    def isWall(self, coord: Coords) -> bool:
        x, y = coord
        return self._grid[x, y] == 1

    def toDisk(self, path: str):
        ...

    @classmethod
    def fromDisk(cls, path: str):
        ...

class EndlessMaze(BaseEnvironment):
    def __init__(self):
        super().__init__()

        self._state: Tuple[Coords, Coords] = (
            (0, 0), # patch coords
            (1, 1), # maze coords within patch
        )

        # configurable specs
        self._patch_size: Coords = (30, 30)
        self._connecting_paths = self._patch_size

        self._rng = np.random.default_rng()

        # TODO: this should be its own class with background loading/off-loading
        self._patches: Dict[Coords, Patch] = {}

        self._patch_seed = 0

        # build center patch
        self.addPatch((0, 0))
        self.addPatch((1, 1))

    # called on every episode, if this is used episodically
    def start(self):
        self._state = (
            (0, 0),
            (1, 1),
        )

        # TODO: need to compute the observable from here
        return None

    # called on every interaction step
    def step(self, action: int) -> Tuple[float, np.ndarray, bool]:
        pid, (x, y) = self._state

        # first look a step ahead to see where we _might_ go
        # we will then check for walls and other patches
        px, py = takeAction((x, y), action)

        # check if we landed in a new patch
        next_pid = pid
        if px >= self._patch_size[0]:
            next_pid = (pid[0] + 1, pid[1])
            px = 0

        elif px < 0:
            next_pid = (pid[0] - 1, pid[1])
            px = self._patch_size[0] - 1

        elif py >= self._patch_size[1]:
            next_pid = (pid[0], pid[1] + 1)
            py = 0

        elif py < 0:
            next_pid = (pid[0], pid[1] - 1)
            py = self._patch_size[1] - 1

        # check if we need to load anything
        if next_pid != pid:
            self._add9x9Patches(next_pid)

        # check for walls
        next_patch = self._patches[next_pid]

        if next_patch.isWall((px, py)):
            px = x
            py = y

            # in theory this shouldn't matter, there should be no walls blocking
            # between the patches if we are in a connecting hallway
            # however, this is just defensive programming
            next_pid = pid

        # TODO: need to compute observations
        # TODO: need to compute rewards
        next_obs = np.zeros(0)
        return -1, next_obs, False

    def show(self):
        mx = 0
        my = 0

        mix = int(1e10)
        miy = int(1e10)

        h = (2 * self._patch_size[0] + 1)
        w = (2 * self._patch_size[1] + 1)
        for coords in self._patches:
            # TODO: this assumes a lot about how the maze is sized
            # probably should be a computed property of the maze
            rx = (coords[0] + 1) * h
            uy = (coords[1] + 1) * w
            lx = coords[0] * h
            dy = coords[1] * w
            if rx > mx:
                mx = rx
            elif lx < mix:
                mix = lx
            if uy > my:
                my = uy
            elif dy < miy:
                miy = dy

        grid = np.zeros((mx - mix, my - miy))

        for coords in self._patches:
            x, y = coords
            sx = x * h - mix
            ex = sx + h
            sy = y * w - miy
            ey = sy + w

            print(coords, sx, ex, sy, ey)
            grid[sx:ex, sy:ey] = self._patches[coords]._grid

        import matplotlib.pyplot as plt
        plt.imshow(grid[::-1, :], cmap='gray_r')
        plt.show()

    def addPatch(self, coords: Coords):
        self._buildPatch(coords)
        self._add9x9Patches(coords)

    # ---------------
    # -- utilities --
    # ---------------

    def _add9x9Patches(self, center: Coords):
        x, y = center

        # from left->right, top->bottom
        to_check = [
            (x - 1, y + 1),
            (x, y + 1),
            (x + 1, y + 1),
            (x - 1, y),
            (x + 1, y),
            (x - 1, y - 1),
            (x, y - 1),
            (x + 1, y - 1),
        ]

        for check in to_check:
            self._buildPatch(check)

    def _buildPatch(self, coords: Coords):
        patch = self._patches.get(coords)
        if patch is not None:
            return patch

        self._patch_seed += 1
        patch = Patch(self._patch_size, self._patch_seed)

        # this needs to be handled by the patch manager
        # because it requires coordinating between two (or more) patches)
        self._attachToSurrounding(patch, coords)
        self._patches[coords] = patch

        return patch

    def _attachToSurrounding(self, me: Patch, coords: Coords):
        my_grid = me._grid

        for axis in range(2):
            for d in range(2):
                other_coords = updateElement(coords, axis, coords[axis] + (2 * d - 1))
                other = self._patches.get(other_coords)

                if other is None:
                    continue

                other_grid = other._grid
                i = self._rng.integers(0, self._patch_size[1 - axis], size=self._connecting_paths[1 - axis]) * 2 + 1

                if axis == 0:
                    my_grid[-d, i] = 0
                    other_grid[d - 1, i] = 0
                else:
                    my_grid[i, -d] = 0
                    other_grid[i, d - 1] = 0
