import numpy as np
from typing import Dict, Tuple
from TheAmazingProc.maze.grid import Coords
from TheAmazingProc.maze.maze import sample
from TheAmazingProc.utils.tuples import updateElement
from RlGlue import BaseEnvironment

RIGHT = 0
DOWN = 1
LEFT = 2
UP = 3

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
        self._patches: Dict[Coords, np.ndarray] = {}

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
        px = x
        py = y
        if action == RIGHT:
            px += 1
        elif action == DOWN:
            py -= 1
        elif action == LEFT:
            px -= 1
        elif action == UP:
            py += 1
        else:
            raise Exception(f'Unknown action: {action}')

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

        if next_patch[px, py] == 1:
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
            grid[sx:ex, sy:ey] = self._patches[coords]

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
        patch = sample(self._patch_size, seed=self._patch_seed)

        self._attachToSurrounding(patch, coords)
        self._patches[coords] = patch

        return patch

    def _attachToSurrounding(self, me: np.ndarray, coords: Coords):
        for axis in range(2):
            for d in range(2):
                other_coords = updateElement(coords, axis, coords[axis] + (2 * d - 1))
                other = self._patches.get(other_coords)

                if other is None:
                    continue

                i = self._rng.integers(0, self._patch_size[1 - axis], size=self._connecting_paths[1 - axis]) * 2 + 1

                if axis == 0:
                    me[-d, i] = 0
                    other[d - 1, i] = 0
                else:
                    me[i, -d] = 0
                    other[i, d - 1] = 0
