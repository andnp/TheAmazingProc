import unittest
import numpy as np
from TheAmazingProc.maze.maze import sample

# TODO: should fill out test coverage with more incremental tests
# for now, moving forward with skeleton MVP coverage
class TestMaze(unittest.TestCase):
    def test_sample(self):
        # using the same seed should yield same mazes
        maze1 = sample((10, 10), 0)
        maze2 = sample((10, 10), 0)

        self.assertTrue(np.all(maze1 == maze2))

        # and a different seed should yield a different maze
        maze1 = sample((10, 10), 0)
        maze2 = sample((10, 10), 1)

        self.assertFalse(np.all(maze1 == maze2))

        # also we should have a fixed amount of whitespace within a maze
        maze = sample((10, 8), 0)
        self.assertEqual(maze.sum(), 198)

        maze = sample((10, 8), 1)
        self.assertEqual(maze.sum(), 198)
