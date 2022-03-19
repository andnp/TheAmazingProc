import unittest
from TheAmazingProc.maze.grid import clip, neighbors, toBoundedState, toCoords

class TestGrid(unittest.TestCase):
    def test_clip(self):
        # clipping happens in integer space
        # upper bound is exclusive
        got = clip(10, 0, 10)
        expected = 9
        self.assertEqual(got, expected)

        got = clip(-1, 0, 10)
        expected = 0
        self.assertEqual(got, expected)

    def test_toBoundedState(self):
        # ensure a coord pair stays within the grid
        shape = (9, 5)

        # check upper bound is exclusive
        got = toBoundedState((1, 8), shape)
        expected = 37
        self.assertEqual(got, expected)

        # check both coords are clipped
        got = toBoundedState((10, 5), shape)
        expected = 44
        self.assertEqual(got, expected)

    def test_toCoords(self):
        # take a state number and give the coords
        got = toCoords(22, (5, 10))
        expected = (2, 4)
        self.assertEqual(got, expected)

        # make sure toBoundedState and toCoords are dual
        shape = (15, 10)
        start = (13, 8)
        state = toBoundedState(start, shape)
        got = toCoords(state, shape)
        expected = start
        self.assertEqual(got, expected)

    def test_neighbors(self):
        # get neighboring states
        # hard to think about state ids, so work in coord-space
        # for these tests
        state = toBoundedState((0, 0), (10, 9))

        got = neighbors(state, (10, 9))._val2idx.keys()
        expected = set([
            toBoundedState((1, 0), (10, 9)),
            toBoundedState((0, 1), (10, 9)),
            toBoundedState((0, 0), (10, 9)),
        ])

        self.assertEqual(got, expected)
