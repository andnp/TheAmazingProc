import unittest
from numba.typed import List as NList
from TheAmazingProc.utils.SamplableSet import SamplableSet

class TestSamplableSet(unittest.TestCase):
    def test_init(self):
        # can build empty set
        s = SamplableSet()
        self.assertEqual(s.length(), 0)

        # can build set with initial data
        s = SamplableSet(NList([1, 2, 3, 3]))
        self.assertEqual(s.length(), 3)

    def test_add(self):
        # can add elements to set
        s = SamplableSet(NList([1]))
        s.add(2)

        self.assertTrue(s.contains(1))
        self.assertTrue(s.contains(2))
        self.assertFalse(s.contains(3))
        self.assertEqual(s.length(), 2)

        # won't add duplicate elements
        s = SamplableSet(NList([1]))
        s.add(1)

        self.assertEqual(s.length(), 1)
        self.assertTrue(s.contains(1))

    def test_remove(self):
        # can remove elements from set
        s = SamplableSet(NList(range(10)))
        self.assertEqual(s.length(), 10)

        # try to remove 3, but first make sure 3 exists
        self.assertTrue(s.contains(3))
        s.remove(3)
        self.assertEqual(s.length(), 9)
        self.assertFalse(s.contains(3))

        # now try to add it back
        s.add(3)
        self.assertEqual(s.length(), 10)
        self.assertTrue(s.contains(3))
