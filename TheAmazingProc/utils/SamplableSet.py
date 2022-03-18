from typing import Dict, Generic, TypeVar
from numba import types
from numba.extending import typeof_impl

T = TypeVar('T')
class SamplableSet(Generic[T]):
    def __init__(self):
        self.vals: Dict[int, T] = {}
        self.idxs: Dict[int, int] = {}

        self._i = 0

    def add(self, val: T):
        self.idxs[self._i] = ...

class SamplableSetType(types.Type):
    def __init__(self):
        super().__init__(name='SamplableSet')

sstype = SamplableSetType()

@typeof_impl.register(SamplableSet)
def typeof_index(val, c):
    return sstype
