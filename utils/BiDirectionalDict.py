from typing import Dict, Generic, List, TypeVar

T = TypeVar("T")
U = TypeVar("U")

class Bidict(dict, Generic[T, U]):
    """Usage:
    bd = bidict({'a': 1, 'b': 2})
    print(bd)                     # {'a': 1, 'b': 2}
    print(bd.inverse)             # {1: ['a'], 2: ['b']}
    bd['c'] = 1                   # Now two keys have the same value (= 1)
    print(bd)                     # {'a': 1, 'c': 1, 'b': 2}
    print(bd.inverse)             # {1: ['a', 'c'], 2: ['b']}
    del bd['c']
    print(bd)                     # {'a': 1, 'b': 2}
    print(bd.inverse)             # {1: ['a'], 2: ['b']}
    del bd['a']
    print(bd)                     # {'b': 2}
    print(bd.inverse)             # {2: ['b']}
    bd['b'] = 3
    print(bd)                     # {'b': 3}
    print(bd.inverse)             # {2: [], 3: ['b']}
    """
    def __init__(self, *args, **kwargs):
        super(Bidict, self).__init__(*args, **kwargs)
        self.inverse: Dict[U, List[T]] = {}
        for key, value in self.items():
            self.inverse.setdefault(value, []).append(key)

    def __setitem__(self, key: T, value: U):
        if key in self:
            self.inverse[self[key]].remove(key)
        super(Bidict, self).__setitem__(key, value)
        self.inverse.setdefault(value, []).append(key)

    def __delitem__(self, key: T):
        self.inverse.setdefault(self[key], []).remove(key)
        if self[key] in self.inverse and not self.inverse[self[key]]:
            del self.inverse[self[key]]
        super(Bidict, self).__delitem__(key)

    def __getitem__(self, key: T) -> U:
        return super().__getitem__(key)