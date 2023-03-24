from itertools import chain, tee
from typing import Iterable, TypeVar

T = TypeVar('T')


def pairwise(iterable: Iterable[T], loop: bool = False) -> Iterable[tuple[T, T]]:
    """
      連続したペアを返す

      loop=Falseのときはpython3.10から導入されるものと同等
      参考: https://docs.python.org/ja/3/library/itertools.html#itertools.pairwise
    """
    if loop:
        # pairwise('ABCDEFG') --> AB BC CD DE EF FG GA
        a = list(iterable)
        return pairwise(chain(a, [a[0]]), loop=False)

    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)
