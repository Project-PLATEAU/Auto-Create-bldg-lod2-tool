from typing import Final


class DisjointSetUnion:
    """素集合データ構造
    """

    _parent: list[int]
    _size: Final[int]

    def __init__(self, size: int) -> None:
        """コンストラクタ

        size(int): 頂点数
        """
        self._parent = [-1] * size
        self._size = size

    def root(self, index: int) -> int:
        """属する木の根の頂点番号を求める

        Args:
            index(int): 頂点番号(0-based)

        Returns:
            int: 入力の頂点番号が属する根の頂点番号(0-based)
        """
        if self._parent[index] < 0:
            return index
        self._parent[index] = self.root(self._parent[index])
        return self._parent[index]

    def unite(self, index1: int, index2: int) -> None:
        """与えられた2つ頂点が属する集合を1つにまとめる

        Args:
            index1(int): 頂点番号(0-based)
            index2(int): 頂点番号(0-based)
        """
        root1, root2 = self.root(index1), self.root(index2)
        if root1 != root2:
            self._parent[root1] = root2

    def groups(self) -> list[list[int]]:
        """グループ毎に属する頂点の番号を求める

        Returns:
            list[list[int]]: グループ毎の属する頂点番号のリスト
        """
        groups = [[] for i in range(self._size)]
        for i in range(self._size):
            groups[self.root(i)].append(i)

        return list(filter(None, groups))
