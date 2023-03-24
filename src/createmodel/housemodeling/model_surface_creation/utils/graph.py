from __future__ import annotations
from copy import deepcopy
from enum import IntEnum
import itertools
from typing import Generator, Optional, Union
import numpy as np
import numpy.typing as npt
from .geometry import Point, Segment


class RoofGraphLabel(IntEnum):
    """屋根線用グラフのエッジラベル

    Note:
        重複した場合には、大きい番号ほど優先される
    """
    DEFAULT = 0
    INNER = 1
    OUTER = 2


class RoofGraph:
    """屋根線をグラフとして扱うためのクラス
    """

    __edges: npt.NDArray[np.int8]  # 隣接行列 (隣接していない場合は-1, 隣接している場合はラベル番号)
    nodes: list[Point]

    def __init__(self, nodes: list[Point]) -> None:
        """コンストラクタ

        Args:
            nodes: 頂点の2次元座標
        """
        self.nodes = nodes
        self.__edges = np.full((len(nodes), len(nodes)), -1, dtype=np.int8)

    @property
    def node_size(self) -> int:
        """頂点数

        Returns:
            int: グラフの頂点数
        """
        return len(self.nodes)

    @property
    def edge_size(self) -> int:
        """エッジ数

        Returns:
            int: グラフのエッジ数
        """
        return (self.__edges >= 0).sum()

    def has_edge(self, a: int, b: int, label: Optional[RoofGraphLabel] = None) -> bool:
        """2頂点間にエッジがあるかを調べる

        Args:
            a(int): 頂点番号(0-based)
            b(int): 頂点番号(0-based)
            label(RoofGraphLabel, optional): エッジラベル 

        Returns:
            bool: 存在するならTrue (labelが指定された場合は、labelも一致している場合のみTrue)
        """

        if label is None:
            return self.__edges[a, b] >= 0
        else:
            return self.__edges[a, b] == label

    def degree(self, node_idx: int, label: Optional[RoofGraphLabel] = None) -> int:
        """頂点の次数を求める

        Args:
            node_idx(int): 頂点番号(0-based)
            label(RoofGraphLabel, optional): エッジラベル

        Returns:
            int: 指定した頂点の次数 (labelが指定された場合は、指定したlabelのエッジのみ考慮する)
        """
        if label is None:
            return (self.__edges[node_idx] >= 0).sum()
        else:
            return (self.__edges[node_idx] == label).sum()

    def move_node(self, node_idx: int, to: Point, permit_to_merge=False) -> None:
        """頂点の位置を移動する

        Args:
            node_idx(int): 頂点番号(0-based)
            to(Point): 移動先
            permit_to_merge(bool): 移動先に頂点があった場合に1つにまとめて良い場合はTrueを指定する 
        """
        same_position_node = self.find_node(to)

        if same_position_node and same_position_node != node_idx and permit_to_merge:
            mask = self.__edges[same_position_node] == -1
            self.__edges[same_position_node, mask] = \
                self.__edges[node_idx][mask]
            self.__edges[mask, same_position_node] = \
                self.__edges[node_idx][mask]
            self.__edges[:, node_idx] = -1
            self.__edges[node_idx] = -1

        self.nodes[node_idx] = to

    def add_node(self, position: Point, force: bool = False) -> int:
        """頂点を追加する

        Args:
            position(Point): 追加する頂点の位置
            force(bool): すでに頂点が存在する位置に追加する場合に新しく作る場合はTrueを指定する

        Returns:
            int: 追加した頂点番号 (すでに頂点が存在した場合にはその頂点の番号)
        """

        index = self.find_node(position)
        if index is not None and not force:
            return index

        self.nodes.append(position)
        self.__edges = np.pad(
            self.__edges,
            ((0, 1), (0, 1)),  # type: ignore
            'constant',  # type: ignore
            constant_values=-1
        )

        return len(self.nodes) - 1

    def add_edge(self, node_idx_1: int, node_idx_2: int, label: RoofGraphLabel = RoofGraphLabel.DEFAULT) -> None:
        """エッジの追加

        Args:
            node_idx_1(int): 追加するエッジの端点
            node_idx_2(int): 追加するエッジの端点
            label(RoofGraphLabel,optional): 追加するエッジのラベル (Default: DEFAULT)

        Note:
            すでにエッジが存在する場合には、Labelの優先順位が高い(値が大きい)ものが優先される
        """
        if node_idx_1 != node_idx_2:
            self.__edges[node_idx_1, node_idx_2] = \
                max(self.__edges[node_idx_1, node_idx_2], label)
            self.__edges[node_idx_2, node_idx_1] = \
                max(self.__edges[node_idx_2, node_idx_1], label)

    def delete_edge(self, node_idx_1: int, node_idx_2: int) -> None:
        """エッジの削除

        Args:
            node_idx_1(int): 削除するエッジの端点
            node_idx_2(int): 削除するエッジの端点
        """
        self.__edges[node_idx_1, node_idx_2] = -1
        self.__edges[node_idx_2, node_idx_1] = -1

    def get_edge_label(self, node_idx_1: int, node_idx_2: int) -> Optional[RoofGraphLabel]:
        """2頂点間にあるエッジのラベルを取得する

        Args:
            node_idx_1(int): 取得するエッジの端点
            node_idx_2(int): 取得するエッジの端点

        Returns:
            Optional[RoofGraphLabel]: エッジラベル、存在しない場合はNoneを返す
        """

        label = self.__edges[node_idx_1, node_idx_2]
        if label == -1:
            return None
        return RoofGraphLabel(label)

    def edge_list(self, label: Optional[RoofGraphLabel] = None) -> Generator[tuple[int, int, RoofGraphLabel], None, None]:
        """エッジの一覧のジェネレータを取得する

        Args:
            label(RoofGraphLabel, optional): 対象とするエッジラベル

        Returns:
            Generator[tuple[int, int, RoofGraphLabel], None, None]: (端点1,端点2,ラベル)のタプルを生成するジェネレータ

        Note:
            ジェネレータを使用中に点やエッジを書き換える場合は、生成されるデータに影響を及ぼす可能性があるため注意する
        """
        num_of_nodes = len(self.nodes)
        for i, j in itertools.product(range(num_of_nodes), repeat=2):
            if i > j:
                continue

            if (label == None and self.__edges[i, j] != -1) or self.__edges[i, j] == label:
                yield (i, j, RoofGraphLabel(self.__edges[i, j]))

    def to_segments(self, label: Optional[RoofGraphLabel] = None) -> list[Segment]:
        """エッジをSegmentに変換して取得する

        Args:
            label(RoofGraphLabel, optional): 対象とするエッジラベル

        Returns:
            list[Segment]: Segmentに変換したエッジのリスト
        """
        return [
            Segment(self.nodes[a], self.nodes[b]) for a, b, _ in self.edge_list(label=label)
        ]

    def find_node(self, query: Point) -> Optional[int]:
        """点の位置から頂点番号を求める

        Args:
            query(Point): 求める位置座標

        Returns:
            Optional[int]: 与えられた位置に頂点が存在する場合は頂点番号、存在しない場合はNone
        """
        for i, node in enumerate(self.nodes):
            if query.is_same(node):
                return i
        return None

    def get_adjacencies(self, node_idx: int, label: Optional[RoofGraphLabel] = None) -> list[int]:
        """隣接頂点を取得する

        Args:
            node_idx(int): 頂点番号
            label(RoofGraphLabel, optional): 対象とするエッジラベル

        Returns:
            list[int]: 与えられた頂点に隣接する頂点番号のリスト (labelが指定された場合には、そのlabelのエッジで隣接する頂点のみ)
        """
        if label is None:
            edge_mask = self.__edges[node_idx] >= 0
        else:
            edge_mask = self.__edges[node_idx] == label

        return np.arange(len(self.nodes))[edge_mask].tolist()

    def simplify(self, delete_unconnected_nodes=False):
        """同じ位置の頂点を1つにまとめ、必要であればエッジが存在しない点を削除する

        Args:
            delete_unconnected_nodes(bool, optional): Trueが指定された場合にはエッジが接続していない頂点を削除する (Default: false)
        """
        new_nodes: list[Point] = []
        new_indices: list[Optional[int]] = [None] * len(self.nodes)

        for i, node in enumerate(self.nodes):
            for j, other in enumerate(new_nodes):
                if node.is_same(other):
                    new_indices[i] = j
                    break

            if delete_unconnected_nodes and self.degree(i) == 0:
                continue

            if new_indices[i] is None:
                new_indices[i] = len(new_nodes)
                new_nodes.append(node)

        self.nodes = new_nodes

        new_edges = np.full(
            (len(new_nodes), len(new_nodes)), -1, dtype=np.int8)

        for i, j in itertools.product(range(len(self.__edges)), repeat=2):
            if new_indices[i] != new_indices[j] and self.__edges[i, j] != -1:
                new_edges[new_indices[i], new_indices[j]] = \
                    max(new_edges[new_indices[i], new_indices[j]],
                        self.__edges[i, j])

        self.__edges = new_edges

    @staticmethod
    def merge(graphs: list[RoofGraph]) -> RoofGraph:
        """複数のRoofGraphを1つにまとめる

        Args:
            graphs(list[RoofGraph]): RoofGraphのリスト

        Returns:
            RoofGraph: まとめた後のRoofGraph
        """

        nodes = list(itertools.chain.from_iterable(
            [g.nodes for g in graphs]
        ))

        graph = RoofGraph(nodes)

        index_begin = 0

        for g in graphs:
            for a, b, label in g.edge_list():
                graph.add_edge(a+index_begin, b+index_begin,
                               RoofGraphLabel(label))
            index_begin += len(g.nodes)

        graph.simplify()

        return graph

    @staticmethod
    def from_segments(segments: list[Segment], labels: Optional[Union[list[RoofGraphLabel], RoofGraphLabel]] = None):
        """線分のリストからRoofGraphを生成する

        Args:
            segments(list[Segment]): 線分のリスト
            labels(list[RoofGraphLabel] or RoofGraphLabel): 線分のラベルのリスト (ラベル単体を指定した場合には全ての線分がそのラベルになる)

        Returns:
            RoofGraph: 生成されたRoofGraphs
        """

        nodes: list[Point] = []
        edges: list[tuple[int, int, int]] = []

        def get_node_index(p: Point):
            for i in range(len(nodes)):
                if nodes[i].is_same(p):
                    return i
            nodes.append(deepcopy(p))
            return len(nodes)-1

        for i, (p1, p2) in enumerate(segments):
            if p1.is_same(p2):
                continue

            p1_index = get_node_index(p1)
            p2_index = get_node_index(p2)

            if p1_index != p2_index:
                label = 0
                if type(labels) is list:
                    label = labels[i]
                elif isinstance(labels, RoofGraphLabel):
                    label = labels

                edges.append((p1_index, p2_index, label))

        graph = RoofGraph(nodes)
        for a, b, label in edges:
            graph.add_edge(a, b, RoofGraphLabel(label))

        return graph
