from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from shapely.geometry import LineString
import enum

from ...custom_itertools import pairwise
from .geometry import Point, is_ccw_order


@dataclass(frozen=True)
class TriangleVertex:
    """三角形の頂点"""

    point_id: int
    order_id: int


@dataclass(frozen=True)
class Triangle:
    """三角形"""

    vertices: list[TriangleVertex]

    def __post_init__(self):
        assert len(self.vertices) == 3, \
            "The number of triangle's vertices must be exactly 3"

    def __iter__(self):
        yield from self.vertices

    def __getitem__(self, pos: int):
        return self.vertices[pos]


class ScoreType(enum.Enum):
    """三角形分割の最適化種類
    """

    MINIMIZE_SUM = enum.auto()
    """隣接する三角形の角度の総和を最小化"""

    MINIMIZE_MAXIMUM = enum.auto()
    """隣接する三角形の角度の最大値を最小化"""

    MAXIMIZE_FLAT = enum.auto()
    """同じ角度の隣接する三角形の組数を最大化"""

    MAXIMIZE_FLAT_WITH_THRESHOLD = enum.auto()
    """一定値(45度)より小さい組を最大化し、その中で同じ角度の隣接する三角形の組数を最大化"""


def triangulation(
    polygon: list[int],
    points: npt.NDArray[np.float_],
    score_type: ScoreType = ScoreType.MINIMIZE_SUM
) -> list[Triangle]:
    """多角形の三角形分割を行う

    Args:
        polygon(list[int]): 多角形の頂点番号のリスト(反時計回り)
        points(NDArray[np.float_]): 頂点の3次元座標 (num of points, 3)
        score_type(ScoreType): 分割時の最適化種類

    Returns:
        list[Triangle]: 分割後の三角形のリスト

    Note:
        環状の区間動的計画法を用いて求める。
        最後に作成した三角形を保持することで、区間の結合で作成した新しい三角形によって増加するコストが計算できる。
    """

    assert points.ndim == 2 and points.shape[1] == 3, \
        "shape of points must be (*, 3)"

    assert score_type == ScoreType.MINIMIZE_SUM, "MINIMIZE_SUMのみ実行可能です"

    # 環状の処理は複雑なため、頂点を繰り返して配列を2倍にして扱う

    N = len(polygon)
    points_xy = np.vstack([points[polygon][:, 0:2], points[polygon][:, 0:2]])

    # 全三角形の法線を計算
    normals_list = []
    for i in polygon:
        # 頂点iを含む三角形の法線を一括計算
        vector_i = points[polygon] - points[i]
        normals_i = np.cross(
            vector_i[:, np.newaxis, :],
            vector_i[np.newaxis, :, :],
        )
        # 単位ベクトルにする
        normals_size = np.linalg.norm(normals_i, axis=2)
        normals_size[normals_size == 0] += 1e-10  # 0割り対策
        normals_i /= normals_size[:, :, np.newaxis]
        # サイズを(2N, 2N, 3)にする
        normals_i = np.concatenate([normals_i, normals_i], axis=0)
        normals_i = np.concatenate([normals_i, normals_i], axis=1)
        normals_list.append(normals_i)

    normals = np.array(normals_list * 2)  # (2N,2N,2N,3)

    # 全頂点間の線分が多角形内部か判定する
    # 条件: 端点以外で他の多角形と交差しない & 端点から内側方向に線分がある
    is_inner_segment = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            if polygon[i] == polygon[j]:
                # 頂点位置が同じ場合は、polygon内のindexも同じ場合のみ内部とする
                is_inner = i == j
            elif i+1 == j or j+1 == i+N:
                # 隣接する頂点を結ぶ線は内部とする
                is_inner = True
            else:
                # 点i,jから線分が内側に伸びていることをチェック
                is_inner = is_ccw_order(
                    Point(*points_xy[i+1]),
                    Point(*points_xy[j]),
                    Point(*points_xy[i-1]),
                    center=Point(*points_xy[i])
                )
                is_inner = is_inner and is_ccw_order(
                    Point(*points_xy[j+1]),
                    Point(*points_xy[i]),
                    Point(*points_xy[j-1]),
                    center=Point(*points_xy[j])
                )

                # 外形線と交差しないかチェック
                for edge in pairwise(range(N), loop=True):
                    # 端点が同じ場合は除く
                    if len({polygon[i], polygon[j]} & {polygon[edge[0]], polygon[edge[1]]}):
                        continue

                    segment1 = LineString(points_xy[[i, j]])
                    segment2 = LineString(points_xy[list(edge)])

                    is_inner = is_inner and not segment1.intersects(segment2)

            # 判定結果を保存
            is_inner_segment[i][j] = is_inner_segment[j][i] = is_inner

    # 初期化
    # [i:左端][j:右端][辺ijを使用する三角形の3つ目の点の位置]
    dp = np.full((N*2, N*2, N*2), np.inf)
    for i in range(N*2-1):
        dp[i][i+1][i] = 0

    # 復元用データ
    # [i][j][k] = (a, b) ... dp[i,j,k]はdp[i][k][a]とdp[k][j][b]から求められたことを示す
    dp_back: list[list[list[tuple[int, int]]]] = [
        [[(-1, -1) for _ in range(N*2)] for _ in range(N*2)] for _ in range(N*2)]

    # 時間計算量: O(N^4)
    for delta in range(2, N):
        for left in range(0, 2*N-delta):
            right = left + delta

            # 線分left-rightが多角形外を通る場合は除く
            if not is_inner_segment[left % N, right % N]:
                continue

            for center in range(left+1, right):
                # dp[left,right,center] が最小となるように
                # [left,center], [center,right] それぞれの区間で最後に作成する三角形を選択する

                if left + 1 == center:
                    # 幅1の時は三角形ではないので追加コストは0
                    index_i = left
                    cost_i = dp[left, center, left]
                else:
                    # dp[L, R, i] + angle(△CLi, △RLC) が最小となるiを求める
                    dot_values = np.dot(normals[left, :, center],
                                        normals[left, center, right])
                    angle_rad = np.arccos(np.clip(dot_values, -1, 1))
                    cost = dp[left, center, :] + angle_rad
                    index_i = int(np.argmin(cost[left+1:center]) + left + 1)
                    cost_i = cost[index_i]

                if center + 1 == right:
                    # 幅1の時は三角形ではないので追加コストは0
                    index_j = center
                    cost_j = dp[center, right, center]
                else:
                    # dp[C, R, j] + angle(△RCj, △RLC) が最小となるjを求める
                    dot_values = np.dot(normals[center, :, right],
                                        normals[left, center, right])
                    angle_rad = np.arccos(np.clip(dot_values, -1, 1))
                    cost = dp[center, right, :] + angle_rad
                    index_j = int(np.argmin(cost[center+1:right]) + center + 1)
                    cost_j = cost[index_j]

                # dp[left,center,index_i]とdp[center,right,index_j]の結合
                dp[left, right, center] = cost_i + cost_j
                dp_back[left][right][center] = (index_i, index_j)

    # 最小値を求める
    min_cost: float = np.inf
    min_cost_index: tuple[int, int, int] = (-1, -1, -1)
    for i in range(N):
        j = i + N - 1
        for k in range(i+1, j):
            if min_cost > dp[i, j, k]:
                min_cost = dp[i, j, k]
                min_cost_index = (i, j, k)

    # 分割方法を復元する
    stack: list[tuple[int, int, int]] = [min_cost_index]
    triangles: list[tuple[int, int, int]] = []

    #assert min_cost_index[0] != -1

    while len(stack):
        i, j, k = stack.pop()

        if abs(i-j) == 1:
            #assert i == k or j == k
            continue

        if i+N != j:
            triangles.append((i, k, j))

        a, b = dp_back[i][j][k]
        stack.append((i, k, a))
        stack.append((k, j, b))

    return [
        Triangle([
            TriangleVertex(polygon[a % N], a % N),
            TriangleVertex(polygon[b % N], b % N),
            TriangleVertex(polygon[c % N], c % N),
        ]) for a, b, c in triangles
    ]


def triangulation_2d(
    polygon: list[int],
    points: npt.NDArray[np.float_],
    score_type: ScoreType = ScoreType.MINIMIZE_SUM
) -> list[Triangle]:
    """多角形の三角形分割を行う

    Args:
        polygon(list[int]): 多角形の頂点番号のリスト(反時計回り)
        points(NDArray[np.float_]): 頂点の2次元座標 (num of points, 2)
        score_type(ScoreType): 分割時の最適化種類

    Returns:
        list[tuple[tuple[int,int],tuple[int,int],tuple[int,int]]]: 分割後の三角形のリスト、点は頂点番号とpolygon内のインデックスのタプル

    Note:
        環状の区間動的計画法を用いて求める。
        最後に作成した三角形を保持することで、区間の結合で作成した新しい三角形によって増加するコストが計算できる。
    """

    assert points.ndim == 2 and points.shape[1] == 2, \
        "shape of points must be (*, 2)"

    assert score_type == ScoreType.MINIMIZE_SUM, "MINIMIZE_SUMのみ実行可能です"

    # 環状の処理は複雑なため、頂点を繰り返して配列を2倍にして扱う

    N = len(polygon)
    points_xy = np.vstack([points[polygon][:, 0:2], points[polygon][:, 0:2]])

    # 全頂点間の線分の距離を計算する
    # 線分が多角形外部を通る場合はnp.infとする
    # 内部の条件: 端点以外で他の多角形と交差しない & 端点から内側方向に線分がある
    distance = np.full((N, N), np.inf)
    for i in range(N):
        for j in range(i, N):
            if polygon[i] == polygon[j]:
                # 頂点位置が同じ場合は、polygon内のindexも同じ場合のみ内部とする
                is_inner = i == j
            elif i+1 == j or j+1 == i+N:
                # 隣接する頂点を結ぶ線は内部とする
                is_inner = True
            else:
                # 点i,jから線分が内側に伸びていることをチェック
                is_inner = is_ccw_order(
                    Point(*points_xy[i+1]),
                    Point(*points_xy[j]),
                    Point(*points_xy[i-1]),
                    center=Point(*points_xy[i])
                )
                is_inner = is_inner and is_ccw_order(
                    Point(*points_xy[j+1]),
                    Point(*points_xy[i]),
                    Point(*points_xy[j-1]),
                    center=Point(*points_xy[j])
                )

                # 外形線と交差しないかチェック
                for edge in pairwise(range(N), loop=True):
                    # 端点が同じ場合は除く
                    if len({polygon[i], polygon[j]} & {polygon[edge[0]], polygon[edge[1]]}):
                        continue

                    segment1 = LineString(points_xy[[i, j]])
                    segment2 = LineString(points_xy[list(edge)])

                    is_inner = is_inner and not segment1.intersects(segment2)

            # 内部の場合、距離を保存
            if is_inner:
                distance[i][j] = distance[j][i] = np.linalg.norm(
                    points_xy[i] - points_xy[j])

    # distance を (2N, 2N) にする
    distance = np.tile(distance, (2, 2))

    # 初期化
    # dp[i:左端][j:右端] = 分割コストの総和
    dp = np.full((N*2, N*2), np.inf)
    np.fill_diagonal(dp[:, 1:], 0)  # dp[i, i+1] = 0

    # 復元用データ
    # [i][j] = a ... dp[i,j]はdp[i][a]とdp[a][j]から求められたことを示す
    dp_back = np.full((N*2, N*2), -1, dtype=np.int_)

    # 時間計算量: O(N^3)
    for delta in range(2, N):
        # R-L=deltaとしたときの各Lについて、
        # dp[L,R] = min_i(dp[L,i] + dp[i,R] + distance[L,R]) をまとめて計算する

        # costs[L, i] = dp[L,i] + dp[i,R] + distance[L,R] を計算
        costs = (dp[:-delta, :] + dp[:, delta:].T) + \
            np.diag(distance, delta)[:, np.newaxis]
        # min_costs[L] = min_i(costs[L, i])と各Lでのiを求める
        min_costs = costs.min(axis=1)
        min_costs_index = np.argmin(costs, axis=1)
        # dp[L,L+delta] = min_costs[L] とする
        np.fill_diagonal(dp[:, delta:], min_costs)
        # dp_back[L,L+delta] = min_costs_index[L] とする
        np.fill_diagonal(dp_back[:, delta:], min_costs_index)

    # dp[i,i+N-1]が最小となるiを求める
    min_cost_index = int(np.argmin(np.diag(dp, N-1)))
    min_cost: float = dp[min_cost_index, min_cost_index+N-1]

    #assert min_cost != np.inf

    # 分割方法を復元する
    stack: list[tuple[int, int]] = [(min_cost_index, min_cost_index+N-1)]
    triangles: list[tuple[int, int, int]] = []

    while len(stack):
        i, j = stack.pop()

        if abs(i-j) == 1:
            continue

        a = dp_back[i, j]
        triangles.append((i, a, j))

        stack.append((i, a))
        stack.append((a, j))

    return [
        Triangle([
            TriangleVertex(polygon[a % N], a % N),
            TriangleVertex(polygon[b % N], b % N),
            TriangleVertex(polygon[c % N], c % N),
        ]) for a, b, c in triangles
    ]
