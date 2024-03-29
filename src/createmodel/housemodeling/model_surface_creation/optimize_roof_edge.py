# -*- coding:utf-8 -*-
from copy import deepcopy
import itertools
from typing import Optional
import numpy as np
import numpy.typing as npt
from shapely.geometry import Polygon, Point as ShapelyPoint, LineString

from .utils.geometry import Segment, Point, get_angle_degree
from .utils.graph import RoofGraph, RoofGraphLabel
from .utils.point_set import PointSet


def optimize_roof_edge(
    shape: Polygon,
    points: npt.NDArray[np.float_],  # (num of points, 2)
    edges: npt.NDArray[np.int_],  # (num of edges, 2)
    *,
    threshold: float = 0.5,  # unit length(通常はmeter)
    same_threshold: float = 0.25,  # unit length(通常はmeter)
    rotate_threshold: float = 10,  # degree
    extension_length: float = 2,  # unit length(通常はmeter)
    straight_threshold: float = 1,  # degree
    rotate_step_size: int = 90,  # degree
):
    """屋根線の最適化

    Args:
        shape(Polygon): 建物外形ポリゴン
        points(NDArray[np.float_]): 屋根面の頂点の2次元座標 (num of points, 2)
        edges(NDArray[np.int_]): 屋根線の頂点のペア, 頂点番号はpointsに対応 (num of edges, 2)

        threshold(float, optional): 汎用的な閾値 (単位はpointsの座標と同じ)
        same_threshold(float, optional): 同一座標とする閾値 (単位はpointsの座標と同じ)
        rotate_threshold(float, optional): 回転で角度を合わせる閾値 (degree)
        extension_length(float, optional): 線分を延長する最大値 (単位はpointsの座標と同じ)
        straight_threshold(float, optional): 一直線上にあると判定する閾値 (degree)
        rotate_step_size(int, optional): 屋根線を回転する際の基準となる角度の間隔 (degree, 90の約数を推奨)

    Returns:
        list[Point]: 線分の端点の座標のリスト
        list[tuple[int,int]]: 内部の屋根線の線分のリスト
        list[tuple[int,int]]: 外形線の線分のリスト
    """

    # shapelyのPolygonをRoofGraphの形式に変更
    shape_coords: list[tuple[float, float]
                       ] = shape.exterior.coords if shape.exterior else []
    footprint_segments = [
        Segment(Point(*point1), Point(*point2))
        for point1, point2 in zip(shape_coords[:-1], shape_coords[1:])
    ]

    # HEATの結果をRoofGraphの形式に変更
    roof_edge_segments = [
        Segment(Point(*points[index1]), Point(*points[index2]))
        for index1, index2 in edges
    ]

    outer_graph = RoofGraph.from_segments(
        footprint_segments, RoofGraphLabel.OUTER)
    inner_graph = RoofGraph.from_segments(
        roof_edge_segments, RoofGraphLabel.INNER
    )

    # 建物の方向を求める
    base_line_vector = calc_base_line_vector(outer_graph, straight_threshold)

    # 最適化処理を実行
    inner_graph, outer_graph = optimize_step_00(inner_graph, outer_graph)
    inner_graph, outer_graph = optimize_step_01(
        inner_graph, outer_graph, threshold, same_threshold, base_line_vector)
    inner_graph, outer_graph = optimize_step_02(
        inner_graph, outer_graph, threshold)
    graph = RoofGraph.merge([inner_graph, outer_graph])
    graph = optimize_step_03(graph, extension_length)
    graph = optimize_step_04(graph, extension_length)
    graph = optimize_step_05(graph, threshold, same_threshold)
    graph = optimize_step_06(graph, rotate_threshold, same_threshold,
                             rotate_step_size, base_line_vector)
    graph = optimize_step_07(graph)
    graph = optimize_step_08(graph, same_threshold)
    graph = optimize_step_05(graph, threshold, same_threshold)
    graph = optimize_step_09(graph)
    graph = optimize_step_10(graph, straight_threshold)

    # 結果の返却

    optimized_points = graph.nodes
    inner_edges = [(a, b) for a, b, _ in graph.edge_list(RoofGraphLabel.INNER)]
    outer_edges = [(a, b) for a, b, _ in graph.edge_list(RoofGraphLabel.OUTER)]

    return optimized_points, inner_edges, outer_edges


def calc_base_line_vector(outer_graph: RoofGraph, threshold: float) -> Point:
    """建物の向きの基準となるベクトルを求める

    平行、垂直の線分が最大の線分のうち最長のものを求め、その線分の向きの単位ベクトルを返す

    Args:
        outer_graph(RoofGraph): 外形線のグラフ表現
        threshold: 垂直、平行と判断する角度の誤差の上限値 (degree)

    Returns:
        Point: 建物の向きの基準となるベクトル
    """
    candidates: list[tuple[Segment, tuple[float, float]]] = []

    for segment1 in outer_graph.to_segments():
        sum_length = 0
        for segment2 in outer_graph.to_segments():
            degree = get_angle_degree(
                Point(0, 0),
                segment1[1] - segment2[0],
                segment2[1] - segment2[0]
            )
            if min(degree % 90, 90 - degree % 90) <= threshold:
                sum_length += segment2.length()

        candidates.append(
            (
                segment1,
                (sum_length, segment1.length())
            )
        )

    base_segment = max(candidates, key=lambda item: item[1])[0]

    return (base_segment[0] - base_segment[1]) / base_segment.length()


def segments_to_polygon(segments: list[Segment]) -> Polygon:
    """線分の一覧から多角形を復元する

    Args:
        segments(list[Segment]): 多角形の線分のリスト

    Returns:
        Polygon: 復元した多角形

    Note:
        グラフが単純多角形以外の場合の挙動は未定義
    """
    points = list(itertools.chain.from_iterable(
        [[segment[0], segment[1]] for segment in segments]))

    segments_idx = [
        (points.index(segment[0]), points.index(segment[1])) for segment in segments
    ]

    prev = segments_idx[0][0]
    cur = segments_idx[0][1]
    polygon = []

    for _ in range(len(segments)):
        polygon.append((points[cur].x, points[cur].y))
        next_segment = list(
            filter(lambda x: cur in x and prev not in x, segments_idx))
        next = list(
            set(segments_idx[segments_idx.index(next_segment[0])]) - {cur})[0]
        cur, prev = next, cur

    return Polygon(polygon)

############
# 最適化処理 #
############


def optimize_step_00(inner_graph: RoofGraph, outer_graph: RoofGraph):
    """外形線外の点を一番近い内部へ移動する

    Args:
        inner_graph(RoofGraph): 内部の屋根線のグラフ表現
        outer_graph(RoofGraph): 外形線のグラフ表現

    Returns:
        RoofGraph: 処理後の内部の屋根線のグラフ表現
        RoofGraph: 処理後の外形線のグラフ表現
    """

    inner_graph, outer_graph = deepcopy(inner_graph), deepcopy(outer_graph)

    outer_polygon = segments_to_polygon(outer_graph.to_segments())

    for inner_idx in range(inner_graph.node_size):
        point = inner_graph.nodes[inner_idx]
        if outer_polygon.contains(ShapelyPoint(point.x, point.y)):
            continue

        # 建物外の場合、最も近い外形線上の点に移動する
        nearest_points = []
        for outer_segment in outer_graph.to_segments():
            nearest_points.append(outer_segment.nearest_point_from(point))

        inner_graph.move_node(
            inner_idx,
            min(nearest_points, key=lambda p: point.distance(p)),
            permit_to_merge=False
        )

    return inner_graph, outer_graph


def optimize_step_01(inner_graph: RoofGraph, outer_graph: RoofGraph, threshold: float, point_threshold: float, base_line_vector: Point):
    """外形線に合わせて内部の頂点の位置を調整する

    Args:
        inner_graph(RoofGraph): 内部の屋根線のグラフ表現
        outer_graph(RoofGraph): 外形線のグラフ表現
        threshold(float): 点を移動する距離の上限値
        point_threshold(float): 端点ではない線分上に移動する場合の点との距離の下限値
        base_line_vector(float): 建物の向きの基準となるベクトル

    Returns:
        RoofGraph: 処理後の内部の屋根線のグラフ表現
        RoofGraph: 処理後の外形線のグラフ表現
    """

    inner_graph, outer_graph = deepcopy(inner_graph), deepcopy(outer_graph)

    # 垂直、平行の線分の数を数える
    def count_horizontal_vertical_line(inner_idx: int, move_to: Point, threshold: float):
        degrees_of_adjacency = [
            get_angle_degree(
                Point(0, 0),
                inner_graph.nodes[adj]-move_to,
                base_line_vector
            ) for adj in inner_graph.get_adjacencies(inner_idx)
        ]

        return len(list(filter(lambda deg: min(deg % 90, 90 - deg % 90) < threshold, degrees_of_adjacency)))

    outer_point_set = PointSet()
    for point in outer_graph.nodes:
        outer_point_set.add(point)

    outer_polygon = segments_to_polygon(outer_graph.to_segments())

    for idx, point in enumerate(inner_graph.nodes):
        # 2. 近い点、線に隣接するスコアを計算する
        # スコア： [誤差5度以内の垂直平行の線分の数、誤差10度以内の垂直平行の線分の数、優先度、移動距離]
        candidates: list[tuple[Point, tuple[float, float, float, float]]] = []

        # 移動しない場合 (優先度：低)
        candidates.append((
            point,
            (
                count_horizontal_vertical_line(idx, point, 5),
                count_horizontal_vertical_line(idx, point, 10),
                0,
                0
            )
        ))

        # 点に合わせて移動する場合 (優先度：高)
        for outer_point in outer_graph.nodes:
            distance = outer_point.distance(point)
            if distance < threshold:
                candidates.append((
                    outer_point,
                    (
                        count_horizontal_vertical_line(idx, outer_point, 5),
                        count_horizontal_vertical_line(idx, outer_point, 10),
                        2,
                        -distance
                    )
                ))

        # 線に合わせて移動する場合 (優先度：中)
        for outer_segment in outer_graph.to_segments():
            distance = outer_segment.distance(point)
            if distance < threshold:
                nearest_point = outer_segment.nearest_point_from(point)

                # 点からの距離が極端に近い場合は除く
                if outer_point_set.find_nearest(nearest_point, point_threshold) is not None:
                    continue

                candidates.append((
                    nearest_point,
                    (
                        count_horizontal_vertical_line(idx, nearest_point, 5),
                        count_horizontal_vertical_line(idx, nearest_point, 10),
                        1,
                        -distance
                    )
                ))

        # 外形線の外側を通る線を移動候補から除く
        filtered_candidates = []
        for candidate in candidates:
            adjacencies = inner_graph.get_adjacencies(idx)
            line_strings = [
                LineString([
                    (inner_graph.nodes[adj].x, inner_graph.nodes[adj].y),
                    (candidate[0].x, candidate[0].y)
                ]) for adj in adjacencies
            ]
            if all([outer_polygon.intersection(line_string).length >= line_string.length * 0.99 for line_string in line_strings]):
                filtered_candidates.append(candidate)

        if len(filtered_candidates) == 0:
            continue

        move_to = max(filtered_candidates, key=lambda item: item[1])[0]
        inner_graph.move_node(idx, move_to, permit_to_merge=False)

    return inner_graph, outer_graph


def optimize_step_02(inner_graph: RoofGraph, outer_graph: RoofGraph, threshold: float):
    """二重線の除去 (細長い三角形を一直線にする)

    Args:
        inner_graph(RoofGraph): 内部の屋根線のグラフ表現
        outer_graph(RoofGraph): 外形線のグラフ表現
        threshold(float): 点を移動を判断する距離の閾値

    Returns:
        RoofGraph: 処理後の内部の屋根線のグラフ表現
        RoofGraph: 処理後の外形線のグラフ表現
    """
    inner_graph, outer_graph = deepcopy(inner_graph), deepcopy(outer_graph)

    for i, j, k in itertools.combinations(range(inner_graph.node_size), 3):
        base_edge: Optional[tuple[int, int]] = None
        moving_point: Optional[tuple[int, Point]] = None

        # TODO: 最適を選択
        for a, b, c in itertools.permutations([i, j, k], 3):
            if not (inner_graph.has_edge(a, b) and inner_graph.has_edge(a, c)):
                continue

            point_a = inner_graph.nodes[a]
            point_b = inner_graph.nodes[b]
            point_c = inner_graph.nodes[c]

            # a - c - b の順に並んでいない場合は考慮しない
            if not (point_a.distance(point_c) <= point_a.distance(point_b)):
                continue

            segment_ab = Segment(point_a, point_b)

            # a-c------b を (ac)------b とする
            if point_c.distance(point_a) < threshold and point_c.distance(point_a) <= point_c.distance(point_b):
                base_edge = (a, b)
                moving_point = (c, point_a)
                break

            #   __c
            # a/----------b を a--c---b とする
            if segment_ab.distance(point_c) < threshold:
                base_edge = (a, b)
                moving_point = (c, segment_ab.project(point_c))
                # a, bと移動先が近い場合はまとめる
                if point_a.distance(moving_point[1]) < threshold:
                    moving_point = (c, point_a)
                elif point_b.distance(moving_point[1]) < threshold:
                    moving_point = (c, point_b)
                break

        if base_edge and moving_point:
            # 既存の線を消す
            inner_graph.delete_edge(i, j)
            inner_graph.delete_edge(j, k)
            inner_graph.delete_edge(i, k)
            # 点を移動して、新しい線を作成する
            inner_graph.move_node(
                moving_point[0], moving_point[1], permit_to_merge=True)
            inner_graph.add_edge(
                base_edge[0], moving_point[0], RoofGraphLabel.INNER)
            inner_graph.add_edge(
                base_edge[1], moving_point[0], RoofGraphLabel.INNER)

    return inner_graph, outer_graph


def optimize_step_03(graph: RoofGraph, extension_length: float):
    """線分の次数1の端点を延長して交点ができるなら延長する

    Args:
        graph(RoofGraph): 外形線を含む屋根線のグラフ表現
        extension_length(float): 線分を延長する長さの上限値

    Returns:
        RoofGraph: 処理後の外形線を含む屋根線のグラフ表現
    """

    graph = deepcopy(graph)

    for a, b, _ in graph.edge_list(label=RoofGraphLabel.INNER):
        if graph.nodes[a].is_same(graph.nodes[b]):  # ノードを移動するため、同じ位置になる場合がある
            continue

        # a, bの移動先候補
        candidates_a: list[Point] = []
        candidates_b: list[Point] = []

        for u, v, _ in graph.edge_list():
            # 同じ辺、または端点を共有しているばあいは除く
            if not set([u, v]).isdisjoint([a, b]):
                continue

            # uとvが同一点の場合は除く
            if graph.nodes[u].is_same(graph.nodes[v]):
                continue

            other_segment = Segment(graph.nodes[u], graph.nodes[v])

            point_a, point_b = graph.nodes[a], graph.nodes[b]
            unit_vector_ab = (point_b - point_a) / (point_b - point_a).abs()
            extended_segment_ab = Segment(
                point_a - unit_vector_ab * extension_length,
                point_b + unit_vector_ab * extension_length
            )

            cross_point = other_segment.get_cross_point(extended_segment_ab)
            if cross_point is None:
                continue

            if cross_point.distance(point_a) < cross_point.distance(point_b):
                if not cross_point.is_same(point_b):
                    candidates_a.append(cross_point)
            else:
                if not cross_point.is_same(point_a):
                    candidates_b.append(cross_point)

        # 次数1の場合、最も近い候補点に移動する
        if len(candidates_a) and graph.degree(a) == 1:
            new_point_a = min(candidates_a,
                              key=lambda point: point_a.distance(point))
            graph.move_node(a, new_point_a, permit_to_merge=False)
        if len(candidates_b) and graph.degree(b) == 1:
            new_point_b = min(candidates_b,
                              key=lambda point: point_b.distance(point))
            graph.move_node(b, new_point_b, permit_to_merge=False)

    graph.simplify()

    return graph


def optimize_step_04(graph: RoofGraph, extension_length: float):
    """孤立点から線を延ばして他の点、線分と繋げる

    Args:
        graph(RoofGraph): 外形線を含む屋根線のグラフ表現
        extension_length(float): 孤立点から延ばす線分の長さの上限値

    Returns:
        RoofGraph: 処理後の外形線を含む屋根線のグラフ表現
    """

    graph = deepcopy(graph)

    for target_idx, target in enumerate(graph.nodes):
        if graph.degree(target_idx) != 1:
            continue

        adjacencies = graph.get_adjacencies(
            target_idx, label=RoofGraphLabel.INNER)
        #assert len(adjacencies) == 1, f"次数1の外形線の頂点が存在します"

        # 他の線分上にある場合は除く
        disjoint_segments = list(filter(
            lambda segment: segment[0] != target_idx and segment[1] != target_idx, graph.edge_list()))
        is_on_other_segment = any([target.is_on(
            Segment(graph.nodes[a], graph.nodes[b])) for a, b, _ in disjoint_segments])

        if is_on_other_segment:
            continue

        # 孤立点と線分で繋ぐ先を列挙する
        candidates: list[Point] = []
        distance_from_adjacency = target.distance(graph.nodes[adjacencies[0]])
        for a, b, _ in disjoint_segments:
            segment = Segment(graph.nodes[a], graph.nodes[b])

            if graph.nodes[adjacencies[0]].is_on(segment):
                continue

            distance = segment.distance(target)
            if distance < distance_from_adjacency and distance < extension_length:
                candidate = segment.project(target)
                # 垂線を下ろした先が線分上でない場合は、端点に寄せる
                if not candidate.is_on(segment):
                    if target.distance(segment[0]) < target.distance(segment[1]):
                        candidate = segment[0]
                    else:
                        candidate = segment[1]

                candidates.append(candidate)

        # 一番近い点を採用する
        if len(candidates):
            added_node_idx = graph.add_node(
                min(candidates, key=lambda candidate: target.distance(candidate))
            )
            graph.add_edge(target_idx, added_node_idx,
                           label=RoofGraphLabel.INNER)

    graph.simplify()

    return graph


def optimize_step_05(graph: RoofGraph, threshold: float, same_threshold: float):
    """外形線上にある内部線を処理

    Args:
        graph(RoofGraph): 外形線を含む屋根線のグラフ表現
        threshold(float): 処理を行う内部線の外形線からの距離の閾値
        same_threshold(float): 同一の点、線とみなす閾値

    Returns:
        RoofGraph: 処理後の外形線を含む屋根線のグラフ表現
    """

    def filter_func_gen(graph, point_a, point_b, same_threshold, segment_ab, threshold):
        """
        線分ab上にある点(距離が少し離れている場合でも許容する)を判定する関数を生成

        Note:
            線分の端点から少しだけ距離の離れた点を含んだ場合に、後の処理で問題が発生するため、場合分けを行っている
            その問題をこの関数で対応することは良くないため、修正が必要
        """
        def filter_func(adj):
            point_adj = graph.nodes[adj]
            # 距離が少し離れている場合も許容
            filter_1_result = all([
                # 線分の端点から一定以上の距離が離れているか
                point_adj.distance(point_a) > same_threshold,
                point_adj.distance(point_b) > same_threshold,
                # 線分の外側に位置していないか
                max(point_adj.distance(point_a), point_adj.distance(
                    point_b)) <= point_a.distance(point_b),
                # 線分との距離が閾値以下であるか
                segment_ab.distance(point_adj) < threshold,
            ])
            # 距離が特に近い点のみ許容、線分の端点付近も含む
            filter_2_result = all([
                # 線分の外側に位置していないか
                max(point_adj.distance(point_a), point_adj.distance(
                    point_b)) <= point_a.distance(point_b),
                # 線分との距離が閾値以下であるか
                segment_ab.distance(point_adj) < same_threshold,
            ])
            return filter_1_result or filter_2_result
        return filter_func

    graph = deepcopy(graph)

    for a, b, _ in graph.edge_list(label=RoofGraphLabel.INNER):
        point_a, point_b = graph.nodes[a], graph.nodes[b]
        segment_ab = Segment(point_a, point_b)

        adjacencies_a = set(graph.get_adjacencies(
            a, label=RoofGraphLabel.OUTER))
        adjacencies_b = set(graph.get_adjacencies(
            b, label=RoofGraphLabel.OUTER))

        # 線分ab上にある点を抽出 (距離が少し離れていても許容する)
        adjacencies_on_ab = list(set(filter(
            filter_func_gen(graph, point_a, point_b,
                            same_threshold, segment_ab, threshold),
            adjacencies_a | adjacencies_b
        )))

        if len(adjacencies_on_ab) == 0:
            continue

        # aから近い順に並べる
        adjacencies_on_ab.sort(
            key=lambda point_idx: point_a.distance(graph.nodes[point_idx]))

        # 線分abを途中の点を通って繋ぐ
        graph.delete_edge(a, b)
        adj = [a, *adjacencies_on_ab, b]
        for u, v in zip(adj[:-1], adj[1:]):
            if not graph.has_edge(u, v):
                graph.add_edge(u, v, label=RoofGraphLabel.INNER)

    graph.simplify()

    return graph


def optimize_step_06(graph: RoofGraph, rotate_threshold: float, same_threshold: float, step_size: int, base_line_vector: Point):
    """フットプリント上の点を移動して、フットプリントの外形と線の向きを合わせる

    Args:
        graph(RoofGraph): 外形線を含む屋根線のグラフ表現
        rotate_threshold(float): 内部の屋根線の回転角度の上限値
        same_threshold(float): 同一の点、線とみなす閾値
        step_size(int): 屋根線を回転する際の基準となる角度の間隔 (degree, 90の約数を推奨)
        base_line_vector(float): 建物の向きの基準となるベクトル        

    Returns:
        RoofGraph: 処理後の外形線を含む屋根線のグラフ表現
    """

    graph = deepcopy(graph)

    # 移動する
    outer_point_set = PointSet()
    for a, b, _ in graph.edge_list(label=RoofGraphLabel.OUTER):
        outer_point_set.add(graph.nodes[a])
        outer_point_set.add(graph.nodes[b])

    def is_movable(idx: int, another_idx: int) -> tuple[bool, Optional[Segment]]:
        if graph.degree(idx, label=RoofGraphLabel.OUTER) >= 1:
            return False, None

        point = graph.nodes[idx]
        another_point = graph.nodes[another_idx]
        for segment in graph.to_segments(label=RoofGraphLabel.OUTER):
            # 端点上と一致する場合は移動しない
            if point.is_same(segment[0]) or point.is_same(segment[1]):
                return False, None

            # 端点ではない線分上にある場合は移動可能 (ただし、もう一方と同じ線分上の場合は除く)
            if point.is_on(segment) and not another_point.is_on(segment):
                return True, segment

        return False, None

    for a, b, _ in graph.edge_list(label=RoofGraphLabel.INNER):
        point_a, point_b = graph.nodes[a], graph.nodes[b]
        segment = Segment(point_a, point_b)

        a_is_movable, segment_on_a = is_movable(a, b)
        b_is_movable, segment_on_b = is_movable(b, a)

        if not a_is_movable and not b_is_movable:
            continue

        degree: float = get_angle_degree(
            Point(0, 0),
            point_a - point_b,
            base_line_vector
        )

        # step_sizeの倍数に近い場合に回転を行う
        diff = min(step_size - degree % step_size, degree % step_size)
        if not (1e-2 < diff < rotate_threshold):
            continue

        # 最も近いstep_sizeの倍数を求める
        new_degree = degree - (degree % step_size if degree % step_size < step_size / 2
                               else degree % step_size - step_size)
        rotate_radian = np.deg2rad(new_degree-degree)

        new_edge: Optional[tuple[Point, Point]] = None  # 移動先

        # 両方の頂点を動かせる場合は中心で回転する
        if a_is_movable and b_is_movable:
            #assert segment_on_a and segment_on_b

            center = (point_a + point_b) / 2
            rotated_a = point_a.rotate(rotate_radian, center)
            rotated_b = point_b.rotate(rotate_radian, center)

            rotated_segment = Segment(rotated_a, rotated_b)
            new_a = segment_on_a.get_cross_point(rotated_segment)
            new_b = segment_on_b.get_cross_point(rotated_segment)

            if new_a and new_b:
                new_edge = (new_a, new_b)

        # 片方を動かす場合
        if not new_edge and a_is_movable:
            #assert segment_on_a
            rotated_a = point_a.rotate(rotate_radian, point_b)
            new_a = segment_on_a.get_cross_point(Segment(rotated_a, point_b))

            if new_a:
                new_edge = (new_a, point_b)

        if not new_edge and b_is_movable:
            #assert segment_on_b
            rotated_b = point_b.rotate(rotate_radian, point_a)
            new_b = segment_on_b.get_cross_point(Segment(point_a, rotated_b))

            if new_b:
                new_edge = (point_a, new_b)

        # 移動先がない場合は移動しない
        if not new_edge:
            continue

        # 外形線の点と一致した場合には合わせる
        new_edge = (
            outer_point_set.adjust_point(new_edge[0], same_threshold),
            outer_point_set.adjust_point(new_edge[1], same_threshold),
        )

        # 移動前の線分上にあった点を移動する
        inner_only_points = set(filter(
            lambda idx:
            graph.degree(idx, label=RoofGraphLabel.INNER) > 0 and
            graph.degree(idx, label=RoofGraphLabel.OUTER) == 0,
            range(graph.node_size)
        ))

        for inner_point_idx in inner_only_points - {a, b}:
            if not graph.nodes[inner_point_idx].is_on(segment):
                continue

            new_inner_point: Optional[Point] = None

            if new_edge[0].is_same(new_edge[1]):
                # エッジが消滅した場合には、その点に移動する
                new_inner_point = new_edge[0]

            elif graph.degree(inner_point_idx) == 1:
                # 次数が1の時は線の向きを維持するために交点を求める
                adjacency_idx = graph.get_adjacencies(inner_point_idx)[0]
                other_segment = Segment(
                    graph.nodes[adjacency_idx],
                    graph.nodes[inner_point_idx] +
                    (graph.nodes[inner_point_idx] - graph.nodes[adjacency_idx])
                )
                cross_point = other_segment.get_cross_point(Segment(*new_edge))

                if cross_point is not None:
                    new_inner_point = cross_point

            if new_inner_point is None:
                new_inner_point = segment.nearest_point_from(
                    graph.nodes[inner_point_idx])

            graph.move_node(inner_point_idx, new_inner_point,
                            permit_to_merge=False)

        # 線を移動する
        graph.move_node(a, new_edge[0], permit_to_merge=False)
        graph.move_node(b, new_edge[1], permit_to_merge=False)

    graph.simplify()

    return graph


def optimize_step_07(graph: RoofGraph):
    """屋根線の交差部分に交点を作成する

    Args:
        graph(RoofGraph): 外形線を含む屋根線のグラフ表現    

    Returns:
        RoofGraph: 処理後の外形線を含む屋根線のグラフ表現    
    """

    graph = deepcopy(graph)

    for a, b, label in graph.edge_list():
        segment = Segment(graph.nodes[a], graph.nodes[b])

        # 交差する部分に頂点を追加
        for u, v, _ in list(graph.edge_list()):
            if u in [a, b] or v in [a, b]:
                continue

            other_segment = Segment(graph.nodes[u], graph.nodes[v])
            if segment.is_same(other_segment):
                continue

            cross_point = segment.get_cross_point(other_segment)
            if cross_point and not segment[0].is_same(cross_point) and not segment[1].is_same(cross_point):
                graph.add_node(cross_point)  # 既存の点と重複する場合には追加されない

        points_on_segment = list(filter(
            lambda idx: graph.nodes[idx].is_on(segment),
            range(graph.node_size)
        ))

        points_on_segment.sort(
            key=lambda idx: graph.nodes[idx].distance(segment[0]))
        #assert points_on_segment[0] == a and points_on_segment[-1] == b

        graph.delete_edge(a, b)
        for u, v in zip(points_on_segment[:-1], points_on_segment[1:]):
            graph.add_edge(u, v, label)

    return graph


def optimize_step_08(graph: RoofGraph, threshold: float):
    """近い点が存在する内部の点を移動してまとめる

    Args:
        graph(RoofGraph): 外形線を含む屋根線のグラフ表現
        threshold(float): まとめる頂点の距離の上限値      

    Returns:
        RoofGraph: 処理後の外形線を含む屋根線のグラフ表現
    """

    graph = deepcopy(graph)

    point_set = PointSet()
    outer_points = itertools.chain.from_iterable(
        graph.edge_list(label=RoofGraphLabel.OUTER))
    # 外形線上の点を優先して合わせる
    for idx in outer_points:
        point_set.add(graph.nodes[idx], 1e-7)
    for idx in range(graph.node_size):
        point_set.add(graph.nodes[idx], threshold)

    for idx, point in enumerate(graph.nodes):
        # 次数が0の点と、外形線の点は除く
        if graph.degree(idx) == 0 or graph.degree(idx, label=RoofGraphLabel.OUTER) > 0:
            continue

        nearest_point = point_set.find_nearest(point, threshold)

        if nearest_point:
            graph.move_node(idx, nearest_point)

    graph.simplify()

    return graph


def optimize_step_09(graph: RoofGraph):
    """次数1の頂点がなくなるまで辺を削除する

    Args:
        graph(RoofGraph): 外形線を含む屋根線のグラフ表現     

    Returns:
        RoofGraph: 処理後の外形線を含む屋根線のグラフ表現
    """

    graph = deepcopy(graph)

    while True:
        exists = False
        for idx in range(graph.node_size):
            if graph.degree(idx) == 1:
                exists = True
                adjacency = graph.get_adjacencies(idx)[0]
                graph.delete_edge(idx, adjacency)

        # 次数1の頂点が存在するまで続ける
        if exists is False:
            break

    graph.simplify()

    return graph


def optimize_step_10(graph: RoofGraph, straight_threshold: float):
    """次数2で直線上にある点を削除

    Args:
        graph(RoofGraph): 外形線を含む屋根線のグラフ表現
        straight_threshold(float): 同一直線上に並んでいると判断する角度の上限値     

    Returns:
        RoofGraph: 処理後の外形線を含む屋根線のグラフ表現
    """

    for idx in range(graph.node_size):
        if graph.degree(idx) != 2:
            continue

        # a - i - b
        a, b = graph.get_adjacencies(idx)

        label_ai = graph.get_edge_label(a, idx)
        label_ib = graph.get_edge_label(idx, b)

        # ラベルが異なる場合は除く
        if label_ai is None or label_ai != label_ib:
            continue

        is_straight = abs(get_angle_degree(
            graph.nodes[a], graph.nodes[idx], graph.nodes[b])) < straight_threshold

        if is_straight:
            graph.delete_edge(a, idx)
            graph.delete_edge(idx, b)
            graph.add_edge(a, b, label_ai)

    graph.simplify()

    return graph
