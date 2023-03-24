from shapely.geometry import Polygon, LineString
from .utils.graph import RoofGraph
from .utils.geometry import Point, get_angle_degree


def extract_roof_surface(
    points: list[Point],
    edges: list[tuple[int, int]],
) -> tuple[list[int], list[list[int]]]:
    """屋根面ポリゴンの抽出

    Args:
        points(list[Point]): 屋根面頂点の2次元座標
        edges(list[tuple[int,int]]): 屋根面の線分のリスト

    Returns:
        list[int]: 外形線の頂点(反時計回り)
        list[list[int]] 内部の多角形のリスト(各多角形の頂点は反時計回り)

    Notes:
        edgesは端点以外で交差しないものを想定している
    """

    # グラフを構築（平面グラフである必要がある）
    graph = RoofGraph(points)
    for point_idx1, point_idx2 in edges:
        graph.add_edge(point_idx1, point_idx2)

    # 平面グラフから多角形を抽出
    outer_polygon, inner_polygons = plane_graph_to_polygons(graph)

    return outer_polygon, inner_polygons


def plane_graph_to_polygons(graph: RoofGraph) -> tuple[list[int], list[list[int]]]:
    """平面グラフから区切られた多角形を抽出する

    Args:
        graph(RoofGraph): 平面グラフ

    Returns:
        list[int]: 平面グラフの外形ポリゴン
        list[list[int]]: 抽出したポリゴン

    Note:
        平面グラフでない場合の挙動は不明
    """
    V = len(graph.nodes)

    adjacencies: list[list[int]] = [[] for _ in range(V)]

    # 隣接ノードを反時計回りで保持する
    for i in range(V):
        adjacencies[i] = graph.get_adjacencies(i)
        adjacencies[i].sort(key=lambda adj: get_angle_degree(
            graph.nodes[i],
            graph.nodes[i] + Point(1, 0),
            graph.nodes[adj],
        ))

    is_used_edge: list[list[bool]] = [
        [False] * len(adjacencies[i]) for i in range(V)]

    outer: list[int] = []
    holes: list[list[int]] = []
    inners: list[list[int]] = []

    # 多角形を抽出
    for i in range(V):
        for j in range(len(adjacencies[i])):
            if is_used_edge[i][j]:
                continue

            polygon: list[int] = []

            prev: int = i
            cur: int = adjacencies[i][j]
            while True:
                polygon.append(cur)

                next = None

                for k in range(len(adjacencies[cur])):
                    if adjacencies[cur][k] == prev:
                        next = adjacencies[cur][(k+1) % len(adjacencies[cur])]
                        is_used_edge[cur][(k+1) % len(adjacencies[cur])] = True

                assert next is not None, "多角形の抽出に失敗しました"

                prev, cur = cur, next

                if prev == i and cur == adjacencies[i][j]:
                    break

            # 頂点が反時計回り順の場合は除く (外形線のため)
            sum_degree = 0
            for k in range(len(polygon)):
                a, b, c = \
                    polygon[k-1], polygon[k], polygon[(k+1) % len(polygon)]
                vec1 = graph.nodes[b] - graph.nodes[a]
                vec2 = graph.nodes[c] - graph.nodes[b]
                sum_degree += get_angle_degree(Point(0, 0), vec1, vec2)

            if sum_degree < 0:  # 360 or -360
                inners.append(polygon)
            else:
                holes.append(polygon)

    assert len(holes) >= 1, "外形線が取得できませんでした"

    holes.sort(key=lambda hole: get_polygon_area(hole, graph.nodes))
    outer = holes[-1]  # 最大の物が外形線
    holes = holes[:-1]

    for hole_graph in holes:
        hole = to_shapely_polygon(hole_graph, graph.nodes)

        for i, inner_graph in enumerate(inners):
            inner = to_shapely_polygon(inner_graph, graph.nodes)

            if not inner.contains(hole):
                continue

            if set(inner_graph) == set(hole_graph):
                continue

            inners[i] = add_hole(inner_graph, hole_graph, graph.nodes)

            break

    outer.reverse()  # 時計回りから反時計回りに変更

    return outer, inners


def add_hole(polygon: list[int], hole: list[int], points: list[Point]) -> list[int]:
    """多角形に穴を追加する

    Args:
        polygon(list[int]): 穴を追加する多角形 (反時計回り)
        hole(list[int]): 穴 (時計回り)
        points(list[Point]): 頂点の2次元座標

    Returns:
        list[int]: 穴を追加した多角形

    Note:
        holeはpolygonと逆の順で頂点リストを渡すことに注意
    """
    polygon_shape = Polygon([(points[i].x, points[i].y) for i in polygon])
    hole_shape = Polygon([(points[i].x, points[i].y) for i in hole])

    for i, polygon_point in enumerate(polygon):
        for j, hole_point in enumerate(hole):
            line = LineString([
                (points[polygon_point].x, points[polygon_point].y),
                (points[hole_point].x, points[hole_point].y)
            ])

            if not polygon_shape.contains(line) or hole_shape.intersection(line).length > 0:
                continue

            return [
                *polygon[:i+1],
                *hole[j:],
                *hole[:j+1],
                *polygon[i:],
            ]

    assert False, "polygonとholeを結ぶ線を求められませんでした"


def to_shapely_polygon(polygon: list[int], points: list[Point]) -> Polygon:
    """多角形の頂点リストからShapelyのPolygonに変換する

    Args:
        polygon(list[int]): 多角形k
        points(list[Point]): 頂点の2次元座標

    Returns:
        Polygon: ShapelyのPolygonオブジェクト
    """
    return Polygon([
        (points[id].x, points[id].y) for id in polygon
    ])


def get_polygon_area(polygon: list[int], points: list[Point]) -> float:
    """多角形の面積を求める

    Args:
        polygon(list[int]): 多角形k
        points(list[Point]): 頂点の2次元座標

    Returns:
        float: 多角形の面積
    """
    return to_shapely_polygon(polygon, points).area
