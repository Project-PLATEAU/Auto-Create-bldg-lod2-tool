from copy import deepcopy
from typing import Final
import numpy as np
import numpy.typing as npt
import pulp

from .utils.triangulation import ScoreType, Triangle, triangulation, triangulation_2d
from .utils.geometry import Point


MIN_BUILDING_HEIGHT: Final[float] = 2  # meters


def estimate_roof_heights(
    points_xy: npt.NDArray[np.float_],
    outer_polygon: list[int],
    inner_polygons: list[list[int]],
    point_cloud: npt.NDArray[np.float_],
    floor_height: float,
) -> tuple[list[float], list[tuple[Triangle, int]]]:
    """屋根面の各頂点の高さを推測する

    Args:
        points_xy(npt.NDArray[np.float_]): 屋根面の頂点の2次元座標
        outer_polygon(list[int]): 屋根面の外形ポリゴン
        inner_polygons(list[list[int]]): 区切られた各屋根面ポリゴン
        point_cloud(NDArray[np.float_]): 点群 (num of points, 3)
        floor_height(float): 床(地面)の高さ

    Returns:
        list[float]: 各頂点の高さ
        list[tuple[Triangle, int]]: 高さ決定時の三角形分割方法(三角形と分割元の屋根面番号のタプル)
    """

    z = point_cloud[:, 2]
    # 地面だと考えられる点群を取り除く
    roof_z = z[z > floor_height + MIN_BUILDING_HEIGHT]

    if len(roof_z):
        roof_bottom_height, roof_top_height = np.percentile(roof_z, [25, 90])
    else:
        roof_bottom_height, roof_top_height = np.percentile(z, [25, 90])
        roof_bottom_height = max(roof_bottom_height, floor_height)
        roof_top_height = max(roof_top_height, floor_height + 0.5)

    # 各屋根面を三角形に分割する
    triangles: list[tuple[Triangle, int]] = []

    points_xyz = np.concatenate([
        points_xy,
        np.zeros((len(points_xy), 1)),
    ], axis=1)

    # 線分上の頂点を取り除いた多角形を作成
    simplified_outer_polygon = simplify_polygon(outer_polygon, points_xy)

    for i in range(2):
        triangles.clear()

        for polygon_idx, polygon in enumerate(inner_polygons):
            if i == 0:
                # 初回のみ2次元での三角形分割を行う
                triangulation_results = triangulation_2d(
                    polygon,
                    points_xyz[:, :2],
                    score_type=ScoreType.MINIMIZE_SUM
                )
            else:
                # 2回目以降はz座標も考慮した三角形分割を行う
                triangulation_results = triangulation(
                    polygon,
                    points_xyz,
                    score_type=ScoreType.MINIMIZE_SUM
                )

            triangles.extend([
                (triangle, polygon_idx) for triangle in triangulation_results
            ])

        points_xyz[:, 2] = solve_linear_programming(
            points_xyz[:, :2],
            triangles,
            simplified_outer_polygon,
            floor_height,
            roof_bottom_height,
            roof_top_height
        )

    return list(points_xyz[:, 2]), triangles


def simplify_polygon(polygon: list[int], points: npt.NDArray[np.float_]):
    """一直線上に並んでいる頂点の端以外の点を取り除き、多角形を単純化する

    Args:
        polygon(list[int]): 多角形の頂点番号のリスト
        points(npt.NDArray(np.float_)): 頂点の2次元座標

    Returns:
        list[int]: 単純化した多角形の頂点番号のリスト
    """
    results = deepcopy(polygon)
    xy = points[:, :2]

    while True:
        updated = False

        for i in range(len(results)):
            next_i = (i + 1) % len(results)

            v_prev = xy[results[i-1]] - xy[results[i]]
            v_next = xy[results[next_i]] - xy[results[i]]

            cross = np.cross(
                v_prev / np.linalg.norm(v_prev),
                v_next / np.linalg.norm(v_next),
            )

            if abs(cross) < 1e-4:
                del results[i]
                updated = True
                break

        if not updated:
            break

    return results


def solve_linear_programming(
    points: npt.NDArray[np.float_],
    triangles: list[tuple[Triangle, int]],
    outer_corners: list[int],
    floor_height: float,
    roof_bottom_height: float,
    roof_top_height: float,
) -> list[float]:
    """屋根の高さの決定する線形計画法を解く

    Args:
        points(npt.NDArray[np.float_]): 屋根面の頂点の2次元座標
        triangles(list[tuple[Triangle, int]]): 屋根面の三角形分割結果(三角形と分割元の屋根面番号のタプル)
        outer_corners(list[int]): 外形ポリゴンの頂点
        floor_height(float): 地面の高さ
        roof_bottom_height(float): 屋根の下端の高さ
        roof_top_height(float): 屋根の上端の高さ

    Returns:
        list[float]: 屋根の高さ
    """
    problem = pulp.LpProblem(sense=pulp.LpMaximize)

    # z変数の範囲を設定
    z_vars: list[pulp.LpVariable] = []
    padding = (roof_top_height - roof_bottom_height) * 0.1
    for i in range(len(points)):
        # 上限値を計算 (45度以下になるようにする)
        distances = np.linalg.norm(
            points[outer_corners] - points[i], axis=1)
        min_distance = distances.min()

        lower_bound = roof_bottom_height - padding
        upper_bound = \
            min(roof_bottom_height + min_distance, roof_top_height) + padding

        z_vars.append(pulp.LpVariable(
            f'z_{i}',
            lowBound=lower_bound,
            upBound=upper_bound,
        ))

    # 評価関数を設定
    objective = []

    # 体積 (加点)
    sum_area = 0
    volume_weight = 1
    objective_volume = []
    for triangle, _ in triangles:
        area_2d = np.cross(
            points[triangle[1].point_id] - points[triangle[0].point_id],
            points[triangle[2].point_id] - points[triangle[0].point_id],
        ) / 2.0
        sum_area += area_2d
        objective_volume.append(
            area_2d *
            (pulp.lpSum([
                z_vars[vertex.point_id] - roof_bottom_height
                for vertex in triangle
            ]))
        )
    objective.append(
        pulp.lpSum(objective_volume) / sum_area * volume_weight
    )

    # 角度 (減点)
    angle_weight = 2.5
    objective_angle = []
    for i in range(len(triangles)):
        for j in range(i+1, len(triangles)):
            triangle_i, group_i = triangles[i]
            triangle_j, group_j = triangles[j]

            if group_i != group_j:
                continue

            # 法線を計算(ただし、z=1に正規化する)
            def cross_normalize_z(v1, v2):
                x1, y1, z1 = v1
                x2, y2, z2 = v2
                cross_z = x1*y2 - y1*x2
                return (
                    (y1*z2 - z1*y2) / cross_z,
                    (z1*x2 - x1*z2) / cross_z,
                    1,
                )

            ps_i = points[[vertex.point_id for vertex in triangle_i]]
            ps_j = points[[vertex.point_id for vertex in triangle_j]]

            normal_i = cross_normalize_z(
                (ps_i[1][0] - ps_i[0][0],
                 ps_i[1][1] - ps_i[0][1],
                 z_vars[triangle_i[1].point_id] - z_vars[triangle_i[0].point_id]),
                (ps_i[2][0] - ps_i[0][0],
                 ps_i[2][1] - ps_i[0][1],
                 z_vars[triangle_i[2].point_id] - z_vars[triangle_i[0].point_id]),
            )
            normal_j = cross_normalize_z(
                (ps_j[1][0] - ps_j[0][0],
                 ps_j[1][1] - ps_j[0][1],
                 z_vars[triangle_j[1].point_id] - z_vars[triangle_j[0].point_id]),
                (ps_j[2][0] - ps_j[0][0],
                 ps_j[2][1] - ps_j[0][1],
                 z_vars[triangle_j[2].point_id] - z_vars[triangle_j[0].point_id]),
            )

            # 絶対値を外すために、plusとminusの変数を用意する
            d_x_plus = pulp.LpVariable(f'd_x_plus_{i}_{j}', lowBound=0)
            d_x_minus = pulp.LpVariable(f'd_x_minus_{i}_{j}', lowBound=0)
            d_y_plus = pulp.LpVariable(f'd_y_plus_{i}_{j}', lowBound=0)
            d_y_minus = pulp.LpVariable(f'd_y_minus_{i}_{j}', lowBound=0)

            problem += d_x_plus - d_x_minus == normal_i[0] - normal_j[0]
            problem += d_y_plus - d_y_minus == normal_i[1] - normal_j[1]

            objective_angle.append(
                pulp.lpSum([d_x_plus, d_x_minus, d_y_plus, d_y_minus])
            )

    if len(objective_angle) >= 1:
        objective.append(
            - pulp.lpSum(objective_angle) / len(objective_angle) * angle_weight
        )

    problem += pulp.lpSum(objective)

    problem.solve(pulp.PULP_CBC_CMD(msg=False))

    #assert problem.status == pulp.LpStatusOptimal, f"LpStatus = {pulp.LpStatus[problem.status]}"

    optimized_heights = []

    for z_var in z_vars:
        height = z_var.value()
        if height is not None:
            optimized_heights.append(height)
        else:
            optimized_heights.append(floor_height)

    return optimized_heights
