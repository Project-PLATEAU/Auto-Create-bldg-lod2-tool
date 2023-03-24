import sys
from dataclasses import dataclass
import itertools
from typing import Union
import numpy as np
import numpy.typing as npt
from shapely.geometry import Polygon
from sklearn.cluster import DBSCAN

from .custom_itertools import pairwise

from ...util.objinfo import BldElementType, ObjInfo
from .model_surface_creation.utils.geometry_3d import get_angle_degree_3d
from .model_surface_creation.utils.disjoint_set_union import DisjointSetUnion
from .model_surface_creation.estimate_roof_height import estimate_roof_heights


@dataclass(frozen=True)
class ModelPoint:
    position_id_2d: int
    position_id_3d: int
    order_id: int


@dataclass
class ModelFace:
    points: list[ModelPoint]
    type: BldElementType
    group_id: int

    def edges_2d(self):
        return pairwise([
            point.position_id_2d for point in self.points
        ], loop=True)

    def edges_3d(self):
        return pairwise([
            point.position_id_3d for point in self.points
        ], loop=True)

    @property
    def position_ids_2d(self):
        return [point.position_id_2d for point in self.points]

    @property
    def position_ids_3d(self):
        return [point.position_id_3d for point in self.points]


class HouseModel:
    """家屋モデルクラス
    """

    _id: str
    _shape: Polygon
    _faces: list[ModelFace]
    _points: npt.NDArray[np.float_]

    def __init__(
        self,
        id: str,
        shape: Polygon,
    ) -> None:
        """コンストラクタ

        Args:
            id(str): 建物ID
            shape(Polygon): 建物外形ポリゴン
        """
        self.id = id
        self.shape = shape
        self._faces = []
        self._points = np.zeros((0, 3), dtype=np.float_)

    @property
    def id(self) -> str:
        """建物ID

        Returns:
            str: 建物ID
        """
        return self._id

    @id.setter
    def id(self, value: str) -> None:
        """建物ID

        Args:
            value (str): 建物ID
        """
        self._id = value

    @property
    def shape(self) -> Polygon:
        """建物外形ポリゴン

        Returns:
            Polygon: 建物外形ポリゴン
        """
        return self._shape

    @shape.setter
    def shape(self, value: Polygon) -> None:
        """建物外形ポリゴン

        Args:
            value (geo.Polygon): 建物外形ポリゴン
        """
        self._shape = value

    def _add_point(self, position: Union[tuple[float, float, float], npt.NDArray[np.float_]]) -> int:
        """点を追加する

        Args:
            position: Union[tuple[float, float, float], npt.NDArray[np.float_]]: 追加する点の3次元座標

        Returns:
            int: 3次元頂点番号

        Note:
            すでに同じ位置に点が存在している場合は、追加せずにその頂点番号を返す
        """
        position_np = np.array(position, dtype=np.float_)

        # 同じ位置の点が存在する場合には追加しない
        if len(self._points) > 0:
            distances = np.linalg.norm(self._points - position_np, axis=1)
            nearest_point_idx = np.argmin(distances)
            if distances[nearest_point_idx] < 1e-10:
                return int(nearest_point_idx)

        # _pointsの末尾に追加する
        self._points = np.concatenate([self._points, np.array([position_np])])
        return len(self._points) - 1

    def _add_roof(self, points: list[ModelPoint], face_group_id: int):
        """屋根面を追加する

        Args:
            points(list[ModelPoint]): 面の点のリスト (反時計回り)
            face_group_id(int): 面のグループ番号
        """
        self._faces.append(
            ModelFace(points, BldElementType.ROOF, face_group_id))

    def _add_ground(self, points: list[ModelPoint], face_group_id: int):
        """地面の面を追加する

        Args:
            points(list[ModelPoint]): 面の点のリスト (反時計回り)
            face_group_id(int): 面のグループ番号
        """
        self._faces.append(
            ModelFace(points, BldElementType.GROUND, face_group_id))

    def _generate_wall(self, face_group_id: int):
        """屋根面同士、屋根面と地面を繋ぐ壁を生成する

        Args:
            face_group_id(int): 面のグループ番号
        """

        assert all([face.type is not BldElementType.WALL for face in self._faces]), \
            "壁がすでに生成されています"

        edge_pairs: dict[tuple[int, int],
                         list[tuple[ModelPoint, ModelPoint]]] = {}

        # 同じ位置の線分毎にペアを作成する
        for face in self._faces:
            for p1, p2 in pairwise(face.points, loop=True):
                a = p1.position_id_2d
                b = p2.position_id_2d

                edge_2d = (min(a, b), max(a, b))
                if edge_2d not in edge_pairs:
                    edge_pairs[edge_2d] = []

                edge_pairs[edge_2d].append((p1, p2))

        # ペアで線分の高さが違う場合には壁を作成する
        for edges in edge_pairs.values():
            assert len(edges) == 2, "屋根面と地面が正しく登録されていません"

            edge1, edge2 = edges
            assert (edge1[0].position_id_2d, edge1[1].position_id_2d) == (edge2[1].position_id_2d, edge2[0].position_id_2d), \
                "屋根面と地面が正しく登録されていません"

            p10 = self._points[edge1[0].position_id_3d]
            p11 = self._points[edge1[1].position_id_3d]
            p20 = self._points[edge2[0].position_id_3d]
            p21 = self._points[edge2[1].position_id_3d]

            # 線分が3次元で交差している(端点のz座標の上下が反転している)場合には追加の頂点を作成する
            if p10[2] != p21[2] and p11[2] != p20[2] and ((p10[2] < p21[2]) ^ (p11[2] < p20[2])):
                cross_point_rate = (p10[2] - p21[2]) / \
                    ((p10[2] - p21[2]) + (p20[2] - p11[2]))
                cross_point_position = (p11 - p10) * cross_point_rate + p10

                cross_point = ModelPoint(
                    -1, self._add_point(cross_point_position), -1
                )

                self._faces.append(
                    ModelFace([
                        ModelPoint(
                            position_id_2d=wall_point.position_id_2d,
                            position_id_3d=wall_point.position_id_3d,
                            order_id=wall_point.position_id_2d  # 壁同士は自由に統合できるためorder_idを書き換える
                        ) for wall_point in [edge1[0], edge2[1], cross_point]
                    ], BldElementType.WALL, face_group_id)
                )

                self._faces.append(
                    ModelFace([
                        ModelPoint(
                            position_id_2d=wall_point.position_id_2d,
                            position_id_3d=wall_point.position_id_3d,
                            order_id=wall_point.position_id_2d  # 壁同士は自由に統合できるためorder_idを書き換える
                        ) for wall_point in [edge2[0], edge1[1], cross_point]
                    ], BldElementType.WALL, face_group_id)
                )
            else:
                # 同じ位置の点を除いて多角形を作成する
                wall_points = [edge1[1], edge1[0]]
                if edge1[0].position_id_3d != edge2[1].position_id_3d:
                    wall_points.append(edge2[1])
                if edge2[0].position_id_3d != edge1[1].position_id_3d:
                    wall_points.append(edge2[0])

                if len(wall_points) >= 3:
                    self._faces.append(
                        ModelFace([
                            ModelPoint(
                                position_id_2d=wall_point.position_id_2d,
                                position_id_3d=wall_point.position_id_3d,
                                order_id=wall_point.position_id_2d  # 壁同士は自由に統合できるためorder_idを書き換える
                            ) for wall_point in wall_points
                        ], BldElementType.WALL, face_group_id))

    def create_model_surface(
        self,
        point_cloud: npt.NDArray[np.float_],
        points_xy: npt.NDArray[np.float_],
        inner_polygons: list[list[int]],
        outer_polygon: list[int],
        ground_height: float,
        balcony_flags: list[bool],
    ) -> None:
        """モデル面の作成

        Args:
            point_cloud(NDarray[np.float_]): DSM点群 (num of points, 3)
            points_xy(NDarray[np.float_]): 屋根面頂点の2次元座標 (num of points, 2)
            inner_polygons(list[list[int]]): 区切られた各屋根面ポリゴン
            outer_polygon(list[int]): 屋根面の外形ポリゴン
            ground_height(float): 地面の高さ
            balcony_height(float): バルコニーの高さ
            balcony_flags(list[float]): inner_polygonsの各屋根面がバルコニーであるかのフラグ
        """

        # 高さを計算
        heights, roof_triangles = estimate_roof_heights(
            points_xy,
            outer_polygon,
            inner_polygons,
            point_cloud,
            ground_height,
        )
        balcony_height: float = max(
            ground_height + 0.1,
            min(heights)
        )

        points_xyz = np.concatenate([
            points_xy,
            np.array(heights)[:, np.newaxis],
        ], axis=1)

        # roof
        for triangle, polygon_idx in roof_triangles:
            face_points = []
            for vertex in triangle:
                xyz = points_xyz[vertex.point_id].copy()
                if balcony_flags[polygon_idx]:
                    xyz[2] = balcony_height

                face_points.append(ModelPoint(
                    position_id_2d=vertex.point_id,
                    position_id_3d=self._add_point(xyz),
                    order_id=vertex.order_id,
                ))

            self._add_roof(face_points, polygon_idx)

        # floor
        ground_points = [
            ModelPoint(
                position_id_2d=position_id_2d,
                position_id_3d=self._add_point((
                    points_xyz[position_id_2d][0],
                    points_xyz[position_id_2d][1],
                    ground_height
                )),
                order_id=i
            ) for i, position_id_2d in enumerate(outer_polygon[::-1])
        ]

        self._add_ground(ground_points, -1)

        # wall
        self._generate_wall(-2)

    def simplify(self, threshold: float):
        """屋根面の単純化

        同じ角度の隣接した面を一つにまとめる

        Args:
            threshold: 同じ角度と判定する閾値 (degree) 
        """

        num_of_faces = len(self._faces)
        dsu = DisjointSetUnion(num_of_faces)

        # 面毎に法線を求める
        normals: list[npt.NDArray[np.float_]] = []
        for face in self._faces:
            normal = np.zeros(3, dtype=np.float_)
            a = face.points[0].position_id_3d
            for b, c in face.edges_3d():
                normal += np.cross(self._points[b] - self._points[a],
                                   self._points[c] - self._points[a])

            normals.append(normal / np.linalg.norm(normal))

        # 統合しない面の組を列挙する
        rules: list[tuple[int, int]] = []
        for i, j in itertools.combinations(range(num_of_faces), 2):
            face_i = self._faces[i]
            face_j = self._faces[j]

            # 位置が同じで、出現順が異なる点を持つ2面は統合しない
            for point_i, point_j in itertools.product(face_i.points, face_j.points):
                if point_i.position_id_2d == point_j.position_id_2d and point_i.order_id != point_j.order_id:
                    rules.append((i, j))

        # 同じ向きの隣り合った面を繋げる
        for i, j in itertools.combinations(range(num_of_faces), 2):
            face_i = self._faces[i]
            face_j = self._faces[j]

            # 面のタイプが異なる場合は除く
            if face_i.type != face_j.type:
                continue

            # 辺を列挙する (ただし片方の辺の向きは逆転させる)
            edges_i = set(face_i.edges_3d())
            edges_j = set([(b, a) for a, b in face_j.edges_3d()])

            # 辺を共有していない場合は除く
            intersection = set(edges_i) & set(edges_j)
            if len(intersection) == 0:
                continue

            # 統合しないペアが統合されないか調べる
            permitted = True
            for a, b in rules:
                # rootの組が一致する場合は、統合した場合に、不許可のペアが統合される
                if {dsu.root(i), dsu.root(j)} == {dsu.root(a), dsu.root(b)}:
                    permitted = False
                    break
            if not permitted:
                continue

            if get_angle_degree_3d(normals[i], normals[j]) < threshold:
                dsu.unite(i, j)

        groups = dsu.groups()

        simplified_faces: list[ModelFace] = []

        # 繋げた面毎に外形線を求める
        # 異なる向きの同じ辺をペアとして消すと、残った辺が外形線になる
        for group in groups:
            # 回転方向を維持するため、元と同じ順で格納する
            unique_edges: set[tuple[int, int]] = set()

            for face_id in group:
                face = self._faces[face_id]

                for a, b in face.edges_3d():
                    if (b, a) in unique_edges:
                        unique_edges.remove((b, a))
                    else:
                        unique_edges.add((a, b))

            assert len(unique_edges) >= 3

            outer = self._to_polygon(list(unique_edges))

            simplified_faces.append(ModelFace(
                [ModelPoint(-1, position_id_3d, -1)
                 for position_id_3d in outer],
                self._faces[group[0]].type,
                self._faces[group[0]].group_id,
            ))

        self._faces = simplified_faces

    def rectify(self):
        """1) 連続点削除
           2) ソリッド非水密エラー修正
              多角形の線分上に頂点が存在する場合、その頂点を多角形に追加する
        """

        def find_onsegment_point(edge_3d: tuple[int, int], skip_list: set[int]) -> int:
            # 線分の頂点
            v0 = self._points[edge_3d[0]]
            v1 = self._points[edge_3d[1]]

            # 線分に存在する頂点を探す
            for i, point in enumerate(self._points):
                if i in skip_list:
                    continue

                dist1 = np.linalg.norm(point - v0)
                dist2 = np.linalg.norm(point - v1)
                dist3 = np.linalg.norm(v0 - v1)
                if dist1 + dist2 - dist3 < 1e-03:
                    return i
            return -1

        # 連続点を探索
        db = DBSCAN(eps=1e-02, min_samples=1).fit(self._points)
        labels = db.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        point_groups: list[list[int]] = list()
        for ci in range(n_clusters):
            inds = np.where(labels == ci)[0]
            point_groups.append(inds.tolist())

        # 連続点をグループ
        new_points = []
        index_conversion: dict[int,int] = dict()
        for group in point_groups:
            if len(group)==1:
                new_points.append(self._points[group[0]])
                index_conversion[group[0]] = len(new_points)-1
            else:
                # 平均点計算
                group_points = self._points[group]
                avg_point = np.mean(group_points, axis=0)
                # 平均点追加
                new_points.append(avg_point)
                for index in group:
                    index_conversion[index] = len(new_points)-1
        self._points = np.array(new_points)

        # 面の頂点リストを修正
        rectified_faces: list[ModelFace] = list()
        for face in self._faces:
            new_position_ids = list()
            for position_id in face.position_ids_3d:
                if index_conversion[position_id] not in new_position_ids:
                    new_position_ids.append(index_conversion[position_id])

            rectified_faces.append(ModelFace(
                [ModelPoint(-1, position_id, -1) for position_id in new_position_ids],
                face.type,
                face.group_id,
            )) 
        self._faces = rectified_faces

        # 面ごとに処理
        rectified_faces: list[ModelFace] = list()

        for face in self._faces:
            # 多角形の線分取得
            edges_3d = set(face.edges_3d())
  
            skip_list = set()
            for i1, i2 in edges_3d:
                skip_list.add(i1)
                skip_list.add(i2)

            # 線分上頂点の探索
            while True:
                rectified_edges: set[tuple[int, int, int]] = set()                
                for edge_3d in edges_3d:
                    i = find_onsegment_point(edge_3d, skip_list)
                    if i >= 0:
                        rectified_edges.add((edge_3d[0], edge_3d[1], i))
                        skip_list.add(i)                      

                for edge in rectified_edges:
                    edges_3d.remove((edge[0], edge[1]))
                    edges_3d.add((edge[0], edge[2]))
                    edges_3d.add((edge[2], edge[1]))

                if len(rectified_edges)==0:
                    break

            outer = self._to_polygon(list(edges_3d))

            rectified_faces.append(ModelFace(
                [ModelPoint(-1, position_id_3d, -1) for position_id_3d in outer],
                face.type,
                face.group_id,
            ))
            
        self._faces = rectified_faces

    def _to_polygon(self, direct_edges: list[tuple[int, int]]) -> list[int]:
        """有向辺から多角形を復元する

        Args:
            direct_edges: 頂点番号のペアのリスト (反時計回り)

        Returns:
            list[int]: 復元した多角形

        Notes:
            単純多角形のみ対応
        """

        cur: int = direct_edges[0][0]
        polygon: list[int] = []

        for _ in range(len(direct_edges)):
            targets = list(filter(lambda e: e[0] == cur, direct_edges))
            assert len(targets) == 1, "単純多角形ではないデータが入力されています"
            polygon.append(cur)
            cur = targets[0][1]

        assert len(set(polygon)) == len(polygon) and cur == direct_edges[0][0], \
            "単純多角形ではないデータが入力されています"

        return polygon

    def output_obj(self, path: str):
        """objファイル出力

        Args:
            path (str): 出力パス
        """
        if len(self._faces) == 0:
            return

        roofs = list(
            filter(lambda face: face.type == BldElementType.ROOF, self._faces))
        walls = list(
            filter(lambda face: face.type == BldElementType.WALL, self._faces))
        grounds = list(
            filter(lambda face: face.type == BldElementType.GROUND, self._faces))

        info = ObjInfo()
        info.append_faces(BldElementType.ROOF, [
                          self._points[roof.position_ids_3d] for roof in roofs])
        info.append_faces(BldElementType.WALL, [
                          self._points[wall.position_ids_3d] for wall in walls])
        info.append_faces(BldElementType.GROUND, [
                          self._points[ground.position_ids_3d] for ground in grounds])

        info.write_file(file_path=path)
