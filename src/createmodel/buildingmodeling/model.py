import sys
import copy
import numpy as np
import shapely.geometry as geo
import networkx as nx
from enum import IntEnum
from typing import Tuple
from sklearn.neighbors import KDTree, NearestNeighbors
from sklearn.cluster import DBSCAN
from scipy.spatial.transform import Rotation
from shapely.ops import unary_union
from numpy.typing import NDArray
from ..message import CreateModelMessage
from ..param import CreateModelParam
from .planeinfo import PlaneInfo
from .geoutil import GeoUtil
from .clusterinfo import ClusterInfo
from ..createmodelexception import CreateModelException
from ...util.objinfo import BldElementType, ObjInfo
from ...util.log import Log, LogLevel, ModuleType


class CompPoint(object):
    """座標値クラス (辞書キー対応版)
    """
    def __init__(self, x: float, y: float, z: float):
        """コンストラクタ

        Args:
            x (float): x 座標値
            y (float): y 座標値
            z (float): z 座標値
        """
        self.x = x
        self.y = y
        self.z = z

    def __eq__(self, other):
        """比較関数
        """
        if not isinstance(other, CompPoint):
            return False
        # return self.x == other.x and self.y == other.y and self.z == other.z

        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        dist = np.sqrt(dx * dx + dy * dy + dz * dz)
        return dist < 0.01  # 距離が1cm未満は同一点扱い
    
    def __hash__(self):
        """ハッシュ関数
        """
        return hash(self.x + self.y + self.z)


class PolylineInfo:
    """輪郭線情報
    """
    def __init__(self, points: NDArray, roof_idx: int) -> None:
        """コンストラクタ

        Args:
            points (NDArray): 輪郭線の頂点列
            roof_idx (int): 屋根id(-1は建物外形,0以上は屋根)
        """
        self.points = points
        self.roof_idx = roof_idx

    # プロパティ
    @property
    def polygon(self) -> geo.Polygon:
        """shapely.geometry.Polygonの取得

        Returns:
            geo.Polygon: ポリゴンデータ
        """
        if self.points is None or len(self.points) < 3:
            return None
        else:
            return geo.Polygon(self.points)


class LineInfo():
    """壁面作成時に使用する線情報
    """
    def __init__(self, line: geo.LineString = None, id1=-1, id2=-1) -> None:
        """コンストラクタ

        Args:
            line (geo.LineString, optional): 壁面位置. Defaults to None.
            id1 (int, optional): 屋根id1. Defaults to -1.
            id2 (int, optional): 屋根id2. Defaults to -1.
        """
        self.line = line
        self.id1 = id1
        self.id2 = id2


class FaceType(IntEnum):
    """面タイプ
    """

    UNKNOWN = 0
    """未知
    """

    ROOF = 1
    """屋根
    """

    WALL = 2,
    """壁
    """

    GROUND = 3
    """地面
    """


class FaceInfo:
    def __init__(self, points: NDArray,
                 type=FaceType.UNKNOWN, label=-1) -> None:
        """面情報

        Args:
            points (NDArray): 頂点座標リスト
            type (FaceType, optional): 面のタイプ. Defaults to FaceType.UNKNOWN.
            label (int, optional): 屋根面ラベル. Defaults to -1.
        """
        self._face_type = type
        self._label = label
        self._points = points

    @property
    def face_type(self) -> FaceType:
        """面タイプ

        Returns:
            FaceType: 面タイプ
        """
        return self._face_type

    @face_type.setter
    def face_type(self, value: FaceType):
        """面タイプ

        Args:
            value (FaceType): 面タイプ
        """
        self._face_type = value

    @property
    def label(self) -> int:
        """屋根面ラベル

        Returns:
            int: 屋根面ラベル
        """
        return self._label

    @label.setter
    def label(self, value: int):
        """屋根面ラベル

        Args:
            value (int): 屋根面ラベル
        """
        self._label = value

    @property
    def points(self) -> NDArray:
        """頂点座標リスト

        Returns:
            NDArray: 頂点座標リスト
        """
        return self._points

    @points.setter
    def points(self, value: NDArray):
        """頂点座標リスト

        Args:
            value (NDArray): 頂点座標リスト
        """
        self._points = value


class InsertPoint:
    """追加予定の不足点情報
    """
    def __init__(
            self, x: float, y: float, roof_id: int,
            index: int, t: float) -> None:
        """コンストラクタ

        Args:
            x (float): x座標
            y (float): y座標
            roof_id (int): 屋根id
            index (int): 追加位置(頂点index)
            t (float): 始点からのベクトル係数\
                (同一追加地点が存在する場合の追加順を決定するための値)
        """
        self.x = x
        self.y = y
        self.roof_id = roof_id
        self.index = index
        self.t = t

    def __lt__(self, other):
        """ソート関数

        Args:
            other (InsertItem): 比較対象

        Returns:
            bool: 比較結果
        """
        if self.roof_id == other.roof_id:
            if self.index == other.index:
                return self.t > other.t             # 降順
            else:
                return self.index > other.index     # 降順
        else:
            return self.roof_id < other.roof_id     # 昇順

    def __repr__(self):
        """オブジェクトを表す公式な文字列の作成

        Returns:
            str: 文字列
        """
        return repr((self.x, self.y, self.roof_id, self.index, self.t))


class Model:
    """建物情報
    """
    @property
    def id(self) -> str:
        """id

        Returns:
            str: id
        """
        return self._id

    @id.setter
    def id(self, value: str):
        """id

        Args:
            value (str): id
        """
        self._id = value

    @property
    def shape(self) -> geo.Polygon:
        """建物外形ポリゴン

        Returns:
            geo.Polygon: 建物外形ポリゴン
        """
        return self._shape

    @shape.setter
    def shape(self, value: geo.Polygon):
        """建物外形ポリゴン

        Args:
            value (geo.Polygon): 建物外形ポリゴン
        """
        self._shape = value

    @property
    def use_hier_classify(self) -> bool:
        """階層分類フラグ

        Returns:
            bool: 階層分類フラグ

        Note:
            Trueの場合、建物外形ポリゴンに隣接する屋根を選出し、
            建物中央にある屋上設備等はモデル化しない
        """
        return self._use_hier_classify

    @use_hier_classify.setter
    def use_hier_classify(self, value: bool):
        """階層分類フラグ

        Args:
            value (bool): 階層分類フラグ

        Note:
            Trueの場合、建物外形ポリゴンに隣接する屋根を選出し、
            建物中央にある屋上設備等はモデル化しない
        """
        self._use_hier_classify = value

    def __init__(
            self, id, shape: geo.Polygon,
            use_hier_classify=False) -> None:
        """コンストラクタ

        Args:
            id (str): 建物ID
            shape (geo.Polygon): 建物外形ポリゴン
            use_hier_classify (bool, optional): 階層分類フラグ. Defaults to False.
        Note:
            use_hier_classify=Trueの場合、建物外形ポリゴンに隣接する屋根を選出し、
            建物中央にある屋上設備等はモデル化しない
        """
        self._DIV_VALUE = 10000.0
        self._GROUND_IDX = -1
        self._NO_PARENT_IDX = -1

        self._id = id
        self._shape = shape     # 建物外形ポリゴン
        self._faces = []        # 内部処理用の面情報
        self._use_hier_classify = use_hier_classify     # 階層分類フラグ
        self._group_list = []   # 親屋根でグルーピングした結果
        # 壁面作成用形状(最外壁と島屋根の外壁)
        self._outlines: list[PolylineInfo] = []

    def _point_to_line_dist(
            self, pos: NDArray, line_pos: NDArray,
            line_vec: NDArray) -> Tuple[float, NDArray, float]:
        """点と直線との距離

        Args:
            pos (NDArray): 点
            line_pos (NDArray): 直線の始点
            line_vec (NDArray): 直線ベクトル

        Raises:
            CreateModelException: \
                点と直線の次元が2次元 or 3次元に統一されていない場合

        Returns:
            float: 距離
            NDArray: 点から下した垂線と直線との交点
            float: 交点までの直線のベクトル係数

        Note:
            GeoUtil.point_to_line_distのラッパー.\
            ベクトル係数値を小数点以下を切り捨てる作業をする\
            (切り捨てる桁数はself._DIV_VALUEで調節する)
        """
        dist, h, t = GeoUtil.point_to_line_dist(
            pos=pos, line_pos=line_pos, line_vec=line_vec)
        
        # 切り捨て
        t = float(int(t * self._DIV_VALUE)) / self._DIV_VALUE
        if GeoUtil.is_zero(t):
            t = 0.0     # -0対策
        
        return dist, h, t

    def _check_inclusion_relationships(self, clusters: list[ClusterInfo]):
        """包含関係の確認

        Args:
            clusters (list[ClusterInfo]): ClusterInfoのリスト
        """
        for i in np.arange(len(clusters)):
            if (clusters[i].roof_polygon is None
                    or GeoUtil.is_zero(clusters[i].roof_polygon.area)):
                continue

            # 屋根iの親屋根を探索する
            target = -1
            area = sys.float_info.max
            for j in np.arange(len(clusters)):
                if i == j:
                    continue

                if (clusters[j].roof_polygon is None
                        or GeoUtil.is_zero(clusters[j].roof_polygon.area)):
                    continue
            
                if clusters[i].roof_polygon.within(clusters[j].roof_polygon):
                    # i番目がj番目に内包されている場合
                    if area > clusters[j].roof_polygon.area:
                        # 入れ子の場合は、面積が小さい屋根が親のはず
                        area = clusters[j].roof_polygon.area
                        target = j

            if target > -1:
                clusters[i].parent = target
                clusters[target].children.append(i)

    def create_model_surface(self, clusters: list[ClusterInfo],
                             ground_height: float):
        """モデル面の作成

        Args:
            clusters (list[ClusterInfo]): ClusterInfoのリスト
            ground_height (float): 地面の高さ
        """
        param = CreateModelParam.get_instance()

        # 頂点マージ
        self._merge_vertex_2d(clusters=clusters)

        # 包含確認
        self._check_inclusion_relationships(clusters)

        # 準備
        self._surface_preparation(
            clusters=clusters,
            angle_th=param.surface_preparation_angle_th,
            sampling_step=param.surface_preparation_sampling_step,
            dist_th=param.surface_preparation_dist_th)

        # 屋根面
        roof_surface = self._create_roof_surface(clusters=clusters)

        # 壁面
        wall_surface = self._create_wall_surface(
            clusters=clusters, ground_height=ground_height)
        
        # 地面
        ground_surface = self._create_ground_surface(
            shape=self._shape, ground_height=ground_height)

        # 結果の保存
        self._faces = roof_surface
        self._faces.extend(wall_surface)
        self._faces.append(ground_surface)

        # 面の表面の順列方向
        if not param.front_is_ccw:
            # 時計回りを表とする場合
            # 頂点列を反転
            for i in range(len(self._faces)):
                self._faces[i].points = np.flipud(self._faces[i].points)

        # ソリッド閉じ対応
        self._solid(dist_th=param.solid_search_edge_th)

        # 頂点マージ
        self._merge_vertex(
            xy_dist=param.model_point_merge_xy_dist,
            reso=param.model_point_merge_z_reso)

    def _create_roof_surface(self, clusters: list[ClusterInfo]) -> list:
        """屋根面作成

        Args:
            clusters (list[ClusterInfo]): クラスタリング情報

        Returns:
            list: 屋根面情報のリスト
        
        Note:
            屋根が入れ子になっている場合や、親屋根を分割して穴のないポリゴンに変換する
        """
        surface = []

        # 屋根面の作成
        for cluster in clusters:
            if cluster.roof_polygon is None or GeoUtil.is_zero(cluster.roof_polygon.area):
                continue

            # 屋根の高さ
            if not self._use_hier_classify:
                # 階層分類を行わない場合
                cluster.roof_height = cluster.points.max[2]

            if len(cluster.children) == 0:
                # 内包する屋根がない場合
                roof: geo.Polygon = cluster.roof_polygon
                points = np.array(roof.exterior.coords)
                if not roof.exterior.is_ccw:
                    # 反時計回りを表とするため反転する
                    points = np.flipud(points)

                points = np.delete(points, obj=len(points) - 1, axis=0)
                tmp_z = np.full((len(points), 1), cluster.roof_height)
                points = np.hstack((points, tmp_z))  # 2次元座標を3次元座標にする
                face = FaceInfo(points=points, type=FaceType.ROOF,
                                label=cluster.id)
                surface.append(face)

            else:
                # 内包する屋根がある場合
                # 屋根を分割する
                exterior = np.array(cluster.roof_polygon.exterior.coords)
                interiors = []

                for i in cluster.children:
                    # 穴の設定
                    child_poly: geo.Polygon = clusters[i].roof_polygon
                    interiors.append(np.array(child_poly.exterior.coords))

                interiors = np.array(interiors)
                polygon = geo.Polygon(shell=exterior, holes=interiors)
                
                # 分割
                parts = [polygon]
                num = 0
                while num != len(parts):
                    num = len(parts)

                    tmp_parts = []
                    for polygon in parts:
                        if len(polygon.interiors) > 0:
                            # 穴がある場合
                            # 穴の中心座標を取得
                            hole: geo.LinearRing = polygon.interiors[0]
                            center = np.array([hole.centroid.x,
                                               hole.centroid.y])
                            
                            # 回転付きバウンディングボックス
                            bbox = polygon.minimum_rotated_rectangle
                            coords = np.array(bbox.exterior.coords)
                            bbox_num = len(coords)

                            # 始点探索
                            start = -1
                            max_dist = sys.float_info.max
                            for i in np.arange(bbox_num - 1):
                                pos1 = np.array([coords[i][0],
                                                coords[i][1]])
                                pos2 = np.array([coords[i + 1][0],
                                                coords[i + 1][1]])

                                vec = pos2 - pos1
                                dist, h, t = self._point_to_line_dist(
                                    pos=center, line_pos=pos1, line_vec=vec)

                                if 0 <= t and t <= 1.0 and dist < max_dist:
                                    start = i
                                    max_dist = dist
                                    start_pos = h
                            
                            if start < 0:
                                # 始点が未発見のため終了
                                break

                            # 終点探索
                            end = -1
                            line = center - start_pos
                            current = 0
                            if start < bbox_num - 2:
                                current = start + 1
                            
                            while(current != start):
                                next = current + 1
                                pos1 = np.array([coords[current][0],
                                                coords[current][1]])
                                pos2 = np.array([coords[next][0],
                                                coords[next][1]])

                                vec = pos2 - pos1
                                ret, h, online1, online2 = GeoUtil.cross_point(
                                    vec1=vec, pos1=pos1,
                                    vec2=line, pos2=start_pos)

                                if ret and online1:
                                    end_pos = h
                                    end = current
                                    break

                                if current < bbox_num - 2:
                                    current += 1
                                else:
                                    current = 0

                            if end < 0:
                                # 終点が未発見のため終了
                                break

                            start = start + 1 if start < bbox_num - 2 else 0
                            end = end + 1 if end < bbox_num - 2 else 0

                            # クリップ枠の作成
                            start_points = np.array([start_pos, end_pos])
                            end_points = np.array([end_pos, start_pos])
                            start_indexs = np.array([start, end])
                            end_indexs = np.array([end, start])
                            for i in np.arange(len(start_indexs)):
                                points = start_points[i]
                                current = start_indexs[i]
                                while current != end_indexs[i]:
                                    pos = np.array([coords[current][0],
                                                    coords[current][1]])
                                    points = np.vstack((points, pos))
                                    if current < bbox_num - 2:
                                        current = current + 1
                                    else:
                                        current = 0

                                points = np.vstack((points, end_points[i]))
                                points = np.vstack(
                                    (points, start_points[i]))
                                    
                                clip = geo.Polygon(points)

                                # 分割
                                and_region = clip.intersection(polygon)

                                # ポリゴンのみを抽出
                                geo_list = GeoUtil.separate_geometry(
                                    and_region)
                                select = [poly for poly in geo_list
                                          if (type(poly) is geo.Polygon)]
                                tmp_parts.extend(select)

                        else:
                            # 穴がない場合
                            tmp_parts.append(polygon)

                    # 更新
                    parts = tmp_parts

                # 屋根面作成
                for roof in parts:
                    points = np.array(roof.exterior.coords)
                    if not roof.exterior.is_ccw:
                        # 反時計回りを表とするため反転する
                        points = np.flipud(points)

                    points = np.delete(points, obj=len(points) - 1, axis=0)
                    tmp_z = np.full((len(points), 1), cluster.roof_height)
                    # 2次元座標を3次元座標にする
                    points = np.hstack((points, tmp_z))
                    face = FaceInfo(points=points, type=FaceType.ROOF,
                                    label=cluster.id)
                    surface.append(face)

        return surface

    def _surface_preparation(
            self, clusters: list[ClusterInfo],
            angle_th=3.0, sampling_step=0.01, dist_th=0.1) -> None:
        """面作成準備

        Args:
            clusters (list[ClusterInfo]): ClusterInfoリスト
            angle_th (float, optional):\
                グルーピング用の辺の角度閾値(deg). Defaults to 3.0.
            sampling_step (float, optional):\
                辺をサンプリングする際の間隔(m). Defaults to 0.01.
            dist_th (float, optional): 距離閾値(m). Defaults to 0.1.

        """
        # 親屋根,子屋根でグルーピング
        self._grouping_roof(clusters=clusters)

        # shapeの最長辺を基準に回転角度を算出する
        angle, pt = self._calc_rotate_angle()

        # 回転オブジェクト
        vec_z = np.array([0.0, 0.0, 1.0])   # 回転軸
        rot: Rotation = GeoUtil.rotate_object(vec=vec_z, angle=-angle)
        rev_rot: Rotation = GeoUtil.rotate_object(vec=vec_z, angle=angle)

        # 回転
        rotate_edges, rotate_points = self._rotate_edge(
            clusters=clusters, rot=rot, pt=pt)

        # 辺の角度でグルーピング
        angle_clusters = self._angle_clustering(
            rotate_edges, angle_th=angle_th)

        # 辺の距離でグルーピング
        edge_clusters = []
        for cluster in angle_clusters:
            tmp_clusters = self._edge_dist_clustering(
                edges=cluster, points=rotate_points,
                dist_th=0.1, sampling_step=sampling_step)
            edge_clusters.extend(tmp_clusters)

        insert_pt_list = []
        for i, cluster in enumerate(edge_clusters):
            # 複数屋根がグルーピングされているか確認
            roofs = [edge[2] for edge in cluster]
            roofs = np.unique(np.array(roofs))
            if len(roofs) < 2:
                continue

            # 辺のサンプリング
            sampling_points = self._sampling_edge(
                cluster=cluster, points=rotate_points,
                sampling_step=sampling_step)

            # 不足点の確認
            self._search_for_missing_vertices(
                edges=cluster, rotate_points=rotate_points,
                sampling_points=sampling_points, rev_rot=rev_rot, base_pt=pt,
                insert_pt_list=insert_pt_list, dist_th=dist_th)

        # 頂点を追加順にソート
        insert_pt_list.sort()

        # 頂点追加
        for insert_pt in insert_pt_list:
            pt = np.array((insert_pt.x, insert_pt.y))
            clusters[insert_pt.roof_id].roof_line = np.insert(
                clusters[insert_pt.roof_id].roof_line,
                insert_pt.index, pt, axis=0)

        # 頂点マージ(追加した不足点と不足点の基になった点を結合する)
        self._merge_vertex_2d(clusters=clusters, dist=dist_th)

        # 外壁用の形状作成
        self._outlines = []     # 外壁用
        for group in self._group_list:
            base_idx = clusters[group[0]].parent

            roof_polygon_list = list()
            for i in group:
                roof_polygon = clusters[i].roof_polygon
                if roof_polygon is not None:
                    if roof_polygon.is_valid:
                        roof_polygon_list.append(roof_polygon)
                    else:
                        roof_polygon_list.append(roof_polygon.buffer(0))                    
            base_polygon = unary_union(roof_polygon_list)

            if base_idx == self._NO_PARENT_IDX:
                # 最外壁の場合
                # 後段の壁面の法線方向判定で使用する
                self._shape = geo.Polygon(base_polygon.exterior)

            geoms = GeoUtil.separate_geometry(base_polygon)
            for geom in geoms:
                if type(geom) is not geo.Polygon:
                    continue

                base_points = np.array(geom.exterior.coords)
                if geom.exterior.is_ccw:
                    # 反時計回りを表とし、裏(穴)は時計回りとする
                    # 双方向エッジにするために建物外形と穴は時計周りとする
                    base_points = np.flipud(base_points)
                self._outlines.append(PolylineInfo(base_points, base_idx))

    def _grouping_roof(self, clusters: list[ClusterInfo]) -> None:
        """親屋根,子屋根でグルーピング

        Args:
            clusters (list[ClusterInfo]): _description_
        """
        # 親屋根,子屋根でグルーピング
        self._group_list = []
        # 親無しのルート屋根
        roots = [
            i for i, cluster in enumerate(clusters)
            if (cluster.roof_polygon is not None
                and cluster.roof_polygon.area > 0
                and cluster.parent == self._NO_PARENT_IDX)]
        self._group_list.append(roots)
        # 子屋根のグループ
        group = [
            cluster.children for cluster in clusters
            if (cluster.roof_polygon is not None
                and cluster.roof_polygon.area > 0
                and len(cluster.children) > 0)]

        self._group_list.extend(group)

    def _calc_rotate_angle(self) -> Tuple[float, NDArray]:
        """shapeの最長辺を基準に回転角度を算出する

        Returns:
            Tuple[float, NDArray]: 角度(deg), 基準点(2D座標)
        """
        points_list = np.array(self._shape.exterior.coords)
        vectors = [points_list[i + 1] - points_list[i]
                   for i in range(len(points_list) - 1)]
        length = [GeoUtil.size(vec) for vec in vectors]
        max_arg = np.argmax(length)
        vec = vectors[max_arg]  # 基準辺
        pt = points_list[max_arg]  # 基準点
        vec_x = np.array([1.0, 0.0])
        angle = GeoUtil.signed_angle(vec_x, vec)    # 回転角

        return angle, pt

    def _rotate_edge(
            self, clusters: list[ClusterInfo],
            rot: Rotation, pt: NDArray) -> Tuple[NDArray, dict]:
        """回転後の辺ベクトルと座標点の作成

        Args:
            clusters (list[ClusterInfo]): ClusterInfoリスト
            rot (Rotation): 回転オブジェクト
            pt (NDArray): 回転中心座標(移動量, 2D座標)

        Returns:
            Tuple[NDArray, dict]: 回転後辺ベクトル配列、屋根ごとの回転後頂点座標
        """
        rotate_edges = []
        rotate_points = {}
        for i, cluster in enumerate(clusters):
            if cluster.roof_line is not None:
                # 移動,回転
                points = cluster.roof_line - pt     # 移動
                tmp_z = np.zeros((len(points), 1))  # z座標は0埋め
                points = np.hstack((points, tmp_z))
                points = rot.apply(points)  # 回転

                # 辺ベクトル
                vectors = [
                    [(points[n + 1] - points[n])[0],
                     (points[n + 1] - points[n])[1]]
                    for n in range(-1, len(points) - 1)]
                vectors = np.array(vectors)

                roof_index = np.full((len(points), 1), i)   # 屋根ID
                # 辺(頂点)ID, 最終点(-1)始まり
                edge_index = np.reshape(range(-1, len(points) - 1), (-1, 1))

                tmp_index = np.hstack((roof_index, edge_index))
                tmp_index = np.hstack((vectors, tmp_index))
                rotate_edges.extend(tmp_index)
                rotate_points[i] = points

        rotate_edges = np.array(rotate_edges)
        return rotate_edges, rotate_points

    def _sampling_edge(
            self, cluster: NDArray, points: dict,
            sampling_step=0.1) -> list:
        """クラスタリングされた辺をサンプリングする処理

        Args:
            cluster (NDArray): 辺クラスタ
            points (dict): 屋根の座標点
            sampling_step (float, optional): サンプリング間隔(m). Defaults to 0.1.

        Returns:
            list: サンプリング結果
        """
        sampling_points = []
        for edge in cluster:
            size = GeoUtil.size(edge[0:2])
            sample_num = round(size / sampling_step)
            # sample_numが丸められた場合は
            # sampling_stepより小さい間隔の等差数列になる
            pt1 = points[int(edge[2])][int(edge[3])]
            pt2 = points[int(edge[2])][int(edge[3]) + 1]
            xs = np.linspace(pt1[0], pt2[0], sample_num)
            ys = np.linspace(pt1[1], pt2[1], sample_num)
            xy = np.c_[xs[..., np.newaxis], ys[..., np.newaxis]]
            sampling_points.append(xy.tolist())

        return sampling_points

    def _search_for_missing_vertices(
            self, edges: list[NDArray], rotate_points: dict,
            sampling_points: list, rev_rot: Rotation, base_pt: NDArray,
            insert_pt_list: list[InsertPoint], dist_th=0.1):
        """不足頂点の確認

        Args:
            edges (list[NDArray]): 同一位置に存在するエッジ情報リスト
            rotate_points (dict): 回転済み屋根頂点列(作業用データ)
            sampling_points (list): 辺をサンプリングした際の頂点列データ
            rev_rot (Rotation): 回転済み屋根頂点を元の座標に戻す際の回転情報
            base_pt (NDArray): 回転済み屋根頂点を元の座標に戻す際の移動量
            insert_pt_list (list[InsertPoint]): 不足頂点の情報(戻り値に相当)
            dist_th (float, optional): 距離閾値. Defaults to 0.1.
        """

        def _check_point(
                sampling_points: NDArray, pt: NDArray,
                dist_th=0.1) -> Tuple[NDArray, float]:
            """途中点確認

            Args:
                sampling_points (NDArray): 辺をサンプリングした頂点列
                pt (NDArray): 確認する頂点座標
                dist_th (float, optional): 距離閾値(m). Defaults to 0.1.

            Returns:
                Tuple[NDArray, float]: 追加頂点座標, 辺の始点からのベクトル係数
                None: 頂点追加が不用な場合
            """
            nn = NearestNeighbors(
                n_neighbors=1, algorithm='kd_tree', leaf_size=10, n_jobs=8)
            nn.fit(sampling_points)
            dists, inds = nn.kneighbors(pt.reshape(1, 2), return_distance=True)
            dists = dists[:, 0]
            inds = inds[:, 0]

            if dists[0] < dist_th:
                vec1 = sampling_points[-1] - sampling_points[0]
                vec2 = sampling_points[inds[0]] - sampling_points[0]
                size1 = GeoUtil.size(vec1)
                size2 = GeoUtil.size(vec2)
                if GeoUtil.is_zero(size1):
                    return None

                t = size2 / size1
                # 切り捨て
                t = float(int(t * self._DIV_VALUE)) / self._DIV_VALUE
                if GeoUtil.is_zero(t) or GeoUtil.is_same_value(t, 1.0):
                    return None

                # 途中点
                dst_pt = np.array(
                    [sampling_points[inds[0]][0],
                        sampling_points[inds[0]][1],
                        0.0])
                return dst_pt, t
            else:
                return None

        ids = [[int(edge[2]), int(edge[3])] for edge in edges]
        for m in range(len(edges) - 1):
            for n in range(m + 1, len(edges)):
                if ids[m][0] == ids[n][0]:
                    # 同一屋根IDの場合
                    continue
                # 辺の頂点を取得
                start_m = rotate_points[ids[m][0]][ids[m][1]][0:2]
                end_m = rotate_points[ids[m][0]][ids[m][1] + 1][0:2]
                start_n = rotate_points[ids[n][0]][ids[n][1]][0:2]
                end_n = rotate_points[ids[n][0]][ids[n][1] + 1][0:2]
                param_list = [
                    [n, start_m], [n, end_m], [m, start_n], [m, end_n]]

                for index, input_pt in param_list:
                    data = _check_point(
                        sampling_points=np.array(sampling_points[index]),
                        pt=input_pt, dist_th=dist_th)
                    if data is not None:
                        src_pt, t = data

                        dst_pt = rev_rot.apply(src_pt)  # 回転
                        dst_pt = dst_pt[0:2] + base_pt  # 移動
                        item = InsertPoint(
                            dst_pt[0], dst_pt[1],
                            int(edges[index][2]),
                            int(edges[index][3]) + 1, t)

                        # 追加予定リスト内の重複確認
                        exist = filter(
                            lambda x: x.roof_id == item.roof_id
                            and x.index == item.index and GeoUtil.is_same_value(x.t, item.t),
                            insert_pt_list)
                        if len(list(exist)) == 0:
                            insert_pt_list.append(item)

    def _create_wall_surface(self, clusters: list[ClusterInfo],
                             ground_height: float) -> list[FaceInfo]:
        """壁面作成処理

        Args:
            clusters (list[ClusterInfo]): ClusterInfoリスト
            ground_height (float): 地面の高さ

        Returns:
            list[FaceInfo]: 壁面情報のリスト
        """
        surface = []

        # 壁面作成位置情報
        line_infos = []

        # グラフ作成
        graph = nx.DiGraph()
        # 屋根
        for cluster in clusters:
            if cluster.roof_polygon is None or cluster.roof_polygon.area <= 0:
                continue

            roof = cluster.roof_line
            for i in range(len(roof)):
                pt1 = CompPoint(roof[i - 1][0], roof[i - 1][1], 0.0)
                pt2 = CompPoint(roof[i][0], roof[i][1], 0.0)
                size = GeoUtil.size(roof[i] - roof[i - 1])
                if GeoUtil.is_zero(size):
                    continue
                graph.add_edge(pt1, pt2, roof=cluster.id)

        # 最外壁、島屋根の外壁
        for outline in self._outlines:
            if outline.points is None:
                continue

            roof = outline.points
            for i in range(len(roof) - 1):
                pt1 = CompPoint(roof[i][0], roof[i][1], 0.0)
                pt2 = CompPoint(roof[i + 1][0], roof[i + 1][1], 0.0)
                size = GeoUtil.size(roof[i + 1] - roof[i])
                if GeoUtil.is_zero(size):
                    continue
                graph.add_edge(pt1, pt2, roof=outline.roof_idx)

        # 双方向エッジの探索
        single = []
        pairs = {}
        attr = nx.get_edge_attributes(graph, 'roof')
        for edge in graph.edges:
            if edge in pairs or edge in pairs.values():
                continue    # 登録済み

            # 反対方向エッジの探索
            try:
                tmp = graph.edges(edge[1])
                reverse_edge = [e for e in tmp if e[1] == edge[0]]
            except Exception:
                continue
            
            if len(reverse_edge) > 0:
                # 反対方向エッジが有れば登録
                pairs[edge] = reverse_edge[0]
                roof_idx1 = attr[edge]
                roof_idx2 = attr[reverse_edge[0]]

                if ((roof_idx1 == self._GROUND_IDX
                     or roof_idx2 == self._GROUND_IDX)
                        and roof_idx2 > roof_idx1):
                    # 地面の場合は、LineInfoのid2に地面(-1)を設定する
                    roof_idx1, roof_idx2 = roof_idx2, roof_idx1
                elif (-1 < roof_idx2 and roof_idx2 < len(clusters)
                        and clusters[roof_idx2].parent == roof_idx1):
                    # 親子関係がある場合は、id1が島id、id2が親idを設定する
                    roof_idx1, roof_idx2 = roof_idx2, roof_idx1

                points = np.array(
                    [[edge[0].x, edge[0].y], [edge[1].x, edge[1].y]])
                line = geo.LineString(points)
                info = LineInfo(line, roof_idx1, roof_idx2)
                line_infos.append(info)
            else:
                # 欠落部
                single.append(edge)

        # 欠落Debug情報
        if len(single) > 0:
            Log.output_log_write(
                LogLevel.DEBUG, ModuleType.MODEL_ELEMENT_GENERATION,
                f'id = {self.id}, missing edge')
            for edge in single:
                Log.output_log_write(
                    LogLevel.DEBUG, ModuleType.MODEL_ELEMENT_GENERATION,
                    f'{edge[0].x}, {edge[0].y} - {edge[1].x}, {edge[1].y}')

        # 壁面作成
        for line_info in line_infos:
            is_hole = False
            cluster1: ClusterInfo = clusters[line_info.id1]
            upper_height = cluster1.roof_height
            if line_info.id2 < 0:
                # 地面の場合
                lower_height = ground_height
                # 壁面の法線ベクトル算出時に使用するポリゴン
                base_polygon = self._shape

            else:
                # 屋根の場合
                cluster2: ClusterInfo = clusters[line_info.id2]
                lower_height = cluster2.roof_height

                # 島屋根(穴)の確認
                if (cluster1.parent > self._NO_PARENT_IDX
                        and line_info.id2 == cluster1.parent
                        and cluster1.roof_height < cluster2.roof_height):
                    # cluster1が島屋根である
                    # cluster2がcluster1の親である
                    # cluster1の高さがcluster2よりも低い(穴)
                    is_hole = True

                # 壁面の法線ベクトル算出時に使用するポリゴン
                base_polygon = cluster1.roof_polygon
                if not is_hole:
                    # 島屋根(穴)ではない場合
                    if cluster1.roof_height < cluster2.roof_height:
                        base_polygon = cluster2.roof_polygon

            if upper_height < lower_height:
                # 上下逆の場合はswap
                upper_height, lower_height = lower_height, upper_height

            base_points = np.array(line_info.line.coords)
            for i in np.arange(len(base_points) - 1):
                pos1 = base_points[i]
                pos2 = base_points[i + 1]
                points = np.array([[pos1[0], pos1[1], upper_height],
                                   [pos1[0], pos1[1], lower_height],
                                   [pos2[0], pos2[1], lower_height],
                                   [pos2[0], pos2[1], upper_height]])

                # 壁面の法線ベクトル作成
                base_pos = (points[3] - points[0]) * 0.5 + points[0]
                normal = self._calc_wall_normal_vector(
                    pos1=points[0], pos2=points[1], pos3=points[2],
                    base_pos=base_pos, shape=base_polygon, is_hole=is_hole)

                # 壁面の順列方向の確認
                if self._check_reverse_wall(wall_points=points, normal=normal):
                    # 反転
                    points = np.flipud(points)

                # 登録
                face = FaceInfo(points=points, type=FaceType.WALL)
                surface.append(face)

        return surface

    def _calc_wall_normal_vector(
            self, pos1: NDArray, pos2: NDArray, pos3: NDArray,
            base_pos: NDArray, shape: geo.Polygon, is_hole=False):
        """壁面の法線ベクトルの算出

        Args:
            pos1 (NDArray): 壁面の頂点1
            pos2 (NDArray): 壁面の頂点2
            pos3 (NDArray): 壁面の頂点3
            base_pos (NDArray): 基準点
            shape (geo.Polygon): 基準とするポリゴンの輪郭線
            is_hole (bool, optional): 屋根同士の関係が包含かつ穴か否か. Defaults to False.

        Returns:
            NDArray: 壁面の法線ベクトル
        """
        # 法線ベクトルの取得
        plane = PlaneInfo()
        plane.calc_plane(pt1=pos1, pt2=pos2, pt3=pos3)
        normal = plane.normal
        # 平面に落とし込む(z座標を無視)
        normal[2] = 0.0
        normal = GeoUtil.normalize(normal)

        # 法線の向き(表裏)の確認
        # 法線方向に少し進んだ地点が屋根の輪郭内か確認する
        # 小領域の屋根の場合に係数が大きいと輪郭線を突き抜ける場合があるため注意
        tmp_pos = normal * 0.01 + base_pos
        pos = geo.Point(tmp_pos[0], tmp_pos[1])
        
        # 内外判定
        is_within = pos.within(shape)

        if (not is_hole and is_within) or (is_hole and not is_within):
            # 穴ではない and 基準とするポリゴン内に点がある場合
            # 穴 and 基準とするポリゴン外に点がある場合
            # 法線の向きを逆にする
            normal *= -1

        return normal

    def _check_reverse_wall(
            self, wall_points: NDArray, normal: NDArray) -> bool:
        """壁面の頂点列の反転確認

        Args:
            wall_points (NDArray): 壁面の頂点列
            normal (NDArray): 壁面の法線ベクトル

        Returns:
            bool: True : 要反転, False : 反転不要
        """
        # 壁面をxy平面に投影するためにz軸との外積ベクトルを軸にして回転する
        vec_z = np.array([0.0, 0.0, 1.0])
        cross_vec = np.cross(vec_z, normal)     # 回転軸
        cross_vec = GeoUtil.normalize(cross_vec)
        angle = GeoUtil.angle(vec_z, normal)    # 回転角

        # 回転オブジェクト
        rot: Rotation = GeoUtil.rotate_object(vec=cross_vec, angle=-angle)

        # 回転後
        rot_points = rot.apply(wall_points)

        # 2次元化
        rot_points_2d = rot_points[:, 0:2]

        # 向き確認
        polygon = geo.Polygon(rot_points_2d)
        if not polygon.exterior.is_ccw:
            # 反時計回りを表とするため、時計回りの場合は要反転
            return True

        return False

    def _create_ground_surface(self, shape: geo.Polygon,
                               ground_height: float) -> FaceInfo:
        """地面作成

        Args:
            shape (geo.Polygon): 建物外形ポリゴン
            ground_height (float): 地面の高さ

        Returns:
            FaceInfo: 地面情報
        """
        points = np.array(shape.exterior.coords)
        if shape.exterior.is_ccw:
            # 反時計回り(表)の場合は、時計回り(裏)とする
            points = np.flipud(points)

        points = np.delete(points, obj=len(points) - 1, axis=0)
        tmp_z = np.full((len(points), 1), ground_height)
        # 2次元座標を3次元座標にする
        points = np.hstack((points, tmp_z))
        face = FaceInfo(points=points, type=FaceType.GROUND)
        return face

    def _solid(self, dist_th=0.001):
        """不足頂点の追加処理

        Args:
            dist_th (float, optional): 辺探索用の距離閾値. Defaults to 0.001.
        """
        # 不足頂点の追加(閉じたソリッドになっていない問題の対応)
        for i in np.arange(len(self._faces)):
            for j in np.arange(len(self._faces)):
                if i == j:
                    continue

                face1: FaceInfo = self._faces[i]
                face2: FaceInfo = self._faces[j]
                if ((face1.face_type == FaceType.ROOF
                        and face2.face_type == FaceType.GROUND)
                    or (face1.face_type == FaceType.GROUND
                        and face2.face_type == FaceType.ROOF)):

                    # 屋根と地面のペアは無視する
                    continue

                # 不足頂点の追加
                new_points = self._add_missing_vertex(
                    target=face1.points, candidates=face2.points,
                    dist_th=dist_th)

                # 更新
                self._faces[i].points = new_points

    def _add_missing_vertex(self, target: NDArray, candidates: NDArray,
                            dist_th=0.001):
        """不足頂点の追加処理

        Args:
            target (NDArray): 追加対象ポリゴンの頂点列
            candidates (NDArray): 追加する頂点の候補
            dist_th (float, optional):
                targetの辺上に位置するか判定するための距離閾値. Defaults to 0.001.

        Returns:
            NDArray: 不足頂点追加後のポリゴン頂点列
        """
        tmp_points = target

        if len(tmp_points) < 3:
            # ポリゴンに満たない場合は何もせずに終了
            return tmp_points

        # 不足頂点の追加
        for i in np.arange(len(candidates)):
            # 追加地点の探索
            index = -1
            min_dist = np.abs(dist_th)

            for j in np.arange(len(tmp_points)):
                next = j + 1 if j < len(tmp_points) - 1 else 0
                pos = tmp_points[j]
                vec = tmp_points[next] - tmp_points[j]
                dist, h, t = self._point_to_line_dist(
                    pos=candidates[i], line_pos=pos, line_vec=vec)

                if 0.0 < t and t < 1.0 and dist < min_dist:
                    index = next
                    min_dist = dist

            if index > -1:
                # 頂点追加
                tmp_points = np.insert(
                    tmp_points, index, candidates[i], axis=0)

        return tmp_points

    def _merge_vertex(self, xy_dist=0.01, reso=0.5) -> None:
        """モデル面の頂点のマージ

        Args:
            xy_dist (float, optional):
                xy平面上でのマージ距離m (0より大きい値). Defaults to 0.01.
            reso (float, optional):
                高さ方向の解像度m (0より大きい値). Defaults to 0.5.
        """
        if xy_dist <= 0.0 or reso <= 0.0:
            return

        for i in np.arange(len(self._faces)):
            face: FaceInfo = self._faces[i]
            points = copy.deepcopy(face.points)
            # face index
            face_label = np.full((len(points), 1), i)
            # point index
            point_label = np.reshape(np.arange(len(points)), (len(points), 1))
            # ラベル付与
            points = np.hstack([points, face_label])
            points = np.hstack([points, point_label])

            # 頂点一覧に追加
            if i == 0:
                tmp_points = points
            else:
                tmp_points = np.vstack([tmp_points, points])

        # xy座標でクラスタリング
        clusters = self._euclidean_clustering(
            cloud=tmp_points, num_th=1, dist_th=xy_dist, ignore_z=True)

        for cluster in clusters:
            # x, yの代表点
            # 壁面の縦方向エッジが傾かないように、xy座標を統一する
            x = cluster[0][0]
            y = cluster[0][1]

            # 高さ方向のグルーピング
            clusters_height = self._euclidean_clustering(
                cloud=cluster, num_th=1, dist_th=reso, ignore_z=False)
            for cluster_height in clusters_height:
                # 最高高さ地点
                max_z = np.max(cluster_height, axis=0)[2]

                # 更新
                for i in np.arange(len(cluster_height)):
                    fid = int(cluster_height[i][3])
                    pid = int(cluster_height[i][4])
                    self._faces[fid].points[pid] = np.array([x, y, max_z])

        # 同一頂点が連続している場合は削除する
        del_face = []
        for i in np.arange(len(self._faces)):
            face = self._faces[i]
            del_target = []
            for j in np.arange(1, len(face.points)):
                ret = (face.points[j] == face.points[j - 1]).all()
                if ret:
                    del_target.append(j)

            # 頂点削除
            del_target = np.flipud(del_target)
            for j in del_target:
                face.points = np.delete(face.points, j, axis=0)

            if (face.points[0] == face.points[-1]).all():
                # 始終点が同一の場合
                face.points = np.delete(face.points, -1, axis=0)

            if len(face.points) < 3:
                del_face.append(i)

        # 不要面の削除
        del_face = np.flipud(del_face)
        for i in del_face:
            del self._faces[i]

    def output_obj(self, path: str):
        """objファイル出力

        Args:
            path (str): 出力パス
        """
        if len(self._faces) == 0:
            return

        roofs = []
        walls = []
        grounds = []

        for face in self._faces:
            if face.face_type == FaceType.ROOF:
                roofs.append(face.points)

            elif face.face_type == FaceType.WALL:
                walls.append(face.points)

            elif face.face_type == FaceType.GROUND:
                grounds.append(face.points)

        info = ObjInfo()
        info.append_faces(BldElementType.ROOF, roofs)
        info.append_faces(BldElementType.WALL, walls)
        info.append_faces(BldElementType.GROUND, grounds)
        info.write_file(file_path=path)

    def _euclidean_clustering(self, cloud: NDArray, num_th: int,
                              dist_th=0.9, ignore_z=False):
        """ユークリッド距離によるクラスタリング

        Args:
            cloud (NDArray): 点群
            num_th (int): 最小クラス点数閾値
            dist_th (float, optional): 距離閾値. Defaults to 0.9.
            ignore_z (bool, optional): z座標を無視するか否か. Defaults to False.

        Raises:
            ValueError: 最小クラスタ点数閾値が0以下の場合
            ValueError: 距離閾値がマイナス値の場合

        Returns:
            list[NDArray]: 点群クラスタ(NDArray)のリスト

        Note:
            cloudは、3次元座標の2次元配列を想定する
                cloud = [[x1, y1, z1],
                         [x2, y2, z2],
                         [x3, y3, z3]]

            xyz座標以外にラベルが付与されている場合も考慮する
                cloud = [[x1, y1, z1, label1],
                         [x2, y2, z2, label2],
                         [x3, y3, z3, label3]]
        """
        class_name = self.__class__.__name__
        func_name = sys._getframe().f_code.co_name
        new_clusters = []

        if num_th < 1:
            msg = '{}.{}, {}'.format(
                class_name, func_name,
                CreateModelMessage.ERR_MSG_MODEL_NUM_TH)
            raise CreateModelException(msg)

        if dist_th < 0:
            msg = '{}.{}, {}'.format(
                class_name, func_name,
                CreateModelMessage.ERR_MSG_MODEL_DIST_TH)
            raise CreateModelException(msg)

        # ラベルの準備
        label = np.full(len(cloud), -1)

        # 点の準備
        if cloud.shape[1] > 3:
            # xyz以降にラベルがついている場合
            tmp_cloud = cloud[:, 0:3]
        else:
            tmp_cloud = cloud

        if ignore_z:
            # z座標を無視する
            tmp_cloud = tmp_cloud[:, 0:2]       # xyのみ取得
            tmp_z = np.zeros((len(cloud), 1))   # z座標は0埋め
            tmp_cloud = np.hstack((tmp_cloud, tmp_z))

        # KDTree
        tree = KDTree(tmp_cloud)

        # クラスタリング
        id = 0
        for n in np.arange(0, len(cloud)):
            if label[n] > -1:
                continue    # 探索済み

            cluster_index = []
            cluster_index.append(n)

            i = 0
            while i < len(cluster_index):
                # 近傍探索
                target = cluster_index[i]
                indexes = tree.query_radius(
                    tmp_cloud[target:target + 1], r=dist_th)
                indexes = indexes[0]

                for index in indexes:
                    if (label[index] > -1):
                        continue    # 探索済み

                    label[index] = id
                    if cluster_index[i] != index:
                        cluster_index.append(index)

                i += 1

            if len(cluster_index) >= num_th:
                # 新規クラスタの追加
                cluster = cloud[label == id]
                new_clusters.append(cluster)

            # 採用/不採用に関わらずidをインクリメント
            id += 1

        return new_clusters

    def _angle_clustering(self, edges: NDArray, angle_th=5.0) -> list[NDArray]:
        """辺の角度によるクラスタリング

        Args:
            edges (NDArray): 辺データ.Note参照
            angle_th (float, optional): 角度閾値(deg). Defaults to 5.0.

        Returns:
            list[NDArray]: クラスタリング結果(入力データをグルーピングした結果)

        Note:
            edgesは、2次元配列を想定する
            1辺のデータは、ベクトルのx座標, y座標, 屋根id, 頂点id
            (辺の始点となる頂点のid)の想定
                edges = [[x1, y1, roof_id1, point_id1],
                         [x2, y2, roof_id1, point_id2],
                         [x3, y3, roof_id2, point_id1]]

            最低限、ベクトルのx座標, y座標のデータが入力されれば動作する
                edges = [[x1, y1],
                         [x2, y2],
                         [x3, y3]]
        """
        new_clusters = []

        if angle_th < 0.0:
            angle_th = 1.0

        # 準備
        if edges.shape[1] > 2:
            # xy以降にラベルがついている場合
            tmp_edges = edges[:, 0:2]
        else:
            tmp_edges = edges

        # 角度算出
        vec_x = np.array((1.0, 0.0))
        tmp_angles = [
            GeoUtil.signed_angle(vec_x, vec) for vec in tmp_edges]
        angles = [
            [angle + 180.0, 0] if angle < 0
            else [angle, 0] for angle in tmp_angles]

        angles = np.array(angles)
        db = DBSCAN(eps=angle_th, min_samples=1).fit(angles)
        labels = db.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        for ci in range(n_clusters):
            inds = np.where(labels == ci)[0]
            new_cluster = edges[inds]
            new_clusters.append(new_cluster)

        return new_clusters

    def _edge_dist_clustering(
            self, edges: NDArray, points: dict,
            dist_th=0.1, sampling_step=0.01) -> list[NDArray]:

        """辺の距離によるクラスタリング

        Args:
            edges (NDArray): 辺データ.Note参照
            points (dict): 屋根ごとの輪郭線の頂点データ.
            dist_th (float, optional): 距離閾値(m). Defaults to 0.1.
            sampling_step (float, optional): サンプリング間隔(m). Defaults to 0.01.

        Returns:
            list[NDArray]: クラスタリング結果(入力データをグルーピングした結果)

        Note:
            edgesは、2次元配列を想定する
            1辺のデータは、ベクトルのx座標, y座標, 屋根id, 頂点id
            (辺の始点となる頂点のid)の想定
                edges = [[x1, y1, roof_id1, point_id1],
                         [x2, y2, roof_id1, point_id2],
                         [x3, y3, roof_id2, point_id1]]

            最低限、ベクトルのx座標, y座標のデータが入力されれば動作する
                edges = [[x1, y1],
                         [x2, y2],
                         [x3, y3]]
        """
        new_clusters = []

        if dist_th < 0.0:
            dist_th = 0.1

        # sampling
        sampling_data = []
        sampling_id = []
        for i, edge in enumerate(edges):
            size = GeoUtil.size(edge[0:2])
            sample_num = round(size / sampling_step)
            # sample_numが丸められた場合はsampling_stepより小さい間隔の等差数列になる
            pt1 = points[int(edge[2])][int(edge[3])]
            pt2 = points[int(edge[2])][int(edge[3]) + 1]
            xs = np.linspace(pt1[0], pt2[0], sample_num)
            ys = np.linspace(pt1[1], pt2[1], sample_num)
            xy = np.c_[xs[..., np.newaxis], ys[..., np.newaxis]]
            sampling_data += xy.tolist()
            sampling_id += [i] * len(xy)
        sampling_data = np.array(sampling_data)
        sampling_id = np.array(sampling_id)
        db = DBSCAN(eps=dist_th, min_samples=1).fit(sampling_data)
        labels = db.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        for ci in range(n_clusters):
            inds = np.where(labels == ci)[0]
            edge_ids = np.unique(sampling_id[inds])
            new_cluster = edges[edge_ids]
            new_clusters.append(new_cluster)

        return new_clusters

    def _merge_vertex_2d(self, clusters: list[ClusterInfo], dist=0.1) -> None:
        """屋根面の頂点のマージ

        Args:
            list[ClusterInfo]: ClusterInfoリスト
            dist (float, optional):
                マージ距離m (0より大きい値). Defaults to 0.1.
        """
        if dist <= 0.0 or len(clusters) == 0:
            return

        indexes = [i for i, cluster in enumerate(clusters)
                   if cluster._roof_line is not None]

        for i, index in enumerate(indexes):
            points = copy.deepcopy(clusters[index].roof_line)

            # index
            label = np.full((len(points), 1), index)
            # point index
            point_label = np.reshape(np.arange(len(points)), (len(points), 1))
            # ラベル付与
            points = np.hstack([points, label])
            points = np.hstack([points, point_label])

            # 頂点一覧に追加
            if i == 0:
                tmp_points = points
            else:
                tmp_points = np.vstack([tmp_points, points])

        db = DBSCAN(eps=dist, min_samples=1).fit(tmp_points[:, 0:2])
        labels = db.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        for ci in range(n_clusters):
            inds = np.where(labels == ci)[0]
            result = tmp_points[inds]

            # x, yの代表点
            x_points = result[:, 0:1]
            y_points = result[:, 1:2]
            x = np.mean(x_points)
            y = np.mean(y_points)

            # 更新
            for i in np.arange(len(result)):
                rid = int(result[i][2])
                pid = int(result[i][3])
                clusters[rid].roof_line[pid] = np.array([x, y])

        # 同一頂点が連続している場合は削除する
        for i in indexes:
            roof = clusters[i]
            del_target = []
            for j in np.arange(len(roof.roof_line)):
                ret = (roof.roof_line[j] == roof.roof_line[j - 1]).all()
                if ret:
                    del_target.append(j)

            # 頂点削除
            del_target = np.flipud(del_target)
            for j in del_target:
                roof.roof_line = np.delete(roof.roof_line, j, axis=0)

            if len(roof.roof_line) < 3:
                clusters[i].roof_line = None
