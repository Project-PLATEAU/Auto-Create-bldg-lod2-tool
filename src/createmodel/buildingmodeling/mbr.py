import numpy as np
import shapely.geometry as geo
import cv2 as cv
import copy
import sys
from numpy.typing import NDArray
from typing import Tuple
from anytree import PostOrderIter, AnyNode
from .clusterinfo import ClusterInfo
from .geoutil import GeoUtil
from ..createmodelexception import CreateModelException
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import MeanShift
from shapely import affinity
from shapely.geometry import CAP_STYLE, JOIN_STYLE
from enum import IntEnum
from ..message import CreateModelMessage
from ..param import CreateModelParam


class RoofPlaneType(IntEnum):
    """屋根と建物外形線の状態タイプ
    """
    UNKNOWN = 0
    """未知
    """
    ONE_NEIGHBOR_LINE = 1
    """隣接建物外形線が1種類
    """
    ORTHOGONAL_CROSSING = 2
    """隣接建物外形線が2種類かつ垂直に交わる
    """
    NOT_ORTHOGONAL_CROSSING = 3
    """隣接建物外形線が2種類かつ垂直に交わらない
    """
    THREE_OR_MORE_NEIGHBOR_LINES = 4
    """隣接建物外形線が3種類以上
    """
    NO_NEIGHBOR_LINE = 5
    """隣接建物外形線が無い
    """


class RoofPlaneRelationType(IntEnum):
    """屋根同士の関係タイプ
    """
    NO_RELATION = 0
    """関係なし
    """
    NEIGHBORING = 1
    """隣接する
    """
    CONTAINING = 2
    """包括する
    """
    CONTAINED = 3
    """包括される
    """


class FootPrint:
    """建物外形クラス
    """

    # プロパティ
    @property
    def poly(self) -> geo.Polygon:
        """建物外形ポリゴン

        Returns:
            geo.Polygon: 建物外形ポリゴン
        """
        return self._poly

    @property
    def poly_xy(self) -> NDArray:
        """建物外形ポリゴンの座標点列

        Returns:
            NDArray: 建物外形ポリゴンの座標点列
        """
        if self._poly is not None:
            return np.array(self._poly.exterior.coords)
        else:
            return None

    @property
    def lines_length(self) -> NDArray:
        """建物外形辺の長さ配列

        Returns:
            NDArray: 建物外形辺の長さ配列
        """
        if self._lines_length is None:
            lines_length = []
            for i in range(len(self.poly_xy) - 1):
                target_pos = self.poly_xy[i]
                next_pos = self.poly_xy[i + 1]
                length = GeoUtil.size(next_pos - target_pos)
                lines_length.append(length)
            self._lines_length = np.array(lines_length)
    
        return self._lines_length

    @property
    def lines_angle(self) -> NDArray:
        """建物外形の辺間角度(deg)配列

        Returns:
            NDArray: 建物外形の辺間角度(deg)配列
        """
        if self._lines_angle is None:
            lines_angle = []
            for i in range(len(self.poly_xy) - 1):
                target_pos = self.poly_xy[i]
                next_pos = self.poly_xy[i + 1]
                vec = next_pos - target_pos
                angle = np.arctan2(vec[1], vec[0])
                angle = np.rad2deg(angle)
                if angle > 90:
                    angle = angle - 180
                elif angle <= -90:
                    angle = angle + 180
                lines_angle.append(angle)
            self._lines_angle = np.array(lines_angle)

        return self._lines_angle

    @property
    def sample_points_xy(self) -> NDArray:
        """サンプリング後の建物外形点列

        Returns:
            NDArray: サンプリング後の建物外形点列
        """
        return self._sample_points_xy

    @property
    def sample_points_line_ids(self) -> NDArray:
        """サンプリング後の建物外形点の辺インデックス番号配列

        Returns:
            NDArray: サンプリング後の建物外形点の辺インデックス番号配列
        """
        return self._sample_points_line_ids

    def __init__(self, polygon: geo.Polygon) -> None:
        """コンストラクタ

        Args:
            polygon (geo.Polygon): 建物外形ポリゴン
        """
        self._poly = polygon
        self._lines_length = None
        self._lines_angle = None

        self._sample_points_xy = None
        self._sample_points_line_ids = None
        self._kdtree = None

    def sampling(self, sampling_step=0.25) -> Tuple[NDArray, NDArray]:
        """建物外形線を指定間隔でサンプリングする

        Args:
            sampling_step (float, optional): サンプリング間隔. Defaults to 0.25.

        Returns:
            Tuple[NDArray, NDArray]:
                サンプリング後の座標点列, インデックス番号配列
        """
        # 建物外形線をサンプリング間隔を基に分割する
        points_xy = self.poly_xy
        sampling_xy = []
        sampling_id = []
        for i in range(len(points_xy) - 1):
            target_pos = points_xy[i]
            next_pos = points_xy[i + 1]
            size = GeoUtil.size(next_pos - target_pos)
            sample_num = round(size / sampling_step)
            # sample_numが丸められた場合はsampling_stepより小さい間隔の等差数列になる
            xs = np.linspace(target_pos[0], next_pos[0], sample_num)
            ys = np.linspace(target_pos[1], next_pos[1], sample_num)
            xy = np.c_[xs[..., np.newaxis], ys[..., np.newaxis]]
            sampling_xy += xy.tolist()
            sampling_id += [i] * len(xy)
        self._sample_points_xy = np.array(sampling_xy)
        self._sample_points_line_ids = np.array(sampling_id)

    def get_neighbor_line_ids(
            self, points_xy: NDArray,
            neighbor_max_dist=0.5, jobs=8) -> NDArray:
        """近傍建物外形線のインデックス番号の取得

        Args:
            points_xy (NDArray): 屋根点群
            neighbor_max_dist (float, optional): 近郷とみなす距離閾値. \
                Defaults to 0.5.
            jobs (int, optional): NearestNeighborsのジョブ数. Defaults to 8.

        Returns:
            NDArray: インデックス番号配列
        """
        if self._kdtree is None:
            self._kdtree = NearestNeighbors(
                n_neighbors=1, algorithm='kd_tree', leaf_size=10, n_jobs=jobs)
            self._kdtree.fit(self.sample_points_xy)

        # 近隣する外形点を探索
        dists, inds = self._kdtree.kneighbors(points_xy, return_distance=True)
        dists = dists[:, 0]
        inds = inds[:, 0]
        
        # 近隣する外形線
        inds = inds[dists < neighbor_max_dist]
        return np.unique(self.sample_points_line_ids[inds])

    def get_distance_from_poly(
            self, poly: geo.Polygon, neighbor_dist_th=0.5) -> float:
        """ポリゴンと建物外形ポリゴンとの距離算出

        Args:
            poly (geo.Polygon): ポリゴン
            neighbor_dist_th (float, optional): 近傍距離最大値. Defaults to 0.5.

        Returns:
            float: ポリゴンと建物外形ポリゴン間の距離

        Note:
            polyの頂点の内、建物外形ポリゴンの内部にあり、
            建物外形線との距離が閾値内のものは近隣点とする。
            近隣点と外形線との最も遠い距離をpolyと建物外形ポリゴンとの距離とする。
        """
        max_dist = 0
        for x, y in np.array(poly.exterior.coords):
            p = geo.Point(x, y)
            if self.poly.contains(p):
                dist = p.distance(geo.LineString(self.poly_xy))
                if dist > neighbor_dist_th:
                    continue
                if dist > max_dist:
                    max_dist = dist
        return max_dist


class RoofPlane:
    """屋根クラス
    """
    def __init__(self) -> None:
        """コンストラクタ
        """
        self._points = None
        self._height = None
        self._angle = None
        self._type = RoofPlaneType.UNKNOWN
        self._poly = None

    # プロパティ
    @property
    def points(self) -> NDArray:
        """屋根点群(2d座標)

        Returns:
            NDArray: 屋根点群(2d座標)
        """
        return self._points

    @points.setter
    def points(self, value: NDArray):
        """屋根点群(2d座標)

        Args:
            value (NDArray): 屋根点群(2d座標)
        """
        self._points = value

    @property
    def poly(self) -> geo.Polygon:
        """屋根形状ポリゴン

        Returns:
            geo.Polygon: 屋根形状ポリゴン
        """
        return self._poly

    @poly.setter
    def poly(self, value: geo.Polygon):
        """屋根形状ポリゴン

        Args:
            value (geo.Polygon): 屋根形状ポリゴン
        """
        self._poly = value

    @property
    def height(self) -> float:
        """屋根の高さ

        Returns:
            float: 屋根の高さ
        """
        return self._height

    @height.setter
    def height(self, value: float):
        """屋根の高さ

        Args:
            value (float): 屋根の高さ
        """
        self._height = value

    @property
    def angle(self) -> float:
        """屋根ポリゴンの角度deg

        Returns:
            float: 屋根ポリゴンの角度deg
        """
        return self._angle

    @property
    def type(self) -> RoofPlaneType:
        """屋根のタイプ

        Returns:
            RoofPlaneType: 屋根のタイプ
        """
        return self._type

    def set_footprint(
            self, footprint: FootPrint, ms_bandwidth: float, ms_jobs: int,
            neighbor_max_dist: float, neighbor_jobs: int,
            angle_ortho_th: float, line_len_th: float) -> None:
        """建物外形線のセット

        Args:
            footprint (FootPrint): 建物外形線の情報
            ms_bandwidth (float): MeanShiftのバンド幅
            ms_jobs (int): MeanShiftのジョブ数
            neighbor_max_dist (float): 近傍建物外形線探索用の最大近傍距離閾値
            neighbor_jobs (int): NearestNeighborのジョブ数
            angle_ortho_th (float): 近傍建物外形線の状態判定用の角度閾値deg
            line_len_th (float): 短い建物外形線分を除外する際の距離閾値
        """
        # 建物外形線を角度でクラスタリング
        ms = MeanShift(
            bandwidth=ms_bandwidth, bin_seeding=False, n_jobs=ms_jobs)
        ms.fit(footprint.lines_angle[..., np.newaxis])

        # 屋根面の隣接建物外形線
        neighbor_line_ids = footprint.get_neighbor_line_ids(
            self.points, neighbor_max_dist=neighbor_max_dist,
            jobs=neighbor_jobs)

        # 短い隣接建物外形線を除外
        tmp_ids = [id for id in neighbor_line_ids
                   if footprint.lines_length[id] > line_len_th]
        neighbor_line_ids = tmp_ids

        # 屋根面タイプ
        if len(neighbor_line_ids):
            # 隣接建物外形線の方向をもとに屋根面のタイプを決定
            labels = ms.labels_[neighbor_line_ids]
            angles = ms.cluster_centers_[np.unique(labels)].flatten()

            if len(angles) == 1:
                # 隣接建物外形線の角度は1種類
                self._type = RoofPlaneType.ONE_NEIGHBOR_LINE
            elif len(angles) == 2:
                if abs(abs(angles[0] - angles[1]) - 90) < angle_ortho_th:
                    # 隣接建物外形線の角度は2種類、かつ、垂直交差
                    self._type = RoofPlaneType.ORTHOGONAL_CROSSING
                else:
                    self._type = RoofPlaneType.NOT_ORTHOGONAL_CROSSING
            else:
                # 隣接建物外形線の角度は3種類以上
                self._type = RoofPlaneType.THREE_OR_MORE_NEIGHBOR_LINES
        else:
            # 隣接建物外形線が無い
            self._type = RoofPlaneType.NO_NEIGHBOR_LINE

        # 屋根ポリコンの角度
        if len(neighbor_line_ids):
            # 最も長い隣接建物外形線の角度
            lengths = footprint.lines_length[neighbor_line_ids]
            angles = footprint.lines_angle[neighbor_line_ids]
            ind = np.argmax(lengths)
            self._angle = angles[ind]
        else:
            self._angle = None

    def get_relation_with_other(self, other) -> RoofPlaneRelationType:
        """屋根同士の関係

        Args:
            other (RoofPlane): 屋根情報

        Returns:
            RoofPlaneRelationType: 屋根関係タイプ
        """
        if self.poly is None or other.poly is None:
            return RoofPlaneRelationType.NO_RELATION
        if self.poly.intersection(other.poly).area / other.poly.area > 0.9:
            return RoofPlaneRelationType.CONTAINING
        elif self.poly.intersection(other.poly).area / self.poly.area > 0.9:
            return RoofPlaneRelationType.CONTAINED
        elif self.poly.distance(other.poly) < 0.5:
            return RoofPlaneRelationType.NEIGHBORING


class MBR:
    """MBRクラス
    """

    def __init__(self) -> None:
        """コンストラクタ
        """
        self._footprint = None
        self._roofplanes = []

    def _split(self, points: NDArray) -> list[NDArray]:
        """ラベリング処理による領域分割

        Args:
            points (NDArray): 分割対象の前景座標点列

        Returns:
            list[NDArray]: 領域分割結果
        """
        # バウンディングボックス
        min_x, min_y, width, height = self._get_bbox(points=points)

        # ラベリング
        img = np.zeros((height, width), np.uint8)
        img[points[:, 1] - min_y, points[:, 0] - min_x] = 255
        num, label_img = cv.connectedComponents(img, connectivity=4)

        # ラベルごとに有効ピクセルの座標を取得
        split_points = []
        for i in range(1, num):
            mask = (label_img == i).astype(np.uint8) * 255
            ys, xs = np.nonzero(mask)
            part_points = (np.c_[xs[..., np.newaxis], ys[..., np.newaxis]]
                           + [min_x, min_y])
            split_points.append(part_points)

        return split_points

    def _create_node(
            self, points: NDArray, parent=None, area_th=0,
            width_th=0.0, slim_rate_th=-1.0) -> AnyNode:
        """ノード作成

        Args:
            points (NDArray): 座標点
            parent (AnyNode, optional): 親ノード. Defaults to None.
            area_th (int, optional): 面積閾値. Defaults to 0.
            width_th (float, optional): 幅閾値. Defaults to 0.
            slim_rate_th (float, optional): 細長度閾値. Defaults to -1.0.

        Returns:
            AnyNode: 作成したノード
        """
        if parent is None:
            hier_id = 0
            node_id = 0
            file_id = "h0n0"
        else:
            hier_id = parent.hier_id + 1
            node_id = len(parent.children)
            file_id = "{}_h{}n{}".format(parent.file_id, hier_id, node_id)

        # 矩形近似
        min_x, min_y, width, height = self._get_bbox(points=points)
        rect_img = np.ones((height, width), np.uint8) * 255
        ys, xs = np.nonzero(rect_img)
        rect_points = (np.c_[xs[..., np.newaxis], ys[..., np.newaxis]]
                       + [min_x, min_y])
        
        # 差分取得
        img = np.zeros((height, width), np.uint8)
        img[points[:, 1] - min_y, points[:, 0] - min_x] = 255
        ys, xs = np.nonzero(np.logical_and(rect_img > 0, img == 0))
        diff_points = (np.c_[xs[..., np.newaxis], ys[..., np.newaxis]]
                       + [min_x, min_y])

        if len(diff_points) == 0:
            # ノード作成
            node = AnyNode(
                parent=parent, hier_id=hier_id, node_id=node_id,
                file_id=file_id, rect_points=rect_points,
                residual_points=diff_points)
            return node

        # 差分分割
        diff_points = self._split(diff_points)

        # 差分を絞り込む
        new_diff_points = []
        for i in range(len(diff_points)):
            if len(diff_points[i]) < area_th:   # 面積
                continue

            if slim_rate_th < 0:    # 細長度
                new_diff_points.append(diff_points[i])
                continue

            # 外形点
            exterior_points = self._get_exterior(diff_points[i])

            # 面積と長さ
            area = len(diff_points[i])
            length = len(exterior_points)

            # 平均幅
            avg_width = 2 * area / length

            # 細長度
            slim_rate = length / avg_width

            # 閾値処理
            if avg_width > width_th:
                new_diff_points.append(diff_points[i])
            elif slim_rate < slim_rate_th:
                new_diff_points.append(diff_points[i])
        diff_points = new_diff_points

        # ノード作成
        node = AnyNode(
            parent=parent, hier_id=hier_id, node_id=node_id, file_id=file_id,
            rect_points=rect_points, residual_points=diff_points)

        return node

    def _get_bbox(self, points: NDArray) -> Tuple[float, float, float, float]:
        """バウンディングボックスの作成

        Args:
            points (NDArray): 点群

        Returns:
            Tuple[float, float, float, float]: 最小x座標, 最小y座標、幅、高さ
        """
        xy_min = np.min(points, axis=0)
        xy_max = np.max(points, axis=0)
        width = xy_max[0] - xy_min[0] + 1
        height = xy_max[1] - xy_min[1] + 1
        return xy_min[0], xy_min[1], width, height

    def _transform(
            self, points: NDArray, angle: float, width: float, height: float,
            nf=False) -> Tuple[NDArray, int, int, NDArray, NDArray]:
        """入力点を画像化

        Args:
            points (NDArray): 点群
            angle (float): 回転角度deg
            width (float): 画像幅
            height (float): 画像高さ
            nf (bool, optional): ノイズ除去フラグ. Defaults to False.

        Returns:
            Tuple[NDArray, int, int, NDArray, NDArray]:
                回転後座標, 回転後画像幅, 回転後画像高さ, 入力点の画像, 回転後の画像
        """
        # 入力点を画像化
        image = np.zeros((height, width), np.uint8)
        image[points[:, 1], points[:, 0]] = 255

        # 入力点画像を中心回転（時計回り）-> 平行移動（前景中心＝回転後画像中心）
        angle_rad = np.radians(angle)
        rotate_width = (int(np.round(width * abs(np.cos(-angle_rad))
                        + height * abs(np.sin(-angle_rad)))))
        rotate_height = (int(np.round(width * abs(np.sin(-angle_rad))
                         + height * abs(np.cos(-angle_rad)))))

        matrix = cv.getRotationMatrix2D(
            center=[width / 2, height / 2], angle=-angle, scale=1)
        matrix[0][2] += -width / 2 + rotate_width / 2
        matrix[1][2] += -height / 2 + rotate_height / 2

        rotate_image = cv.warpAffine(
            image, matrix, (rotate_width, rotate_height),
            flags=cv.INTER_NEAREST, borderValue=(0, 0, 0))

        # ノイズフィルタリング
        if nf:
            # 回転した入力点座標を取得
            ys, xs = np.nonzero(rotate_image)
            rotate_points = np.c_[xs[..., np.newaxis], ys[..., np.newaxis]]
            _, _, width, height = self._get_bbox(rotate_points)
            # ノイズフィルタリング
            if width > 5 and height > 5:
                kernel = np.ones((3, 3), np.uint8)
                rotate_image = cv.morphologyEx(
                    rotate_image, cv.MORPH_OPEN, kernel)
                rotate_image = cv.morphologyEx(
                    rotate_image, cv.MORPH_CLOSE, kernel)
 
        # 回転した入力点座標を取得
        ys, xs = np.nonzero(rotate_image)
        rotate_points = np.c_[xs[..., np.newaxis], ys[..., np.newaxis]]

        return rotate_points, rotate_width, rotate_height, image, rotate_image

    def _reverse_transform(
            self, points: NDArray, angle: float, width: int, height: int,
            rotate_width: int, rotate_height: int) -> NDArray:
        """画像化した点群から作成したポリゴンを入力点群の座標系に戻す

        Args:
            points (NDArray): ポリゴンの座標点列
            angle (float): 回転角度deg
            width (int): 画像幅
            height (int): 画像高さ
            rotate_width (int): 回転時の画像幅
            rotate_height (int): 回転時の画像高さ

        Returns:
            NDArray: ポリゴン座標点列
        """
        # 屋根外形点を回転後画像中心で回転（反時計回り）-> 平行移動
        center = [rotate_width / 2, rotate_height / 2]
        matrix = cv.getRotationMatrix2D(center=center, angle=angle, scale=1)
        matrix[0][2] += -rotate_width / 2 + width / 2
        matrix[1][2] += -rotate_height / 2 + height / 2
        matrix = np.array([matrix[0][0], matrix[0][1],
                           matrix[1][0], matrix[1][1],
                           matrix[0][2], matrix[1][2]])

        polygon = geo.Polygon(points)
        polygon = affinity.affine_transform(polygon, matrix)
        rotate_points = np.array(polygon.exterior.coords)
        return rotate_points

    def _get_exterior(self, points: NDArray) -> NDArray:
        """外形点の取得

        Args:
            points (NDArray): 有効領域の点

        Returns:
            NDArray: 外形点
        """
        min_x, min_y, width, height = self._get_bbox(points)
        img = np.zeros((height + 2, width + 2), np.uint8)
        img[points[:, 1] - min_y + 1, points[:, 0] - min_x + 1] = 255

        kernels = np.array(
            [[[-1, 0, 0], [0, +1, 0], [0, 0, 0]],
             [[0, -1, 0], [0, +1, 0], [0, 0, 0]],
             [[0, 0, -1], [0, +1, 0], [0, 0, 0]],
             [[0, 0, 0], [0, +1, -1], [0, 0, 0]],
             [[0, 0, 0], [0, +1, 0], [0, 0, -1]],
             [[0, 0, 0], [0, +1, 0], [0, -1, 0]],
             [[0, 0, 0], [0, +1, 0], [-1, 0, 0]],
             [[0, 0, 0], [-1, +1, 0], [0, 0, 0]]], dtype='int')
        mask = np.zeros_like(img)
        for kernel in kernels:
            mask_ = cv.morphologyEx(img, cv.MORPH_HITMISS, kernel)
            mask[mask_ > 0] = 255

        ys, xs = np.nonzero(mask)
        exterior_points = (np.c_[xs[..., np.newaxis], ys[..., np.newaxis]]
                           + [min_x - 1, min_y - 1])
        return exterior_points

    def _get_contour(self, mask: NDArray) -> NDArray:
        """外形線作成

        Args:
            mask (NDArray): マスク

        Returns:
            NDArray: 外形線点列
        """
        # 外形点を抽出
        contours, _ = cv.findContours(
            mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        if len(contours) > 1:
            # 最大面積の領域で上書き
            areas = [geo.Polygon(np.squeeze(con)).area for con in contours]
            index = np.argmax(areas)
            contours = [contours[index]]

        #assert len(contours) == 1, "# of contours is not one"
        contour_points = contours[0].reshape(-1, 2)

        # コーナー点を抽出
        contour_image = np.zeros_like(mask)
        contour_image[contour_points[:, 1], contour_points[:, 0]] = 255
        contour_image = cv.copyMakeBorder(
            contour_image, 1, 1, 1, 1, cv.BORDER_CONSTANT, value=0)
        contour_points += np.array([1, 1], int)
        corner_points = []
        for x, y in contour_points:
            l_pixel = contour_image[y, x - 1]
            r_pixel = contour_image[y, x + 1]
            if l_pixel == 255 and r_pixel == 255:
                continue
            t_pixel = contour_image[y - 1, x]
            b_pixel = contour_image[y + 1, x]
            if t_pixel == 255 and b_pixel == 255:
                continue
            corner_points.append([x, y])
        corner_points -= np.array([1, 1], int)
        return corner_points

    def _mbr(
            self, points: NDArray, angle: float,
            rect_area_th=10, width_th=0.0, slim_rate_th=-1.0,
            max_hiers=10) -> NDArray:
        """mbr処理

        Args:
            points (NDArray): MBR対象点群
            angle (float): 回転角
            rect_area_th (int, optional): 矩形のpixel数閾値. Defaults to 10.
            width_th (float, optional): 幅閾値. Defaults to 0.
            slim_rate_th (float, optional): 細長度閾値. Defaults to -1.0.
            max_hiers (int, optional): 最大階層数. Defaults to 10.

        Returns:
            NDArray: mbr結果の座標点列. 作成に失敗した場合はNone.
        """
        # バウンディングボックス
        min_x, min_y, width, height = self._get_bbox(points)
        # 屋根を平行移動
        trans_points = points - np.array([min_x, min_y])
        # 屋根点を回転
        rotate_points, rotate_width, rotate_height, image, rotate_image \
            = self._transform(points=trans_points, angle=angle,
                              width=width, height=height)
        
        # モルフォロジー処理で屋根点が全消去の場合
        if len(rotate_points) == 0:
            return None

        # root node の作成
        root_node = self._create_node(
            points=rotate_points, parent=None, area_th=rect_area_th,
            width_th=width_th, slim_rate_th=slim_rate_th)

        unprocessed_nodes = []   # 未処理リスト
        if len(root_node.residual_points):
            unprocessed_nodes += [root_node]

        # MBR処理(階層ノードの作成)
        while len(unprocessed_nodes):
            current = unprocessed_nodes.pop(0)
            for residual_points in current.residual_points:
                node = self._create_node(
                    points=residual_points, parent=current,
                    area_th=rect_area_th, width_th=width_th)
                if len(node.residual_points):
                    unprocessed_nodes += [node]
            if current.hier_id > max_hiers:
                break

        # 屋根外形の作成
        for node in PostOrderIter(root_node):
            if node.is_leaf:
                # 葉ノードの場合は、矩形形状をそのままノードの形状点とする
                node.points = node.rect_points
                continue

            img = np.zeros((rotate_height, rotate_width), np.uint8)
            # 注目ノードを前景設定
            img[node.rect_points[:, 1], node.rect_points[:, 0]] = 255
            # 子ノード部分を背景設定
            for child in node.children:
                img[child.points[:, 1], child.points[:, 0]] = 0

            # 注目ノードの形状点
            ys, xs = np.nonzero(img)
            node.points = np.c_[xs[..., np.newaxis], ys[..., np.newaxis]]
        
        mask_simplified = np.zeros((rotate_height, rotate_width), np.uint8)
        mask_simplified[root_node.points[:, 1], root_node.points[:, 0]] = 255

        corner_points = self._get_contour(mask=mask_simplified)
        # assert len(corner_points) > 3, "# of corner_points is less than 3"
        if len(corner_points) <= 3:
            return None

        # 屋根外形点の逆回転
        rotate_corner_points = self._reverse_transform(
            points=corner_points, angle=angle, width=width, height=height,
            rotate_width=rotate_width, rotate_height=rotate_height)

        # 平行移動
        trans_corner_points = rotate_corner_points + np.array([min_x, min_y])

        return trans_corner_points

    def _rectify(self):
        """屋根整形
        """
        param = CreateModelParam.get_instance()
        # 形状の簡略化
        for roofplane in self._roofplanes:
            if roofplane.poly is None:
                continue
            poly = roofplane.poly.simplify(tolerance=param.simplify_roof_th)
            if type(poly) is geo.Polygon and poly.area > 0:
                roofplane.poly = poly
            else:
                roofplane.poly = None

        # 建物外形点の最も近い屋根面を探す
        nn_roofplane_ids = np.zeros(
            len(self._footprint.sample_points_xy), int)
        nn_roofplane_dists = np.zeros(
            len(self._footprint.sample_points_xy), float)
        for i, xy in enumerate(self._footprint.sample_points_xy):
            dists = np.full(len(self._roofplanes), 999.0)
            for j, roofplane in enumerate(self._roofplanes):
                if roofplane.poly is not None:
                    dists[j] = geo.Point(xy[0], xy[1]).distance(roofplane.poly)
            ind = np.argmin(dists)
            nn_roofplane_ids[i] = ind
            nn_roofplane_dists[i] = dists[ind]

        # 屋根面の膨張サイズを計算
        dilation_size = np.zeros(len(self._roofplanes))
        for i in range(len(self._roofplanes)):
            inds = nn_roofplane_ids == i
            if np.sum(inds):
                dilation_size[i] = np.max(nn_roofplane_dists[inds])

        # 屋根面を膨張
        for i, roofplane in enumerate(self._roofplanes):
            if roofplane.poly is not None:
                dilation = roofplane.poly.buffer(
                    dilation_size[i] + 0.05, cap_style=CAP_STYLE.flat,
                    join_style=JOIN_STYLE.mitre)
                roofplane.poly = dilation

        # 残存領域のマージ
        self._merge_remain_region(merge_area_th=0.0, is_merge_dilation=True)

        # ONE_NEIGHBOR_LINEとORTHOGONAL_CROSSINGの屋根面でその他の屋根面を整形
        for i, roofplane_i in enumerate(self._roofplanes):
            if (roofplane_i.type == RoofPlaneType.ONE_NEIGHBOR_LINE
                    or roofplane_i.type == RoofPlaneType.ORTHOGONAL_CROSSING
                    or roofplane_i.poly is None):
                continue

            poly_i = copy.deepcopy(roofplane_i.poly)
            for j, roofplane_j in enumerate(self._roofplanes):
                if ((roofplane_j.type != RoofPlaneType.ONE_NEIGHBOR_LINE
                    and roofplane_j.type != RoofPlaneType.ORTHOGONAL_CROSSING)
                        or roofplane_j.poly is None):
                    continue
                
                roof_type = roofplane_i.get_relation_with_other(roofplane_j)
                if roof_type is RoofPlaneRelationType.NEIGHBORING:
                    poly_i = poly_i.difference(roofplane_j.poly)
            roofplane_i.poly = poly_i

        # 隣接するONE_NEIGHBOR_LINEとORTHOGONAL_CROSSINGの屋根面を整形
        processed_pairs = {}
        for i, roofplane_i in enumerate(self._roofplanes):
            if ((roofplane_i.type != RoofPlaneType.ONE_NEIGHBOR_LINE
                    and roofplane_i.type != RoofPlaneType.ORTHOGONAL_CROSSING)
                    or roofplane_i.poly is None):
                continue

            poly_i = copy.deepcopy(roofplane_i.poly)
            for j, roofplane_j in enumerate(self._roofplanes):
                if i == j:
                    continue
                if ((roofplane_j.type != RoofPlaneType.ONE_NEIGHBOR_LINE
                        and roofplane_j.type
                        != RoofPlaneType.ORTHOGONAL_CROSSING)
                        or roofplane_j.poly is None):
                    continue

                # 屋根面jは屋根面iより整形されたか確認
                if j in processed_pairs and i in processed_pairs[j]:
                    continue

                roof_type = roofplane_i.get_relation_with_other(roofplane_j)
                if roof_type is RoofPlaneRelationType.NEIGHBORING:
                    difference = poly_i.difference(roofplane_j.poly)
                    if (type(difference) is geo.MultiPolygon
                            or type(difference) is geo.Polygon):

                        if type(difference) is geo.MultiPolygon:
                            # 最大面積の領域で上書き
                            areas = [tmp_poly.area
                                     for tmp_poly in difference.geoms]
                            index = np.argmax(areas)
                            difference = difference.geoms[index]

                        n_points_before = len(
                            np.array(poly_i.exterior.coords))
                        n_points_after = len(
                            np.array(difference.exterior.coords))
                        if n_points_before > n_points_after:
                            poly_i = difference
                            if i in processed_pairs:
                                processed_pairs[i].append(j)
                            else:
                                processed_pairs[i] = [j]
            roofplane_i.poly = poly_i

        # 屋根面の高さ順(降順)のインデックス列を作成
        heights = np.array(
            [roofplane.height for roofplane in self._roofplanes])
        indexes = np.argsort(heights)[::-1]

        # 建物外形で屋根面を整形
        target_polygon = copy.deepcopy(self._footprint.poly)
        for i in indexes:
            if GeoUtil.is_zero(target_polygon.area):
                # shape面積が無くなったため、割り当て不可
                self._roofplanes[i].poly = None
                continue

            if self._roofplanes[i].poly is None:
                continue

            # ノイズフィルタリング
            erosion = self._roofplanes[i].poly.buffer(
                param.noise_canceling_buffer1, cap_style=CAP_STYLE.flat,
                join_style=JOIN_STYLE.mitre)
            dilation = erosion.buffer(
                param.noise_canceling_buffer2, cap_style=CAP_STYLE.flat,
                join_style=JOIN_STYLE.mitre)
            erosion = dilation.buffer(
                param.noise_canceling_buffer3, cap_style=CAP_STYLE.flat,
                join_style=JOIN_STYLE.mitre)
            dilation = erosion.buffer(
                param.noise_canceling_buffer4, cap_style=CAP_STYLE.flat,
                join_style=JOIN_STYLE.mitre)

            intersection = target_polygon.intersection(dilation)
            # 最大ポリコンのみ残す
            geoms = GeoUtil.separate_geometry(intersection)
            max_area = 0.0
            for geom in geoms:
                if type(geom) is geo.Polygon and geom.area >= max_area:
                    max_area = geom.area
                    intersection = geom

            if GeoUtil.is_zero(max_area):
                # 他の屋根面がスペースを使用済みの場合
                self._roofplanes[i].poly = None
                continue
            
            self._roofplanes[i].poly = geo.Polygon(
                intersection.exterior.coords)
            target_polygon = target_polygon.difference(dilation)
            #assert (type(target_polygon) is geo.Polygon
            #        or type(target_polygon) is geo.MultiPolygon)

        # 残存領域のマージ
        self._merge_remain_region(
            merge_area_th=0.0001, is_merge_dilation=False)

    def _to_orthoimg_coordinate(
            self, src_points: NDArray, grid_size: float) -> NDArray:
        """地理座標系をオルソ画像の座標系に変換する

        Args:
            src_points (NDArray): 地理座標系の座標点
            grid_size (float): 解像度

        Returns:
            NDArray: オルソ画像座標系の座標点
            offset_x: X方向のオフセット
            offset_y: Y方向のオフセット
        """
        dst_points = copy.deepcopy(src_points)
        # 座標変換
        offset_x = np.floor(np.min(dst_points[:, 0]) / grid_size) * grid_size
        offset_y = np.ceil(np.max(dst_points[:, 1]) / grid_size) * grid_size
        dst_points[:, 0] = dst_points[:, 0] - offset_x
        dst_points[:, 0] = dst_points[:, 0] / grid_size
        dst_points[:, 1] = dst_points[:, 1] - offset_y
        dst_points[:, 1] = dst_points[:, 1] / (-grid_size)

        dst_points = np.round(dst_points).astype(np.int)
        return dst_points, offset_x, offset_y

    def _to_geo_coordinate(
            self, src_points: NDArray, grid_size: float,
            offset_x: float, offset_y: float) -> NDArray:
        """オルソ画像座標系を地理座標系に変換する

        Args:
            src_points (NDArray): オルソ画像座標系の座標点
            grid_size (float): 解像度
            offset_x: X方向のオフセット
            offset_y: Y方向のオフセット

        Returns:
            NDArray: 地理座標系の座標点
        """
        # 座標変換
        dst_points = src_points.astype(np.float32)

        dst_points[:, 0] = dst_points[:, 0] * grid_size
        dst_points[:, 0] = dst_points[:, 0] + offset_x
        dst_points[:, 1] = dst_points[:, 1] * (-grid_size)
        dst_points[:, 1] = dst_points[:, 1] + offset_y

        return dst_points

    def execute(self, src_clusters: list[ClusterInfo],
                shape: geo.Polygon,
                grid_size=0.25,
                sampling_step=0.25, neighbor_jobs=8,
                mean_shift_jobs=8, angle_ms_bandwidth=5.0,
                neightbor_max_dist=0.5, roof_angle_ortho_th=5.0,
                line_length_th=1.0, valid_pixel_num=25,
                width_th=5.0, slim_rate_th=10.0,
                max_hiers=100) -> list[ClusterInfo]:
        """mbr実行

        Args:
            src_clusters (list[ClusterInfo]): mbr対象クラスタ
            shape (geo.Polygon): 建物外形ポリゴン
            grid_size (float): 解像度
            sampling_step (float, optional): \
                建物外形ポリゴン辺のサンプリング間隔. Defaults to 0.25.
            neighbor_jobs (int, optional): \
                NearestNeightborのジョブ数. Defaults to 8.
            mean_shift_jobs (int, optional): \
                建物外形辺の角度のMeanShiftのジョブ数. Defaults to 8.
            angle_ms_bandwidth (float, optional): \
                建物外形辺の角度のMeanShiftのバンド幅. Defaults to 5.0.
            neightbor_max_dist (float, optional): \
                近傍建物外形線探索用の最大近傍距離閾値. Defaults to 0.5.
            roof_angle_ortho_th (float, optional): \
                近傍建物外形線の状態判定用の角度閾値deg. Defaults to 5.0.
            line_length_th (float, optional): \
                短い建物外形線分を除外する際の距離閾値. Defaults to 1.0.
            valid_pixel_num (int, optional): 有効矩形のpixel数. Defaults to 25.
            width_th (float, optional): 幅閾値. Defaults to 5.0.
            slim_rate_th (float, optional): 細長度閾値. Defaults to 10.0.
            max_hiers (int, optional): 最大階層数. Defaults to 100.

        Raises:
            ValueError: クラスタ情報が無い場合

        Returns:
            list[ClusterInfo]: MBR結果付きのクラスタ情報リスト
        """

        if len(src_clusters) == 0:
            # クラスタ情報がない
            class_name = self.__class__.__name__
            func_name = sys._getframe().f_code.co_name
            msg = '{}.{}, {}'.format(
                class_name, func_name,
                CreateModelMessage.ERR_MSG_MBR_NO_CLUSTER)
            raise CreateModelException(msg)

        # 屋根クラスタの近傍にある建物外形線の探索と屋根の角度算出
        self._footprint = FootPrint(shape)
        self._footprint.sampling(sampling_step=sampling_step)
        self._roofplanes = [RoofPlane() for _ in range(len(src_clusters))]
        for i, cluster in enumerate(src_clusters):
            self._roofplanes[i].points = cluster.points.get_points()[:, 0:2]
            self._roofplanes[i].height = cluster.roof_height
            self._roofplanes[i].set_footprint(
                footprint=self._footprint,
                ms_bandwidth=angle_ms_bandwidth,
                ms_jobs=mean_shift_jobs,
                neighbor_max_dist=neightbor_max_dist,
                neighbor_jobs=neighbor_jobs,
                angle_ortho_th=roof_angle_ortho_th,
                line_len_th=line_length_th)

        # 屋根ごとにMBR
        for i, roofplane in enumerate(self._roofplanes):
            if roofplane.type is RoofPlaneType.NO_NEIGHBOR_LINE:
                roofplane.poly = None
                continue

            # オルソ画像の座標系に変換
            xy_points, offset_x, offset_y = self._to_orthoimg_coordinate(
                src_points=roofplane.points, grid_size=grid_size)

            # MBR
            tmp_width_th = 0.0
            tmp_slim_rate_th = -1.0
            if (roofplane.type is RoofPlaneType.ONE_NEIGHBOR_LINE
                    or roofplane.type is RoofPlaneType.ORTHOGONAL_CROSSING):
                tmp_width_th = width_th
                tmp_slim_rate_th = slim_rate_th

            polygon_xy = self._mbr(
                points=xy_points, angle=roofplane.angle,
                rect_area_th=valid_pixel_num, width_th=tmp_width_th,
                slim_rate_th=tmp_slim_rate_th, max_hiers=max_hiers)

            if polygon_xy is None:
                # 屋根形状の作成に失敗
                roofplane.poly = None
                continue

            # 地理座標系に変換
            polygon_xy = self._to_geo_coordinate(
                src_points=polygon_xy,
                grid_size=grid_size,
                offset_x=offset_x,
                offset_y=offset_y)

            # 矩形領域のセット
            # shapeからはみ出ている可能性があるため後でクリップする
            roofplane.poly = geo.Polygon(polygon_xy)
        
        # 屋根整形
        self._rectify()

        # クラスタ作成
        heights = np.array(
            [roofplane.height for roofplane in self._roofplanes])
        indexes = np.argsort(heights)[::-1]  # 屋根面の高さ順(降順)のインデックス列

        new_clusters = list()
        for i in indexes:
            new_cluster = copy.deepcopy(src_clusters[i])
            if self._roofplanes[i].poly is not None:
                points = np.array(
                    self._roofplanes[i].poly.exterior.coords)
                if not self._roofplanes[i].poly.exterior.is_ccw:
                    # 反時計回りを表とするため反転する
                    points = np.flipud(points)
                # 始終点が同一座標のため終点を削除
                points = np.delete(points, len(points) - 1, axis=0)
                new_cluster.roof_line = points
            new_cluster.id = len(new_clusters)
            new_clusters.append(new_cluster)

        return new_clusters

    def _merge_remain_region(self, merge_area_th=0.0, is_merge_dilation=True):
        """建物外形から屋根領域を除いた際に発生する残存領域を屋根領域にマージする

        Args:
            merge_area_th (float, optional):\
                マージ対象とする領域の面積閾値. Defaults to 0.0.
            is_merge_dilation (bool, optional):\
                膨張領域をマージするか否か. Defaults to True.
        """
        # 建物外形の残り領域を作成
        target_polygon = copy.deepcopy(self._footprint.poly)
        for roofplane in self._roofplanes:
            if target_polygon.area > 0.0 and roofplane.poly is not None:
                target_polygon = target_polygon.difference(roofplane.poly)
        remain_geoms = GeoUtil.separate_geometry(target_polygon)

        # 残り領域を近傍屋根面にマージする
        for geom in remain_geoms:
            if not(type(geom) is geo.Polygon and geom.area > merge_area_th):
                continue

            merge_id = None
            max_intersection = 0.0
            dilation = geom.buffer(
                0.01, cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre)
            # RoofPlaneType.ONE_NEIGHBOR_LINE,
            # RoofPlaneType.ORTHOGONAL_CROSSING 以外の屋根にマージ
            for i, roofplane in enumerate(self._roofplanes):
                if (roofplane.type == RoofPlaneType.ONE_NEIGHBOR_LINE
                        or roofplane.type == RoofPlaneType.ORTHOGONAL_CROSSING
                        or roofplane.poly is None):
                    continue
                intersection = dilation.intersection(roofplane.poly)
                if intersection.area > max_intersection:
                    max_intersection = intersection.area
                    merge_id = i

            if merge_id is None:
                # RoofPlaneType.ONE_NEIGHBOR_LINE,
                # RoofPlaneType.ORTHOGONAL_CROSSING の屋根にマージ
                for i, roofplane in enumerate(self._roofplanes):
                    if ((roofplane.type != RoofPlaneType.ONE_NEIGHBOR_LINE
                        and roofplane.type
                         != RoofPlaneType.ORTHOGONAL_CROSSING)
                            or roofplane.poly is None):
                        continue

                    intersection = dilation.intersection(roofplane.poly)
                    if intersection.area > max_intersection:
                        max_intersection = intersection.area
                        merge_id = i

            #assert merge_id is not None

            if is_merge_dilation:
                self._roofplanes[merge_id].poly = \
                    self._roofplanes[merge_id].poly.union(dilation)

            else:
                # 屋根の補正処理の最後に行う残存領域マージ用
                tmp = self._roofplanes[merge_id].poly.union(geom)
                # assert type(tmp) is geo.Polygon
                if type(tmp) is not geo.Polygon:
                    continue

                # 動作検証時に発生した不正な屋根面(自己交差のように見えるスパイク形状)
                # を解消するための対応
                # 注目頂点の前後点を繋ぐベクトル同士の角度が0degのものは除外する
                # 始終点が同一頂点のためベクトル同士の角度が算出不能(0deg扱い)のため、
                # 削除対象からは除外する
                points = np.array(tmp.exterior.coords)
                angles = np.zeros(len(points), dtype=np.float32)
                for i in range(len(points)):
                    prev = i - 1
                    next = i + 1 if i < len(points) - 1 else 0
                    vec1 = points[prev] - points[i]
                    vec2 = points[next] - points[i]
                    angles[i] = GeoUtil.angle(vec1, vec2)
                inds = [i for i in range(1, len(points) - 1)
                        if GeoUtil.is_zero(angles[i])]
                points = np.delete(points, inds, axis=0)
                self._roofplanes[merge_id].poly = geo.Polygon(points)

            #assert type(self._roofplanes[merge_id].poly) is geo.Polygon
