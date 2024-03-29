import sys
import os
import glob
import multiprocessing
from typing import Optional
import numpy as np
from numpy.typing import NDArray
import shapely.geometry as geo
import laspy
from .message import CreateModelMessage
from .createmodelexception import CreateModelException
from ..util.log import Log, LogLevel, ModuleType


class PointCloud:
    """点群データ管理クラス
    """
    _cloud: NDArray[np.float_]
    _colors: NDArray[np.float_]
    _init: bool
    _init_color: bool
    _index: list[int]

    @property
    def index(self) -> list[int]:
        """点群のインデックス情報のゲッター

        Returns:
            list[int]: インデックス番号のリスト
        """
        return self._index

    @index.setter
    def index(self, value: list[int]):
        """インデックス情報のセッター

        Args:
            value (list[int]): 点群のインデックス情報(紐づけたいデータがある場合)
        """
        self._index = value

    def __init__(self) -> None:
        """コンストラクタ
        """
        self._cloud = np.empty([0, 0])
        self._colors = np.empty([0, 0])
        self._init = False
        self._init_color = False
        self._index = []

    def add_points(self, points: NDArray):
        """点群追加

        Args:
            points (NDArray): 点群配列
        """
        if self._init:
            self._cloud = np.append(self._cloud, points, axis=0)
        else:
            self._cloud = points
            self._init = True

    def get_points(self, offset=np.array([0.0, 0.0, 0.0])):
        """点群取得

        Args:
            offset (NDArray, optional): 座標点のオフセット値.
                                       Defaults to np.array([0.0, 0.0, 0.0]).

        Returns:
            NDArray: 点群配列
        """
        cloud = self._cloud
        offset_size = np.linalg.norm(offset, ord=2)
        if (len(cloud) > 0 and offset_size != 0):
            cloud = cloud + offset

        return cloud

    def add_colors(self, colors: NDArray):
        """色追加

        Args:
            colors (NDArray): 色配列
        """
        if self._init_color:
            self._colors = np.append(self._colors, colors, axis=0)
        else:
            self._colors = colors
            self._init_color = True

    def get_colors(self):
        """色取得

        Returns:
            NDArray: 色配列
        """
        return self._colors

    @property
    def min(self) -> Optional[NDArray[np.float_]]:
        """各座標の最小値

        Returns:
            NDArray:
                各座標の最小値配列
                点が格納されていない場合はNoneを返却

        """
        if len(self._cloud) < 1:
            return None

        return np.min(self._cloud, axis=0)

    @property
    def max(self) -> Optional[NDArray[np.float_]]:
        """各座標の最大値

        Returns:
            NDArray:
                各座標の最大値配列
                点が格納されていない場合はNoneを返却
        """
        if len(self._cloud) < 1:
            return None

        return np.max(self._cloud, axis=0)


class LasFileInfo:
    """LASファイル情報クラス
    """
    @property
    def path(self) -> str:
        """ファイルパス

        Returns:
            str: ファイルパス
        """
        return self._path

    @path.setter
    def path(self, value: str):
        """ファイルパス

        Args:
            value (str): ファイルパス
        """
        self._path = value

    @property
    def min_x(self) -> float:
        """最小x座標

        Returns:
            float: 最小x座標
        """
        return self._min_x

    @min_x.setter
    def min_x(self, value: float):
        """最小x座標

        Args:
            value (float): 最小x座標
        """
        self._min_x = value

    @property
    def min_y(self) -> float:
        """最小y座標

        Returns:
            float: 最小y座標
        """
        return self._min_y

    @min_y.setter
    def min_y(self, value: float):
        """最小y座標

        Args:
            value (float): 最小y座標
        """
        self._min_y = value

    @property
    def max_x(self) -> float:
        """最大x座標

        Returns:
            float: 最大x座標
        """
        return self._max_x

    @max_x.setter
    def max_x(self, value: float):
        """最大x座標

        Args:
            value (float): 最大x座標
        """
        self._max_x = value

    @property
    def max_y(self) -> float:
        """最大y座標

        Returns:
            float: 最大y座標
        """
        return self._max_y

    @max_y.setter
    def max_y(self, value: float):
        """最大y座標

        Args:
            value (float): 最大y座標
        """
        self._max_y = value

    def __init__(self, path: str, min_x: float, min_y: float,
                 max_x: float, max_y: float):
        """コンストラクタ

        Args:
            path (str): ファイルパス
            min_x (float): 最小座標x
            min_y (float): 最小座標y
            max_x (float): 最大座標x
            max_y (float): 最大座標y
        """
        self.path = path
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y

    def get_area_polygon(self):
        """データ範囲のバウンディングボックス

        Returns:
            shapely.geometry.box: LASデータ範囲のバウンディングボックス
        """
        box = geo.box(self.min_x, self.min_y, self.max_x, self.max_y)
        return box


class LasManager:
    """LASデータマネージャー
    """

    def __init__(self, swap_xy=False) -> None:
        """コンストラクタ

        Args:
            swap_xy (bool, optional):\
                True:xy座標を入れ替えて保持する,\
                False:入力値のまま保持する. Defaults to False.
        """
        self._min_pos = np.array([0, 0])
        self._max_pos = np.array([0, 0])
        self._target_files = []
        self._building_polygon = geo.Polygon()
        self._ground_polygon = geo.Polygon()
        self._is_search_ground = False
        self._swap_xy = swap_xy

        # RGB情報を保持しているレコードフォーマットの番号
        self._COLOR_RECORD_FORMATS = [2, 3, 5, 7, 8, 10]

    def get_area_size(self):
        """点群範囲

        Returns:
            float, float: 幅[m], 高さ[m]
        """
        width = self._max_pos[0] - self._min_pos[0]
        height = self._max_pos[1] - self._min_pos[1]
        return width, height

    def read_header(self, folder_path, polygon):
        """ヘッダ情報の読込

        Args:
            folder_path (string): LASフォルダパス
            polygon (shapely.geometry.Polygon): 読込対象範囲(平面直角座標系)

        Raises:
            CreateModelException: LASフォルダが存在しない
            CreateModelException: LASファイルが存在しない
            CreateModelException: 読込対象範囲の点群データがない
        """
        class_name = self.__class__.__name__
        func_name = sys._getframe().f_code.co_name

        self._target_files = []     # 初期化
        if not os.path.isdir(folder_path):
            # フォルダが存在しない場合
            msg = '{}.{}, {}'.format(
                class_name, func_name,
                CreateModelMessage.ERR_MSG_LAS_MNG_LAS_FOLDER_NOT_FOUND)
            raise CreateModelException(msg)

        files = glob.glob(os.path.join(folder_path, '*.las'))
        if len(files) == 0:
            # lasファイルが存在しない場合
            msg = '{}.{}, {}'.format(
                class_name, func_name,
                CreateModelMessage.ERR_MSG_LAS_MNG_LAS_NOT_FOUND)
            raise CreateModelException(msg)

        min_pos = np.array([sys.float_info.max, sys.float_info.max])
        max_pos = np.array([-sys.float_info.max, -sys.float_info.max])
        for file in files:
            try:
                # headerの読み込み(pointにアクセスしないためopenを使用)
                with laspy.open(file) as las:
                    if self._swap_xy:
                        # xy座標を入れ替える
                        x_min = las.header.y_min
                        x_max = las.header.y_max
                        y_min = las.header.x_min
                        y_max = las.header.x_max
                    else:
                        # 入力値をそのまま使用する
                        x_min = las.header.x_min
                        x_max = las.header.x_max
                        y_min = las.header.y_min
                        y_max = las.header.y_max

                    # polygonとの重畳確認
                    las_polygon = geo.Polygon(
                        [(x_min, y_min), (x_min, y_max),
                         (x_max, y_max), (x_max, y_min), (x_min, y_min)])

                    # 重畳しない場合はskip
                    if (las_polygon.disjoint(polygon)):
                        continue

                    file_info = LasFileInfo(file, x_min, y_min, x_max, y_max)

                    self._target_files.append(file_info)

                    if x_min < min_pos[1]:
                        min_pos[1] = x_min
                    if y_min < min_pos[0]:
                        min_pos[0] = y_min
                    if x_max > max_pos[1]:
                        max_pos[1] = x_max
                    if y_max > max_pos[0]:
                        max_pos[0] = y_max
            except Exception:
                # ヘッダ情報取得時のエラー
                msg = '{}.{}, {} ({})'.format(
                    class_name, func_name,
                    CreateModelMessage.ERR_MSG_FAILED_TO_READ_LAS_FILE,
                    os.path.basename(file))
                Log.output_log_write(
                    LogLevel.WARN, ModuleType.MODEL_ELEMENT_GENERATION,
                    msg)

        if (len(self._target_files) == 0):
            # 点群データ取得対象のデータがない場合
            msg = '{}.{}, {}'.format(
                class_name, func_name,
                CreateModelMessage.ERR_MSG_LAS_MNG_NO_LAS_FILE)
            raise CreateModelException(msg)

        self._min_pos = min_pos
        self._max_pos = max_pos

    def get_points(self, bilding_polygon, ground_polygon=None):
        """点群データの取得

        Args:
            bilding_polygon (shapely.geometry.Polygon): 取得対象範囲
            ground_polygon (shapely.geometry.Polygon): 地面探索範囲

        Returns:
            PointCloud: 点群データ
            float:      最低地面の高さm(地面探索を行う場合、行わない場合はNone)
            float:      前処理用の地面の高さm(地面探索を行う場合、行わない場合はNone)
        """

        # 並列処理の準備
        cpu_num = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(cpu_num)

        class_name = self.__class__.__name__
        func_name = sys._getframe().f_code.co_name

        self._building_polygon = geo.polygon.orient(bilding_polygon)
        if ground_polygon is not None:
            # 地面探索を行う場合
            self._ground_polygon = geo.polygon.orient(ground_polygon)
            self._is_search_ground = True
        else:
            # 地面探索を行わない場合
            self._ground_polygon = geo.Polygon()
            self._is_search_ground = False

        # 点群の準備
        cloud = PointCloud()
        cloud_ground = PointCloud()

        min_height = None
        for file in self._target_files:
            box = file.get_area_polygon()
            if (not self._building_polygon.disjoint(box)
                or (self._is_search_ground
                    and not self._ground_polygon.disjoint(box))):
                # 建物外形/地面探索範囲とLASデータ範囲が接している場合

                las = laspy.read(file.path)
                # 座標値の取得
                if self._swap_xy:
                    # xy座標を入れ替える
                    points = np.stack(
                        [las.y, las.x, las.z], axis=0).transpose((1, 0))
                else:
                    # 入力値をそのまま使用する
                    points = np.stack(
                        [las.x, las.y, las.z], axis=0).transpose((1, 0))

                # 色情報の取得
                pf: laspy.PointFormat = las.header.point_format
                if pf.id in self._COLOR_RECORD_FORMATS:
                    colors = np.stack(
                        [las.red, las.green, las.blue],
                        axis=0).transpose((1, 0))

                    # 8bitデータ対応
                    if np.max(colors) < 256: # 8bit画像？
                        colors *= 256
                else:
                    msg = '{}.{}, {}'.format(
                        class_name, func_name,
                        CreateModelMessage.ERR_MSG_LAS_MNG_UNSUPPORTED_LAS_FORMAT)
                    raise CreateModelException(msg)

                # polygonの最小外接長方形でpointをfilterする
                polygon_mbr: tuple[float, float, float, float] = (
                    self._ground_polygon
                    if self._is_search_ground else self._building_polygon
                ).bounds  # type: ignore
                in_mbr = ((polygon_mbr[0] <= points[:, 0])
                          & (points[:, 0] <= polygon_mbr[2])
                          & (polygon_mbr[1] <= points[:, 1])
                          & (points[:, 1] <= polygon_mbr[3]))

                # 並列化処理
                try:
                    # 屋根点取得 + 地面探索
                    ret = pool.map(
                        self._check_point_in_polygon, points[in_mbr])
                    conv_ret = np.zeros((len(points), 2), dtype=np.int_)
                    # list -> NDArray
                    conv_ret[in_mbr] = np.array(ret, dtype=np.int_)
                    # polygon内の点のみ取得
                    ex_points = points[conv_ret[:, 0] == 1]
                    cloud.add_points(ex_points)

                    if pf.id in self._COLOR_RECORD_FORMATS:
                        # polygon内の点のみ取得
                        ex_colors = colors[conv_ret[:, 0] == 1]
                        cloud.add_colors(colors=ex_colors)
                    else:
                        msg = '{}.{}, {}'.format(
                            class_name, func_name,
                            CreateModelMessage.ERR_MSG_LAS_MNG_UNSUPPORTED_LAS_FORMAT)
                        raise CreateModelException(msg)

                    if self._is_search_ground:
                        # 地面探索範囲内の点のみ取得
                        target_points = points[
                            np.logical_and(conv_ret[:, 0] == 0,
                                           conv_ret[:, 1] == 1)]
                        cloud_ground.add_points(target_points)

                        if len(target_points) > 0:
                            # z座標の最小値
                            height = target_points[:, 2].min()

                            if min_height is None:
                                min_height = height
                            else:
                                if min_height > height:
                                    min_height = height  # 最小値の更新

                except Exception as e:
                    # 点群取得時のエラー
                    class_name = self.__class__.__name__
                    func_name = sys._getframe().f_code.co_name
                    msg = '{}.{}, {}'.format(class_name, func_name, e)
                    Log.output_log_write(
                        LogLevel.WARN, ModuleType.MODEL_ELEMENT_GENERATION,
                        msg)

        if len(cloud.get_points()) == 0:
            # 建物点群がない場合
            msg = '{}.{}, {}'.format(
                class_name, func_name,
                CreateModelMessage.ERR_MSG_LAS_MNG_NO_POINTS)
            raise CreateModelException(msg)

        ground_height = None
        if self._is_search_ground:
            points_xyz = cloud_ground.get_points()
            if len(points_xyz) == 0:
                # 地面点群がない場合
                msg = '{}.{}, {}'.format(
                    class_name, func_name,
                    CreateModelMessage.ERR_MSG_LAS_MNG_NO_GROUOND_POINTS)
                raise CreateModelException(msg)

            zs = points_xyz[:, 2]
            z_min = np.min(zs)
            z_max = np.max(zs)
            inds = np.logical_and(zs >= z_min, zs <= (z_min + z_max) / 2)
            zs = zs[inds]
            n_bins = int((np.max(zs) - np.min(zs) + 0.5) // 1.0)
            n_bins = max(n_bins, 2)
            hist, bins = np.histogram(zs, bins=n_bins)
            ind = np.argmax(hist)
            ground_height = (bins[ind] + bins[ind + 1]) / 2

        return cloud, min_height, ground_height

    def _check_point_in_polygon(self, pos: NDArray):
        """座標点のポリゴン内外判定

        Args:
            pos (NDArray): 座標点(x,y,z)を想定

        Returns:
            int: 1の場合はポリゴン内, 0の場合はポリゴン外
        """
        # 建物外形内か確認する
        pt = geo.Point(pos[0], pos[1])
        is_building = 0
        if pt.within(self._building_polygon):
            is_building = 1

        # 地面探索実施時は、地面探索範囲内か確認する
        is_ground = 0
        if self._is_search_ground and pt.within(self._ground_polygon):
            is_ground = 1

        return [is_building, is_ground]
