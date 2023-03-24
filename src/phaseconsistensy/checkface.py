import sys
import math
from logging import getLogger
import numpy as np
from numpy.typing import NDArray
from enum import IntEnum
from ..util.objinfo import ObjInfo, BldElementType
from ..util.faceinfo import FaceInfo, IndexInfo
from .geoutil import GeoUtil
from .linemanager import TriangleInfo, LineManager
from ..util.parammanager import ParamManager

logger = getLogger(__name__)


class TestResultType(IntEnum):
    """検査・補正結果
    """
    NO_ERROR = 0                    # エラーなし
    ERROR = 1                       # エラー有り
    AUTO_CORRECTED = 2              # エラー有り、自動補正に成功
    AUTO_CORRECTION_FAILURE = 3     # エラー有り、自動補正に失敗


class CheckFace:
    """面情報検査処理クラス
    """

    def __init__(self, obj_info: ObjInfo, face: FaceInfo, param: ParamManager):
        """コンストラクタ

        Args:
            obj_info(ObjInfo): 建物情報
            face (FaceInfo): 面情報
            param (PramManager): パラメータ情報
        """
        self._obj_info = obj_info       # 建物情報
        self._face = face               # 面情報(インデックス値リスト)の参照
        self._v_list = obj_info.v_list_manger          # 座標リストの参照
        self._err_list = []             # エラー座標リスト
        self._param = param             # パラメータ情報
    
    @property
    def err_list(self) -> list:
        return self._err_list

    def check_double_point(self) -> TestResultType:
        """連続頂点重複検査/補正処理

        Returns:
            TestResultType: 検査・補正結果
        """

        ret = TestResultType.NO_ERROR   # エラー無し

        logger.debug(f'len(self._face.indx) = {len(self._face.indx)}')
        logger.debug(f'self._v_list.get_point_num() = \
                        {self._v_list.get_point_num()}')

        delete_list = list()
        for i in range(len(self._face.indx)):                
            cur_pos = self._v_list.get_pos(self._face.indx[i].pos)
            pre_pos = self._v_list.get_pos(self._face.indx[i-1].pos)

            if GeoUtil.is_zero(pre_pos.x - cur_pos.x) \
                and GeoUtil.is_zero(pre_pos.y - cur_pos.y) \
                and GeoUtil.is_zero(pre_pos.z - cur_pos.z):
                delete_list.append(i)

                # エラーリストに追加
                self._err_list.append(cur_pos)

        if len(delete_list):
            if len(self._face.indx) <= 3:
                ret = TestResultType.AUTO_CORRECTION_FAILURE
            else:
                delete_list.reverse()
                for i in delete_list:
                    del self._face.indx[i]

                # エラー有り、自動補正済み
                ret = TestResultType.AUTO_CORRECTED
       
        return ret

    def check_intersection(self) -> bool:
        """自己交差・自己接触検査処理

        Returns:
            bool: True: エラーなし, False: エラーあり
        """

        # Point リスト作成
        p_list = []
        for index in self._face.indx:
            p_list.append(self._v_list.get_pos(index.pos))

        # 平面に投影
        xy_list = GeoUtil.conv_2d(p_list)

        # logger.debug(f'xy_list = {xy_list}')
        # 自己交差チェック
        err_index = []
        if GeoUtil.is_cross_2d(xy_list, err_index):
            # print('  intersect line')
            # エラーリストに追加
            for index in err_index:
                pos = self._face.indx[index].pos
                self._err_list.append(self._v_list.get_pos(pos))
        
        # 自己接触チェック
        err_index = []
        if GeoUtil.is_touch_2d(xy_list, err_index):
            # logger.debug('  toutch point')
            # エラーリストに追加
            for index in err_index:
                pos = self._face.indx[index].pos
                self._err_list.append(self._v_list.get_pos(pos))

        if len(self._err_list) > 0:
            # エラー情報設定
            return False
        
        return True

    def check_non_plane(self) -> TestResultType:
        """非平面検査処理

        Returns:
            TestResultType: 検査・補正結果
        """
        # Point リスト作成
        p_list = []
        for index in self._face.indx:
            p_list.append(self._v_list.get_pos(index.pos))

        # 面の法線算出
        normal = GeoUtil.get_normal(p_list)

        r_flag = TestResultType.NO_ERROR    # エラー無し
        # 厚み検査
        if not self._check_non_plane_thickness(p_list, normal):
            r_flag = TestResultType.ERROR

        # 法線角度検査
        if not self._check_non_plane_normal(p_list, normal):
            r_flag = TestResultType.ERROR

        if r_flag is TestResultType.ERROR:
            r_flag = TestResultType.AUTO_CORRECTED

            try:
                element_type = CheckFaces._get_element_type(p_list)
                # 三角形分割
                multi_point_list = GeoUtil.divide_triangle(p_list, check_intersect=True)
                self._face.indx.clear()
                for i in range(len(multi_point_list)):
                    pos_list = []
                    dp_flag = False
                    for j in range(3):
                        # 三角形に連続頂点がないか確認,ある場合は出力しない
                        #if (multi_point_list[i][j]
                        #        == multi_point_list[i][(j + 1) % 3]):
                        if abs(multi_point_list[i][j].x - multi_point_list[i][j-1].x) < 1e-06 \
                            and abs(multi_point_list[i][j].y - multi_point_list[i][j-1].y) < 1e-06 \
                            and abs(multi_point_list[i][j].z - multi_point_list[i][j-1].z) < 1e-06:
                            dp_flag = True
                        pos_list.append(multi_point_list[i][j])
                    if not dp_flag:
                        if self._face.get_str() == "":
                            for pos in pos_list:
                                # 座標リストに追加
                                index_no = self._v_list.append_pos(pos)
                                # インデックス値更新
                                self._face.append(IndexInfo(index_no))
                        else:
                            self._obj_info.append_point_list(
                                element_type, pos_list)

                self._err_list.extend(p_list)
            except Exception:
                # 予期せぬエラーで補正失敗
                r_flag = TestResultType.AUTO_CORRECTION_FAILURE

        return r_flag

    def _check_non_plane_thickness(self, p_list: list,
                                   normal: NDArray) -> bool:
        """非平面厚み検査

        Args:
            p_list (list): ポリゴン座標列
            normal (NDArray): 法線

        Returns:
            bool: True: エラーなし, False: エラーあり
        """

        # 基準点
        base_pos = np.array([p_list[0].x, p_list[0].y, p_list[0].z])

        height_min = sys.float_info.max
        height_max = -sys.float_info.max
        for pos in p_list[1:]:
            vec = np.array([pos.x, pos.y, pos.z]) - base_pos
            # 法線方向の基準点との高さ算出
            height = np.dot(vec, normal)
            height_min = GeoUtil.min(height, height_min)
            height_max = GeoUtil.max(height, height_max)
        
        # 厚み算出
        thickness = abs(height_max - height_min)
        
        if thickness > self._param.non_plane_thickness:
            logger.debug(f'thickness = {thickness}')
            return False
        
        return True

    def _check_non_plane_normal(self, p_list: list,
                                normal: NDArray) -> bool:
        """非平面法線角度検査

        Args:
            p_list (list): ポリゴン座標列
            normal (NDArray): 法線

        Returns:
            bool: True: エラーなし, False: エラーあり
        """

        cross_list = []
        point_num = len(p_list)
        for i in range(point_num):
            v0 = np.array([p_list[i].x, p_list[i].y, p_list[i].z])

            for j in range(1, point_num - 1):
                index1 = (i + j) % point_num
                index2 = (i + j + 1) % point_num
                v1 = np.array([p_list[index1].x,
                               p_list[index1].y,
                               p_list[index1].z]) - v0
                v2 = np.array([p_list[index2].x,
                               p_list[index2].y,
                               p_list[index2].z]) - v0
                v1 = GeoUtil.normalize(v1)
                v2 = GeoUtil.normalize(v2)

                cross = np.cross(v1, v2)
                cross = GeoUtil.normalize(cross)

                dot = GeoUtil.dot(v1, v2)
                degree = abs(math.degrees(math.acos(dot)))
                if degree >= 170.0 or degree <= 10.0:
                    continue

                if np.dot(cross, normal) < 0.0:
                    cross *= -1.0
                cross_list.append(cross)
        
        cross_len = len(cross_list)
        angle_max = 0.0
        for i in range(cross_len):
            for j in range(i + 1, cross_len):
                dot = GeoUtil.dot(cross_list[i], cross_list[j])
                angle = abs(math.degrees(math.acos(dot)))
                if angle > angle_max:
                    angle_max = angle

        if angle_max > self._param.non_plane_angle:
            logger.debug(f'angle = {angle}')
            return False
        
        return True

    def check_zero_area(self) -> bool:
        """面積 0 ポリゴン検査

        Returns:
            bool: True: エラーなし, False: エラーあり
        """

        # Point リスト作成
        p_list = []
        for index in self._face.indx:
            p_list.append(self._v_list.get_pos(index.pos))
        
        if GeoUtil.is_zero_area(p_list):
            self._err_list.extend(p_list)
            return False
        
        return True


class CheckFaces:
    """面情報群検査処理クラス
    """

    _element_angle_margin = 2.0

    def __init__(self, obj_info: ObjInfo, param: ParamManager):
        """コンストラクタ

        Args:
            obj_info (ObjInfo): 建物情報
        """
        self._obj_info = obj_info        # 建物情報
        self._err_list = []              # エラー座標リスト
        self._param = param              # パラメータ情報
   
    @property
    def err_list(self) -> list:
        return self._err_list
    
    def check_face_intersection(self) -> bool:
        """面同士交差処理

        Returns:
            bool: True: エラーなし, False: エラーあり
        """
        # 三角形分割
        multi_triangle_list = []  # 分割前のポリゴン x 三角形分割のリスト格納用
        triangle_list = []    # 分割三角形のリスト格納用
        for f_key, f_value in self._obj_info.faces_list.items():
            multi_polygon_list = self._obj_info.get_polygon_list(f_key)
            logger.debug(f'multi_polygon_list len = {len(multi_polygon_list)}')
            for point_list in multi_polygon_list:
                triangle_list = GeoUtil.divide_triangle(point_list)
                triangle_info_list = []
                for triangle in triangle_list:
                    triangle_info = TriangleInfo(triangle)
                    triangle_info_list.append(triangle_info)
                multi_triangle_list.append(triangle_info_list)

        ret_flag = True
        # 交差チェック
        polygon_num = len(multi_triangle_list)
        logger.debug(f'polygon_num = {polygon_num}')
        for i in range(polygon_num):
            triangle_list1 = multi_triangle_list[i]
            logger.debug(f'i = {i}')
            for j in range(i + 1, polygon_num):
                # logger.debug(f'j = {j}')
                triangle_list2 = multi_triangle_list[j]

                for triangle1 in triangle_list1:
                    for triangle2 in triangle_list2:
                        cross_point_list = []
                        if GeoUtil.is_cross_triangle(triangle1, triangle2,
                                                     cross_point_list):
                            # エラーあり
                            for pos in cross_point_list:
                                self._err_list.append(pos)
                                ret_flag = False
        
        return ret_flag

    def check_solid(self) -> TestResultType:
        """ソリッド閉合検査/補正処理

        Returns:
            TestResultType: 検査・補正結果
        """

        # 三角形分割/線分リスト作成
        line_manager = LineManager()
        for f_key in self._obj_info.faces_list.keys():
            multi_polygon_list = self._obj_info.get_polygon_list(f_key)
            for point_list in multi_polygon_list:
                triangle_list = GeoUtil.divide_triangle(point_list, check_intersect=False)
                for triangle in triangle_list:
                    line_manager.append_polygon(triangle)

        # 同一線分除外
        line_manager.delete_pair_line()

        # tmp_list = line_manager.get_list()
        # for line in tmp_list:
        #     logger.debug(f'delete pair line = {line.str()}')

        # 線分マージ
        line_manager.marge_lines()

        # tmp_list = line_manager.get_list()
        # for line in tmp_list:
        #     logger.debug(f'marge line       = {line.str()}')

        # 同一線分除外
        line_manager.delete_pair_line()

        # tmp_list = line_manager.get_list()
        # for line in tmp_list:
        #     logger.debug(f'delete pair line2= {line.str()}')

        # 同一線分の重複部分を削除/分割
        line_list = line_manager.delete_overlap_lines()

        # for line in line_list:
        #     logger.debug(f'delete overlap   = {line.str()}')

        # 残った線分を開口部と判定
        ret = TestResultType.NO_ERROR   # エラー無し
        line_num = len(line_list)
        if line_num != 0:
            for i in range(line_num):
                logger.debug(f'i = {i}, line = {line_list[i].str()}')

            try:
                # 補正処理
                for i in range(line_num):
                    if line_list[i] is None:
                        continue
                    polygon_list = []
                    line_info = line_list[i]
                    line_list[i].get_polygon_set(line_list, polygon_list)
                    polygon_line_len = len(polygon_list)
                    logger.debug(f'polygon_line_len = {polygon_line_len}')
                    logger.debug(f'polygon_list = {polygon_list}')

                    # for line in polygon_list:
                    #     logger.debug(f'poly line = {line.str()}')

                    if polygon_line_len != 0:

                        # 始終点が一致しているかどうか
                        if GeoUtil.is_same_point(
                                polygon_list[0].pos1,
                                polygon_list[polygon_line_len - 1].pos2) is False:
                            continue

                        # 開口部分のポリゴン作成
                        point_list = []
                        for line_info in reversed(polygon_list):
                            point_list.append(line_info.pos1)

                        # エラー座標値追加
                        for pos in point_list:
                            self._err_list.append(pos)

                        # 部材算出
                        element_type = self._get_element_type(point_list)
                        if element_type != BldElementType.NONE:
                            # ポリゴンを ObjInfo に追加
                            self._obj_info.append_point_list(
                                element_type, point_list)
                        
                        line_list[i] = None

                ret = TestResultType.AUTO_CORRECTED  # 自動補正
            except Exception:
                # 予期せぬ例外が発生して、自動補正が失敗
                ret = TestResultType.AUTO_CORRECTION_FAILURE
        
        return ret

    @classmethod
    def _get_element_type(self, point_list: list) -> BldElementType:
        """面データの部材を算出

           面の法線が下向きなら地面、地面と平行なら壁、その他を屋根と判定する

        Args:
            point_list (list<Point>): 面の座標列

        Returns:
            BldElementType: 部材タイプ
        """
        normal_vec = GeoUtil.get_normal(point_list)
        under_direcion_vec = np.array([0.0, 0.0, -1.0])

        normal_length = np.linalg.norm(normal_vec, ord=2)

        if GeoUtil.is_same_value(normal_length, 0.0):
            return BldElementType.NONE
        else:
            dot = GeoUtil.dot(normal_vec, under_direcion_vec)
            angle = abs(math.degrees(math.acos(dot)))

            if angle < self._element_angle_margin:
                # 地面
                return BldElementType.GROUND
            elif 90 - angle < self._element_angle_margin:
                # 壁
                return BldElementType.WALL
            else:
                # 屋根
                return BldElementType.ROOF
