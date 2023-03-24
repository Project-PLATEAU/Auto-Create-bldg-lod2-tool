# -*- coding:utf-8 -*-
import sys
import numpy as np
from numpy.typing import NDArray
import shapely.geometry as geo
from scipy.spatial.transform import Rotation
from ..createmodelexception import CreateModelException


class GeoUtil:
    @classmethod
    def is_same_value(self, value1: float, value2: float) -> bool:
        """浮動小数点の同一比較判定

        Args:
            value1 (float): 値1
            value2 (float): 値2

        Returns:
            bool: True: 同一と判定, False: 同一でないと判定
        """
        if abs(value1 - value2) < sys.float_info.epsilon:
            return True
        return False

    @classmethod
    def is_zero(self, value: float) -> bool:
        """浮動小数点が0であるかチェック

        Args:
            value (float): 値
 
        Returns:
            bool: True: 0である, False: 0でない
        """
        return GeoUtil.is_same_value(value, 0.0)

    @staticmethod
    def size(vec: NDArray) -> float:
        """ベクトルの大きさ

        Args:
            vec (NDArray): ベクトル

        Returns:
            float: ユークリッド距離
        """
        return np.linalg.norm(vec, ord=2)

    @staticmethod
    def normalize(vec: NDArray) -> NDArray:
        """ベクトルの正規化

        Args:
            vec (NDArray): 2/3次元ベクトル

        Returns:
            NDArray: 正規化済みの2/3次元ベクトル
        """
        is_3d = True if len(vec) == 3 else False
        
        d = np.linalg.norm(vec, ord=2)
        if GeoUtil.is_zero(d):
            if is_3d:
                return np.array([0.0, 0.0, 0.0])
            else:
                return np.array([0.0, 0.0])

        else:
            mag = 1.0 / d
            a = vec[0] * mag
            b = vec[1] * mag

            if is_3d:
                c = vec[2] * mag
                return np.array([a, b, c])
            else:
                return np.array([a, b])

    @staticmethod
    def angle(a: NDArray, b: NDArray) -> float:
        """ベクトル同士の角度の算出

        Args:
            a (NDArray): 3次元ベクトル
            b (NDArray): 3次元ベクトル

        Returns:
            float: 角度(deg)
        """

        # ab = |a||b|cos(theta)
        norm_a = GeoUtil.normalize(a)
        norm_b = GeoUtil.normalize(b)
        size_a = np.linalg.norm(norm_a, ord=2)
        size_b = np.linalg.norm(norm_b, ord=2)
        if GeoUtil.is_zero(size_a) or GeoUtil.is_zero(size_b):
            return 0.0
        
        cos = np.dot(norm_a, norm_b)
        if cos > 1.0:
            cos = 1.0
        if cos < -1.0:
            cos = -1.0

        rad = np.arccos(cos)
        deg = np.rad2deg(rad)

        return deg

    @staticmethod
    def signed_angle(a: NDArray, b: NDArray) -> float:
        """ベクトル同士の符号つき角度の算出

        Args:
            a (NDArray): 2次元ベクトル
            b (NDArray): 2次元ベクトル
        Returns:
            float: 角度(deg)
        """
        tmp_a = np.array([a[0], a[1], 0.0])
        tmp_b = np.array([b[0], b[1], 0.0])
        deg = GeoUtil.angle(tmp_a, tmp_b)

        norm_a = GeoUtil.normalize(tmp_a)
        norm_b = GeoUtil.normalize(tmp_b)
        norm_base = np.array([0.0, 0.0, 1.0])

        cross_vec = np.cross(norm_a, norm_b)
        dot_vec = np.dot(norm_base, cross_vec)
        if dot_vec < 0:
            deg *= -1.0

        return deg

    @staticmethod
    def point_to_line_dist(pos: NDArray, line_pos: NDArray,
                           line_vec: NDArray):
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
            np.array: 点から下した垂線と直線との交点
            float: 交点までの直線のベクトル係数
        """
        is3d = (len(pos) == 3 and len(line_pos) == 3 and len(line_vec) == 3)
        is2d = (len(pos) == 2 and len(line_pos) == 2 and len(line_vec) == 2)
        if not (is3d or is2d):
            class_name = GeoUtil.__name__
            func_name = sys._getframe().f_code.co_name
            msg = '{}.{}, {}'.format(
                class_name, func_name,
                'Please unify the coordinates in 3-D or 2-D coordinates.')
            raise CreateModelException(msg)

        lensqv = GeoUtil.size(line_vec)
        lensqv *= lensqv
        t = 0.0

        if lensqv > 0.0:
            t = np.dot(line_vec, pos - line_pos) / lensqv
        
        if GeoUtil.is_zero(t):
            t = 0.0
        
        h = line_vec * t + line_pos
        dist = GeoUtil.size(h - pos)

        return dist, h, t

    @staticmethod
    def cross_point(vec1: NDArray, pos1: NDArray,
                    vec2: NDArray, pos2: NDArray):
        """2直線の交点

        Args:
            vec1 (NDArray): 直線1のベクトル
            pos1 (NDArray): 直線1の始点
            vec2 (NDArray): 直線2のベクトル
            pos2 (NDArray): 直線2の始点

        Returns:
            bool: 処理結果 True : 成功, False : 失敗
            np.array: 交点
            bool: 交点が直線1上か否か
            bool: 交点が直線2上か否か
        """
        h = np.array([0.0, 0.0])
        online1 = False
        online2 = False

        is2d = (len(pos1) == 2 and len(vec1) == 2
                and len(pos2) == 2 and len(vec2) == 2)
        if not is2d:
            # 入力データが2次元座標ではない
            return False, h, online1, online2

        if GeoUtil.is_zero(GeoUtil.size(vec1)) or GeoUtil.is_zero(GeoUtil.size(vec2)):
            # 入力ベクトルがゼロベクトル
            return False, h, online1, online2

        # vec1に垂直なベクトル
        k = np.sqrt(1.0 / (vec1[0] ** 2 + vec1[1] ** 2))
        vertical_vec = np.array([vec1[1] * k, vec1[0] * -k])
        vertical_vec = GeoUtil.normalize(vertical_vec)
        
        # 交点探索
        # l(s) = p + su (vec2)
        # m(t) = q + tv (vec1)
        # mに垂直な単位ベクトルn
        # s = n(q - p) / nu
        # 内積nv = 0
        tmp_val = np.dot(vertical_vec, vec2)
        if GeoUtil.is_zero(tmp_val):
            return False, h, online1, online2

        s = np.dot(vertical_vec, pos1 - pos2) / tmp_val
        # 交点
        h = vec2 * s + pos2

        tmp_vec = h - pos1
        t = GeoUtil.size(tmp_vec) / GeoUtil.size(vec1)
        angle = GeoUtil.angle(tmp_vec, vec1)
        if angle > 90.0:
            t *= -1

        if 0 <= t and t <= 1:
            online1 = True
        if 0 <= s and s <= 1:
            online2 = True

        return True, h, online1, online2

    @staticmethod
    def rotate_object(vec: NDArray, angle: float) -> Rotation:
        """3次元回転オブジェクトの作成

        Args:
            vec (NDArray): 回転時の基準軸
            angle (float): 回転角度[deg]

        Returns:
            Rotation: 3次元回転オブジェクト
        """
        rad = np.deg2rad(angle * 0.5)

        # クォータニオン
        qx = vec[0] * np.sin(rad)
        qy = vec[1] * np.sin(rad)
        qz = vec[2] * np.sin(rad)
        qw = np.cos(rad)
        quat = np.array([qx, qy, qz, qw])
        rot = Rotation.from_quat(quat)

        # 回転オブジェクトを返却
        return rot

    def separate_geometry(geom) -> list:
        """shapelyの複数ジオメトリクラスをジオメトリごとに分割する

        Args:
            geom (shapelyのジオメトリ): ジオメトリ

        Returns:
            list: ジオメトリのリスト
        """
        geoms = []

        if geom is not None:
            if type(geom) is geo.GeometryCollection:
                for tmp in geom.geoms:
                    tmp_list = GeoUtil.separate_geometry(tmp)
                    if len(tmp_list) > 0:
                        geoms.extend(tmp_list)

            elif (type(geom) is geo.MultiLineString
                    or type(geom) is geo.MultiPolygon
                    or type(geom) is geo.MultiPoint):
                geoms.extend(geom.geoms)

            else:
                geoms.append(geom)

        return geoms
