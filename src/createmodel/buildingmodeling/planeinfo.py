# -*- coding:utf-8 -*-
import sys
import numpy as np
from numpy.typing import NDArray
from .geoutil import GeoUtil
from ..createmodelexception import CreateModelException


class PlaneInfo:
    """平面情報クラス
    """

    @property
    def a(self) -> float:
        """平面パラメータa

        Returns:
            float: 平面パラメータa
        """
        return self._a

    @a.setter
    def a(self, value: float):
        """平面パラメータa

        Args:
            value (float): 平面パラメータa
        """
        self._a = value

    @property
    def b(self) -> float:
        """平面パラメータb

        Returns:
            float: 平面パラメータb
        """
        return self._b

    @b.setter
    def b(self, value: float):
        """平面パラメータb

        Args:
            value (float): 平面パラメータb
        """
        self._b = value

    @property
    def c(self) -> float:
        """平面パラメータc

        Returns:
            float: 平面パラメータc
        """
        return self._c

    @c.setter
    def c(self, value: float):
        """平面パラメータc

        Args:
            value (float): 平面パラメータc
        """
        self._c = value

    @property
    def d(self) -> float:
        """平面パラメータd

        Returns:
            float: 平面パラメータd
        """
        return self._d

    @d.setter
    def d(self, value: float):
        """平面パラメータd

        Args:
            value (float): 平面パラメータd
        """
        self._d = value

    @property
    def normal(self) -> NDArray:
        """法線ベクトル

        Returns:
            NDArray: 法線ベクトル
        """
        return self._normal

    @normal.setter
    def normal(self, value: NDArray):
        """法線ベクトル

        Args:
            value (NDArray): 法線ベクトル
        """
        self._normal = value

    def __init__(self, a: float = None, b: float = None,
                 c: float = None, d: float = None) -> None:
        """コンストラクタ

        Args:
            a (float, optional): 平面パラメータa. Defaults to None.
            b (float, optional): 平面パラメータb. Defaults to None.
            c (float, optional): 平面パラメータc. Defaults to None.
            d (float, optional): 平面パラメータd. Defaults to None.
        
        Note:
            ax + by + cz + d = 0
        """
        self._a = 0.0 if a is None else a
        self._b = 0.0 if b is None else b
        self._c = 0.0 if c is None else c
        self._d = 0.0 if d is None else d
        self._normal = np.array([self.a, self.b, self.c])
        
    def get_normal_size(self) -> float:
        """平面の法線ベクトルの大きさ

        Returns:
            float: 平面の法線ベクトルの大きさ
        """
        return np.linalg.norm(self.normal, ord=2)

    def normalize(self) -> NDArray:
        """正規化済み法線ベクトル

        Returns:
            NDArray: 正規化済み法線ベクトル
        """
        return GeoUtil.normalize(self.normal)

    def distance(self, pos: NDArray) -> float:
        """平面と点との距離の算出

        Args:
            pos (NDArray): 座標点

        Raises:
            ValueError: 平面のパラメータが不正の場合

        Returns:
            float: 平面に対して点から垂線を下した際の距離
        """
        denominator = self.get_normal_size()
        if GeoUtil.is_zero(denominator):
            name = self.__class__.__name__
            func_name = sys._getframe().f_code.co_name
            msg = '{}.{}, plane parameter(a, b, c) is zero.'.format(
                name, func_name)
            raise CreateModelException(msg)
        
        dist = np.abs((self.a * pos[0] + self.b * pos[1]
                       + self.c * pos[2] + self.d)) / denominator

        return dist

    def intersection_point(self, pos: NDArray, vec: NDArray) -> NDArray:
        """平面と直線の交点の算出

        Args:
            pos (NDArray): 直線の始点
            vec (NDArray): 直線のベクトル

        Raises:
            ValueError: 直線のベクトルの大きさが0の場合

        Returns:
            NDArray: 交点座標
        """
        dist = np.linalg.norm(vec, ord=2)
        if GeoUtil.is_zero(dist):
            name = self.__class__.__name__
            func_name = sys._getframe().f_code.co_name
            msg = '{}.{}, vec size is zero.'.format(name, func_name)
            raise CreateModelException(msg)
        
        a = -(self.a * pos[0] + self.b * pos[1] + self.c * pos[2] + self.d)
        b = self.a * vec[0] + self.b * vec[1] + self.c * vec[2]
        t = a / b

        cross_pos = pos + (vec * t)

        return cross_pos

    def calc_plane(self, pt1: NDArray, pt2: NDArray, pt3: NDArray) -> None:
        """3点から平面のパラメータを算出する

        Args:
            pt1 (NDArray): 座標点1
            pt2 (NDArray): 座標点2
            pt3 (NDArray): 座標点3
        """
        vec1 = pt2 - pt1
        vec2 = pt3 - pt1
        cross_vec = np.cross(vec1, vec2)
        self.a = cross_vec[0]
        self.b = cross_vec[1]
        self.c = cross_vec[2]
        self.d = -self.a * pt1[0] - self.b * pt1[1] - self.c * pt1[2]
        self.normal = np.array([self.a, self.b, self.c])
