from __future__ import annotations
from copy import deepcopy
import math
from typing import Optional, Final
import numpy as np


EPS = 1e-8  # 計算時の許容誤差
AcceptableError = 1e-6  # 判定時の許容誤差のデフォルト値


class Point:
    """2次元平面上の点クラス
    """

    _x: Final[float]
    _y: Final[float]

    def __init__(self, x: float, y: float) -> None:
        """コンストラクタ

        Args:
            x(float): x座標
            y(float): y座標 
        """
        self._x = x
        self._y = y

    @property
    def x(self):
        """x座標

        Returns:
            float: x座標
        """
        return self._x

    @property
    def y(self):
        """y座標

        Returns:
            float: y座標
        """
        return self._y

    def __add__(self, p: Point) -> Point:
        """点をベクトルとみなして足し算を行う

        Args:
            p(Point): 点

        Returns:
            Point: 各要素毎に足し算を行った値の点
        """
        return self.__class__(self.x + p.x, self.y + p.y)

    def __sub__(self, p: Point) -> Point:
        """点をベクトルとみなして引き算を行う

        Args:
            p(Point): 点

        Returns:
            Point: 各要素毎に引き算を行った値の点
        """
        return self.__class__(self.x - p.x, self.y - p.y)

    def __mul__(self, t: float):
        """各要素の値をt倍する

        Args:
            t(float): かける値

        Returns:
            Point: 各要素の値をt倍した点
        """
        return self.__class__(self.x * t, self.y * t)

    def __truediv__(self, t: float):
        """各要素の値を1/t倍する

        Args:
            t(float): わる値

        Returns:
            Point: 各要素の値を1/t倍した点
        """
        return self.__class__(self.x / t, self.y / t)

    def __str__(self) -> str:
        """オブジェクトの内容を表現する文字列を作成

        Returns:
            str: 点の位置を表す文字列
        """
        return f'({self.x}, {self.y})'

    def __eq__(self, p: Point) -> bool:
        """点が完全に一致しているか調べる

        Args:
            p(Point): 点

        Returns:
            bool: 一致している場合True

        Note:
            誤差を許容する場合にはis_sameを使用する
        """
        return self.is_same(p)

    def norm(self) -> float:
        """(0,0)からの距離の2乗を求める

        Returns:
            float: (0,0)からの距離の2乗
        """
        return math.pow(self.x, 2) + math.pow(self.y, 2)

    def abs(self) -> float:
        """(0,0)からの距離を求める

        Returns:
            float: (0,0)からの距離
        """
        return math.sqrt(self.norm())

    def dot(self, p: Point) -> float:
        """点をベクトルみなして内積を求める

        Args:
            p(Point): 点

        Returns:
            float: 内積
        """
        return self.x * p.x + self.y * p.y

    def cross(self, p: Point) -> float:
        """点をベクトルとみなして外積を求める

        Args:
            p(Point): 点

        Returns:
            float: 外積のz要素
        """
        return self.x * p.y - self.y * p.x

    def distance(self, p: Point) -> float:
        """他の点との距離を求める

        Args:
            p(Point): 点

        Returns:
            float: 点pとの距離
        """
        return math.sqrt(math.pow(self.x - p.x, 2) +
                         math.pow(self.y - p.y, 2))

    def is_same(self, p: Point, acceptable_error=AcceptableError) -> bool:
        """点が同一であるか調べる

        Args:
            p(Point): 点
            acceptable_error(float, optional): 許容誤差

        Returns:
            bool: 点の位置が同じである場合にはTrue        
        """
        return self.distance(p) < acceptable_error

    def is_on(self, s: Segment, acceptable_error=AcceptableError) -> bool:
        """点が線分上にあるか調べる

        Args:
            s(Segment): 線分
            acceptable_error(float, optional): 許容誤差

        Returns:
            bool: 線分上に点がある場合にはTrue
        """

        return s.distance(self) < acceptable_error

    def rotate(self, radian: float, center: Optional[Point] = None) -> Point:
        """centerを中心として回転する

        Args:
            radian(float): 回転する角度
            center(Point, optional): 回転の中心とする点 (Default: (0,0))
        """
        if center is None:
            center = Point(0, 0)

        rot = Point(math.cos(radian), math.sin(radian))
        moved = self - center
        rotated = Point(moved.x * rot.x - moved.y * rot.y,
                        moved.x * rot.y + moved.y * rot.x)

        return rotated + center


class Segment:
    """2次元平面上の線分クラス
    """

    _points: tuple[Point, Point]

    def __init__(self, p1: Point, p2: Point) -> None:
        """コンストラクタ

        Args:
            p1: 線分の端点
            p2: 線分の端点
        """
        self._points = (deepcopy(p1), deepcopy(p2))

    def __getitem__(self, index) -> Point:
        """インデックスで線分の端点にアクセス出来るようにする

        Args:
            index(int): インデックス(0 or 1)

        Returns:
            Point: 線分の端点
        """
        return self._points[index]

    def __str__(self) -> str:
        """オブジェクトの内容を表現する文字列を作成

        Returns:
            str: 線分の位置を表す文字列
        """
        return f'[{self[0]}, {self[1]}]'

    def is_same(self, s: Segment, acceptable_error=AcceptableError) -> bool:
        """線分が同じであるか調べる

        Args:
            s(Segment): 線分
            acceptable_error(float, optional): 許容誤差

        Returns:
            bool: 線分が一致している場合Trueを返す
        """
        return (self[0].is_same(s[0], acceptable_error) and self[1].is_same(s[1], acceptable_error)) \
            or (self[1].is_same(s[0], acceptable_error) and self[0].is_same(s[1], acceptable_error))

    def length(self):
        """線分の長さを求める

        Returns:
            float: 線分の長さ
        """
        return self[0].distance(self[1])

    def distance(self, p: Point) -> float:
        """与えられた点との距離を求める

        Args:
            p(Point): 点の2次元座標

        Returns:
            float: 点pとの距離

        Note:
            線分と点の距離は、点から最も近い線分上の点との距離で定義
        """
        if ((self[1] - self[0]).dot(p - self[0]) < EPS):
            return p.distance(self[0])
        if ((self[0] - self[1]).dot(p - self[1]) < EPS):
            return p.distance(self[1])
        return abs((self[1] - self[0]).cross(p - self[0])) / self.length()

    def nearest_point_from(self, p: Point) -> Point:
        """与えられた点から最も近い線分上の点を求める

        Args:
            p(Point): 点の2次元座標

        Returns:
            Point: 点pから最も近い線分上の座標
        """
        if ((self[1] - self[0]).dot(p - self[0]) < EPS):
            return deepcopy(self[0])
        if ((self[0] - self[1]).dot(p - self[1]) < EPS):
            return deepcopy(self[1])
        return self.project(p)

    def is_intersected(self, s: Segment, acceptable_error=AcceptableError) -> bool:
        """線分と交差しているかを判定する

        Args:
            s(Segment): 線分
            acceptable_error(float, optional): 許容誤差

        Returns:
            bool: 交差している場合True
        """
        self_v = (self[1] - self[0]) / self.length() * acceptable_error
        s_v = (s[1] - s[0]) / s.length() * acceptable_error

        p0 = self[0] - self_v
        p1 = self[1] + self_v
        q0 = s[0] - s_v
        q1 = s[1] + s_v

        if p0.is_same(q0) or p0.is_same(q1) or p1.is_same(q0) or p1.is_same(q1):
            return True

        def unit_cross(a, b):
            return (a / a.abs()).cross(b / b.abs())

        a = unit_cross(p0 - p1, q0 - p0)
        b = unit_cross(p0 - p1, q1 - p0)
        c = unit_cross(q0 - q1, p0 - q0)
        d = unit_cross(q0 - q1, p1 - q0)

        # 同一直線上
        if (abs(a) < EPS and abs(b) < EPS) or (abs(c) < EPS and abs(d) < EPS):
            return False

        return a * b < EPS and c * d < EPS

    def get_cross_point(self, s: Segment, acceptable_error=AcceptableError) -> Optional[Point]:
        """線分との交点を求める

        Args:
            s(Segment): 線分
            acceptable_error(float, optional): 許容誤差

        Returns:
            Optional[Point]: 交点 (交点が存在しない場合にはNone)
        """
        if not self.is_intersected(s, acceptable_error):
            return None

        self_vec = self[1] - self[0]
        s_vec = s[1] - s[0]

        return self[0] + self_vec * s_vec.cross(s[0] - self[0]) / s_vec.cross(self_vec)

    # pを通る垂線との交点
    def project(self, point: Point) -> Point:
        """与えられた点を通る線分の垂線と線分の交点を求める

        Args:
            point(Point): 2次元座標

        Returns:
            Point: 垂線と線分の交点

        Note:
            正確には、線分を延長した直線との交点を求めているため、求められた交点が線分上の点とは限らない
        """
        base = self[1] - self[0]
        return self[0] + base * (point - self[0]).dot(base) / base.norm()


def get_angle_degree(center: Point, p1: Point, p2: Point) -> float:
    """centerを中心として、2点のなす角を返す

    Args:
        center: 中心点の2次元座標
        p1: 1つめの点の2次元座標
        p2: 2つめの点の2次元座標

    Returns:
        float: p2を基準としたp1の角度 (-180以上180以下)
    """
    v1 = p1 - center
    v2 = p2 - center

    return np.rad2deg(math.atan2(v2.cross(v1), v2.dot(v1)))


def is_ccw_order(a: Point, b: Point, c: Point, center: Point = Point(0, 0)) -> bool:
    """点a,b,cがcenter(default:(0,0))反時計回り順であるか判定する"""

    a = a - center
    b = b - center
    c = c - center

    if a.cross(c) >= 0:
        return a.cross(b) > EPS and b.cross(c) > EPS
    else:
        return a.cross(b) > EPS or b.cross(c) > EPS
