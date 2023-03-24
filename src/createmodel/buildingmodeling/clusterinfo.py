# -*- coding:utf-8 -*-
from .geoutil import GeoUtil
from ..lasmanager import PointCloud
import shapely.geometry as geo
import alphashape
from numpy.typing import NDArray


class ClusterInfo(object):
    """点群と推定面を管理するデータクラス
    """
    @property
    def id(self) -> int:
        """id

        Returns:
            int: id
        """
        return self._id
    
    @id.setter
    def id(self, value: int):
        """id

        Args:
            value (int): id
        """
        self._id = value

    @property
    def points(self) -> PointCloud:
        """点群データ

        Returns:
            PointCloud: 点群データ
        """
        return self._points
    
    @points.setter
    def points(self, value: PointCloud):
        """点群データ

        Args:
            value (PointCloud): 点群データ
        """
        self._points = value

    @property
    def roof_polygon(self) -> geo.Polygon:
        """屋根形状

        Returns:
            shapely.geometry.Polygon: 屋根形状
        """
        if self._roof_line is not None:
            return geo.Polygon(self._roof_line)
        else:
            return None

    # @roof_polygon.setter
    # def roof_polygon(self, value: geo.Polygon):
    #     """屋根形状

    #     Args:
    #         value (shapely.geometry.Polygon): 屋根形状
    #     """
    #     self._roof_polygon = value

    @property
    def parent(self) -> int:
        """親屋根id

        Returns:
            int: 親屋根id

        Node:
            -1の場合は親屋根なし
        """
        return self._parent

    @parent.setter
    def parent(self, value: int):
        """親屋根id

        Args:
            value (int): 親屋根id

        Node:
            -1の場合は親屋根なし
        """
        self._parent = value

    @property
    def children(self) -> list[int]:
        """子屋根リスト

        Returns:
            list[int]: 子屋根idリスト
        """
        return self._children

    @children.setter
    def children(self, value: list[int]):
        """子屋根リスト

        Args:
            value (list[int]): 子屋根idリスト
        """
        self._children = value

    @property
    def roof_height(self) -> float:
        """屋根の高さ

        Returns:
            float: 屋根の高さ
        """
        return self._roof_height

    @roof_height.setter
    def roof_height(self, value: float):
        """屋根の高さ

        Args:
            value (float): 屋根の高さ
        """
        self._roof_height = value

    @property
    def roof_line(self) -> NDArray:
        """2D屋根の形状の頂点座標列(終点は始点と異なる座標)

        Returns:
            NDArray: 2D屋根の形状の頂点座標列(終点は始点と異なる座標)
        """
        return self._roof_line

    @roof_line.setter
    def roof_line(self, value: NDArray):
        """2D屋根の形状の頂点座標列(終点は始点と異なる座標)

        Args:
            value (NDArray): 2D屋根の形状の頂点座標列(終点は始点と異なる座標)
        """
        self._roof_line = value

    def __init__(self, id=None, points=None) -> None:
        """コンストラクタ

        Args:
            id (int, optional): id. Defaults to None.
            points (PointCloud, optional): 点群. Defaults to None.
        """
        self._id = 0 if id is None else id
        self._points = PointCloud()
        if points is not None:
            self._points = points

        self._roof_polygon = None
        self._parent = -1
        self._children = []
        self._roof_height = 0
        self._roof_line = None

    def __lt__(self, other):
        """ソート関数

        Args:
            other (ClusterInfo): 比較対象

        Returns:
            bool: 比較結果
        """
        self_pt = len(self.points.get_points())
        other_pt = len(other.points.get_points())
        return self_pt < other_pt

    def __repr__(self):
        """オブジェクトを表す公式な文字列の作成

        Returns:
            str: 文字列
        """
        return repr((self.id, self.points,
                     self.roof_polygon, self.parent,
                     self.children, self.roof_height))

    def get_contours(self) -> list[geo.Polygon]:
        """点群のalpha shape形状を取得する

        Returns:
            list[geo.Polygon]: ポリゴンのリスト
        """
        list = []
        try:
            if len(self._points.get_points()) > 0:
                geom = alphashape.alphashape(
                    self._points.get_points()[:, 0:2], alpha=2.0)
                separate_geoms = GeoUtil.separate_geometry(geom)
                list = [poly for poly in separate_geoms
                        if (type(poly) is geo.Polygon and poly.area > 0)]
        except Exception:
            pass

        return list
