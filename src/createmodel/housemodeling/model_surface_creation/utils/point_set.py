from copy import deepcopy
from typing import Optional
from .geometry import AcceptableError, Point, Segment

AcceptableError = 0.5
EPS = 1e-8


class PointSet:
    """
    点の集合を管理するクラス
    """

    points: list[Point]

    def __init__(self) -> None:
        """コンストラクタ
        """
        self.points = []

    def __len__(self) -> int:
        """点の数を返す
        """
        return len(self.points)

    def find_nearest(self, p: Point, limit: float = float('inf')) -> Optional[Point]:
        """最も近い点を求める

        Args:
            p(Point): 基準となる点
            limit(float, optional): 距離の上限

        Returns:
            Optional[Point]: limitより近い最近点、存在しない場合はNone
        """
        minimum_distance = float('inf')
        nearest_point = None
        for point in self.points:
            distance = point.distance(p)
            if distance < limit and distance < minimum_distance:
                minimum_distance = distance
                nearest_point = point
        return nearest_point

    def add(self, point: Point, acceptable_error=AcceptableError) -> tuple[bool, Point]:
        """点の追加

        Args:
            point(Point): 追加する点
            acceptable_error(float, optional): 同一の点と見なす距離の上限(Default: 0.5) 

        Returns:
            bool: 点の追加が発生した場合True
            Point: 追加した点または、同一の点とみなされた点
        """

        nearest_point = self.find_nearest(point, limit=acceptable_error)
        if not nearest_point:
            self.points.append(point)
            return True, point

        return False, nearest_point

    def adjust_point(self, point: Point, acceptable_error=AcceptableError) -> Point:
        """Set内の頂点の位置に合わせる

        Args:
            point(Point): 点
            acceptable_error(float, optional): 調整を行う上限距離

        Returns:
            Point: 近い点が存在する場合はその点、そうでなければ入力の点
        """
        nearest_point = self.find_nearest(point, limit=acceptable_error)
        return deepcopy(nearest_point) if nearest_point else deepcopy(point)
