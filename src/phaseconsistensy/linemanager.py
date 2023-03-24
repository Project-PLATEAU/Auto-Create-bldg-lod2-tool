import math
from logging import getLogger

from shapely.geometry import Point

from .geoutil import GeoUtil

logger = getLogger(__name__)


class TriangleInfo:
    """三角形情報クラス
    """
    def __init__(self, point_list: list):
        """コンストラクタ

        Args:
            point_list (list<Point>): 頂点列
        """
        self._point_list = point_list   # 頂点列
        min, max = GeoUtil.get_minmax(point_list)
        self._min = min     # バウンディングボックス(min)
        self._max = max     # バウンディングボックス(max)

    @property
    def min(self) -> Point:
        return self._min

    @property
    def max(self) -> Point:
        return self._max
    
    @property
    def point_list(self) -> list:
        return self._point_list


class LineInfo:
    """線分情報クラス
    """
    id = 0  # 識別番号

    def __init__(self, pos1: Point, pos2: Point):
        """コンストラクタ

        Args:
            pos1 (Point): 始点座標
            pos2 (Point): 終点座標
        """
        self.pos1 = pos1        # 始点
        self.pos2 = pos2        # 終点
        self.id = LineInfo.id   # ID
        LineInfo.id += 1

    def str(self):
        return 'id: ' + str(self.id) + ', ' + str(self.pos1) + " - " + str(self.pos2)

    def is_pair_line(self, other) -> bool:
        """同一の始終点を持ち、反対方向の線分かどうかの判定
           (接している線分)

        Args:
            other (LineInfo): 比較対象のインスタンス

        Returns:
            bool: True: 同一の始終点/反対方向の線分 False: 左記以外
        """
        if not isinstance(other, LineInfo):
            return False
        if GeoUtil.is_same_point(self.pos1, other.pos2) \
                and GeoUtil.is_same_point(self.pos2, other.pos1):
            return True
        return False

    def merge_lines(self, other) -> bool:
        """同一線上にあり、繋がっている線分の場合に一つの線分にマージ

        Args:
            other (LineInfo): 他の線分

        Returns:
            bool: True: マージ実施, False: マージ対象外
        """
        if GeoUtil.is_same_point(self.pos2, other.pos1):
            angle = GeoUtil.interior_angle(self.pos1, self.pos2, other.pos2)
            logger.debug(f'id = {self.id}, id2 = {other.id}, angle = {angle}')
            if GeoUtil.is_same_value(angle, math.pi):
                logger.debug(f'case 1: b {self.str()}')
                logger.debug(f'case 1: o {other.str()}')
                self.pos2 = Point(other.pos2)
                logger.debug(f'case 1: a {self.str()}')
                return True
        
        elif GeoUtil.is_same_point(self.pos1, other.pos2):
            angle = GeoUtil.interior_angle(other.pos1, other.pos2, self.pos2)
            logger.debug(f'id = {self.id}, id2 = {other.id}, angle = {angle}')
            if GeoUtil.is_same_value(angle, math.pi):
                logger.debug(f'case 2: b {self.str()}')
                logger.debug(f'case 2: o {other.str()}')
                self.pos1 = Point(other.pos1)
                logger.debug(f'case 2: a {self.str()}')
                return True

        return False

    def get_polygon_set(self, line_list: list, polygon_list: list):
        """指定されたリスト内の繋がっている線分を抽出

        Args:
            line_list (list<LineInfo>): 入力線分リスト
            polygon_list (list<LineInfo>): 該当線分の出力先
        """
        line_num = len(line_list)
        same_list = []

        logger.debug('----------------------')
        logger.debug(f'line_list = {line_list}')
        logger.debug(f'self = {self.str()}')

        # つながる線分探索
        for i in range(line_num):
            if line_list[i] is None:
                continue
            if id == line_list[i].id:
                continue
            if GeoUtil.is_same_point(self.pos2, line_list[i].pos1):
                same_list.append(i)

        same_list_len = len(same_list)
        logger.debug(f'same_list_len = {same_list_len}')
        logger.debug(f'same_list = {same_list}')
        index = 0
        if same_list_len == 0:
            return
        elif same_list_len == 1:
            index = same_list[0]
        else:
            # 複数ある場合、内角が小さいものを選択
            min_angle = GeoUtil.interior_angle(self.pos1, self.pos2,
                                               line_list[same_list[0]].pos1)
            index = same_list[0]
            for i in range(1, same_list_len):
                angle = GeoUtil.interior_angle(self.pos1, self.pos2,
                                               line_list[same_list[i]].pos1)
                if angle < min_angle:
                    index = same_list[i]
                    min_angle = angle
        
        target_line = line_list[index]
        logger.debug(f'index = {index}, target_line = {target_line.str()}')
        polygon_list.append(target_line)
        line_list[index] = None

        # 再帰的につなぐ処理を実施
        if target_line is not None:
            target_line.get_polygon_set(line_list, polygon_list)


class LineManager:
    """複数線分管理クラス
    """
    def __init__(self):
        """コンストラクタ
        """
        self._hash_table = dict()   # ハッシュテーブル
    
    def append(self, line: LineInfo):
        """線分の追加

        Args:
            line (LineInfo): 追加する線分
        """
        hash_key = self.hash_val(line)
        if not (hash_key in self._hash_table):
            self._hash_table[hash_key] = []
        self._hash_table[hash_key].append(line)
    
    def append_polygon(self, point_list: list):
        """面の頂点列を追加

           面の頂点列から線分を作成して追加する

        Args:
            point_list (list<Point>): 面の頂点列
        """
        point_num = len(point_list)
        for i in range(point_num):
            if i == point_num - 1:
                line = LineInfo(point_list[i], point_list[0])
            else:
                line = LineInfo(point_list[i], point_list[i + 1])
            self.append(line)

    def hash_val(self, line: LineInfo) -> int:
        """ハッシュテーブル用のハッシュ値算出

        Args:
            line (LineInfo): 線分

        Returns:
            int: ハッシュ値
        """
        return abs(int(line.pos1.x * 1000
                       + line.pos1.y * 1000
                       + line.pos1.z * 1000
                       + line.pos2.x * 1000
                       + line.pos2.y * 1000
                       + line.pos2.z * 1000))
    
    def is_same_line(self, line: LineInfo) -> bool:
        """始終点が一致し、逆方向を向いた線分かどうかの判定

        Args:
            line (LineInfo): 比較対象の線分

        Returns:
            bool: True: 始終点一致/逆方向の線分, False: 左記以外
        """
        hask_key = self.hash_val(line)
        for line2 in self._hash_table[hask_key]:
            if line.is_pair_line(line2):
                return True
        
        return False
    
    def delete_pair_line(self):
        """始終点が一致し、逆方向を向いた線分を削除する
           (接している辺)
        """
        for hash_key in self._hash_table.keys():
            line_list = self._hash_table[hash_key]
            delete_list = []
            line_num = len(line_list)
            for i in range(line_num):
                if line_list[i] is None:
                    continue
                for j in range(i + 1, line_num):
                    if line_list[i] is None or line_list[j] is None:
                        continue
                    if line_list[i].is_pair_line(line_list[j]):
                        delete_list.append(i)
                        delete_list.append(j)
                        line_list[i] = None
                        line_list[j] = None
            
            # logger.debug(f'line_num = {line_num}')
            delete_list.sort()
            for index in reversed(delete_list):
                # logger.debug(f'  index = {index}')
                del line_list[index]

    def marge_lines(self):
        """同一線上で分割されている線分をマージしたものに置き換える
        """
        line_list = []
        for hash_key in self._hash_table.keys():
            line_list.extend(self._hash_table[hash_key])

        self._hash_table.clear()
        line_num = len(line_list)
        for i in range(line_num):
            if line_list[i] is None:
                continue
            for j in range(i + 1, line_num):
                if line_list[i] is None or line_list[j] is None:
                    continue
                if line_list[i].merge_lines(line_list[j]):
                    line_list[j] = None
                    break
            self.append(line_list[i])

    def get_list(self) -> list:
        """ハッシュテーブル内の線分からリストを作成

        Returns:
            list<LineInfo>: 線分のリスト
        """
        line_list = []
        for hash_key in self._hash_table.keys():
            line_list.extend(self._hash_table[hash_key])
        return line_list

    def delete_overlap_lines(self) -> list:
        """重複している線分(同一線分/逆方向のもの)を削除

        Returns:
            list<LineInfo>: 削除後の線分のリスト
        """
        line_list = self.get_list()

        output_line_list = []
        line_num = len(line_list)
        for i in range(line_num):
            if line_list[i] is None:
                continue
            same_line = False
            for j in range(i + 1, line_num):
                if line_list[j] is None:
                    continue
                if GeoUtil.is_on_same_line(line_list[i].pos1,
                                           line_list[i].pos2,
                                           line_list[j].pos1,
                                           line_list[j].pos2):
                    logger.debug(f'  on same line: id1 = {line_list[i].id}')
                    logger.debug(f'                id2 = {line_list[j].id}')
                    line1, line2 = self._delete_overlap_on_same_line(
                        line_list[i], line_list[j])
                    same_line = True
                    if line1 is not None:
                        output_line_list.append(line1)
                    if line2 is not None:
                        output_line_list.append(line2)
                    line_list[j] = None
                    break
            if same_line:
                line_list[i] = None
            else:
                output_line_list.append(line_list[i])
        
        return output_line_list
    
    def _delete_overlap_on_same_line(self, line1: LineInfo, line2: LineInfo) \
            -> tuple:
        """同一線上の重畳部分除外処理

        Args:
            line1 (LineInfo): 入力線分1
            line2 (LineInfo): 入力線分2
        
        Returns:
            LineInfo1, LineInfo2 : 重畳部分除外した線分
        """
        new_line1 = None
        new_line2 = None
        if not GeoUtil.is_same_point(line1.pos1, line2.pos2):
            new_line1 = LineInfo(line1.pos1, line2.pos2)
        
        if not GeoUtil.is_same_point(line2.pos1, line1.pos2):
            new_line2 = LineInfo(line2.pos1, line1.pos2)

        return new_line1, new_line2

