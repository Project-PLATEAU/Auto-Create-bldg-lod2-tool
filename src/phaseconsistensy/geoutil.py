import sys
import math
from logging import getLogger
import numpy as np
from numpy.typing import NDArray
from shapely.geometry import Point, LineString, Polygon

logger = getLogger(__name__)


class GeoUtil:

    delta_threshold = 0.001

    @classmethod
    def min(self, val1, val2):
        if val1 < val2:
            return val1
        return val2
    
    @classmethod
    def max(self, val1, val2):
        if val1 > val2:
            return val1
        return val2

    @classmethod
    def get_point_str(self, pos: Point):
        return '(' + str(pos.x) + ', ' + str(pos.y) + ', ' + str(pos.z) + ')'

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
    def float_is_zero(self, value: float) -> bool:
        """浮動小数点が0であるかチェック
    
        Args:
            value (float): 値
     
        Returns:
            bool: True: 0である, False: 0でない
        """
        if abs(value) < sys.float_info.epsilon:
            return True
        return False

    @classmethod
    def normalize(self, vec: NDArray) -> NDArray:
        """単位ベクトルに正規化

        Args:
            vec (NDArray): 入力ベクトル

        Returns:
            NDArray: 正規化後のベクトル
        """

        x = np.linalg.norm(vec)
        if GeoUtil.float_is_zero(x):
            return np.zeros(len(vec))
        return vec / x

    @classmethod
    def dot(self, vec1: NDArray, vec2: NDArray) -> float:
        """ベクトルの内積算出 (正規化されたベクトル)

        Args:
            vec1 (NDArray): ベクトル1
            vec2 (NDArray): ベクトル2

        Returns:
            float: 内積値
        """
        norm1 = GeoUtil.normalize(vec1)
        norm2 = GeoUtil.normalize(vec2)
        ret_value = np.dot(norm1, norm2)
        if ret_value > 1.0:
            ret_value = 1.0
        elif ret_value < -1.0:
            ret_value = -1.0
        return ret_value

    @classmethod
    def get_normal(self, p_list: list) -> NDArray:
        """面の法線ベクトルを算出

        Args:
            p_list (list<Point>): 入力頂点列

        Returns:
            NDArray: 法線ベクトル
        """
        normal = np.zeros(3)
        list_len = len(p_list)
        if list_len < 3:
            return normal
        
        i = 1
        v0 = np.array([p_list[0].x, p_list[0].y, p_list[0].z])
        while (i < list_len - 1):
            v1 = np.array([p_list[i].x, p_list[i].y, p_list[i].z]) - v0
            v2 = np.array([p_list[i + 1].x,
                           p_list[i + 1].y,
                           p_list[i + 1].z]) - v0
            normal += np.cross(v2, v1)
            i += 1
        
        return self.normalize(normal)

    @classmethod
    def interior_angle(self, p1: Point, p2: Point, p3: Point) -> float:
        """3 頂点が作る角度算出

        Args:
            p1 (Point): 頂点 1
            p2 (Point): 頂点 2
            p3 (Point): 頂点 3

        Returns:
            float: 角度 (ラジアン)
        """
        v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
        v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
        norm1 = np.linalg.norm(v1, ord=2)
        norm2 = np.linalg.norm(v2, ord=2)
        if GeoUtil.is_same_value(norm1, 0.0) \
                or GeoUtil.is_same_value(norm2, 0.0):
            return 0.0
        dot = GeoUtil.dot(v1, v2)
        return math.acos(dot)

    @classmethod
    def conv_2d(self, p_list: list) -> list:
        """3次元の面を 2次元に変換

           法線に垂直な平面に投影した頂点列を返す

        Args:
            p_list (list<Point>): 入力頂点列

        Returns:
            list<Point>: 出力頂点列
        """

        # 法線ベクトル算出
        z_vec = self.get_normal(p_list)

        # （原点の）先頭頂点から一番遠い点を使用して平面ベクトル算出
        max_d = 0.0
        v1 = np.zeros(3)
        p0 = p_list[0]
        max_pos = 0
        for i, pos in enumerate(p_list):
            dist = pos.distance(p0)
            if dist > max_d:
                max_d = dist
                v1 = np.array([pos.x - p0.x, pos.y - p0.y, pos.z - p0.z])
                max_pos = i
        v1 = self.normalize(v1)

        # z_vec の長さが 0 の場合
        # 先頭頂点から二番目に遠い点を使用して法線方向のベクトル算出
        if GeoUtil.float_is_zero(np.linalg.norm(z_vec)):
            v2 = np.zeros(3)
            max_d = 0.0
            for i, pos in enumerate(p_list):
                if i == max_pos:
                    continue
                dist = pos.distance(p0)
                if dist > max_d:
                    max_d = dist
                    v2 = np.array([pos.x - p0.x, pos.y - p0.y, pos.z - p0.z])
            v2 = self.normalize(v2)
            z_vec = np.cross(v2, v1)
            z_vec = self.normalize(z_vec)
        
        new_list = []
        if GeoUtil.float_is_zero(z_vec[0]) \
           and GeoUtil.float_is_zero(z_vec[1]) \
           and GeoUtil.float_is_zero(z_vec[2]):
            for pos in p_list:
                new_list.append(pos)
            return new_list

        y_vec = np.cross(z_vec, v1)
        y_vec = self.normalize(y_vec)
        x_vec = np.cross(y_vec, z_vec)
        x_vec = self.normalize(x_vec)

        conv_mat = np.array([[x_vec[0], y_vec[0], z_vec[0], 0.0],
                             [x_vec[1], y_vec[1], z_vec[1], 0.0],
                             [x_vec[2], y_vec[2], z_vec[2], 0.0],
                             [0.0, 0.0, 0.0, 1.0]])

        inv_mat = np.linalg.inv(conv_mat)

        for i, pos in enumerate(p_list):
            in_pos = np.array([pos.x-p0.x , pos.y-p0.y, pos.z-p0.z, 1.0])
            out_pos = inv_mat @ in_pos
            new_list.append(Point(out_pos[0], out_pos[1]))
        
        return new_list

    @classmethod
    def get_line_list(self, p_list: list) -> list:
        """線分リストを取得する

        Args:
            p_list (list<Point>): 入力頂点列

        Returns:
            list<LineString>: 線分リスト
        """
        line_list = []
        point_len = len(p_list)
        for i in range(point_len):
            if i == (point_len - 1):
                line = LineString([(p_list[i].x, p_list[i].y),
                                   (p_list[0].x, p_list[0].y)])
            else:
                line = LineString([(p_list[i].x, p_list[i].y),
                                   (p_list[i + 1].x, p_list[i + 1].y)])
            line_list.append(line)
        return line_list

    @classmethod
    def is_cross_or_touch(self, p_list: list, err_list: list = None) -> bool:
        """面の自己交差/接触判定

        Args:
            p_list (list<Point>): 入力頂点列
            err_list (list<Point>): エラー頂点列

        Returns:
            bool: True: 交差/接触あり, False: 交差/接触なし
        """

        if err_list is None:
            err_list = list()

        # 平面に投影
        xy_list = self.conv_2d(p_list)

        # 自己交差チェック
        err_index = []
        if self.is_cross_2d(xy_list, err_index):
            for i in err_index:
                err_list.append(p_list[i])
        
        # 自己接触チェック
        err_index = []
        if self.is_touch_2d(xy_list, err_index):
            for i in err_index:
                err_list.append(p_list[i])

        if len(err_list) > 0:
            return True        
        return False
    
    @classmethod
    def is_cross_2d(self, p_list: list, err_list: list) -> bool:
        """面の自己交差判定 (2次元)

        Args:
            p_list (list): 入力頂点列 (2次元)
            err_list (list<int>): エラー位置格納先 (p_list 内インデックス値のリスト)

        Returns:
            bool: True: 交差あり, False: 交差なし
        """

        r_flag = False

        line_list = self.get_line_list(p_list)
        line_len = len(line_list)
        for i in range(line_len):
            line1 = line_list[i]
            for j in range(i + 2, line_len):
                if i == 0 and j == (line_len - 1):
                    continue
                line2 = line_list[j]
                if line1.crosses(line2):
                    logger.debug(f'line1 = {line1}')
                    logger.debug(f'line2 = {line2}')
                    logger.debug('  cross')
                    r_flag = True
                    if j == line_len - 1:
                        add_list = [i, i + 1, j, 0]
                    else:
                        add_list = [i, i + 1, j, j + 1]
                    err_list += add_list
        
        return r_flag

    @classmethod
    def is_touch_2d(self, p_list: list, err_list: list) -> bool:
        """自己接触判定

        Args:
            p_list (list<Point>): 入力頂点列 (2次元)
            err_list (list<int>): エラー位置格納先 (p_list 内インデックス値のリスト)

        Returns:
            bool: True: 接触あり, False: 接触なし
        """

        r_flag = False

        line_list = self.get_line_list(p_list)
        for i in range(len(p_list)):
            p1 = Point(line_list[i].coords[0])
            p2 = Point(line_list[i].coords[1])
            for j in range(0, len(p_list)):
                if j == i or j == ((i+1) % len(p_list)):
                    continue
                p = Point(p_list[j].x, p_list[j].y)
                if p.distance(line_list[i]) <= self.delta_threshold:
                    r_flag = True
                    err_list.append(j)     
                    
        return r_flag
    
    @classmethod
    def is_convex_polygon(self, p_list: list) -> bool:
        """凸多角形かどうかを判定

        Args:
            p_list (list<Point>): 入力頂点列

        Returns:
            bool: True: 凸多角形, False: 凹多角形
        """

        pos_len = len(p_list)
        if pos_len <= 3:
            return True
        
        for i in range(pos_len):
            v0 = np.array([p_list[i].x, p_list[i].y, p_list[i].z])
            base_normal = np.zeros(3)

            for j in range(2, pos_len):
                p1 = p_list[(i + j - 1) % pos_len]
                p2 = p_list[(i + j) % pos_len]

                v01 = np.array([p1.x, p1.y, p1.z]) - v0
                v02 = np.array([p2.x, p2.y, p2.z]) - v0
                v01 = self.normalize(v01)
                v02 = self.normalize(v02)

                normal = np.cross(v02, v01)
                normal = self.normalize(normal)

                if (j == 2):
                    base_normal = normal
                else:
                    if np.dot(base_normal, normal) < 0.0:
                        return False
        
        return True

    @classmethod
    def is_zero_area(self, p_list: list) -> bool:
        """面積 0 のポリゴンかどうか判定

        Args:
            p_list (list<Point>): 入力ポリゴンの頂点リスト

        Returns:
            bool: True: 面積 0 と判定, False: 面積 0 ではない
        """
        
        # 面の法線算出
        normal = GeoUtil.get_normal(p_list)
        # 法線の長さ算出
        dist_square = normal[0] * normal[0] \
            + normal[1] * normal[1] \
            + normal[2] * normal[2]
        # 長さが 0
        if abs(dist_square) < sys.float_info.epsilon:
            return True
        
        return False

    @classmethod
    def divide_triangle_for_convec_polygon(self, p_list: list) -> list:
        """凸ポリゴンを三角形に分割

        Args:
            p_list (list<Point>): 入力ポリゴンの頂点リスト

        Returns:
            list<list<Point>>: 分割後の頂点列リスト
        """

        multi_point_list = []

        vertex_list = p_list.copy()

        while len(vertex_list) >= 3:
            vertex_num = len(vertex_list)
            cur_index = 0
            prev_index = cur_index - 1
            if prev_index < 0:
                prev_index = vertex_num - 1
            post_index = (cur_index + 1) % vertex_num

            triangle_list = []
            triangle_list.append(vertex_list[prev_index])
            triangle_list.append(vertex_list[cur_index])
            triangle_list.append(vertex_list[post_index])
            multi_point_list.append(triangle_list)
            del vertex_list[cur_index]

        return multi_point_list

    @classmethod
    def divide_triangle_for_concave_polygon(self, p_list: list, check_intersect: bool = True) -> list:
        """凹ポリゴンを三角形に分割

        Args:
            p_list (list<Point>): 入力ポリゴンの頂点リスト
            check_interset (bool): 自己交差・接触チェックするか

        Returns:
            list<list<Point>>: 分割後の頂点列リスト
        """

        # 2次元に投影
        xy_list = self.conv_2d(p_list)

        # 原点から一番離れている頂点を算出し、その頂点の 2 つの辺から法線算出
        dist_max = 0.0
        index = 0
        #for i, pos in enumerate(xy_list):
        for i, pos in enumerate(p_list):
            #dist_square = pos.x * pos.x + pos.y * pos.y
            dist_square = pos.x * pos.x + pos.y * pos.y + pos.z * pos.z
            if dist_square > dist_max:
                dist_max = dist_square
                index = i
        index1 = index - 1
        if index1 < 0:
            index1 = len(xy_list) - 1
        index2 = (index + 1) % len(xy_list)
        v0 = np.array([xy_list[index].x, xy_list[index].y])
        v01 = np.array([xy_list[index1].x, xy_list[index1].y]) - v0
        v02 = np.array([xy_list[index2].x, xy_list[index2].y]) - v0
        v01 = self.normalize(v01)
        v02 = self.normalize(v02)
        base_normal = np.cross(v01, v02)

        multi_point_list = []
        vertex_list = p_list.copy()
        
        #cur_index = index
        cur_index = 0
        loop_ct = 0
        while len(xy_list) > 3:
            loop_ct += 1
            vertex_num = len(xy_list)
            if loop_ct > vertex_num * 2 + 10:
                # 分割失敗
                triangle_list = []
                for i in range(vertex_num):
                    triangle_list.append(vertex_list[i])

                # 自己交差チェック
                if not check_intersect:
                    multi_point_list.append(triangle_list)
                elif not self.is_cross_or_touch(triangle_list):
                    multi_point_list.append(triangle_list)
                    
                return multi_point_list

            cur_index %= vertex_num
            prev_index = cur_index - 1
            if prev_index < 0:
                prev_index = vertex_num - 1
            post_index = (cur_index + 1) % vertex_num

            cur_pos = Point(xy_list[cur_index].x, xy_list[cur_index].y)
            prev_pos = Point(xy_list[prev_index].x, xy_list[prev_index].y)
            post_pos = Point(xy_list[post_index].x, xy_list[post_index].y)
            cur_vec = np.array([cur_pos.x, cur_pos.y])
            prev_vec = np.array([prev_pos.x, prev_pos.y]) - cur_vec
            post_vec = np.array([post_pos.x, post_pos.y]) - cur_vec
            prev_vec = self.normalize(prev_vec)
            post_vec = self.normalize(post_vec)

            # 向きが異なる場合スキップ
            if base_normal * np.cross(prev_vec, post_vec) < 0.0:
                cur_index += 1
                continue

            # 自己交差チェック
            triangle_list = []
            triangle_list.append(vertex_list[prev_index])
            triangle_list.append(vertex_list[cur_index])
            triangle_list.append(vertex_list[post_index])
            if check_intersect and self.is_cross_or_touch(triangle_list):
                cur_index += 1
                continue

            # 三角形内に他の頂点が入っている場合はスキップ
            triangle = Polygon([(post_pos.x, post_pos.y),
                                (cur_pos.x, cur_pos.y),
                                (prev_pos.x, prev_pos.y)])
            skip = False
            for i, pos in enumerate(xy_list):
                if i == prev_index or i == cur_index or i == post_index:
                    continue
                if triangle.contains(pos):
                    skip = True
                    break
            if skip:
                cur_index += 1
                continue

            # 三角形作成 (投影前の頂点から作成)
            triangle_list = []
            triangle_list.append(vertex_list[prev_index])
            triangle_list.append(vertex_list[cur_index])
            triangle_list.append(vertex_list[post_index])
            multi_point_list.append(triangle_list)

            # TODO:反対方向を見ているかチェック

            loop_cnt = 0
            del xy_list[cur_index]
            del vertex_list[cur_index]

        # 最後の三角形を追加
        if len(xy_list) == 3:
            triangle_list = []
            triangle_list.append(vertex_list[0])
            triangle_list.append(vertex_list[1])
            triangle_list.append(vertex_list[2])
            # 自己交差チェック
            if not check_intersect:
                multi_point_list.append(triangle_list)
            elif not self.is_cross_or_touch(triangle_list):
                # TODO:反対方向を見ているかチェック
                multi_point_list.append(triangle_list)

        return multi_point_list

    @classmethod
    def divide_triangle(self, p_list: list, check_intersect: bool = True) -> list:
        """面を三角形に分割

        Args:
            p_list (list<Point>): 面の頂点リスト
            check_interset (bool): 自己交差・接触チェックするか

        Returns:
            list<list<Point>>: 分割後の頂点列リスト
        """

        multi_point_list = []
        pos_len = len(p_list)
        if pos_len <= 3:
            multi_point_list.append(p_list)
            return multi_point_list

        if self.is_convex_polygon(p_list):
            return self.divide_triangle_for_convec_polygon(p_list)
        else:
            return self.divide_triangle_for_concave_polygon(p_list, check_intersect)
    
    @classmethod
    def get_min(self, pos1: Point, pos2: Point) -> Point:
        """入力座標の最小の座標値(x, y, z)を算出

        Args:
            pos1 (Point): 入力座標1
            pos2 (Point): 入力座標2

        Returns:
            Point: 最小座標値
        """
        pos_x = self.min(pos1.x, pos2.x)
        pos_y = self.min(pos1.y, pos2.y)
        pos_z = self.min(pos1.z, pos2.z)

        return Point(pos_x, pos_y, pos_z)

    @classmethod
    def get_max(self, pos1: Point, pos2: Point) -> Point:
        """入力座標の最大の座標値(x, y, z)を算出

        Args:
            pos1 (Point): 入力座標1
            pos2 (Point): 入力座標2

        Returns:
            Point: 最大座標値
        """
        pos_x = self.max(pos1.x, pos2.x)
        pos_y = self.max(pos1.y, pos2.y)
        pos_z = self.max(pos1.z, pos2.z)

        return Point(pos_x, pos_y, pos_z)

    @classmethod
    def get_minmax(self, p_list: list) -> tuple:
        """入力ポリゴンのバウンディングボックス座標を算出

        Args:
            p_list (list<Point>): 入力ポリゴン座標リスト

        Returns:
            tuple (Point, Point): バウンディングボックス
        """
        point_num = len(p_list)
        if point_num < 1:
            return Point(), Point()
        p_min = p_list[0]
        p_max = p_list[0]
        for i in range(1, point_num):
            p_min = self.get_min(p_list[i], p_min)
            p_max = self.get_max(p_list[i], p_max)
        
        return p_min, p_max

    @classmethod
    def is_same_point(self, pos1: Point, pos2: Point) -> bool:
        """2 頂点が同一座標かどうかを判定

        Args:
            pos1 (Point): 入力座標1
            pos2 (Point): 入力座標2

        Returns:
            bool: True: 同一座標, False: 同一座標でない
        """
        if GeoUtil.is_same_value(pos1.x, pos2.x) \
            and GeoUtil.is_same_value(pos1.y, pos2.y) \
            and GeoUtil.is_same_value(pos1.z, pos2.z):  
            return True
        return False

    @classmethod
    def is_zero(self, length: float) -> bool:
        if abs(length) < 0.01:
            return True
        return False
    
    @classmethod
    def is_in_triangle(self, normal: NDArray, pos: NDArray, triangle: list) \
            -> bool:
        """頂点が三角形内部にあるかどうかを判定

        Args:
            normal (NDArray): 三角形の法線
            pos (NDArray): 頂点座標
            triangle (list<Point>): 三角形頂点座標リスト

        Returns:
            bool: True: 内部にある, False: 内部にない
        """
        if len(triangle) != 3:
            return False
        
        plus_flag = False
        minus_flag = False
        for i in range(3):
            next_i = (i + 1) % 3
            vv = np.array([triangle[next_i].x,
                           triangle[next_i].y,
                           triangle[next_i].z]) \
                - np.array([triangle[i].x, triangle[i].y, triangle[i].z])
            vp = pos - np.array([triangle[i].x, triangle[i].y, triangle[i].z])
            vv = self.normalize(vv)
            vp = self.normalize(vp)

            cross = np.cross(vv, vp)
            cross = self.normalize(cross)
            dot = np.dot(normal, cross)
            if self.is_zero(dot):
                return False
            if dot >= 0.0:
                plus_flag = True
            else:
                minus_flag = True
        
        if plus_flag != minus_flag:
            return True
        
        return False

    @classmethod
    def is_cross_triangle(self, triangle_info1: list, triangle_info2: list,
                          cross_point_list: list) -> bool:
        """2つの三角形の交差判定
           交差ありの場合には交点座標を返す

        Args:
            triangle1 (list<TriangleInfo>): 三角形1
            triangle2 (list<TriangleInfo>): 三角形2
            cross_point_list (list<Point>): 交点格納先

        Returns:
            bool: True: 交差あり, False: 交差なし
        """

        triangle1 = triangle_info1.point_list
        triangle2 = triangle_info2.point_list
        len1 = len(triangle1)
        len2 = len(triangle2)

        if len1 != 3 or len2 != 3:
            return False

        # バウンディングボックス交差判定
        t1_min = triangle_info1.min
        t1_max = triangle_info1.max
        t2_min = triangle_info2.min
        t2_max = triangle_info2.max
        if (t2_max.x < t1_min.x or t2_max.y < t1_min.y
                or t2_max.z < t1_min.z or t1_max.x < t2_min.x
                or t1_max.y < t2_min.y or t1_max.z < t2_min.z):
            return False
        
        # 辺同士の接触
        for i in range(3):
            for j in range(3):
                if (self.is_same_point(triangle1[i], triangle2[j])
                    and self.is_same_point(triangle1[(i + 1) % 3],
                                           triangle2[(j + 1) % 3])):
                    return False
                if (self.is_same_point(triangle1[i], triangle2[(j + 1) % 3])
                    and self.is_same_point(triangle1[(i + 1) % 3],
                                           triangle2[j])):
                    return False

        if GeoUtil.is_zero_area(triangle1) or GeoUtil.is_zero_area(triangle2):
            return False
        
        # 交差判定
        v1_0 = np.array([triangle1[0].x, triangle1[0].y, triangle1[0].z])
        v1_01 = np.array([triangle1[1].x, triangle1[1].y, triangle1[1].z]) \
            - v1_0
        v1_02 = np.array([triangle1[2].x, triangle1[2].y, triangle1[2].z]) \
            - v1_0
        v1_01 = self.normalize(v1_01)
        v1_02 = self.normalize(v1_02)

        v1_normal = np.cross(v1_01, v1_02)
        v1_normal = self.normalize(v1_normal)

        # 平面との符号付き距離算出
        dist_list = [0.0, 0.0, 0.0]
        for i in range(3):
            v_p = np.array([triangle2[i].x, triangle2[i].y, triangle2[i].z]) \
                - v1_0
            dist = np.dot(v1_normal, v_p)
            if self.is_zero(dist):
                continue
            dist_list[i] = dist
        
        if (dist_list[0] >= 0
            and dist_list[1] >= 0
            and dist_list[2] >= 0) \
            or (dist_list[0] <= 0
                and dist_list[1] <= 0
                and dist_list[2] <= 0):
            # 各頂点が平面の片側にある
            return False

        # v1 平面との線分の交点が、三角形内かどうか判定
        for i in range(3):
            next_i = (i + 1) % 3
            if GeoUtil.float_is_zero(dist_list[i]) \
                or GeoUtil.float_is_zero(dist_list[next_i]):    # 平面上に存在
                continue
            if dist_list[i] * dist_list[next_i] > 0.0:         # 平面と交差しない
                continue

            p1 = np.array([triangle2[i].x, triangle2[i].y, triangle2[i].z])
            p2 = np.array([triangle2[next_i].x,
                           triangle2[next_i].y,
                           triangle2[next_i].z])

            if GeoUtil.float_is_zero(dist_list[next_i] - dist_list[i]):
                continue
            alpha = (-dist_list[i]) / (dist_list[next_i] - dist_list[i])
            cross_pos = (p2 - p1) * alpha + p1

            # 三角形の内側にあるかチェック
            if self.is_in_triangle(v1_normal, cross_pos, triangle1):
                ret_pos = Point(cross_pos[0], cross_pos[1], cross_pos[2])
                cross_point_list.append(ret_pos)
                return True
        
        return False

    @classmethod
    def is_on_same_line(self, p1: Point, p2: Point, q1: Point, q2: Point) \
            -> bool:
        """2つの線分が同一線上且つ逆方向かどうか判定

        Args:
            p1 (Point): 線分 1 の始点
            p2 (Point): 線分 1 の終点
            q1 (Point): 線分 2 の始点
            q2 (Point): 線分 2 の終点

        Returns:
            bool: True: 同一線上且つ逆方向, False: 同一線上且つ逆方向でない
        """
        angle1 = GeoUtil.interior_angle(p1, q1, p2)
        angle2 = GeoUtil.interior_angle(p1, q2, p2)
        # logger.debug(f'angle1 = {angle1}, angle2 ={angle2}')
        if (GeoUtil.is_same_value(angle1, math.pi)
                and GeoUtil.is_same_value(angle2, math.pi)) \
                or (GeoUtil.is_same_value(angle1, 0.0)
                    and GeoUtil.is_same_value(angle2, 0.0)) \
                or (GeoUtil.is_same_value(angle1, 0.0)
                    and GeoUtil.is_same_value(angle2, math.pi)) \
                or (GeoUtil.is_same_value(angle1, math.pi)
                    and GeoUtil.is_same_value(angle2, 0.0)):
            v1 = np.array([p2.x - p1.x, p2.y - p1.y, p2.z - p1.z])
            v2 = np.array([q2.x - q1.x, q2.y - q1.y, q2.z - q1.z])
            dot = GeoUtil.dot(v1, v2)
            # logger.debug(f'  angle3 = {math.acos(dot)}')
            if GeoUtil.is_same_value(math.acos(dot), math.pi):
                # ベクトルが逆方向 (なす角が 180°)
                return True
        
        return False
