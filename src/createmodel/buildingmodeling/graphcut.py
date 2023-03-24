# -*- coding:utf-8 -*-
import sys
import numpy as np
from typing import Tuple
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from maxflow.fastmin import aexpansion_grid
from numpy.typing import NDArray
from .clusterinfo import ClusterInfo
from ..createmodelexception import CreateModelException
from ..lasmanager import PointCloud
from ..message import CreateModelMessage


class GraphCut:
    def __init__(self) -> None:
        """コンストラクタ
        """
        pass

    def _get_distance_to_plane(self, points: NDArray, offset: NDArray,
                               alpha: float, beta: float, gamma: float):
        """各点と平面との距離

        Args:
            points (NDArray): 座標点列の配列
            offset (NDArray): 座標点のオフセット値
            alpha (float): 線形回帰の係数
            beta (float): 線形回帰の係数
            gamma (float): 線形回帰の切片

        Returns:
            NDArray: 各点と平面との距離の配列
        """
        move_points = points - offset

        # 平面の方程式1
        # z = alpha * x + beta * y + gamma
        a = alpha
        b = beta
        c = -1
        d = gamma

        # 平面の方程式2
        # ax + by + cz * d = 0
        # 点(x0, y0, z0)と平面の距離
        # |a*x0 + b*y0 + c*z0 * d| / sqrt(a^2 + b^2 + c^2)
        denominator = np.sqrt(a**2 + b**2 + c**2)
        nominator = np.abs(np.dot(move_points, np.array([a, b, c])) + d)
        dist = nominator / denominator

        return dist

    def _get_bbox(self, xy_points: NDArray):
        """バウンディングボックスの作成

        Args:
            xy_points (NDArray): 2次元座標点群

        Returns:
            タプル: 最小x値、最小y値、幅(x座標の幅)、高さ(y座標の幅)
        """
        min = np.min(xy_points, axis=0)
        max = np.max(xy_points, axis=0)
        width = max[0] - min[0] + 1
        height = max[1] - min[1] + 1

        return min[0], min[1], width, height

    def _check_ground_cluster(
            self, clusters: list[ClusterInfo],
            ground_height: float) -> list[bool]:
        """地面の確認

        Args:
            clusters (list[ClusterInfo]): クラスタ情報のリスト
            ground_height (float): 地面の高さm

        Returns:
            list[bool]: 判定結果のリスト(True:地面クラスタ, False:屋根クラスタ)
        """

        # 候補面のポリゴン
        candidate_polys = []
        for cluster in clusters:
            candidate_polys.append(cluster.get_contours())

        # 候補面の面積と外形長さ
        candidate_area = []
        candidate_length = []
        for polys in candidate_polys:
            area = 0.0
            length = 0.0
            for poly in polys:
                area += poly.area
                length += poly.length
            candidate_area.append(area)
            candidate_length.append(length)

        # 候補面の平均幅
        candidate_width = []
        for i, polys in enumerate(candidate_polys):
            if len(polys):
                candidate_width.append(candidate_area[i] / candidate_length[i])
            else:
                candidate_width.append(0.0)

        # 候補面の包括関係
        CANDIDATE_WITHIN_RATE = 0.9

        candidate_contains = []
        for i, polys_i in enumerate(candidate_polys):
            # 候補面iに含まれる候補面を検索
            contains = []
            for j, polys_j in enumerate(candidate_polys):
                if i == j:
                    continue
                if len(polys_j) == 0:
                    continue

                # 候補面iとjとの重畳面積
                intersect_area = 0.0
                for poly_j in polys_j:
                    for poly_i in polys_i:
                        intersect_area += poly_j.intersection(poly_i).area
                # 候補面jは候補面iに含まれているか
                if intersect_area / candidate_area[j] > CANDIDATE_WITHIN_RATE:
                    contains.append(j)
            candidate_contains.append(contains)

        # 候補面は地面点か判定
        CANDIDATE_AREA_TH = 20
        CANDIDATE_WIDTH_TH = 1.0

        candidate_is_ground = []
        for i, cluster in enumerate(clusters):
            # 候補面のポリゴン作成失敗の場合は地面点とする
            if len(candidate_polys[i]) == 0:
                candidate_is_ground.append(True)
                continue

            cluster_height = cluster.points.get_points()[:, 2].mean()
            if cluster_height < ground_height:
                # 候補面iは他の候補面に含まれるか
                is_contained = False
                for contains in candidate_contains:
                    if i in contains:
                        is_contained = True
                        break
                # 候補面iは他の候補面に含まれる
                # また面積が小さい
                # また平均幅が小さい場合は地面点とする
                if (is_contained or candidate_area[i] < CANDIDATE_AREA_TH
                        or candidate_width[i] < CANDIDATE_WIDTH_TH):
                    candidate_is_ground.append(True)
                else:
                    candidate_is_ground.append(False)
            else:
                candidate_is_ground.append(False)

        # ##################### 暫定対応 #####################
        roofs = [i for i in range(len(candidate_is_ground))
                 if candidate_is_ground[i] is False]
        if len(roofs) == 0:
            # 屋根が1つもない場合
            # 面積が最大のものを屋根とする
            index = candidate_area.index(max(candidate_area))
            candidate_is_ground[index] = False
        # ###################################################

        return candidate_is_ground

    def _separate_points(
            self, src_points: NDArray,
            clusters: list[ClusterInfo],
            ground_height: float) -> Tuple[PointCloud, PointCloud, list[bool]]:
        """地面点と屋根点の分割

        Args:
            src_points (NDArray): 建物点群
            clusters (list[ClusterInfo]): クラスタリング情報のリスト
            ground_height (float): 地面の高さm

        Returns:
            Tuple[PointCloud, PointCloud, list[bool]]:
                PointCloud: 建物点群
                PointCloud: 地面点群
                list[bool]: クラスタの地面判定結果
        """

        # 地面クラスタの確認
        ground_flags = self._check_ground_cluster(
            clusters=clusters, ground_height=ground_height)

        # 地面点と屋根点の分割
        building_points_ind = np.arange(len(src_points))
        cluster_points_ind = []
        for cluster in clusters:
            cluster_points_ind.extend(cluster.points.index)
        cluster_points_ind = np.array(cluster_points_ind)
        # クラスタリングされなかった建物点群のインデックス
        other_points_ind = np.setdiff1d(
            building_points_ind, cluster_points_ind)

        # 地面の高さより低い位置にある点群
        ground_points_ind = other_points_ind[
            src_points[other_points_ind, 2] < ground_height]
        for i, cluster in enumerate(clusters):
            if ground_flags[i]:
                # 地面判定されているクラスタの場合
                ground_points_ind = np.concatenate(
                    [ground_points_ind, np.array(cluster.points.index)])
        roof_points_ind = np.setdiff1d(building_points_ind, ground_points_ind)

        ground_cloud = PointCloud()
        ground_cloud.add_points(src_points[ground_points_ind])
        ground_cloud.index = ground_points_ind
        roof_cloud = PointCloud()
        roof_cloud.add_points(src_points[roof_points_ind])
        roof_cloud.index = roof_points_ind

        return roof_cloud, ground_cloud, ground_flags

    def graph_cut(
            self, src_points: NDArray,
            src_clusters: list[ClusterInfo],
            ground_height: float,
            grid_size=0.25,
            smooth_weight=2.0,
            invalid_point_dist=99.0,
            height_diff_th=2.0,
            dbscan_eps=0.5,
            dbscan_point_th=5) -> list[ClusterInfo]:
        """Graph Cut

        Args:
            src_points (NDArray): 建物点群
            src_clusters (list[ClusterInfo]): クラスタリング情報
            ground_height (float): 地面の高さm
            grid_size (float): 解像度m
            smooth_weight (float, optional): 平滑化の重み. Defaults to 2.0.
            invalid_point_dist (float, optional): \
                無効点とする距離m. Defaults to 99.0.
            height_diff_th (float, optional): 高さ差分閾値. Defaults to 2.0.
            dbscan_eps (float, optional): dbscanの探索半径. Defaults to 0.5.
            dbscan_point_th (int, optional): \
                dbscanのcore点判定用の点数閾値. Defaults to 5.
        Raises:
            CreateModelException: 点群に対応するオルソ画像がない場合

        Returns:
            list[ClusterInfo]: Graph Cutによるクラスタリング結果
        """
        class_name = self.__class__.__name__
        func_name = sys._getframe().f_code.co_name

        ######################
        # 地面点と屋根点分割
        ######################
        roof_cloud, ground_cloud, ground_flags = self._separate_points(
            src_points=src_points, clusters=src_clusters,
            ground_height=ground_height)

        if len(roof_cloud.get_points()) == 0:
            # 屋根点が取得でき無かった場合
            msg = '{}.{}, {}'.format(
                class_name, func_name,
                CreateModelMessage.ERR_MSG_GRAPHCUT_NO_ROOF_POINTS)
            raise CreateModelException(msg)

        #####################
        # 屋根点処理
        #####################
        #
        # 距離：
        #          無効点ラベル     屋根候補面ラベル
        # 無効点    0               ∞
        # 屋根点    ∞               屋根候補面への鉛直距離
        #
        # 初期ラベル：
        #   屋根候補面点：屋根候補面のラベル
        #   屋根候補面以外の屋根点：無効点ラベル
        #   無効点：無効点ラベル

        # 座標の最小値
        offset = np.min(src_points, axis=0)

        # 屋根点と屋根候補面との距離
        dists = []
        for i, cluster in enumerate(src_clusters):
            if ground_flags[i]:
                continue    # 地面判定の場合

            # 屋根の高さ
            cluster_height = cluster.points.get_points()[:, 2].mean()

            # 屋根候補面
            alpha = 0.0
            beta = 0.0
            gamma = cluster_height - offset[2]

            # 屋根点と屋根候補面との距離
            d = self._get_distance_to_plane(
                points=roof_cloud.get_points(), offset=offset,
                alpha=alpha, beta=beta, gamma=gamma)
            dists.append(d)
        dists = np.array(dists)

        # ラベル数（無効ラベルを含まない数、無効ラベル番号：n_labels）
        n_labels = len(dists)

        # 屋根点の初期ラベルの作成（非屋根候補面点の初期ラベル：無効ラベル）
        labels = np.full(len(src_points), n_labels, int)
        cnt = 0
        for i, cluster in enumerate(src_clusters):
            if ground_flags[i]:
                continue    # 地面判定の場合

            labels[cluster.points.index] = cnt
            cnt += 1
        labels = labels[roof_cloud.index]

        # 画像座標に変換
        xy_points = roof_cloud.get_points()[:, 0:2].copy()

        x_min = np.floor(np.min(xy_points[:, 0]) / grid_size) * grid_size
        y_max = np.ceil(np.max(xy_points[:, 1]) / grid_size) * grid_size
        xy_points[:, 0] = xy_points[:, 0] - x_min
        xy_points[:, 0] = xy_points[:, 0] / grid_size
        xy_points[:, 1] = xy_points[:, 1] - y_max
        xy_points[:, 1] = xy_points[:, 1] / (-grid_size)

        xy_points = np.round(xy_points).astype(np.int)

        tlx, tly, range_x, range_y = self._get_bbox(xy_points=xy_points)
        xs = xy_points[:, 0] - tlx
        ys = xy_points[:, 1] - tly

        # label画像
        mask = np.zeros((range_y, range_x), np.uint8)
        mask[ys, xs] = 255

        label_image = np.full((range_y, range_x), n_labels, int)
        label_image[ys, xs] = labels

        # data term行列
        #          無効点ラベル(n_labels)   屋根候補面ラベル
        # 無効点    0                       ∞
        # 屋根点    ∞                       屋根候補面への鉛直距離
        D = np.zeros((range_y, range_x, n_labels + 1), np.double)
        for i in range(n_labels):
            D[ys, xs, i] = dists[i]
        D[ys, xs, n_labels] = (np.ones(len(roof_cloud.get_points()))
                               * invalid_point_dist)

        invalid_ys, invalid_xs = np.nonzero(mask == 0)
        for i in range(n_labels):
            D[invalid_ys, invalid_xs, i] = invalid_point_dist

        # smoothness term行列
        V = (1 - np.eye(n_labels + 1)) * smooth_weight

        # α拡張
        labels = aexpansion_grid(D, V, max_cycles=100000, labels=label_image)
        roof_labels = labels[ys, xs]

        ##################
        # 高さでマージし、水平連続性で分割する
        ##################
        label_heights = []
        for i, cluster in enumerate(src_clusters):
            if ground_flags[i]:
                continue    # 地面判定の場合
            cluster_height = cluster.points.get_points()[:, 2].mean()
            label_heights.append(cluster_height)
        label_heights = np.array(label_heights)

        label_height_delta = np.zeros((n_labels, n_labels), float)
        for i in range(n_labels):
            height_i = label_heights[i]
            for j in range(n_labels):
                if i == j:
                    continue
                height_j = label_heights[j]
                label_height_delta[i, j] = abs(height_i - height_j)

        query_labels = list(range(n_labels))
        merged_labels = []
        while len(query_labels) > 0:
            i = query_labels.pop(0)
            merged_labels_ = [i]
            query_labels_copy = query_labels.copy()
            for j in query_labels_copy:
                height_delta_ij = label_height_delta[i, j]
                if height_delta_ij < height_diff_th:
                    merged_labels_.append(j)
                    query_labels.remove(j)
            merged_labels.append(merged_labels_)

        new_roof_points = []
        new_roof_labels = []
        new_label_heights = []
        cur_label = 0
        for merged_labels_ in merged_labels:
            merged_points_xyz = []
            merged_points_label = []
            for i in merged_labels_:
                inds = roof_labels == i
                if np.sum(inds) == 0:
                    continue
                merged_points_xyz += roof_cloud.get_points()[inds].tolist()
                merged_points_label += [i] * len(roof_cloud.get_points()[inds])
            if len(merged_points_xyz) == 0:
                continue
            merged_points_xyz = np.array(merged_points_xyz)
            merged_points_label = np.array(merged_points_label)

            db = DBSCAN(
                eps=dbscan_eps,
                min_samples=dbscan_point_th).fit(merged_points_xyz[:, 0:2])
            db_labels = db.labels_
            for i in range(db_labels.max() + 1):
                inds = db_labels == i
                new_roof_points += merged_points_xyz[inds].tolist()
                new_roof_labels += [cur_label] * len(merged_points_xyz[inds])
                new_label_heights.append(
                    np.mean(
                        label_heights[np.unique(merged_points_label[inds])]))
                cur_label += 1
        
        if len(new_roof_points) == 0:
            # 屋根点が取得できなかった場合
            msg = '{}.{}, {}'.format(
                class_name, func_name,
                CreateModelMessage.ERR_MSG_GRAPHCUT_NO_MERGED_ROOF_POINTS)
            raise CreateModelException(msg)

        roof_points = np.array(new_roof_points)
        roof_labels = np.array(new_roof_labels)
        label_heights = np.array(new_label_heights)
        n_labels = cur_label

        if len(ground_cloud.get_points()) > 0:
            #####################
            # 地面点処理
            #####################
            nn = NearestNeighbors(
                n_neighbors=1, algorithm='kd_tree', leaf_size=10, n_jobs=8)
            nn.fit(roof_points[:, 0:2])
            inds = nn.kneighbors(
                ground_cloud.get_points()[:, 0:2], return_distance=False)[:, 0]
            ground_labels = roof_labels[inds]

            #####################
            # クラスタ作成
            #####################
            building_points = np.r_[roof_points, ground_cloud.get_points()]
            building_labels = np.r_[roof_labels, ground_labels]
        else:
            # 地面点が無い場合
            building_points = roof_points
            building_labels = roof_labels

        gc_clusters = []
        for i in range(n_labels):
            inds = building_labels == i
            points = building_points[inds]
            if len(points) == 0:
                continue

            pc = PointCloud()
            pc.add_points(points)
            pc.index = inds

            cluster = ClusterInfo()
            cluster.points = pc
            cluster.roof_height = label_heights[i]
            gc_clusters.append(cluster)

        # ソート
        gc_clusters.sort(reverse=True)

        # id設定
        for i in range(len(gc_clusters)):
            gc_clusters[i].id = i

        return gc_clusters
