# -*- coding:utf-8 -*-
import sys
import numpy as np
import shapely.geometry as geo
from sklearn.cluster import DBSCAN, MeanShift
from sklearn.neighbors import NearestNeighbors
from jakteristics import compute_features
from .clusterinfo import ClusterInfo
from ..lasmanager import PointCloud
from .graphcut import GraphCut
from .mbr import MBR
from ..param import CreateModelParam
from ..message import CreateModelMessage
from ..createmodelexception import CreateModelException


class Preprocess:
    """建物点群取得後から、モデル化前に行う前処理クラス
    """
    def __init__(self) -> None:
        """コンストラクタ
        """
        self._XYZ = 'xyz'
        self._RGB = 'rgb'
        self._IND = 'ind'
        pass

    def _height_clustering(
            self, cloud: PointCloud, z_band_width=0.5) -> list:
        """高さによるクラスタリング

        Args:
            cloud (PointCloud): 点群
            z_band_width (float, optional): 高さのバンド幅. Defaults to 0.5.

        Returns:
            list: クラスタリング結果のリスト
        """
        # 高さでクラスタリング(MeanShift)
        zs = cloud.get_points()[:, 2]
        z_min = np.min(zs)
        zs = zs - z_min

        clustering = MeanShift(
            bandwidth=z_band_width,
            bin_seeding=True, n_jobs=8).fit(zs[..., np.newaxis])
        labels = clustering.labels_

        hier_points = []
        for label in np.unique(labels):
            inds = labels == label
            hier_points.append({
                self._XYZ: cloud.get_points()[inds],
                self._RGB: cloud.get_colors()[inds],
                self._IND: cloud.index[inds]})
        
        return hier_points

    def _color_clustering(
            self, hier_points: list, rgb_band_width=20) -> list:
        """色によるクラスタリング

        Args:
            hier_points (list): 点群のリスト
            rgb_band_width (int, optional): 色のバンド幅. Defaults to 20.

        Returns:
            list: クラスタリング結果のリスト
        """
        # 色クラスタリング（MeanShift）★Option：色+法線領域拡張/色領域拡張
        hier_cluster_points = []
        for hier_points_ in hier_points:
            clustering = MeanShift(
                bandwidth=rgb_band_width, bin_seeding=True,
                n_jobs=8).fit(hier_points_[self._RGB] / 256)
            labels = clustering.labels_

            for label in np.unique(labels):
                inds = labels == label
                hier_cluster_points.append({
                    self._XYZ: hier_points_[self._XYZ][inds],
                    self._RGB: hier_points_[self._RGB][inds],
                    self._IND: hier_points_[self._IND][inds]})

        return hier_cluster_points

    def _dbscan(self, hier_clusters: list,
                eps=0.5, point_th=5, n=20) -> list:
        """DBSCAN

        Args:
            hier_clusters (list): 点群のリスト
            eps (float, optional): dbscanの探索半径. Defaults to 0.5.
            point_th (int, optional): core点判定用の点数閾値. Defaults to 5.
            n (int, optional): 最小クラスタ点数閾値. Defaults to 20.

        Returns:
            list: クラスタリング結果のリスト
        """
        new_hier_cluster_points = []
        for i in range(len(hier_clusters)):
            db = DBSCAN(
                eps=eps, min_samples=point_th).fit(hier_clusters[i][self._XYZ])
            labels = db.labels_

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            for ci in range(n_clusters):
                inds = np.where(labels == ci)[0]
                if len(inds) < n:
                    continue
    
                new_hier_cluster_points.append({
                    self._XYZ: hier_clusters[i][self._XYZ][inds],
                    self._RGB: hier_clusters[i][self._RGB][inds],
                    self._IND: hier_clusters[i][self._IND][inds]})

        return new_hier_cluster_points

    def _search_cluster_close_to_polygon(
            self, hier_clusters: list,
            shape: geo.Polygon,
            sample_step=0.25,
            search_radius=1.0) -> list:
        """点群近傍に存在する建物外形ポリゴン辺の探索

        Args:
            hier_clusters (list): クラスタ点群のリスト
            shape (geo.Polygon): 建物外形ポリゴン
            sample_step (float, optional): \
                建物外形ポリゴン辺のサンプリング間隔m. Defaults to 0.25.
            search_radius (float, optional): \
                近傍探索時の探索半径m. Defaults to 1.0.

        Returns:
            list: クラスタリング結果のリスト
        """
        # 外形ポリゴンに近いクラスタを探索する
        shape_xy = np.array(shape.exterior.coords).copy()

        # 建物外形線のサンプリング
        footprint_points_xy = []
        for i in range(len(shape_xy) - 1):
            point_i = shape_xy[i]
            point_j = shape_xy[i + 1]
            length = np.linalg.norm(point_i - point_j)
            n_points = round(length / sample_step)
            xs = np.linspace(point_i[0], point_j[0], n_points)
            ys = np.linspace(point_i[1], point_j[1], n_points)
            xy = np.c_[xs[..., np.newaxis], ys[..., np.newaxis]]
            footprint_points_xy += xy.tolist()

        # 建物外形線の近傍点（クラスタリングした階層点群）を探索
        points_xyz = []     # クラスタリングした階層点群格納
        points_hier_cluster_id = []
        for i, hier_cluster in enumerate(hier_clusters):
            points_xyz += hier_cluster[self._XYZ].tolist()
            points_hier_cluster_id += [i] * len(hier_cluster[self._XYZ])
        points_xyz = np.array(points_xyz)
        points_hier_cluster_id = np.array(points_hier_cluster_id)

        hier_clusters_fpnn = []
        if len(points_xyz) > 0:
            nn = NearestNeighbors(
                radius=search_radius,
                algorithm='ball_tree', leaf_size=10, n_jobs=8)
            nn.fit(points_xyz[:, 0:2])
            inds = nn.radius_neighbors(
                footprint_points_xy, return_distance=False)

            # 建物外形に近隣する階層
            fpnn_hier_cluster_ids = []
            for inds_ in inds:
                fpnn_hier_cluster_ids += points_hier_cluster_id[inds_].tolist()
            fpnn_hier_cluster_ids = np.unique(fpnn_hier_cluster_ids)

            for i, _ in enumerate(hier_clusters):
                if i in fpnn_hier_cluster_ids:
                    hier_clusters[i]["is_fpnn"] = True
                else:
                    hier_clusters[i]["is_fpnn"] = False

            for i, hier_cluster in enumerate(hier_clusters):
                if hier_cluster["is_fpnn"]:
                    hier_clusters_fpnn.append({
                        self._XYZ: hier_cluster[self._XYZ],
                        self._RGB: hier_cluster[self._RGB],
                        self._IND: hier_cluster[self._IND]})

        return hier_clusters_fpnn

    def _merge(self, hier_clusters_fpnn: list, height_th=2.0,
               dbscan_eps=0.5, dbscan_point_th=5) -> list:
        """高さが近く水平連続のクラスタをマージする

        Args:
            hier_clusters_fpnn (list): クラスタのリスト
            height_th (float, optional): 高さ差分閾値m. Defaults to 2.0.
            dbscan_eps (float, optional): dbscanの探索半径. Defaults to 0.5.
            dbscan_point_th (int, optional): \
                core点判定用の点数閾値. Defaults to 5.

        Returns:
            list: クラスタリング結果のリスト
        """
        # 高さが近く、水平連続のクラスタをマージする
        n_clusters = len(hier_clusters_fpnn)
        clusters_height_delta = np.zeros((n_clusters, n_clusters), float)
        for i in range(n_clusters):
            height_i = np.mean(hier_clusters_fpnn[i][self._XYZ][:, 2])
            for j in range(n_clusters):
                if i == j:
                    continue
                height_j = np.mean(hier_clusters_fpnn[j][self._XYZ][:, 2])
                clusters_height_delta[i, j] = abs(height_i - height_j)

        query_clusters = list(range(n_clusters))
        merged_clusters = []
        while len(query_clusters) > 0:
            i = query_clusters.pop(0)
            merged_clusters_ = [i]
            query_clusters_copy = query_clusters.copy()
            for j in query_clusters_copy:
                height_delta_ij = clusters_height_delta[i, j]
                if height_delta_ij < height_th:
                    # クラスタの平均高さの差分が閾値未満の場合はマージする
                    merged_clusters_.append(j)
                    query_clusters.remove(j)
            merged_clusters.append(merged_clusters_)

        new_hier_clusters_fpnn = []
        for merged_clusters_ in merged_clusters:
            points_xyz = []
            points_rgb = []
            points_ind = []
            for i in merged_clusters_:
                points_xyz += hier_clusters_fpnn[i][self._XYZ].tolist()
                points_rgb += hier_clusters_fpnn[i][self._RGB].tolist()
                points_ind += hier_clusters_fpnn[i][self._IND].tolist()
            points_xyz = np.array(points_xyz)
            points_rgb = np.array(points_rgb)
            points_ind = np.array(points_ind)

            # マージ対象の点に対してDBSCAN
            db = DBSCAN(
                eps=dbscan_eps,
                min_samples=dbscan_point_th).fit(points_xyz[:, 0:2])
            db_labels = db.labels_
            for i in range(db_labels.max() + 1):
                inds = (db_labels == i)
                new_hier_clusters_fpnn.append({
                    self._XYZ: points_xyz[inds],
                    self._RGB: points_rgb[inds],
                    self._IND: points_ind[inds]})
        
        return new_hier_clusters_fpnn

    def preprocess(
            self, cloud: PointCloud,
            shape: geo.Polygon,
            ground_height: float,
            grid_size: float) -> list[ClusterInfo]:
        """前処理

        Args:
            cloud (PointCloud): 建物点群
            shape (geo.Polygon): 建物外形ポリゴン
            ground_height (float): 地面の高さm
            grid_size (float): 解像度m

        Returns:
            list[ClusterInfo]: 点群クラスタリスト
        
        Note:
            建物点群を屋根ごとの点群に分割し、\
            分割した点群を基に屋根の形状ポリゴンを作成する
        """
        class_name = self.__class__.__name__
        func_name = sys._getframe().f_code.co_name
        param = CreateModelParam.get_instance()

        # 特徴量計算
        feature_names = ['planarity', 'linearity', 'verticality']
        features = compute_features(
            cloud.get_points(),
            search_radius=param.verticality_search_radius,
            feature_names=feature_names)

        verticality = features[:, 2]    # 高さ方向特徴量
        vert_inds = verticality < param.verticality_th
        input_cloud = PointCloud()
        input_cloud.add_points(cloud.get_points()[vert_inds])
        input_cloud.add_colors(cloud.get_colors()[vert_inds])
        building_points_ind = np.arange(len(cloud.get_points()))
        input_cloud.index = building_points_ind[vert_inds]

        # 高さでクラスタリング
        hier_clusters = self._height_clustering(
            cloud=input_cloud, z_band_width=param.height_band_width)
        
        # 色でクラスタリング
        color_clusters = self._color_clustering(
            hier_points=hier_clusters, rgb_band_width=param.color_band_width)

        # 連続性クラスタリング
        dbscan_clusters = self._dbscan(
            hier_clusters=color_clusters,
            eps=param.dbscan_search_radius,
            point_th=param.dbscan_point_th,
            n=param.dbscan_cluster_point_th)

        # 外形ポリゴンに近いクラスタを探索
        ex_clusters = self._search_cluster_close_to_polygon(
            hier_clusters=dbscan_clusters, shape=shape,
            sample_step=param.search_near_polygon_sample_step,
            search_radius=param.search_near_polygon_search_radius)

        # 高さが近く、水平連続のクラスタをマージ
        merge_clusters = self._merge(
            hier_clusters_fpnn=ex_clusters,
            height_th=param.merge_height_diff_th,
            dbscan_eps=param.merge_dbscan_radius,
            dbscan_point_th=param.merge_dbscan_point_th)

        # 屋根面推定
        clusters: list[ClusterInfo] = []
        for i, cluster in enumerate(merge_clusters):
            points = PointCloud()
            points.add_points(cluster[self._XYZ])
            points.add_colors(cluster[self._RGB])
            points.index = cluster[self._IND]
            info = ClusterInfo(points=points)
            info.id = i
            clusters.append(info)

        if len(clusters) == 0:
            # 屋根クラスタの取得に失敗
            msg = '{}.{}, {}'.format(
                class_name, func_name,
                CreateModelMessage.ERR_MSG_PREPROC_NO_ROOF_CLUSTER)
            raise CreateModelException(msg)

        graphcut = GraphCut()
        gc_clusters = graphcut.graph_cut(
            src_points=cloud.get_points(),
            src_clusters=clusters,
            ground_height=ground_height + param.graphcut_height_offset,
            grid_size=grid_size,
            smooth_weight=param.graphcut_smooth_weight,
            invalid_point_dist=param.graphcut_invalid_point_dist,
            height_diff_th=param.graphcut_height_diff_th,
            dbscan_eps=param.graphcut_dbscan_radius,
            dbscan_point_th=param.graphcut_dbscan_point_th)

        # MBR
        mbr = MBR()
        mbr_clusters = mbr.execute(
            src_clusters=gc_clusters,
            shape=shape,
            grid_size=grid_size,
            sampling_step=param.mbr_sampling_step,
            neighbor_jobs=param.mbr_neighbor_jobs,
            mean_shift_jobs=param.mbr_angle_ms_jobs,
            angle_ms_bandwidth=param.mbr_angle_ms_bandwidth,
            neightbor_max_dist=param._mbr_neighbor_max_dist,
            roof_angle_ortho_th=param._mbr_roof_angle_ortho_th,
            line_length_th=param._mbr_line_length_th,
            valid_pixel_num=param.mbr_valid_pixel_num,
            width_th=param.mbr_width_th,
            slim_rate_th=param.mbr_slim_rate_th,
            max_hiers=param.mbr_max_hiers)

        # ソート
        mbr_clusters.sort(reverse=True)
        for i in np.arange(len(mbr_clusters)):
            mbr_clusters[i].id = i

        return mbr_clusters
