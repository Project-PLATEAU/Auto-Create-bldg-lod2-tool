# -*- coding:utf-8 -*-
from os.path import join, dirname
from .createmodelexception import CreateModelException


class CreateModelParam():
    """モデル要素生成モジュールのパラメータクラス(シングルトンクラス)

    Note:
        インスタンスの取得は get_instance() で取得すること.
        ユニークインスタンスが生成済みの状態でコンストラクタを呼び出すと、
        例外を発する
    """
    # クラス変数
    _instance = None
    """ユニークインスタンス
    """

    def _set_init_param(self):
        """パラメータ初期化関数
        """
        # 地面探索
        self._ground_search_dist = 1.0           # 地面探索範囲

        # 高さ方向特徴量
        self._verticality_search_radius = 2.0    # 高さ特徴量探索範囲
        self._verticality_th = 0.15              # 高さ特徴量閾値

        # 平面推定
        # 高さによるクラスタリング
        self._height_band_width = 0.5            # バンド幅
        # 色によるクラスタリング
        self._color_band_width = 20              # バンド幅
        # 連続性クラスタリング
        self._dbscan_search_radius = 0.5         # 探索半径
        self._dbscan_point_th = 5                # コア点判定用の点数閾値
        self._dbscan_cluster_point_th = 20       # 採用クラスタの最小点数
        # 外形ポリゴンに近いクラスタ探索処理
        # 外形ポリゴンのサンプリング間隔
        self._search_near_polygon_sample_step = 0.25
        self._search_near_polygon_search_radius = 1.0    # 探索半径
        # 高さが近く水平連続となるクラスタのマージ
        self._merge_height_diff_th = 2.0         # クラスタの平均高さの差分閾値
        self._merge_dbscan_radius = 0.5          # 探索半径
        self._merge_dbscan_point_th = 5          # コア点判定用の点数閾値
        # GraphCut
        self._graphcut_height_offset = 4.0       # 地面高さのoffset値
        #self._graphcut_smooth_weight = 25.0          # 平滑化の重み
        self._graphcut_smooth_weight = 20.0          # 平滑化の重み
        self._graphcut_invalid_point_dist = 100.0    # 無効点とする距離閾値
        self._graphcut_height_diff_th = 2.0      # 高さ差分閾値
        self._graphcut_dbscan_radius = 0.5       # dbscanの探索半径
        self._graphcut_dbscan_point_th = 5       # dbscanのコア点判定用の点数閾値

        # MBR
        # 建物外形ポリゴン辺のサンプリング間隔
        self._mbr_sampling_step = 0.25
        # 近傍建物外形線探索用の最大近傍距離閾値
        self._mbr_neighbor_max_dist = 0.5
        # 近傍建物外形線探索時のNearestNeighborのジョブ数
        self._mbr_neighbor_jobs = 8
        # 短い建物外形線分を除外する際に距離閾値
        self._mbr_line_length_th = 1.0
        # 近傍建物外形線の状態判定用の角度閾値deg
        self._mbr_roof_angle_ortho_th = 5.0
        # 建物外形線間の角度に対するMeanShiftのバンド幅
        self._mbr_angle_ms_bandwidth = 5.0
        # 建物外形線間の角度に対するMeanShiftのジョブ数
        self._mbr_angle_ms_jobs = 8
        self._mbr_valid_pixel_num = 25           # 有効矩形のpixel数
        self._mbr_width_th = 5.0                 # 幅閾値pix
        self._mbr_slim_rate_th = 10.0            # 細長度
        self._mbr_max_hiers = 100                # 最大階層数

        # 屋根形状補正
        # 屋根形状の簡略化閾値
        self._simplify_roof_th = 0.5
        # ノイズ除去時の収縮処理のバッファサイズ
        self._noise_canceling_buffer1 = -0.25
        self._noise_canceling_buffer2 = 0.5
        self._noise_canceling_buffer3 = -0.25
        self._noise_canceling_buffer4 = 0.01

        # モデル面作成準備
        # エッジグルーピング用角度(deg)
        self._surface_preparation_angle_th = 3.0
        # エッジサンプリング用ステップ幅(m)
        self._surface_preparation_sampling_step = 0.01
        # エッジグルーピング用距離(m)
        self._surface_preparation_dist_th = 0.1

        # モデル補正
        # ソリッド閉じ対応
        self._solid_search_edge_th = 0.001       # 辺探索用の距離閾値
        # 頂点マージ
        self._model_point_merge_xy_dist = 0.01   # xy平面上でのマージ距離
        self._model_point_merge_z_reso = 0.01     # 高さ方向解像度

        # モデルの表面の順列方向
        self._front_is_ccw = True    # 反時計回りが表

        # 学習データ(建物分類、非陸屋根モデル様)
        dir_name = dirname(__file__)
        self._classifier_checkpoint_path = join(
            dir_name, 'data', 'classifier_parameter.pkl')
        self._balcony_segmentation_checkpoint_path = join(
            dir_name, 'data', 'balcony_segmentation_parameter.pkl')
        self._roof_edge_detection_checkpoint_path = join(
            dir_name, 'data', 'roof_edge_detection_parameter.pth')

        # GPU使用フラグ
        self._use_gpu = True

    def __init__(self) -> None:
        """コンストラクタ

        Raises:
            CreateModelException:
                2回目以降のコンストラクタ呼び出し時(シングルトンのため)
        """
        if CreateModelParam._instance is not None:
            raise CreateModelException(
                'CreateModelParam Class is Singleton class. \
                If you get instance, you shoud use get_instance() method.')
        else:
            CreateModelParam._instance = self
            self._set_init_param()  # 初期パラメータ設定

    @staticmethod
    def get_instance():
        """インスタンス取得

        Returns:
            CreateModelParam: インスタンス
        """
        if CreateModelParam._instance is None:
            CreateModelParam()

        return CreateModelParam._instance

    # プロパティ
    @property
    def ground_search_dist(self) -> float:
        """地面探索範囲

        Returns:
            float: 地面探索範囲
        """
        return self._ground_search_dist

    @ground_search_dist.setter
    def ground_search_dist(self, value: float):
        """地面探索範囲

        Args:
            value (float): 地面探索範囲
        """
        self._ground_search_dist = value

    @property
    def verticality_search_radius(self) -> float:
        """高さ特徴量探索範囲

        Returns:
            float: 高さ特徴量探索範囲
        """
        return self._verticality_search_radius

    @verticality_search_radius.setter
    def verticality_search_radius(self, value: float):
        """高さ特徴量探索範囲

        Args:
            value (float): 高さ特徴量探索範囲
        """
        self._verticality_search_radius = value

    @property
    def verticality_th(self) -> float:
        """高さ方向特徴量閾値

        Returns:
            float: 高さ方向特徴量閾値
        """
        return self._verticality_th

    @verticality_th.setter
    def verticality_th(self, value: float):
        """高さ方向特徴量閾値

        Args:
            value (float): 高さ方向特徴量閾値
        """
        self._verticality_th = value

    @property
    def height_band_width(self) -> float:
        """高さによるクラスタリングのバンド幅

        Returns:
            float: 高さによるクラスタリングのバンド幅
        """
        return self._height_band_width

    @height_band_width.setter
    def height_band_width(self, value: float):
        """高さによるクラスタリングのバンド幅

        Args:
            value (float): 高さによるクラスタリングのバンド幅
        """
        self._height_band_width = value

    @property
    def color_band_width(self) -> int:
        """色によるクラスタリングのバンド幅

        Returns:
            int: 色によるクラスタリングのバンド幅
        """
        return self._color_band_width

    @color_band_width.setter
    def color_band_width(self, value: int):
        """色によるクラスタリングのバンド幅

        Args:
            value (int): 色によるクラスタリングのバンド幅
        """
        self._color_band_width = value

    @property
    def dbscan_search_radius(self) -> float:
        """連続性クラスタリングの探索半径

        Returns:
            float: 連続性クラスタリングの探索半径
        """
        return self._dbscan_search_radius

    @dbscan_search_radius.setter
    def dbscan_search_radius(self, value: float):
        """連続性クラスタリングの探索半径

        Args:
            value (float): 連続性クラスタリングの探索半径
        """
        self._dbscan_search_radius = value

    @property
    def dbscan_point_th(self) -> int:
        """連続性クラスタリングのコア点判定用の点数閾値

        Returns:
            int: 連続性クラスタリングのコア点判定用の点数閾値
        """
        return self._dbscan_point_th

    @dbscan_point_th.setter
    def dbscan_point_th(self, value: int):
        """連続性クラスタリングのコア点判定用の点数閾値

        Args:
            value (int): 連続性クラスタリングのコア点判定用の点数閾値
        """
        self._dbscan_point_th = value

    @property
    def dbscan_cluster_point_th(self) -> float:
        """連続性クラスタリングの採用クラスタの最小点数

        Returns:
            float: 連続性クラスタリングの採用クラスタの最小点数
        """
        return self._dbscan_cluster_point_th

    @dbscan_cluster_point_th.setter
    def dbscan_cluster_point_th(self, value: float):
        """連続性クラスタリングの採用クラスタの最小点数

        Args:
            value (float): 連続性クラスタリングの採用クラスタの最小点数
        """
        self._dbscan_cluster_point_th = value

    @property
    def search_near_polygon_sample_step(self) -> float:
        """外形ポリゴンに近いクラスタ探索処理の外形ポリゴンのサンプリング間隔

        Returns:
            float: サンプリング間隔
        """
        return self._search_near_polygon_sample_step

    @search_near_polygon_sample_step.setter
    def search_near_polygon_sample_step(self, value: float):
        """外形ポリゴンに近いクラスタ探索処理の外形ポリゴンのサンプリング間隔

        Args:
            value (float): サンプリング間隔
        """
        self._search_near_polygon_sample_step = value

    @property
    def search_near_polygon_search_radius(self) -> float:
        """外形ポリゴンに近いクラスタ探索処理の探索半径

        Returns:
            float: 外形ポリゴンに近いクラスタ探索処理の探索半径
        """
        return self._search_near_polygon_search_radius

    @search_near_polygon_search_radius.setter
    def search_near_polygon_search_radius(self, value: float):
        """外形ポリゴンに近いクラスタ探索処理の探索半径

        Args:
            value (float): 外形ポリゴンに近いクラスタ探索処理の探索半径
        """
        self._search_near_polygon_search_radius = value

    @property
    def merge_height_diff_th(self) -> float:
        """高さが近く水平連続となるクラスタのマージ処理のクラスタの高さ差分閾値

        Returns:
            float: 高さ差分閾値
        """
        return self._merge_height_diff_th

    @merge_height_diff_th.setter
    def merge_height_diff_th(self, value: float):
        """高さが近く水平連続となるクラスタのマージ処理のクラスタの高さ差分閾値

        Args:
            value (float): 高さ差分閾値
        """
        self._merge_height_diff_th = value

    @property
    def merge_dbscan_radius(self) -> float:
        """高さが近く水平連続となるクラスタのマージ処理のdbscan探索範囲

        Returns:
            float: dbscan探索範囲
        """
        return self._merge_dbscan_radius

    @merge_dbscan_radius.setter
    def merge_dbscan_radius(self, value: float):
        """高さが近く水平連続となるクラスタのマージ処理のdbscan探索範囲

        Args:
            value (float): dbscan探索範囲
        """
        self._merge_dbscan_radius = value

    @property
    def merge_dbscan_point_th(self) -> int:
        """高さが近く水平連続となるクラスタのマージ処理の
           dbscanのコア点判定用の点数閾値

        Returns:
            int: dbscanのコア点判定用の点数閾値
        """
        return self._merge_dbscan_point_th

    @merge_dbscan_point_th.setter
    def merge_dbscan_point_th(self, value: int):
        """高さが近く水平連続となるクラスタのマージ処理の
           dbscanのコア点判定用の点数閾値

        Args:
            value (int): dbscanのコア点判定用の点数閾値
        """
        self._merge_dbscan_point_th = value

    @property
    def graphcut_height_offset(self) -> float:
        """GraphCutの地面高さのoffset値

        Returns:
            float: GraphCutの地面高さのoffset値
        """
        return self._graphcut_height_offset

    @graphcut_height_offset.setter
    def graphcut_height_offset(self, value: float):
        """GraphCutの地面高さのoffset値

        Args:
            value (float): GraphCutの地面高さのoffset値
        """
        self._graphcut_height_offset = value

    @property
    def graphcut_smooth_weight(self) -> float:
        """GraphCutの平滑化の重み

        Returns:
            float: GraphCutの平滑化の重み
        """
        return self._graphcut_smooth_weight

    @graphcut_smooth_weight.setter
    def graphcut_smooth_weight(self, value: float):
        """GraphCutの平滑化の重み

        Args:
            value (float): GraphCutの平滑化の重み
        """
        self._graphcut_smooth_weight = value

    @property
    def graphcut_invalid_point_dist(self) -> float:
        """GraphCutの無効点とする距離閾値

        Returns:
            float: GraphCutの無効点とする距離閾値
        """
        return self._graphcut_invalid_point_dist

    @graphcut_invalid_point_dist.setter
    def graphcut_invalid_point_dist(self, value: float):
        """GraphCutの無効点とする距離閾値

        Args:
            value (float): GraphCutの無効点とする距離閾値
        """
        self._graphcut_invalid_point_dist = value

    @property
    def graphcut_height_diff_th(self) -> float:
        """GraphCutの高さ差分閾値

        Returns:
            float: GraphCutの高さ差分閾値
        """
        return self._graphcut_height_diff_th

    @graphcut_height_diff_th.setter
    def graphcut_height_diff_th(self, value: float):
        """GraphCutの高さ差分閾値

        Args:
            value (float): GraphCutの高さ差分閾値
        """
        self._graphcut_height_diff_th = value

    @property
    def graphcut_dbscan_radius(self) -> float:
        """GraphCut処理でのdbscanの探索半径

        Returns:
            float: GraphCut処理でのdbscanの探索半径
        """
        return self._graphcut_dbscan_radius

    @graphcut_dbscan_radius.setter
    def graphcut_dbscan_radius(self, value: float):
        """GraphCut処理でのdbscanの探索半径

        Args:
            value (float): GraphCut処理でのdbscanの探索半径
        """
        self._graphcut_dbscan_radius = value

    @property
    def graphcut_dbscan_point_th(self) -> int:
        """GraphCut処理でのdbscanのコア点判定用の点数閾値

        Returns:
            int: GraphCut処理でのdbscanのコア点判定用の点数閾値
        """
        return self._graphcut_dbscan_point_th

    @graphcut_dbscan_point_th.setter
    def graphcut_dbscan_point_th(self, value: int):
        """GraphCut処理でのdbscanのコア点判定用の点数閾値

        Args:
            value (int): GraphCut処理でのdbscanのコア点判定用の点数閾値
        """
        self._graphcut_dbscan_point_th = value

    @property
    def mbr_sampling_step(self) -> float:
        """MBR処理時の建物外形ポリゴン辺のサンプリング間隔

        Returns:
            float: MBR処理時の建物外形ポリゴン辺のサンプリング間隔
        """
        return self._mbr_sampling_step

    @mbr_sampling_step.setter
    def mbr_sampling_step(self, value: float):
        """MBR処理時の建物外形ポリゴン辺のサンプリング間隔

        Args:
            value (float): MBR処理時の建物外形ポリゴン辺のサンプリング間隔
        """
        self._mbr_sampling_step = value

    @property
    def mbr_neighbor_max_dist(self) -> float:
        """近傍建物外形線探索用の最大近傍距離閾値

        Returns:
            float: 近傍建物外形線探索用の最大近傍距離閾値
        """
        return self._mbr_neighbor_max_dist

    @mbr_neighbor_max_dist.setter
    def mbr_neighbor_max_dist(self, value: float):
        """近傍建物外形線探索用の最大近傍距離閾値

        Args:
            value (float): 近傍建物外形線探索用の最大近傍距離閾値
        """
        self._mbr_neighbor_max_dist = value

    @property
    def mbr_neighbor_jobs(self) -> int:
        """近傍建物外形線探索時のNearestNeighborのジョブ数

        Returns:
            int: 近傍建物外形線探索時のNearestNeighborのジョブ数
        """
        return self._mbr_neighbor_jobs

    @mbr_neighbor_jobs.setter
    def mbr_neighbor_jobs(self, value: int):
        """近傍建物外形線探索時のNearestNeighborのジョブ数

        Args:
            value (int): 近傍建物外形線探索時のNearestNeighborのジョブ数
        """
        self._mbr_neighbor_jobs = value

    @property
    def mbr_line_length_th(self) -> float:
        """短い建物外形線分を除外する際に距離閾値

        Returns:
            float: 短い建物外形線分を除外する際に距離閾値
        """
        return self._mbr_line_length_th

    @mbr_line_length_th.setter
    def mbr_line_length_th(self, value: float):
        """短い建物外形線分を除外する際に距離閾値

        Args:
            value (float): 短い建物外形線分を除外する際に距離閾値
        """
        self._mbr_line_length_th = value

    @property
    def mbr_roof_angle_ortho_th(self) -> float:
        """近傍建物外形線の状態判定用の角度閾値deg

        Returns:
            float: 近傍建物外形線の状態判定用の角度閾値deg
        """
        return self._mbr_roof_angle_ortho_th

    @mbr_roof_angle_ortho_th.setter
    def mbr_roof_angle_ortho_th(self, value: float):
        """近傍建物外形線の状態判定用の角度閾値deg

        Args:
            value (float): 近傍建物外形線の状態判定用の角度閾値deg
        """
        self._mbr_roof_angle_ortho_th = value

    @property
    def mbr_angle_ms_bandwidth(self) -> float:
        """建物外形線間の角度に対するMeanShiftのバンド幅

        Returns:
            float: 建物外形線間の角度に対するMeanShiftのバンド幅
        """
        return self._mbr_angle_ms_bandwidth

    @mbr_angle_ms_bandwidth.setter
    def mbr_angle_ms_bandwidth(self, value: float):
        """建物外形線間の角度に対するMeanShiftのバンド幅

        Args:
            value (float): 建物外形線間の角度に対するMeanShiftのバンド幅
        """
        self._mbr_angle_ms_bandwidth = value

    @property
    def mbr_angle_ms_jobs(self) -> int:
        """建物外形線間の角度に対するMeanShiftのジョブ数

        Returns:
            int: 建物外形線間の角度に対するMeanShiftのジョブ数
        """
        return self._mbr_angle_ms_jobs

    @mbr_angle_ms_jobs.setter
    def mbr_angle_ms_jobs(self, value: int):
        """建物外形線間の角度に対するMeanShiftのジョブ数

        Args:
            value (int): 建物外形線間の角度に対するMeanShiftのジョブ数
        """
        self._mbr_angle_ms_jobs = value

    @property
    def mbr_valid_pixel_num(self) -> int:
        """MBRの有効矩形pixel数

        Returns:
            int: MBRの有効矩形pixel数
        """
        return self._mbr_valid_pixel_num

    @mbr_valid_pixel_num.setter
    def mbr_valid_pixel_num(self, value: int):
        """MBRの有効矩形pixel数

        Args:
            value (int): MBRの有効矩形pixel数
        """
        self._mbr_valid_pixel_num = value

    @property
    def mbr_width_th(self) -> float:
        """幅閾値pix

        Returns:
            float: 幅閾値pix
        """
        return self._mbr_width_th

    @mbr_width_th.setter
    def mbr_width_th(self, value: float):
        """幅閾値pix

        Args:
            value (float): 幅閾値pix
        """
        self._mbr_width_th = value

    @property
    def mbr_slim_rate_th(self) -> float:
        """細長度

        Returns:
            float: 細長度
        """
        return self._mbr_slim_rate_th

    @mbr_slim_rate_th.setter
    def mbr_slim_rate_th(self, value: float):
        """細長度

        Args:
            value (float): 細長度
        """
        self._mbr_slim_rate_th = value

    @property
    def mbr_max_hiers(self) -> float:
        """MBRの最大階層数

        Returns:
            float: MBRの最大階層数
        """
        return self._mbr_max_hiers

    @mbr_max_hiers.setter
    def mbr_max_hiers(self, value: float):
        """MBRの最大階層数

        Args:
            value (float): MBRの最大階層数
        """
        self._mbr_max_hiers = value

    @property
    def simplify_roof_th(self) -> float:
        """屋根形状簡略化の閾値

        Returns:
            float: shapley.geometry.Polygon.simplyfy()のtolerance値
        """
        return self._simplify_roof_th

    @simplify_roof_th.setter
    def simplify_roof_th(self, value: float):
        """屋根形状簡略化の閾値

        Args:
            value (float): shapley.geometry.Polygon.simplyfy()のtolerance値
        """
        self._simplify_roof_th = value

    @property
    def noise_canceling_buffer1(self) -> float:
        """屋根形状補正のバッファサイズ1

        Returns:
            float: バッファサイズ(-の場合は収縮, +の場合は膨張)
        """
        return self._noise_canceling_buffer1

    @noise_canceling_buffer1.setter
    def noise_canceling_buffer1(self, value: float):
        """屋根形状補正のバッファサイズ1

        Args:
            value (float): 屋根形状補正のバッファサイズ1
        """
        self._noise_canceling_buffer1 = value

    @property
    def noise_canceling_buffer2(self) -> float:
        """屋根形状補正のバッファサイズ2

        Returns:
            float: バッファサイズ(-の場合は収縮, +の場合は膨張)
        """
        return self._noise_canceling_buffer2

    @noise_canceling_buffer2.setter
    def noise_canceling_buffer2(self, value: float):
        """屋根形状補正のバッファサイズ2

        Args:
            value (float): 屋根形状補正のバッファサイズ2
        """
        self._noise_canceling_buffer2 = value

    @property
    def noise_canceling_buffer3(self) -> float:
        """屋根形状補正のバッファサイズ3

        Returns:
            float: バッファサイズ(-の場合は収縮, +の場合は膨張)
        """
        return self._noise_canceling_buffer3

    @noise_canceling_buffer3.setter
    def noise_canceling_buffer3(self, value: float):
        """屋根形状補正のバッファサイズ3

        Args:
            value (float): 屋根形状補正のバッファサイズ3
        """
        self._noise_canceling_buffer3 = value

    @property
    def noise_canceling_buffer4(self) -> float:
        """屋根形状補正のバッファサイズ4

        Returns:
            float: バッファサイズ(-の場合は収縮, +の場合は膨張)
        """
        return self._noise_canceling_buffer4

    @noise_canceling_buffer4.setter
    def noise_canceling_buffer4(self, value: float):
        """屋根形状補正のバッファサイズ4

        Args:
            value (float): 屋根形状補正のバッファサイズ4
        """
        self._noise_canceling_buffer4 = value

    @property
    def surface_preparation_angle_th(self) -> float:
        """モデル面作成準備のエッジグルーピング時の角度閾値(deg)

        Returns:
            float: 角度(deg)
        """
        return self._surface_preparation_angle_th

    @surface_preparation_angle_th.setter
    def surface_preparation_angle_th(self, value: float):
        """モデル面作成準備のエッジグルーピング時の角度閾値(deg)

        Args:
            value (float): 角度(deg)
        """
        self._surface_preparation_angle_th = value

    @property
    def surface_preparation_sampling_step(self) -> float:
        """モデル面作成準備のエッジサンプリング時のステップ幅(m)

        Returns:
            float: ステップ幅(m)
        """
        return self._surface_preparation_sampling_step

    @surface_preparation_sampling_step.setter
    def surface_preparation_sampling_step(self, value: float):
        """モデル面作成準備のエッジサンプリング時のステップ幅(m)

        Args:
            value (float): ステップ幅(m)
        """
        self._surface_preparation_sampling_step = value

    @property
    def surface_preparation_dist_th(self) -> float:
        """モデル面作成準備のエッジグルーピング時の距離閾値(deg)

        Returns:
            float: 距離(m)
        """
        return self._surface_preparation_dist_th

    @surface_preparation_dist_th.setter
    def surface_preparation_dist_th(self, value: float):
        """モデル面作成準備のエッジグルーピング時の距離閾値(deg)

        Args:
            value (float): 距離(m)
        """
        self._surface_preparation_dist_th = value

    @property
    def solid_search_edge_th(self) -> float:
        """ソリッド閉じ対応時の近傍辺探索用の距離閾値

        Returns:
            float: ソリッド閉じ対応時の近傍辺探索用の距離閾値
        """
        return self._solid_search_edge_th

    @solid_search_edge_th.setter
    def solid_search_edge_th(self, value: float):
        """ソリッド閉じ対応時の近傍辺探索用の距離閾値

        Args:
            value (float): ソリッド閉じ対応時の近傍辺探索用の距離閾値
        """
        self._solid_search_edge_th = value

    @property
    def model_point_merge_xy_dist(self) -> float:
        """頂点マージ処理におけるxy平面上でのマージ距離

        Returns:
            float: 頂点マージ処理におけるxy平面上でのマージ距離
        """
        return self._model_point_merge_xy_dist

    @model_point_merge_xy_dist.setter
    def model_point_merge_xy_dist(self, value: float):
        """頂点マージ処理におけるxy平面上でのマージ距離

        Args:
            value (float): 頂点マージ処理におけるxy平面上でのマージ距離
        """
        self._model_point_merge_xy_dist = value

    @property
    def model_point_merge_z_reso(self) -> float:
        """頂点マージ処理における高さ方向解像度

        Returns:
            float: 頂点マージ処理における高さ方向解像度
        """
        return self._model_point_merge_z_reso

    @model_point_merge_z_reso.setter
    def model_point_merge_z_reso(self, value: float):
        """頂点マージ処理における高さ方向解像度

        Args:
            value (float): 頂点マージ処理における高さ方向解像度
        """
        self._model_point_merge_z_reso = value

    @property
    def front_is_ccw(self) -> bool:
        """モデル面の表面の順列方向

        Returns:
            bool:\
            True 反時計回りを表面,
            False 時計回りを表面
        """
        return self._front_is_ccw

    @front_is_ccw.setter
    def front_is_ccw(self, value: bool):
        """モデル面の表面の順列方向

        Args:
            value (bool):\
                True 反時計回りを表面,\
                False 時計回りを表面
        """
        self._front_is_ccw = value

    @property
    def classifier_checkpoint_path(self) -> str:
        """建物分類の学習済みモデルファイルパス

        Returns:
            str: 建物分類の学習済みモデルファイルパス
        """
        return self._classifier_checkpoint_path

    @property
    def balcony_segmentation_checkpoint_path(self) -> str:
        """バルコニーのセグメンテーションの学習済みモデルファイルパス
        Returns:
            str: バルコニーのセグメンテーションの学習済みモデルファイルパス
        """
        return self._balcony_segmentation_checkpoint_path

    @property
    def roof_edge_detection_checkpoint_path(self) -> str:
        """屋根線検出の学習済みモデルファイルパス
        Returns:
            str: 屋根線検出の学習済みモデルファイルパス
        """
        return self._roof_edge_detection_checkpoint_path

    @property
    def use_gpu(self) -> bool:
        """GPU使用フラグ
        Returns:
            str: GPU使用フラグ
        """
        return self._use_gpu
