# -*- coding:utf-8 -*-
import sys
from shapely.geometry import Polygon
from shapely.geometry import JOIN_STYLE
from .lasmanager import LasManager
from .message import CreateModelMessage
from .param import CreateModelParam
from .createmodelexception import CreateModelException
from .buildingmodeling import CreateModel as CreateBuildingModel
from .housemodeling import CreateModel as CreateHouseModel
from .buildingclassification import BuildingClass, ClassifyBuilding


class Building:
    """建物クラス
    """

    def __init__(
            self, id: str, shape: list,
            dsm_folder_path: str,
            grid_size: float,
            output_folder_path: str) -> None:
        """コンストラクタ

        Args:
            id (str): 建物id
            shape (list): 建物外形形状
            dsm_folder_path (str): dsm画像フォルダパス
            grid_size (float): 解像度m
            output_folder_path (str): 出力フォルダパス

        Raises:
            CreateModelException: 頂点列が4点未満の場合
            CreateModelException: 建物外形形状の面積が0の場合
        """
        class_name = self.__class__.__name__
        func_name = sys._getframe().f_code.co_name

        if (len(shape) < 4):
            # 頂点列が4点未満の場合
            msg = '{}.{}, {}'.format(
                class_name, func_name,
                CreateModelMessage.ERR_MSG_CITY_GML_POLYGON_DATA)
            raise CreateModelException(msg)

        polygon = Polygon(shape)
        if (polygon.area == 0):
            # 建物外形形状の面積が0の場合
            msg = '{}.{}, {}'.format(
                class_name, func_name,
                CreateModelMessage.ERR_MSG_CITY_GML_POLYGON_NO_AREA)
            raise CreateModelException(msg)

        # 建物外形形状の保持
        self._id = id
        self._shape = polygon

        # 点群探索範囲の設定
        # 建物外形形状の外側のみを膨張して、地面範囲を追加する
        param = CreateModelParam.get_instance()
        self._points_search_area = self._shape.buffer(
            param.ground_search_dist, join_style=JOIN_STYLE.mitre,
            single_sided=True)

        # 地面探索範囲の設定
        # 建物外形との差分を取得することで枠ポリゴンを作成
        self._ground_area = self._points_search_area.difference(self._shape)

        # 入力DSMフォルダパス
        self._dsm_folder_path = dsm_folder_path

        # 入力DSM解像度(m)
        self._grid_size = grid_size

        # 出力objフォルダパス
        self._output_folder_path = output_folder_path

    def create(self, las_swap_xy=False) -> None:
        """モデル生成, obj出力

        Args:
            las_swap_xy (bool, optional):\
                lasのxyを入れ替えフラグ. Defaults to False.
        """
        param = CreateModelParam.get_instance()

        # 点群データの取得
        # lasファイルの座標値をそのまま使用する
        las_mng = LasManager(swap_xy=las_swap_xy)
        
        # ヘッダファイルの読み込み
        las_mng.read_header(self._dsm_folder_path, self._points_search_area)

        # 建物点群の取得
        cloud, min_ground_height, graphcut_height = las_mng.get_points(
            self._shape, self._ground_area)

        building_class = ClassifyBuilding(
            cloud=cloud,
            shape=self._shape,
            classifier_checkpoint_path=param.classifier_checkpoint_path,
            use_gpu=param.use_gpu,
            grid_size=0.25,
            expand_rate_for_house_model=0.25 / 0.08,
        )

        if building_class == BuildingClass.FLAT:
            # 陸屋根の場合
            CreateBuildingModel(
                cloud=cloud, shape=self._shape,
                graphcut_height=graphcut_height,
                grid_size=self._grid_size,
                building_id=self._id,
                min_ground_height=min_ground_height,
                output_folder_path=self._output_folder_path)

        elif building_class == BuildingClass.NON_FLAT:
            # GPU無し環境の対応：
            # return

            # 非陸屋根の場合
            CreateHouseModel(
                cloud=cloud,
                shape=self._shape,
                building_id=self._id,
                min_ground_height=min_ground_height,
                output_folder_path=self._output_folder_path,
                balcony_segmentation_checkpoint_path=param.balcony_segmentation_checkpoint_path,
                roof_edge_detection_checkpoint_path=param.roof_edge_detection_checkpoint_path,
                use_gpu=param.use_gpu,
                grid_size=self._grid_size,
                expand_rate=0.25 / 0.08,
            )

        else:
            assert False, f"Unsupported building class, {building_class.name}"
