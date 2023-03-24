import os
import sys
import numpy as np
import numpy.typing as npt

from ..createmodelexception import CreateModelException

# ./thirdparty/heat を読み込む
heat_directory_path = os.path.join(os.path.dirname(__file__), 'roof_edge_detection_model', 'thirdparty', 'heat')  # noqa
sys.path.append(heat_directory_path)  # noqa
from .roof_edge_detection_model.thirdparty.heat.model import HEAT
del sys.path[sys.path.index(heat_directory_path)]  # noqa


class RoofEdgeDetectionCheckpointReadException(CreateModelException):
    """屋根線検出の学習済みモデル読み込みエラークラス
    """

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class RoofEdgeDetection:
    """屋根線検出クラス
    """

    _model: HEAT

    def __init__(self, checkpoint_path: str, use_gpu: bool) -> None:
        """コンストラクタ

        Args:
            checkpoint_path(str): 学習済みモデルファイルのパス 
            use_gpu(bool): 推論にGPUを使用する場合はtrue
        """
        self._model = HEAT(use_gpu)
        try:
            self._model.load_checkpoint(checkpoint_path)
        except Exception as e:
            class_name = self.__class__.__name__
            func_name = sys._getframe().f_code.co_name
            msg = '{}.{}, {}'.format(class_name, func_name, e)
            raise RoofEdgeDetectionCheckpointReadException(msg)

    def infer(self, rgb_image: npt.NDArray[np.uint8]) -> tuple[npt.NDArray, npt.NDArray]:
        """屋根線の検出を行う

        Args:
            rgb_image(NDArray[np.uint8]): (image_size, image_size, 3)のRGB画像データ

        Returns:
            NDArray: 屋根面の頂点の位置 (num of corners, 2)
            NDArray: 屋根線(頂点の番号の組)のリスト (num of edges, 2)
        """
        corners, edges = self._model.infer(rgb_image)

        return corners, edges
