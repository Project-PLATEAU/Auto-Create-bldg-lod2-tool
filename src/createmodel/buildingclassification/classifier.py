import sys
from enum import IntEnum
import numpy as np
import numpy.typing as npt
import torch

from ..createmodelexception import CreateModelException
from .classifier_model import ClassifierModel


class ClassifierCheckpointReadException(CreateModelException):
    """建物分類の学習済みモデル読み込みエラークラス
    """

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class BuildingClass(IntEnum):
    """建物種別
    """

    FLAT = 1
    """陸屋根"""

    NON_FLAT = 2
    """非陸屋根(家屋)"""


class Classifier:
    """建物分類クラス
    """

    _model: torch.nn.Module
    _device: torch.device

    def __init__(self, checkpoint_path: str, use_gpu: bool) -> None:
        """コンストラクタ

        Args:
            checkpoint_path(str): 学習済みモデルファイルパス
            use_gpu(bool): 推論時にGPUを使用する場合はTrue
        """

        self._model = ClassifierModel(in_channels=3, n_classes=2)

        # Choose to infer on CPU or GPU
        if use_gpu:
            assert torch.cuda.is_available(), "CUDA is not available."
            self._device = torch.device('cuda:0')
        else:
            self._device = torch.device('cpu')

        self._model.to(self._device)

        # Load checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self._device)
            if 'criterion.weight' in checkpoint.keys():
                checkpoint.pop('criterion.weight')
            self._model.load_state_dict(checkpoint)
        except Exception as e:
            class_name = self.__class__.__name__
            func_name = sys._getframe().f_code.co_name
            msg = '{}.{}, {}'.format(class_name, func_name, e)
            raise ClassifierCheckpointReadException(msg)

    def classify(self, image: npt.NDArray[np.uint8]) -> BuildingClass:
        """画像から建物種類を求める

        Args:
            image(NDArray[np.uint8]): RGB画像 (height, width, 3)

        Returns:
            BuildingClass: 建物種別
        """

        self._model.eval()

        X = image[:, :, ::-1].astype(np.float32)  # RGB -> BGR
        X = X.transpose(2, 0, 1) / 255.
        X = torch.from_numpy(X[np.newaxis, :, :, :]).to(
            dtype=torch.float, device=self._device)

        with torch.inference_mode():
            outputs = self._model(X)
            _, preds = torch.max(outputs, 1)
            pred = preds.detach().cpu().numpy()[0]

        if pred == 0:
            return BuildingClass.NON_FLAT
        elif pred == 1:
            return BuildingClass.FLAT
        else:
            assert False, "Error"
