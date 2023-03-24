import sys
import torch
import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw

from ..createmodelexception import CreateModelException

from .model_surface_creation.utils.geometry import Point
from .balcony_segmentation_model.model import SegmentationModel


class BalconySegmentationCheckpointReadException(CreateModelException):
    """バルコニーセグメンテーションの学習済みモデル読み込みエラークラス
    """

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class BalconyDetection:
    """バルコニー検出クラス
    """

    _image_size: int
    _model: torch.nn.Module
    _device: torch.device

    def __init__(self, checkpoint_path: str, use_gpu: bool) -> None:
        """コンストラクタ

        Args:
            checkpoint_path(str): 学習済みモデルファイルのパス 
            use_gpu(bool): 推論にGPUを使用する場合はtrue
        """
        self._image_size = 256

        self._model = SegmentationModel(
            num_classes=2,
            image_size=self._image_size,
            in_channels=4,
        )

        # Choose to infer on CPU or GPU
        if use_gpu:
            assert torch.cuda.is_available(), "CUDA is not available."
            self._device = torch.device('cuda:0')
        else:
            self._device = torch.device('cpu')

        self._model.to(self._device)

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self._device)
            if 'criterion.weight' in checkpoint.keys():
                checkpoint.pop('criterion.weight')
            self._model.load_state_dict(checkpoint)
        except Exception as e:
            class_name = self.__class__.__name__
            func_name = sys._getframe().f_code.co_name
            msg = '{}.{}, {}'.format(class_name, func_name, e)
            raise BalconySegmentationCheckpointReadException(msg)

    def infer(
        self,
        rgb_image: npt.NDArray[np.uint8],
        depth_image: npt.NDArray[np.uint8],
        points: list[Point],
        polygons: list[list[int]],
        threshold: float = 0.5,
    ) -> list[bool]:
        """バルコニー領域の推論を行う

        Args:
            rgb_image(NDArray[np.uint8]): (image_size, image_size, 3)のRGB画像データ
            depth_image(NDArray[np.uint8]): (image_size, image_size)の高さのグレースケール画像データ
            points(list[Point]): polygonsの各頂点の位置
            polygons(list[list[int]]): 屋根面の一覧(各要素はpointsのindexを接続順に並べたリスト)
            threshold(float, optional): バルコニーと判定する割合の閾値 (Default: 0.5)

        Returns:
            list[bool]: polygonsの各面のバルコニー判定結果
        """
        # バルコニー領域のセグメンテーションを行う
        self._model.eval()

        X = np.concatenate([
            rgb_image[:, :, ::-1],  # RGB -> BGR
            depth_image[:, :, np.newaxis]
        ], axis=2).astype(np.float32)
        X = X.transpose(2, 0, 1) / 255.
        X = torch.from_numpy(X[np.newaxis, :, :, :]).to(
            dtype=torch.float, device=self._device
        )

        with torch.inference_mode():
            outputs = self._model(X)
            _, preds = torch.max(outputs, 1)
            pred = preds.detach().cpu().numpy()[0].astype(np.bool8)

        balcony_flags: list[bool] = []

        # 各polygonのバルコニー判定を行う
        for polygon in polygons:
            mask_image = Image.new(
                '1', (self._image_size, self._image_size), 0)
            draw = ImageDraw.Draw(mask_image)
            draw.polygon([
                (points[point_idx].x, points[point_idx].y)
                for point_idx in polygon
            ], fill=1)

            mask = np.array(mask_image)
            polygon_area = mask.sum()
            balcony_area = (pred[mask] > 0).sum()

            is_balcony = polygon_area > 0 and balcony_area / polygon_area >= threshold

            balcony_flags.append(is_balcony)

        return balcony_flags
