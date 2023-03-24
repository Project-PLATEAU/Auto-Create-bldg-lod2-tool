# -*- coding:utf-8 -*-
import glob
import os
import sys
import cv2 as cv
import numpy as np
from ..message import CreateModelMessage
from ..createmodelexception import CreateModelException
from ...util.log import Log, LogLevel, ModuleType


class TFWReadException(CreateModelException):
    """TFW読み込みエラークラス
    """
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class TIFFReadException(CreateModelException):
    """TIFF読み込みエラークラス
    """
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class ImageInfo:
    """画像情報クラス
    """
    def __init__(self, tiff_file_path: str) -> None:
        """コンストラクタ

        Args:
            tiff_file_path (str): TIFF画像パス
        """
        self.tiff_path = tiff_file_path
        file_name = os.path.splitext(os.path.basename(tiff_file_path))[0]
        folder_path = os.path.dirname(tiff_file_path)
        self.tfw_path = os.path.join(folder_path, f'{file_name}.tfw')
        self.matrix = self._read_tfw(self.tfw_path)

        self._read_image(tiff_file_path)

    def _read_tfw(self, file_path: str):
        """tfwファイルの読み込み

        Args:
            file_path (str): tfwファイルパス

        Raises:
            TFWReadException: TFW読み込みに失敗した場合

        Returns:
            辞書: tfw行列
                  A : 1ピクセルのX方向の長さ
                  D : 行の回転パラメータ
                  B : 列の回転パラメータ
                  E : 1ピクセルのY方向の長さ
                  C : TLピクセルの中心位置のX座標
                  F : TLピクセルの中心位置のY座標
        Note:
            X(東方向(右)) = Ax + By + C
            Y(北方向(上)) = Dx + Ey + F
        """
        mat = {}

        try:
            with open(file_path, 'rt') as f:
                line = f.readlines()
                line = [s.strip() for s in line]
                mat['A'] = float(line[0])  # 1ピクセルのX方向の長さ
                mat['D'] = float(line[1])  # 行の回転パラメータ
                mat['B'] = float(line[2])  # 列の回転パラメータ
                mat['E'] = float(line[3])  # 1ピクセルのY方向の長さ
                mat['C'] = float(line[4])  # 左上ピクセルの中心座標のx座標
                mat['F'] = float(line[5])  # 左上ピクセルの中心座標のy座標
        except Exception as e:
            class_name = self.__class__.__name__
            func_name = sys._getframe().f_code.co_name
            msg = '{}.{}, {}'.format(class_name, func_name, e)
            raise TFWReadException(msg)

        return mat

    def _read_image(self, file_path: str, flags=cv.IMREAD_COLOR,
                    dtype=np.uint8, keep=False):
        """画像読み込み

        Args:
            file_path (str): 画像ファイルパス
            flags (cv.ImreadModes, optional): \
                画像読み込みモード. Defaults to cv.IMREAD_COLOR.
            dtype (type, optional): データ型. Defaults to np.uint8.
            keep (bool, optional): \
                画像データを保持するか否か. Defaults to False.

        Raises:
            TIFFReadException: TIFF読み込みに失敗した場合
        """
        try:
            self.img = None
            n = np.fromfile(file_path, dtype)
            img = cv.imdecode(n, flags)
            self.height, self.width, ch = img.shape

            if keep:
                self.img = img

        except Exception as e:
            class_name = self.__class__.__name__
            func_name = sys._getframe().f_code.co_name
            msg = '{}.{}, {}'.format(class_name, func_name, e)
            raise TIFFReadException(msg)

    def contains(self, geo_x: float, geo_y: float) -> bool:
        """画像内に地理座標系で指定した座標が含まれる確認する

        Args:
            geo_x (float): 東方向の座標
            geo_y (float): 北方向の座標

        Returns:
            bool: True : 含む, False : 含まない
        """
        # X(東方向(右)) = Ax + By + C
        # Y(北方向(上)) = Dx + Ey + F
        # 回転(B,D)未考慮
        img_x = (geo_x - self.matrix['C']) / self.matrix['A']
        img_y = (geo_y - self.matrix['F']) / self.matrix['E']

        if (0 <= img_x and img_x < self.width
                and 0 <= img_y and img_y < self.height):
            return True
        else:
            return False


class ImageManager:
    """画像管理クラス
    """

    def __init__(self) -> None:
        """コンストラクタ
        """
        self._images = []

    def read_files(self, folder_path: str):
        """TIFFファイルの読み込み処理

        Args:
            folder_path (str): TIFFフォルダパス

        Raises:
            FileNotFoundError: 入力フォルダが存在しない場合
            FileNotFoundError: フォルダ内にTIFFファイルが存在しない場合
            Exception: 全てのTIFFファイルの読み込みに失敗した場合
        """
        class_name = self.__class__.__name__
        func_name = sys._getframe().f_code.co_name

        # 初期化
        self._images = []

        if not os.path.isdir(folder_path):
            # フォルダが存在しない場合
            msg = '{}.{}, {}'.format(
                class_name, func_name,
                CreateModelMessage.ERR_MSG_IMG_MNG_FOLDER_NOT_FOUND)
            raise CreateModelException(msg)

        files = glob.glob(os.path.join(folder_path, '*.tif'))
        tiff_files = glob.glob(os.path.join(folder_path, '*.tiff'))
        files.extend(tiff_files)
        if len(files) == 0:
            # tiffファイルが存在しない場合
            msg = '{}.{}, {}'.format(
                class_name, func_name,
                CreateModelMessage.ERR_MSG_IMG_MNG_IMAGE_NOT_FOUND)
            raise CreateModelException(msg)

        for file in files:
            try:
                # tfwの読み込み
                info = ImageInfo(file)
                self._images.append(info)

            except TFWReadException:
                # TFWの読み込みに失敗した場合
                msg = '{}.{}, {}'.format(
                    class_name, func_name,
                    CreateModelMessage.ERR_MSG_IMG_MNG_TFW_READ)
                Log.output_log_write(
                    LogLevel.WARN, ModuleType.MODEL_ELEMENT_GENERATION, msg)
                Log.output_log_write(
                    LogLevel.WARN, ModuleType.MODEL_ELEMENT_GENERATION, file)
            except TIFFReadException:
                # TIFF読み込みに失敗した場合
                msg = '{}.{}, {}'.format(
                    class_name, func_name,
                    CreateModelMessage.ERR_MSG_IMG_MNG_TIFF_READ)
                Log.output_log_write(
                    LogLevel.WARN, ModuleType.MODEL_ELEMENT_GENERATION, msg)
                Log.output_log_write(
                    LogLevel.WARN, ModuleType.MODEL_ELEMENT_GENERATION, file)
            except Exception as e:
                # そのたのエラー
                msg = '{}.{}, {}'.format(class_name, func_name, e)
                Log.output_log_write(
                    LogLevel.WARN, ModuleType.MODEL_ELEMENT_GENERATION, msg)
                Log.output_log_write(
                    LogLevel.WARN, ModuleType.MODEL_ELEMENT_GENERATION, file)

        if len(self._images) == 0:
            # 全TIFFファイルの読み込みに失敗
            msg = '{}.{}, {}'.format(
                class_name, func_name,
                CreateModelMessage.ERR_MSG_IMG_MNG_READ)
            raise CreateModelException(msg)

        if len(self._images) != len(files):
            # 一部のTIFFファイルの読み込みに失敗
            Log.output_log_write(
                LogLevel.WARN, ModuleType.MODEL_ELEMENT_GENERATION,
                CreateModelMessage.WARN_MSG_IMG_MNG_READ)

    def search_image(self, geo_x: float, geo_y: float) -> ImageInfo:
        """画像探索

        Args:
            geo_x (float): 地理座標系のx座標
            geo_y (float): 地理座標系のy座標

        Returns:
            ImageInfo: 対応する画像情報
                       対応する画像情報がない場合はNone
        """
        ret = None
        for info in self._images:
            if info.contains(geo_x, geo_y):
                ret = info
                break
        
        return ret
