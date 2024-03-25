import os
import cv2
import numpy as np
from numpy.typing import NDArray


class Cv2Japanese:
    def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
        """OpenCvで日本語パスを読み込む

        Args:
            filename (string): 読み込みファイルパス
            flags (int, optional): カラーモード. Defaults to cv2.IMREAD_COLOR.
            dtype (numpy, optional): データ型. Defaults to np.uint8.

        Returns:
            NDArray: 読み込み画像
        """
        try:
            dec = np.fromfile(filename, dtype)
            image = cv2.imdecode(dec, flags)
            return image
        except Exception as e:
            print(e)
            return None

    def imwrite(filename: str, img: NDArray, params=None):
        """OpenCvで日本語を使ったパスで保存する

        Args:
            filename (string): 書き込みファイルパス
            img (NDArray): 書き込み画像
            params (list, optional): データ型固有パラメータ. Defaults to None.

        Returns:
            bool: 書き込み成功(True)/書き込み失敗(False)
        """
        try:
            ext = os.path.splitext(filename)[1]
            result, n = cv2.imencode(ext, img, params)

            if result:
                with open(filename, mode='w+b') as f:
                    n.tofile(f)
                return True
            else:
                return False
        except Exception as e:
            print(e)
            return False
