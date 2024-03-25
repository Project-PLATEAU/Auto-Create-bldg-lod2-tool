import math
from typing import Optional
import cv2
import numpy as np
import pathlib


def find_start(UV, w: int, h: int):
    """UV座標から、アトラス化画像内にある各画像の左上の位置を取得

    Args:
        UV (np.ndarray): UV座標
        w (int): 横方向の画像サイズ
        h (int): 縦方向の画像サイズ

    Returns
        tuple[int, int]: 左上のピクセル座標
    """
    U_min = UV[:, 0].min()
    V_max = UV[:, 1].max()
    return math.floor((1-V_max)*h), math.floor(U_min*w)


def find_end(UV, w: int, h: int):
    """UV座標から、アトラス化画像内にある各画像の右下の位置を取得

    Args:
        UV (np.ndarray): UV座標
        w (int): 横方向の画像サイズ
        h (int): 縦方向の画像サイズ

    Returns
        tuple[int, int]: 右下のピクセル座標
    """
    U_max = UV[:, 0].max()
    V_min = UV[:, 1].min()
    return math.ceil((1-V_min)*h), math.ceil(U_max*w)


def write_wall_to_atlas(
    proj_img,
    dst_image,
    x_start: int,
    y_start: int,
):
    """画像をアトラス化画像に貼り付ける

    Args:
        proj_img (np.ndarray): 逆射影変換された画像
        dst_image (np.ndarray): 貼り付け先の画像データ(入力atlas化画像)
        x_start (int): 横方向のスタート位置
        y_start (int): 縦方向のスタート位置

    Returns
        np.ndarray: 貼り付け後の画像データ
    """
    h, w, _ = proj_img.shape
    if h == 1 and w == 1:
        result_image = dst_image.copy()
    else:
        result_image = dst_image.copy()
        result_image[y_start:y_start+h, x_start:x_start+w] = proj_img

    return result_image


def write_roof_to_atlas(
    src_image,
    dst_image,
    x_start: int,
    y_start: int,
    x_end: Optional[int] = None,
    y_end: Optional[int] = None,
):
    """屋根の画像をアトラス化画像に貼り付ける

    Args:
        src_image (np.ndarray): 入力atlas化画像
        dst_image (np.ndarray): 出力入力atlas化画像
        x_start (int): 横方向のスタート位置
        y_start (int): 縦方向のスタート位置
        x_end (int): 横方向の終了位置
        y_end (int): 縦方向の終了位置

    """

    result_image = dst_image.copy()
    result_image[y_start:y_end, x_start:x_end] \
        = src_image[y_start:y_end, x_start:x_end]

    return result_image


class Put:
    """逆射影変換を施すクラス

        Attributes:
            logger:
            seitaika_logs (list[str]): ログファイル名のリスト
            proj_images (list[str]): 逆射影された画像名のリスト
            roof_infos (list[str]): 正対化ツールによる屋根の情報のファイル名のリスト
            new_w (int): アトラス化画像の横方向のサイズ
            new_h (int): アトラス化画像の縦方向のサイズ
            UVs (list[np.ndarray]): 壁面画像のUV座標
            UVs_roof (list[np.ndarray]): 屋根画像のUV座標
    """

    def __init__(self, logger, seitaika_logs, proj_images, roof_infos):
        self.logger = logger
        self.seitaika_logs = seitaika_logs  # 正対化ログ

        self.proj_images = proj_images  # 逆射影した画像
        self.new_w = None  # アトラス化画像の横方向のサイズ
        self.new_h = None  # アトラス化画像の縦方向のサイズ
        self.roof_infos = roof_infos  # 正対化ツールによる屋根の情報出力

        self.UVs = []  # 壁面画像のUV座標
        self.UVs_roof = []  # 屋根画像のUV座標

    def read_UVs(self):
        """
        logファイルから逆変換した画像UVを読み出す (texture)
        """
        for seitaika_log in self.seitaika_logs:
            UV = seitaika_log["texture"]
            self.UVs.append(np.array(UV))
        return

    def read_UVs_roof(self):
        """
        logファイルから逆変換した画像UVを読み出す (texture)
        """
        for roof_info in self.roof_infos:
            UV = roof_info["texture"]
            self.UVs_roof.append(np.array(UV))
        return

    def read_default_atlas(self):
        """
        入力のアトラス化画像のサイズ入手
        """
        self.atlas = pathlib.Path(self.seitaika_logs[0]["texture_file_path"])
        self.new_h = self.seitaika_logs[0]["h"]
        self.new_w = self.seitaika_logs[0]["w"]
        return

    def write(self):
        """
        元画像のUVsとHWから、ピクセルを割り出しマスクで足していく
        Args:
            output_dir: 新規作成するアトラス化した画像の出力先のディレクトリ
        """
        assert type(self.new_w) == int and type(self.new_h) == int
        assert self.atlas != None

        src_image = cv2.imread(str(self.atlas))

        w, h = self.new_w, self.new_h
        result_image = src_image.copy()
        for i, proj_img in enumerate(self.proj_images):
            y, x = find_start(self.UVs[i], w, h)
            result_image = write_wall_to_atlas(proj_img, result_image, x, y)

        for i in range(len(self.UVs_roof)):
            y, x = find_start(self.UVs_roof[i], w, h)
            y_end, x_end = find_end(self.UVs_roof[i], w, h)
            result_image = write_roof_to_atlas(
                src_image, result_image, x, y, x_end, y_end)

        return result_image
