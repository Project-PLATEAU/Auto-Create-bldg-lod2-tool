import cv2
import numpy as np
import json
import math
import pathlib


def read_json(log_path: pathlib.Path):
    """jsonを読む関数

    Args:
        log_path (str): ログファイル名

    Returns
        dict: ログ情報
    """
    with open(log_path, "r") as f:
        dic = json.load(f)
    return dic


def get_hw(src):
    """画像の範囲を計算する関数

    Args:
        src (np.ndarray): テクスチャ情報

    Returns
        list[float]: 範囲
    """
    U_min = src[:, 0].min()
    V_min = src[:, 1].min()
    U_max = src[:, 0].max()
    V_max = src[:, 1].max()
    return V_max, V_min, U_max, U_min


class CalcInvProj:
    """逆射影変換を施すクラス

        Attributes:
            image_seitaika (Path): 正対化された画像パス
            json_log (dict): ログ情報

    """

    def __init__(self, logger, seitaika_log, syn_fig):

        self.image_seitaika = syn_fig

        self.logger = logger
        self.json_log = seitaika_log


    def inv_proj(self):
        """逆射影変換を行う関数

        Args
            output_image_path (str): 出力パス
        """
        assert self.json_log["src"] != None
        assert self.json_log["dst"] != None
        src = np.array(self.json_log["src"])
        dst = np.array(self.json_log["dst"])
        atlas_path = self.json_log["texture_file_path"]

        im_atlas = cv2.imread(str(atlas_path))
        h_atlas, w_atlas, _ = im_atlas.shape

        homo, _ = cv2.findHomography(
            # 座標がはみ出ている場合は内側に入るように調整する(ずれは発生する)
            dst.clip([0, 0], [self.json_log["w_dst"], self.json_log["h_dst"]]),
            src
        )
        v_max, v_min, u_max, u_min = get_hw(src)
        v_max, v_min, u_max, u_min = math.ceil(v_max), math.floor(
            v_min), math.ceil(u_max), math.floor(u_min)

        # 外側の色が薄くなるのを防ぐため、外側も塗りつぶしマスクで切り取る
        image = self.image_seitaika

        # Return value as is when image size is 1 x 1
        if image.shape[0] == 1 and image.shape[1] == 1:
            if self.logger is not None:
                self.logger.info(f"Warning: Return value as is when image size is 1 x 1")
            return image
        
        dst_image = cv2.warpPerspective(
            image, homo, (w_atlas, h_atlas),
            borderMode=cv2.BORDER_REPLICATE,
            borderValue=(255, 255, 255)
        )
        mask_image = cv2.warpPerspective(
            np.full(image.shape[:2], 255, dtype=np.uint8),
            homo,
            (w_atlas, h_atlas),
            borderValue=0
        )
        dst_image[mask_image == 0] = 255
        dst_image = dst_image[v_min:v_max, u_min:u_max]
        
        return dst_image
