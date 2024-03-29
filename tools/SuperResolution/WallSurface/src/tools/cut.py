
import cv2
import numpy as np
import json
import pathlib
import math


def to_square(image, size: int):
    """画像を正方形にする。余白は黒埋めする。

    Args
        image (np.ndarray): 画像データ
        size(int): 指定のサイズ

    Returns
        np.ndarray: 画像データ
    """
    H, W, C = image.shape
    assert H <= size and W <= size

    im = np.zeros((size, size, C), dtype=int)
    im[:H, :W, :] = image[:, :, :]
    return im


def elongation(image, size: int):
    """長辺がsizeになるまで引き延ばし、正方形にする。画像の余白は黒埋めする。

    Args:
        image (np.ndarray): 画像データ
        size(int): 指定のサイズ

    Returns:
        np.ndarray: 画像データ
    """
    H, W, _ = image.shape
    if H > W:
        w = W * (size / H)
        w = round(w)
        image = cv2.resize(image, (w, size))
    else:
        h = H * (size / W)
        h = round(h)
        image = cv2.resize(image, (size, h))
    image = to_square(image, size)
    return image


class Cut:
    """CycleGANインプット用の画像の編集クラス

    Attributes:
        overlap (float): ラップ率
        size (int): CycleGANのインプット画像サイズ
        input_path (Path): 整形する正対化済み画像のパス
        residual_h (int): 縦方向に黒埋めする余白のサイズ
        residual_w (int): 横方向に黒埋めする余白のサイズ
        output_dir (Path): 出力フォルダ
        output_image_name_format (str): 整形画像の名称
        result_images (list): 加工後の画像

    """

    def __init__(self, logger, seitaika_fig, output_dir: pathlib.Path, overlap=0.1, size=256):
        self.logger = logger
        self.overlap = overlap  # ラップ率
        self.size = size  # CycleGANのインプット画像サイズ
        self.residual_h = 0  # 縦方向に黒埋めする余白のサイズ
        self.residual_w = 0  # 横方向に黒埋めする余白のサイズ
        self.output_dir = output_dir  # 出力フォルダ
        self.output_image_name_format = "output_{i}_{iw}_{ih}.jpg"  # 整形画像の名称
        self.result_images = []

        self.input_path = seitaika_fig['path']
        self.img = seitaika_fig['img']
        
        self.height = self.img.shape[0]
        self.width = self.img.shape[1]

    def calc_nH(self):
        """
           画像をcycleGANのインプット用に整形する際に縦方向に画像何枚文必要か計算
           余白を黒で画像埋めるピクセル数を計算
        """
        h, _, _ = self.img.shape
        self.nh = 1
        space = self.size - round(self.overlap*self.size)
        n = math.ceil((h - self.size) / space)
        if n >= 1:
            self.nh += n
        residual = h - self.size
        residual -= (self.nh - 1) * (self.size - round(self.overlap*self.size))

        if residual < 0:
            self.residual_h = -residual
        self.nh = round(self.nh)
        return

    def calc_nW(self):
        """
           画像をcycleGANのインプット用に整形する際に横方向に画像何枚文必要か計算
           余白を黒で画像埋めるピクセル数を計算
        """
        _, w, _ = self.img.shape
        self.nw = 1
        space = self.size - round(self.overlap*self.size)
        n = math.ceil((w - self.size) / space)
        if n >= 1:
            self.nw += n
        residual = w - self.size
        residual -= (self.nw - 1) * (self.size - round(self.overlap*self.size))

        if residual < 0:
            self.residual_w = -residual
        self.nw = round(self.nw)
        return

    def cut(self):
        """
           画像をcycleGANのインプット用に整形
        """
        h, w, _ = self.img.shape
        self.ratio = h/w
        if self.nh == 1 and self.nw == 1:
            result_image = elongation(self.img, self.size).astype(np.uint8)
            self.result_images = [[result_image]]
            return
        else:
            self.nh = 1
            self.nw = 1
            result_image = elongation(self.img, self.size).astype(np.uint8)
            self.result_images = [[result_image]]
            return

        # The following processes are not executed
        h, w, _ = self.img.shape
        space = self.size - round(self.overlap*self.size)
        start_ws = [i*space for i in range(self.nw)]
        start_hs = [i*space for i in range(self.nh)]
        for iw in range(self.nw):
            result_im_tmp = []
            start_w = start_ws[iw]
            for ih in range(self.nh):
                start_h = start_hs[ih]
                if self.nh - 1 == ih or self.nw - 1 == iw:
                    if self.nh - 1 == ih and self.nw - 1 != iw:
                        cropped = self.img[start_h: h, start_w:start_w+self.size]
                        cropped = to_square(cropped, self.size)
                    elif self.nh - 1 != ih and self.nw - 1 == iw:
                        cropped = self.img[start_h:start_h+self.size, start_w: w]
                        cropped = to_square(cropped, self.size)
                    elif self.nh - 1 == ih and self.nw - 1 == iw:
                        cropped = self.img[start_h: h, start_w: w]
                        cropped = to_square(cropped, self.size)
                    else:
                        raise ValueError()
                else:
                    cropped = self.img[start_h:start_h+self.size,
                                       start_w:start_w+self.size]
                
                cropped = cropped.astype(np.uint8)
                result_im_tmp.append(cropped)

            self.result_images.append(result_im_tmp)

    def save(self, i: int):
        """
           画像を保存
           Args:
               i (int): 画像インデックス
           Returns:
               list[list[Path]]: 出力画像パス

        """
        output_figs = []
        for iw in range(self.nw):
            output_fig = []
            for ih in range(self.nh):
                output_path = self.output_dir.joinpath(
                    self.output_image_name_format.format(iw=iw, ih=ih, i=i)
                )
                if self.logger is not None:
                    cv2.imwrite(str(output_path), self.result_images[iw][ih])
                self.result_images[iw][ih] = cv2.cvtColor(self.result_images[iw][ih], cv2.COLOR_BGR2RGB)
                output_fig.append({'img': self.result_images[iw][ih], 'path': str(output_path)})
            output_figs.append(output_fig)
        return output_figs

    def output_log(self, log_file_name: str):
        """
           処理情報のログ出力
           Args:
               log_file_name (str): ログ出力ファイル名

        """
        
        h, w, _ = self.img.shape
        dic = {
            "overlap": self.overlap,
            "size": self.size,
            "imagename": str(self.input_path),
            "residual_h": self.residual_h,
            "residual_w": self.residual_w,
            "nh": self.nh,
            "nw": self.nw,
            "H": h,
            "W": w,
            "ratio": self.ratio,
        }
        if self.size >= h and self.size >= w:
            dic["cutting"] = False
        else:
            # dic["cutting"] = True
            dic["cutting"] = False

        log_path = None
        if self.logger is not None:
            log_path = self.output_dir.joinpath(log_file_name)
            with open(log_path, "w") as f:
                json.dump(dic, f, indent=4)
            log_path = log_path.name
        
        return dic, log_path
