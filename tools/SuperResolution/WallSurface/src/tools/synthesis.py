import cv2
import numpy as np
import pathlib

class Synthesis:
    """CycleGANインプット用の画像の編集クラス"""


    def __init__(self, output_figs, seitaika_fig, output_dir: pathlib.Path):
        
        """クラスの初期化メソッド"""
        self.overlap = 0  # ラップ率
        self.size = 256  # CycleGANのインプット画像サイズ
        self.output_figs = output_figs  # CycleGANの出力画像
        self.seitaika_fig = seitaika_fig  # 正対化の出力画像
        self.residual_h = None  # 縦方向に黒埋めする余白のサイズ
        self.residual_w = None  # 横方向に黒埋めする余白のサイズ
        self.nh = None  # 縦方向の枚数
        self.nw = None  # 横方向の枚数
        self.cut_log = None  # CycleGANインプットの作成処理の情報(ログファイルから読み取る)
        self.output_dir = output_dir  # 出力フォルダ
        self.result_image = None  # 合成結果画像


    def load(self, cut_log):
        """CycleGANのインプット作成処理の情報をログファイルから読み取る"""
        if len(self.output_figs[0]) == 0:
            return

        self.cut_log = cut_log

        self.nh = self.cut_log["nh"]
        self.nw = self.cut_log["nw"]
        self.residual_h = self.cut_log["residual_h"]
        self.residual_w = self.cut_log["residual_w"]
        self.h = self.cut_log['H']
        self.w = self.cut_log['W']
        self.overlap = self.cut_log['overlap']
        self.size = self.cut_log['size']


    def merge(self):

        if self.cut_log is None:
            self.result_image = self.seitaika_fig
            return self.result_image
        
        """画像を合成する。"""
        assert type(self.nh) == int and type(self.nw) == int
        assert type(self.residual_h) == int and type(self.residual_w) == int

        H, W = self.h, self.w
        im_count = np.zeros((H, W, 3), dtype=int)
        im_syn = np.zeros((H, W, 3), dtype=int)
        space = self.size - round(self.overlap * self.size)
        start_ws = [i * space for i in range(self.nw)]
        start_hs = [i * space for i in range(self.nh)]


        if self.nw == 1 and self.nh == 1 and not self.cut_log["cutting"]:
            ratio = self.cut_log["ratio"]
            image = self.output_figs[0][0]['img']

            h, w, _ = image.shape
            if H < W:
                w_enlarge = w
                h_enlarge = round(h * ratio)
            else:
                h_enlarge = h
                w_enlarge = round(w / ratio)
            im_syn = image[:h_enlarge, :w_enlarge, :]
            im_syn = cv2.resize(im_syn, (W, H))
            self.result_image = im_syn
            return self.result_image


        for iw in range(self.nw):
            start_w = start_ws[iw]
            for ih in range(self.nh):
                start_h = start_hs[ih]
                im = self.output_figs[iw][ih]['img']
                if self.nh - 1 == ih or self.nw - 1 == iw:
                    if self.nh - 1 == ih and self.nw - 1 != iw:
                        im_syn[start_h:H, start_w:start_w +
                               self.size] += im[:self.size - self.residual_h, :self.size]
                        im_count[start_h:H, start_w:start_w+self.size] += 1
                    elif self.nh - 1 != ih and self.nw - 1 == iw:
                        im_syn[start_h:start_h+self.size,
                               start_w:W] += im[:self.size, :self.size - self.residual_w]
                        im_count[start_h:start_h+self.size, start_w:W] += 1
                    elif self.nh - 1 == ih and self.nw - 1 == iw:
                        im_syn[start_h:H, start_w:W] += im[:self.size -
                                                           self.residual_h, :self.size - self.residual_w]
                        im_count[start_h:H, start_w:W] += 1
                else:
                    im_syn[start_h:start_h+self.size, start_w:start_w +
                           self.size] += im[:self.size, :self.size]
                    im_count[start_h:start_h+self.size,
                             start_w:start_w+self.size] += 1


        if self.cut_log["cutting"]:
            im_syn = im_syn / im_count


        self.result_image = im_syn
        return self.result_image


    def save(self, i: int):
        """合成した画像を保存"""
        assert self.result_image is not None

        output_path = self.output_dir.joinpath(f"syn_{i}.jpg")
        cv2.imwrite(str(output_path), self.result_image)
 