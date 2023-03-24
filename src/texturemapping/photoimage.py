# -*- coding:utf-8 -*-

import numpy as np
import math
import os
from ..util.cvsupportjp import Cv2Japanese
from ..util.parammanager import ParamManager


class PhotoImage():
    """写真情報クラス
    """
    def __init__(self) -> None:
        """コンストラクタ
        """
        self.filename = None
        self.photodir = None
        # self.img = None
        self._focallength = 0         # 焦点距離[mm]
        self._ppx = 0                 # カメラの主点X座標[mm]
        self._ppy = 0                 # カメラの主点Y座標[mm]
        self._adjustedfocallen = 0     # キャリブレーション後の焦点距離
        self._adjustedfocallen2 = 0    # キャリブレーション後の焦点距離 * -1 (-m_fAdjustedFolcalLen)
        self._calibparam = [0 for i in range(5)]

        self._paramomega = 0
        self._paramphi = 0
        self._paramkappa = 0
        self._focalpos = [0 for i in range(3)]

        self._rot_matrix = np.zeros((3, 3))                   # 回転行列の要素
        self._imagesize = [0 for i in range(2)]             # 画像サイズ(画素数) width/height
        self._sensorsize = [0 for i in range(2)]            # センサーサイズ(mm) width/height
        self._valid_imagerange = [0 for i in range(2)]       # 有効画像座標範囲(画像サイズ - 1) width/height
        self._calibration_flag = False                       # キャリブレーションフラグ
        # 外部標定要素から算出する回転行列のモード
        self._rorate_matrix_mode = ParamManager.RotateMatrixMode.XYZ

    def set_photo_param(
            self, photodir: str, excalib: str, caminfo: str, calibflag: bool,
            rotate_matrix_mode: ParamManager.RotateMatrixMode):
        """写真情報をセットする

        Args:
            photodir (string): 入力写真フォルダパス
            excalib (string): 外部標定要素情報
            caminfo (string): カメラ情報
            calibflag (bool): キャリブレーションフラグ
            rotate_matrix_mode (ParamManager.RotateMatrixMode):\
                外部標定要素から算出する回転行列のモード

        Returns:
            bool: 関数実行結果
        """
        # 写真ファイル名
        self.filename = excalib[0]
        self.photodir = photodir

        # 画像の有無を確認
        photo_path = os.path.join(self.photodir, self.filename)
        if not os.path.isfile(photo_path):
            return False

        # 画像サイズをセットする
        img = Cv2Japanese.imread(photo_path)
        # self.img = img
        self.set_imagesize([img.shape[1], img.shape[0]])
        # 画像サイズ(pixel)×1pixelサイズ(mm)=センサーサイズ(mm)
        # カメラ情報と実画像で縦横が逆になる場合があるため、カメラ情報として入力したサイズと一致したものをセンササイズとする
        # 誤差は0.01とする(暫定)
        if (math.isclose(img.shape[1] * float(caminfo[3]) / 1000,
                         float(caminfo[1]),
                         rel_tol=0.01)
           or math.isclose(img.shape[1] * float(caminfo[3]) / 1000,
                           float(caminfo[2]),
                           rel_tol=0.01)):
            self._sensorsize[0] = img.shape[1] * float(caminfo[3]) / 1000
        if (math.isclose(img.shape[1] * float(caminfo[4]) / 1000,
                         float(caminfo[1]),
                         rel_tol=0.01)
           or math.isclose(img.shape[1] * float(caminfo[4]) / 1000,
                           float(caminfo[2]),
                           rel_tol=0.01)):
            self._sensorsize[1] = img.shape[0] * float(caminfo[4]) / 1000

        # カメラの焦点距離(mm)
        self._focallength = float(caminfo[0])

        # カメラの主点(mm)
        self._ppx = float(caminfo[5])
        self._ppy = float(caminfo[6])

        # 外部標定要素
        self._focalpos[0] = float(excalib[1])
        self._focalpos[1] = float(excalib[2])
        self._focalpos[2] = float(excalib[3])
        self._paramomega = float(excalib[4])
        self._paramphi = float(excalib[5])
        self._paramkappa = float(excalib[6])

        # キャリブレーションフラグ
        self._calibration_flag = calibflag

        if self._calibration_flag:
            # 補正焦点距離(mm)
            self._calibparam[0] = float(caminfo[3])
            # 主点ズレ(x)(mm)
            self._calibparam[1] = float(caminfo[4])
            # 主点ズレ(y)(mm)
            self._calibparam[2] = float(caminfo[5])
            # 半径方向歪み係数(3次項)
            self._calibparam[3] = float(caminfo[6])
            # 半径方向歪み係数(5次項)
            self._calibparam[4] = float(caminfo[7])

        # キャリブレーション後の焦点距離
        self._adjustedfocallen = self._focallength
        # キャリブレーション後の焦点距離 * -1 (_adjustedfocallen)
        self._adjustedfocallen2 = -1.0 * self._adjustedfocallen

        # 回転行列の要素を求める
        self.set_rotmatrix(rotate_matrix_mode)

        return True

    def get_photo_pos(self, point) -> None:
        """ 撮影中心位置を取得する

        Args:
            point (float[]): 撮影中心位置(x,y,z)
             
        """
        point[0] = self._focalpos[0]
        point[1] = self._focalpos[1]
        point[2] = self._focalpos[2]

    def set_imagesize(self, size) -> None:
        """画像サイズをセットする

        Args:
            size (float[]): 画像サイズ(x,y)
        """
        self._imagesize = size
        self._valid_imagerange[0] = self._imagesize[0] - 1
        self._valid_imagerange[1] = self._imagesize[1] - 1

    def get_imagesize(self):
        """画像サイズを取得する
        """
        return self._imagesize

    def get_imagepos(self, point, imagepos):
        """絶対座標に対応する画像座標と、画像内に座標が存在するかを判定する

        Args:
            point (float[]): 絶対座標(x,y,z)
            imagepos (float[]): 画像座標(x,y)

        Returns:
            int: 画像内に座標が存在するか(0:存在しない　1:存在する)
        """
        # 写真座標
        photopos = [0 for i in range(2)]  # 写真座標

        # 絶対座標に対応する写真座標を求める
        point_3d = np.array([point[0], point[1], point[2]], np.double)
        point_3d = point_3d - np.array([self._focalpos[0],
                                        self._focalpos[1],
                                        self._focalpos[2]], np.double)
        # point_3d = np.dot(self._rot_matrix.transpose(), point_3d)
        point_3d = np.dot(self._rot_matrix.T, point_3d)
        photopos[0] = point_3d[0] / point_3d[2]
        photopos[1] = point_3d[1] / point_3d[2]

        # キャリブレーション
        if self._calibration_flag:
            # 中心(写真座標原点)からの距離の2乗を求める
            length = photopos[0] * photopos[0] + photopos[1] * photopos[1]

            photopos[0] -= (
                photopos[0] * length
                * (self._calibparam[3] + self._calibparam[4] * length)
                - self._calibparam[1])
            photopos[1] -= (
                photopos[1] * length
                * (self._calibparam[3] + self._calibparam[4] * length)
                - self._calibparam[2])

        imagepos[0] = (self._imagesize[0] / 2 
                       + (self._ppx - photopos[0] * self._adjustedfocallen)
                       * self._imagesize[0] / self._sensorsize[0])
        imagepos[1] = (self._imagesize[1] / 2 
                       + (self._ppy - photopos[1] * self._adjustedfocallen)
                       * self._imagesize[1] / self._sensorsize[1])
        imagepos[1] = self._imagesize[1] - imagepos[1]
        # 写真中心からの距離を求める
        # distance = photopos[0] * photopos[0] + photopos[1] * photopos[1]

        if (imagepos[0] < 0 or imagepos[1] < 0
           or imagepos[0] > self._valid_imagerange[0]
           or imagepos[1] > self._valid_imagerange[1]):
            return 0

        return 1

    def set_rotmatrix(
            self, rotate_matrix_mode: ParamManager.RotateMatrixMode) -> None:
        """回転行列の要素を求める

        Args:
            rotate_matrix_mode (ParamManager.RotateMatrixMode): 回転行列のモード
        """
        omega_ = self._paramomega * math.pi / 180
        phi_ = self._paramphi * math.pi / 180
        kappa_ = self._paramkappa * math.pi / 180

        sin_omega = math.sin(omega_)
        sin_kappa = math.sin(kappa_)
        sin_phi = math.sin(phi_)
        cos_omega = math.cos(omega_)
        cos_kappa = math.cos(kappa_)
        cos_phi = math.cos(phi_)

        r_omega = np.array(
            [[1., 0., 0.],
             [0., cos_omega, -sin_omega],
             [0., sin_omega, cos_omega]])
        r_phi = np.array(
            [[cos_phi, 0., sin_phi],
             [0., 1., 0.],
             [-sin_phi, 0., cos_phi]])
        r_kappa = np.array(
            [[cos_kappa, -sin_kappa, 0.],
             [sin_kappa, cos_kappa, 0.],
             [0., 0., 1.]])

        if rotate_matrix_mode is ParamManager.RotateMatrixMode.ZYX:
            # R = Rz(κ)Ry(Φ)Rx(ω)
            r_kappa_phi = np.dot(r_kappa, r_phi)
            self._rot_matrix = np.dot(r_kappa_phi, r_omega)
        else:
            # R = Rx(ω)Ry(Φ)Rz(κ)
            r_omega_phi = np.dot(r_omega, r_phi)
            self._rot_matrix = np.dot(r_omega_phi, r_kappa)
