from typing import Final, Optional
import cv2
import numpy as np
import numpy.typing as npt
import shapely.geometry as geo
from shapely.geometry import Point
from sklearn.neighbors import NearestNeighbors
import laspy
import re
import math

from ..lasmanager import PointCloud

# LAS
def write_las(fpath, xyz, rgb=None, classification=None):
    x = xyz[:,0]
    y = xyz[:,1]
    z = xyz[:,2]

    if re.match('^1\.', laspy.__version__):
        header = laspy.header.Header(file_version=1.2, point_format=2)
        las_file = laspy.file.File(fpath, mode='w', header=header)

        las_file.header.scale = [0.01, 0.01, 0.01]
        las_file.header.min = [np.min(x), np.min(y), np.min(z)]
        las_file.header.max = [np.max(x), np.max(y), np.max(z)]
        las_file.header.offset = [(np.min(x) + np.max(x)) / 2, 
                                  (np.min(y) + np.max(y)) / 2,
                                  (np.min(z) + np.max(z)) / 2]

        # data
        las_file.x = x
        las_file.y = y
        las_file.z = z

        if rgb is not None:
            las_file.red = rgb[:,0]
            las_file.green = rgb[:,1]
            las_file.blue = rgb[:,2]

        if classification is not None:
            las_file.raw_classification = classification.astype(np.uint16)

        las_file.close() 

    if re.match('^2\.', laspy.__version__):
        header = laspy.LasHeader(version='1.2', point_format=2)
        header.scales = [0.01, 0.01, 0.01]
        header.mins = [np.min(x), np.min(y), np.min(z)]
        header.maxs = [np.max(x), np.max(y), np.max(z)]
        header.offsets = [(np.min(x) + np.max(x)) / 2, 
                          (np.min(y) + np.max(y)) / 2,
                          (np.min(z) + np.max(z)) / 2]

        las_file = laspy.LasData(header=header)

        # data
        las_file.x = x
        las_file.y = y
        las_file.z = z

        if rgb is not None:
            las_file.red = rgb[:,0]
            las_file.green = rgb[:,1]
            las_file.blue = rgb[:,2]

        if classification is not None:
            las_file.raw_classification = classification.astype(np.uint16)
    
        las_file.write(fpath) 


class Preprocess:
    """
    建物点群から、バルコニー判定用モデルに読み込ませるデータに変換する
    """

    _grid_size: Final[float]
    _expand_rate: Final[float]

    def __init__(
        self,
        grid_size: float,
        expand_rate: Optional[float] = None
    ) -> None:
        """コンストラクタ

        Args:
            grid_size(float): 点群の間隔(meter)
            expand_rate(float, optional): 画像の拡大率 
        """

        self._grid_size = grid_size
        self._expand_rate = expand_rate if expand_rate is not None else 1

    def _calc_rotation_angle(
        self,
        shape: geo.Polygon
    ) -> float:
        """建物外形ポリゴンの回転角度を求める

        Args:
            shape(geo.Polygon): 建物外形ポリゴン

        Returns:
            float: 回転角度(degree)
        """

        # 建物外形点座標
        coords = shape.exterior.coords if shape.exterior else []
        building_poly_xy = np.array(coords).copy()

        # 建物外形の直線
        footprint_lines: list[tuple[npt.NDArray[np.float_],
                                    npt.NDArray[np.float_]]] = []
        for point_i, point_j in zip(building_poly_xy[:-1], building_poly_xy[1:]):
            footprint_lines.append((point_i, point_j))

        # 建物外形の直線の角度
        footprint_lines_angle: list[float] = []
        for point_i, point_j in footprint_lines:
            delta_x, delta_y = point_j - point_i
            angle = np.rad2deg(np.arctan2(delta_y, delta_x))
            footprint_lines_angle.append(angle)

        # 建物外形の直線の長さ
        footprint_lines_length: list[float] = []
        for point_i, point_j in footprint_lines:
            length = float(np.linalg.norm(point_i - point_j))
            footprint_lines_length.append(length)

        # 最も長い外形直線の角度
        maximum_length_idx = np.argmax(footprint_lines_length)
        maximum_length_angle = footprint_lines_angle[maximum_length_idx]

        return maximum_length_angle

    def _rotate_point_cloud(
        self,
        cloud: PointCloud,
        rotation_angle: float,
        footprint: geo.Polygon
    ) -> npt.NDArray[np.uint8]:
        """点群を回転させ、画像に変換する

        Args:
            cloud(PointCloud): 点群
            rotation_angle(float): 回転角度 (degree)
            footprint(geo.Polygon): 建物外形ポリゴン

        Returns:
            NDArray[np.uint8]: 回転後の建物画像 (128, 128, 3)
        """

        if False:
            xyz = cloud.get_points().copy()
            rgb = cloud.get_colors().copy()        

            x_min = np.min(xyz[:, 0])
            y_max = np.max(xyz[:, 1])

            # 回転前画像の作成
            pixel_xy = np.dstack([
                (xyz[:, 0] - x_min) / self._grid_size,
                (y_max - xyz[:, 1]) / self._grid_size,
            ])[0, :, :]
            pixel_xy = np.round(pixel_xy).astype(np.int_) 

            width, height = pixel_xy.max(axis=0) + np.array([1, 1])
            img_rgb = np.zeros((height, width, 3), np.uint8)
            img_rgb[pixel_xy[:, 1], pixel_xy[:, 0]] = rgb / 255.

        else:
            # 回転前画像の作成
            pc_xyz = cloud.get_points().copy()
            pc_rgb = cloud.get_colors().copy()

            pc_x_min, pc_y_min, _ = pc_xyz.min(axis=0)
            pc_x_max, pc_y_max, _ = pc_xyz.max(axis=0)

            width = math.ceil((pc_x_max - pc_x_min) / self._grid_size) + 1
            height = math.ceil((pc_y_max - pc_y_min) / self._grid_size) + 1
        
            xs = np.arange(width) * self._grid_size + pc_x_min
            ys = -np.arange(height) * self._grid_size + pc_y_max 
            xx, yy = np.meshgrid(xs, ys)
            xy = np.dstack([xx, yy]).reshape(-1,2)

            nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree', leaf_size=30, n_jobs=4)
            nn.fit(pc_xyz[:,0:2])
            inds = nn.kneighbors(xy, return_distance=False)[:, 0]

            img_rgb = pc_rgb[inds] / 256
            for i, xy_ in enumerate(xy):
                p = Point(xy_[0], xy_[1]) 
                if not footprint.contains(p):
                    img_rgb[i] = 0

            img_rgb = img_rgb.reshape(height, width, 3).astype(np.uint8)

        # 回転行列
        center = [width/2, height/2]
        rotation_mat = cv2.getRotationMatrix2D(center, -rotation_angle, 1)

        # 画像の回転
        rotation_angle_rad = np.deg2rad(rotation_angle)
        width_rot = int(np.round(width*abs(np.cos(-rotation_angle_rad)) +
                        height*abs(np.sin(-rotation_angle_rad))))
        height_rot = int(np.round(width*abs(np.sin(-rotation_angle_rad)) +
                                  height*abs(np.cos(-rotation_angle_rad))))
        rotation_mat[0][2] += -width/2 + width_rot/2
        rotation_mat[1][2] += -height/2 + height_rot/2

        rotated_img_rgb = cv2.warpAffine(
            img_rgb, rotation_mat, (width_rot, height_rot), flags=cv2.INTER_LINEAR, borderValue=(0,))

        # padding除去
        mask = (rotated_img_rgb > 0).all(axis=2)
        ys, xs = np.nonzero(mask)
        xy = np.dstack([xs, ys])[0, :, :]
        min_x, min_y = xy.min(axis=0)
        max_x, max_y = xy.max(axis=0)

        rotated_img_rgb = rotated_img_rgb[min_y:max_y+1, min_x:max_x+1]

        # モデル入力用の正方形画像に変換
        square_rotated_image_rgb = cv2.resize(
            rotated_img_rgb, dsize=(128, 128))

        return np.array(square_rotated_image_rgb, dtype=np.uint8)

    def preprocess(
        self,
        cloud: PointCloud,
        shape: geo.Polygon,
    ) -> npt.NDArray[np.uint8]:
        """前処理

        Args:
            cloud (PointCloud): 建物点群
            shape (geo.Polygon): 建物外形ポリゴン
            ground_height (float): 地面の高さm
            img_folder (str): オルソ画像フォルダパス

        Returns:
          NDArray[np.uint8]: (width, height, 3)のデータ
        """

        rotation_angle = self._calc_rotation_angle(shape)
        rotated_image = self._rotate_point_cloud(cloud, rotation_angle, shape)

        return rotated_image
