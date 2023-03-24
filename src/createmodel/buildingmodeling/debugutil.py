# -*- coding:utf-8 -*-
import os
import numpy as np
import random
import open3d as o3d
import colorsys
import datetime
import shapely.geometry as geo
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from typing import Tuple
from numpy.typing import NDArray
from .clusterinfo import ClusterInfo
from .geoutil import GeoUtil
matplotlib.use('Agg')


# 定数
GM = (np.sqrt(5) - 1.0) / 2.0
W = 8.0
H = W * GM
WIN_SIZE = (W, H)


class DebugUtil:
    """デバッグユーティリティ
    """
    @staticmethod
    def sigmoid(x: float, gain=10.0, offset=0.2):
        """sigmoid method

        Args:
            x (float): 0.0 - 1.0
            gain (float, optional): ゲイン. Defaults to 10.0.
            offset (float, optional): オフセット. Defaults to 0.2.

        Returns:
            float: sigmoid関数値
        """
        return ((np.tanh(((x + offset) * gain) / 2.0) + 1.0) / 2.0)

    @staticmethod
    def sigmoid_color(x: float):
        """sigmod関数によるカラー

        Args:
            x (float): 0.0 - 1.0

        Returns:
            float: 赤
            float: 緑
            float: 青
        """
        gain = 10.0
        offset = 0.2
        green_offset = 0.6
        tmp = x * 2.0 - 1.0
        r = DebugUtil.sigmoid(tmp, gain=gain, offset=-offset)
        b = 1.0 - DebugUtil.sigmoid(tmp, gain=gain, offset=offset)
        g = (DebugUtil.sigmoid(tmp, offset=green_offset)
             + (1.0 - DebugUtil.sigmoid(tmp, offset=-green_offset))
             - 1.0)
        return r, g, b
    
    @staticmethod
    def get_color(x: int):
        """固定色またはランダムによるカラー

        Args:
            x (int): インデックス

        Returns:
            float: 赤
            float: 緑
            float: 青
        
        Note:
            x < 20までは固定色、20 <= xからはランダムカラー
        """
        # 色順
        # 赤, 緑, 青, 橙, 黄, 紫, 桃, 黄緑, 水色, 赤紫,
        # 山吹, 青紫, 茶, 深緑, 葡萄, 空, 臙脂, 紺, 鮭, 薄茶

        red = np.array([1.0, 0.0, 0.0, 1.0, 1.0,
                        0.8, 1.0, 0.8, 0.8, 1.0,
                        1.0, 0.6, 0.6, 0.2, 0.4,
                        0.4, 0.8, 0.0, 1.0, 0.8])
        
        green = np.array([0.0, 1.0, 0.0, 0.6, 1.0,
                          0.0, 0.6, 1.0, 1.0, 0.0,
                          0.8, 0.6, 0.2, 0.2, 0.0,
                          0.8, 0.0, 0.0, 0.6, 0.6])

        blue = np.array([0.0, 0.0, 1.0, 0.0, 0.0,
                         1.0, 1.0, 0.4, 1.0, 0.6,
                         0.0, 1.0, 0.0, 0.0, 0.4,
                         1.0, 0.2, 0.4, 0.6, 0.0])

        if x > -1 and x < len(red):
            return red[x], green[x], blue[x]

        else:
            # random
            r = random.uniform(0.0, 1.0)
            g = random.uniform(0.0, 1.0)
            b = random.uniform(0.0, 1.0)

            return r, g, b

    @staticmethod
    def get_colors(num: int):
        hsv = [(x * 1.0 / num, 1.0, 1.0) for x in range(num)]
        rgb = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv))
        return rgb

    @staticmethod
    def get_color_code(id: int):
        """matplotlib用のカラーコード(ex. #FF003E)

        Args:
            id (int): インデックス

        Returns:
            str: カラーコード
        """
        r_ratio, g_ratio, b_ratio = DebugUtil.get_color(id)
        r = int(255 * r_ratio)
        g = int(255 * g_ratio)
        b = int(255 * b_ratio)
        color_code = '#{:02X}{:02X}{:02X}'.format(r, g, b)
        return color_code

    @staticmethod
    def ignorez_points(points: NDArray):
        """2/3次元点のz座標を0にする

        Args:
            points (NDArray): 2/3次元点の点群

        Returns:
            NDArray: z座標0の点群(入力が2次元配列でない場合はNone)
        """
        if points.ndim == 2:
            # 2次元配列の場合
            if points.shape[1] == 3:
                # 3次元点
                # xyのみ取得
                points = points[:, 0:2]
                # z座標を0にする
                zeros = np.zeros((len(points), 1))
                return np.hstack([points, zeros])

            elif points.shape[1] == 2:
                # 2次元点
                zeros = np.zeros((len(points), 1))
                return np.hstack([points, zeros])
        else:
            return None

    @staticmethod
    def create_path(polygon: geo.base.BaseGeometry):
        """matplotlibのPathPatch用のPath作成

        Args:
            polygon (geo.base.BaseGeometry): ポリゴン
        """
        def coding(polygon):
            coods = np.ones(len(polygon.coords), Path.code_type) * Path.LINETO
            coods[0] = Path.MOVETO
            return coods

        points = np.concatenate(
            [np.asarray(polygon.exterior)[:, :2]]
            + [np.asarray(inner)[:, :2] for inner in polygon.interiors])

        codes = np.concatenate(
            [coding(polygon.exterior)]
            + [coding(inner) for inner in polygon.interiors])

        return Path(points, codes)

    @staticmethod
    def draw_cloud(cloud: NDArray, colors: NDArray = None,
                   ignore_z=False, swapxy=True,
                   title: str = None, save=False,
                   save_folder_path: str = None):
        """点群描画

        Args:
            cloud (NDArray): 点群
            colors (NDArray, optional): \
                点ごとの描画色(RGB, 0-1.0の値). Defaults to None.
            ignore_z (bool, optional): z座標を無視するか否か. Defaults to False.
            swapxy (bool, optional): \
                xy座標を反転して描画するか否か. Defaults to True.
            title (str, optional): タイトル. Defaults to None.
            save (bool, optional): \
                True=画像保存, False=Window表示. Defaults to False.
            save_folder_path (str, optional): \
                保存先フォルダパス. Defaults to None.
        """
        if len(cloud) == 0:
            return

        name = 'points' if title is None else title
        if ignore_z:
            points = DebugUtil.ignorez_points(cloud)
        else:
            points = cloud

        if swapxy:
            x = points[:, 0:1]
            y = points[:, 1:2]
            z = points[:, 2:3]
            points = np.hstack([y, x])
            points = np.hstack([points, z])

        o3d_cloud = o3d.geometry.PointCloud()
        o3d_cloud.points = o3d.utility.Vector3dVector(points)
        if colors is not None and len(colors) == len(points):
            o3d_cloud.colors = o3d.utility.Vector3dVector(colors)

        if save:
            # 画像保存
            filename = '{}.png'.format(name)
            if save_folder_path:
                if not os.path.isdir(save_folder_path):
                    os.makedirs(save_folder_path)
                filename = os.path.join(save_folder_path, filename)

            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False)
            vis.add_geometry(o3d_cloud)
            vis.update_geometry(o3d_cloud)
            vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image(filename)
            vis.destroy_window()
        else:
            # window表示
            o3d.visualization.draw_geometries([o3d_cloud], window_name=name)

    @staticmethod
    def draw_clusters(clusters: list[ClusterInfo],
                      base_polygon: geo.Polygon = None,
                      ignore_z=False, swapxy=True,
                      title=None, save=False,
                      save_folder_path: str = None):
        """クラスタの描画

        Args:
            clusters (list[ClusterInfo]): クラスタ情報リスト
            base_polygon (geo.Polygon, optional): 基準ポリゴン. Defaults to None.
            ignore_z (bool, optional): z座標を無視するか否か. Defaults to False.
            swapxy (bool, optional):
                xy座標を反転して描画するか否か. Defaults to True.
            title (str, optional): タイトル. Defaults to None.
            save (bool, optional):
                True=画像保存, False=Window表示. Defaults to False.
            save_folder_path (str, optional):
                保存先フォルダパス. Defaults to None.
        """
        if len(clusters) == 0:
            return

        linewidth = 1.5
        alpha = 0.5
        name = 'ransac' if title is None else title
        fig = plt.figure(figsize=WIN_SIZE, dpi=90)
        plt.subplots_adjust(wspace=0.6, hspace=0.6)
        fig.set_frameon(True)
        if ignore_z:
            ax = fig.add_subplot(111)
        else:
            ax = Axes3D(fig)

        ax.set_title(name)

        init_flag = True
        colors = []
        for cluster in clusters:
            points = cluster.points.get_points()
            if swapxy:
                x = points[:, 1:2]
                y = points[:, 0:1]
            else:
                x = points[:, 0:1]
                y = points[:, 1:2]

            r = 0.0
            g = 0.0
            b = 0.0
            while True:
                r, g, b = DebugUtil.get_color(cluster.id)
                if colors.count([r, g, b]) == 0:
                    break
            colors.append([r, g, b])

            if ignore_z:
                ax.scatter(x, y, color=(r, g, b), marker='.')
            else:
                z = points[:, 2:3]
                ax.scatter(x, y, z, color=(r, g, b), marker='.')
            
            if init_flag:
                xmin = x.min()
                ymin = y.min()
                xmax = x.max()
                ymax = y.max()
                if not ignore_z:
                    zmin = z.min()
                    zmax = z.max()
                init_flag = False
            else:
                xmin = x.min() if xmin > x.min() else xmin
                ymin = y.min() if ymin > y.min() else ymin
                xmax = x.max() if xmax < x.max() else xmax
                ymax = y.max() if ymax < y.max() else ymax
                if not ignore_z:
                    zmin = z.min() if zmin > z.min() else zmin
                    zmax = z.max() if zmax < z.max() else zmax

        if base_polygon is not None:
            tmp_xmin, tmp_ymin, tmp_xmax, tmp_ymax \
                = DebugUtil.plot_geometry(
                    base_polygon, ax, swapxy, '#FFFFFF', '#000000',
                    linewidth, alpha, 0)
            if init_flag:
                xmin = tmp_xmin
                ymin = tmp_ymin
                xmax = tmp_xmax
                ymax = tmp_ymax
                init_flag = False
            else:
                xmin = tmp_xmin if xmin > tmp_xmin else xmin
                ymin = tmp_ymin if ymin > tmp_ymin else ymin
                xmax = tmp_xmax if xmax < tmp_xmax else xmax
                ymax = tmp_ymax if ymax < tmp_ymax else ymax

        offset = 5
        step = 10
        xmin = int(xmin - offset)
        ymin = int(ymin - offset)
        xmax = int(xmax + offset)
        ymax = int(ymax + offset)
        ax.set_xlim(xmin, xmax)
        ax.set_xticks(np.arange(xmin, xmax + 1, step))
        ax.set_ylim(ymin, ymax)
        ax.set_yticks(np.arange(ymin, ymax + 1, step))
        if not ignore_z:
            zmin -= offset
            zmax += offset
            ax.set_zlim(zmin, zmax)
            ax.set_zticks(np.arange(zmin, zmax + 1, step))
            ax.set_aspect('auto')
        else:
            ax.set_aspect('equal')

        if swapxy:
            ax.set_xlabel('y')
            ax.set_ylabel('x')
        else:
            ax.set_xlabel('x')
            ax.set_ylabel('y')

        if save:
            # 画像保存
            filename = '{}.png'.format(name)
            if save_folder_path:
                if not os.path.isdir(save_folder_path):
                    os.makedirs(save_folder_path)
                filename = os.path.join(save_folder_path, filename)
            plt.savefig(filename)
        else:
            # window表示
            plt.show(block=True)
        
        plt.clf()
        plt.close(fig=fig)

    @staticmethod
    def draw_clusters_open3d(
            clusters: list[ClusterInfo], ignore_z=False,
            title=None, save=False, save_folder_path: str = None):
        """クラスタの描画

        Args:
            clusters (list[ClusterInfo])): クラスタ情報リスト
            ignore_z (bool, optional): z座標を無視するか否か. Defaults to False.
            title (str, optional): タイトル. Defaults to None.
            save (bool, optional):
                True=画像保存, False=Window表示. Defaults to False.
            save_folder_path (str, optional):
                保存先フォルダパス. Defaults to None.
        """
        if len(clusters) == 0:
            return

        name = 'cluster' if title is None else title

        if save:
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False)

            colors = []
            for cluster in clusters:
                o3d_cloud = o3d.geometry.PointCloud()
                if ignore_z:
                    cloud = DebugUtil.ignorez_points(
                        cluster.points.get_points())
                else:
                    cloud = cluster.points.get_points()
                o3d_cloud.points = o3d.utility.Vector3dVector(cloud)

                r = 0.0
                g = 0.0
                b = 0.0
                while True:
                    r, g, b = DebugUtil.get_color(cluster.id)
                    if colors.count([r, g, b]) == 0:
                        break
                colors.append([r, g, b])
                o3d_cloud.paint_uniform_color([r, g, b])
                vis.add_geometry(o3d_cloud)
                vis.update_geometry(o3d_cloud)

            filename = '{}.png'.format(name)
            if save_folder_path:
                if not os.path.isdir(save_folder_path):
                    os.makedirs(save_folder_path)
                filename = os.path.join(save_folder_path, filename)

            vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image(filename)
            vis.destroy_window()

        else:
            # 座標点をo3dで表示
            points = []
            colors = []
            for cluster in clusters:
                o3d_cloud = o3d.geometry.PointCloud()
                if ignore_z:
                    cloud = DebugUtil.ignorez_points(
                        cluster.points.get_points())
                else:
                    cloud = cluster.points.get_points()
                o3d_cloud.points = o3d.utility.Vector3dVector(cloud)

                r = 0.0
                g = 0.0
                b = 0.0
                while True:
                    r, g, b = DebugUtil.get_color(cluster.id)
                    if colors.count([r, g, b]) == 0:
                        break
                colors.append([r, g, b])
                o3d_cloud.paint_uniform_color([r, g, b])
                points.append(o3d_cloud)

            o3d.visualization.draw_geometries(points, window_name=name)

    @staticmethod
    def plot_polygon(
            polygon: geo.Polygon, ax, swapxy=True, facecolor='#FFFFFF',
            edgecolor='#000000', linewidth=1.5,
            alpha=0.5, zorder=1) -> Tuple[float, float, float, float]:
        """ポリゴンの描画

        Args:
            polygon (geo.Polygon): ポリゴン
            ax (Axes): グラフ
            swapxy (bool, optional): \
                xy座標を反転して描画するか否か. Defaults to True.
            facecolor (str, optional): 面色. Defaults to '#FFFFFF'.
            edgecolor (str, optional): 線色. Defaults to '#000000'.
            linewidth (float, optional): 線幅. Defaults to 1.5.
            alpha (float, optional): 透過率. Defaults to 0.5.
            zorder (int, optional): zorder. Defaults to 1.

        Returns:
            Tuple[float, float, float, float]: xmin, ymin, xmax, ymax
        """
        points = np.array(polygon.exterior.coords)
        inners = [np.asarray(inner)[:, :2] for inner in polygon.interiors]
        if swapxy:
            x = points[:, 0:1]
            y = points[:, 1:2]
            points = np.hstack([y, x])

            for i in range(len(inners)):
                x = inners[i][:, 0:1]
                y = inners[i][:, 1:2]
                inners[i] = np.hstack([y, x])

        polygon = geo.Polygon(points, inners)
        path = DebugUtil.create_path(polygon)
        patch = PathPatch(path, facecolor=facecolor,
                          edgecolor=edgecolor, linewidth=linewidth,
                          alpha=alpha, zorder=zorder)

        ax.add_patch(patch)
        xmin, ymin, xmax, ymax = polygon.bounds
        return xmin, ymin, xmax, ymax

    @staticmethod
    def plot_point(
            pt: geo.Point, ax, swapxy=True, color='#000000',
            alpha=0.5, zorder=1) -> Tuple[float, float, float, float]:
        """点の描画

        Args:
            pt (geo.Point): 点
            ax (Axes): グラフ
            swapxy (bool, optional): \
                xy座標を反転して描画するか否か. Defaults to True.
            color (str, optional): 色. Defaults to #000000'.
            alpha (float, optional): 透過率. Defaults to 0.5.
            zorder (int, optional): zorder. Defaults to 1.

        Returns:
            Tuple[float, float, float, float]: xmin, ymin, xmax, ymax
        """
        x, y = pt.xy
        if swapxy:
            x, y = y, x
        ax.plot(x, y, 'o', color=color, alpha=alpha, zorder=zorder)
        return x[0], y[0], x[0], y[0]

    @staticmethod
    def plot_line(
            line, ax, swapxy=True, color='#000000', linewidth=1.5,
            alpha=0.5, zorder=1) -> Tuple[float, float, float, float]:
        """線の描画

        Args:
            line (LineString, LinearRing): 線
            ax (Axes): グラフ
            swapxy (bool, optional): \
                xy座標を反転して描画するか否か. Defaults to True.
            color (str, optional): 色. Defaults to '#000000'.
            linewidth (float, optional): 線幅. Defaults to 1.5.
            alpha (float, optional): 透過率. Defaults to 0.5.
            zorder (int, optional): zorder. Defaults to 1.

        Returns:
            Tuple[float, float, float, float]: xmin, ymin, xmax, ymax
        """
        x, y = line.xy
        if swapxy:
            x, y = y, x
        ax.plot(x, y, color=color, linewidth=linewidth,
                alpha=alpha, zorder=zorder)

        return min(x), min(y), max(x), max(y)

    @staticmethod
    def plot_geometry(
            geometry, ax, swapxy=True, facecolor='#FFFFFF',
            edgecolor='#000000', linewidth=1.5,
            alpha=0.5, zorder=1) -> Tuple[float, float, float, float]:
        """ジオメトリの描画

        Args:
            geometry (LineString, LinearRing, Point, Polygon): ジオメトリ
            ax (Axes): グラフ
            swapxy (bool, optional): \
                xy座標を反転して描画するか否か. Defaults to True.
            facecolor (str, optional): 面色. Defaults to '#FFFFFF'.
            edgecolor (str, optional): 線色. Defaults to '#000000'.
            linewidth (float, optional): 線幅. Defaults to 1.5.
            alpha (float, optional): 透過率. Defaults to 0.5.
            zorder (int, optional): zorder. Defaults to 1.

        Returns:
            Tuple[float, float, float, float]: xmin, ymin, xmax, ymax
        """
        xmin = 0
        ymin = 0
        xmax = 0
        ymax = 0
        if type(geometry) is geo.Polygon:
            xmin, ymin, xmax, ymax = DebugUtil.plot_polygon(
                geometry, ax, swapxy, facecolor, edgecolor,
                linewidth, alpha, zorder)

        elif (type(geometry) is geo.LineString
                or type(geometry) is geo.LinearRing):
            xmin, ymin, xmax, ymax = DebugUtil.plot_line(
                geometry, ax, swapxy, edgecolor, linewidth, alpha, zorder)

        elif type(geometry) is geo.Point:
            xmin, ymin, xmax, ymax = DebugUtil.plot_point(
                geometry, ax, swapxy, facecolor, alpha, zorder)

        return xmin, ymin, xmax, ymax

    @staticmethod
    def draw_geometries(geometries: list, base_polygon: geo.Polygon = None,
                        swapxy=True, title=None, save=False,
                        save_folder_path: str = None) -> None:
        """ジオメトリの描画

        Args:
            geometries (list): ジオメトリリスト
            base_polygon (geo.Polygon, optional): 基準ポリゴン. Defaults to None.
            swapxy (bool, optional): \
                xy座標を反転して描画するか否か. Defaults to True.
            title (str, optional): タイトル. Defaults to None.
            save (bool, optional): \
                True=画像保存, False=Window表示. Defaults to False.
            save_folder_path (str, optional):
                保存先フォルダパス. Defaults to None.
        """
        
        if len(geometries) == 0:
            return

        name = 'geometries' if title is None else title

        linewidth = 1.5
        alpha = 0.5
        zorder = 2
        offset = 10
        step = 10
        fig = plt.figure(figsize=WIN_SIZE, dpi=90)
        plt.subplots_adjust(wspace=0.6, hspace=0.6)
        fig.set_frameon(True)
        ax = fig.add_subplot(111)
        ax.set_title(name)

        # geometriesの描画
        init_flag = True
        for i in np.arange(len(geometries)):
            color_code = DebugUtil.get_color_code(i)

            if (type(geometries[i]) is geo.Polygon
                    or type(geometries[i]) is geo.LineString
                    or type(geometries[i]) is geo.LinearRing
                    or type(geometries[i]) is geo.Point):
                # ポリゴン、線、リング、点の場合
                tmp_xmin, tmp_ymin, tmp_xmax, tmp_ymax \
                    = DebugUtil.plot_geometry(
                        geometries[i], ax, swapxy, color_code, color_code,
                        linewidth, alpha, zorder)
            elif (type(geometries[i]) is geo.MultiPolygon
                    or type(geometries[i]) is geo.MultiLineString
                    or type(geometries[i]) is geo.MultiPoint
                    or type(geometries[i]) is geo.GeometryCollection):
                # 複数ポリゴン、複数線、複数点、複数ジオメトリの場合の場合
                tmp_init_flag = True
                for geom in geometries[i].geoms:

                    x1, y1, x2, y2 = DebugUtil.plot_geometry(
                        geom, ax, swapxy, color_code, color_code,
                        linewidth, alpha, zorder)

                    if tmp_init_flag:
                        tmp_xmax = x2
                        tmp_xmin = x1
                        tmp_ymax = y2
                        tmp_ymin = y1
                        tmp_init_flag = False
                    else:
                        tmp_xmax = x2 if tmp_xmax < x2 else tmp_xmax
                        tmp_xmin = x1 if tmp_xmin > x1 else tmp_xmin
                        tmp_ymax = y2 if tmp_ymax < y2 else tmp_ymax
                        tmp_ymin = y1 if tmp_ymin > y1 else tmp_ymin

            if init_flag:
                xmin = tmp_xmin
                ymin = tmp_ymin
                xmax = tmp_xmax
                ymax = tmp_ymax
                init_flag = False
            else:
                xmin = tmp_xmin if xmin > tmp_xmin else xmin
                ymin = tmp_ymin if ymin > tmp_ymin else ymin
                xmax = tmp_xmax if xmax < tmp_xmax else xmax
                ymax = tmp_ymax if ymax < tmp_ymax else ymax

        if base_polygon is not None:
            tmp_xmin, tmp_ymin, tmp_xmax, tmp_ymax \
                = DebugUtil.plot_geometry(
                    base_polygon, ax, swapxy, '#FFFFFF', '#000000',
                    linewidth, alpha, 0)
            if init_flag:
                xmin = tmp_xmin
                ymin = tmp_ymin
                xmax = tmp_xmax
                ymax = tmp_ymax
                init_flag = False
            else:
                xmin = tmp_xmin if xmin > tmp_xmin else xmin
                ymin = tmp_ymin if ymin > tmp_ymin else ymin
                xmax = tmp_xmax if xmax < tmp_xmax else xmax
                ymax = tmp_ymax if ymax < tmp_ymax else ymax

        xmin = int(np.floor(xmin) - offset)
        ymin = int(np.floor(ymin) - offset)
        xmax = int(np.floor(xmax) + offset)
        ymax = int(np.floor(ymax) + offset)
        ax.set_xlim(xmin, xmax)
        ax.set_xticks(np.arange(xmin, xmax + 1, step))
        ax.set_ylim(ymin, ymax)
        ax.set_yticks(np.arange(ymin, ymax + 1, step))
        ax.set_aspect('equal')
        if swapxy:
            ax.set_xlabel('y')
            ax.set_ylabel('x')
        else:
            ax.set_xlabel('x')
            ax.set_ylabel('y')

        if save:
            # 画像保存
            filename = '{}.png'.format(name)
            if save_folder_path:
                if not os.path.isdir(save_folder_path):
                    os.makedirs(save_folder_path)
                filename = os.path.join(save_folder_path, filename)
            plt.savefig(filename)
        else:
            # window表示
            plt.show(block=True)

        plt.clf()
        plt.close(fig=fig)

    @staticmethod
    def capture_cloud_sigmoid_color(cloud: NDArray, value: NDArray,
                                    ignore_z=False, swapxy=True, title=None,
                                    save_folder_path: str = None):
        """点群のキャプチャ画像取得

        Args:
            cloud (NDArray): 点群
            value (NDArray): sigmoid color用の値(0.0 - 1.0)
            ignore_z (bool, optional): z座標を無視するか否か. Defaults to False.
            swapxy (bool, optional):
                xy座標を反転して描画するか否か. Defaults to True.
            title (str, optional): タイトル. Defaults to None.
            save_folder_path (str, optional):
                保存先フォルダパス. Defaults to None.
        """
        if len(cloud) == 0:
            return

        name = 'points' if title is None else title

        if ignore_z:
            points = DebugUtil.ignorez_points(cloud)
        else:
            points = cloud

        if swapxy:
            x = points[:, 0:1]
            y = points[:, 1:2]
            z = points[:, 2:3]
            points = np.hstack([y, x])
            points = np.hstack([points, z])

        # 座標点をo3dで表示
        o3d_cloud = o3d.geometry.PointCloud()
        for i in np.arange(len(points)):
            o3d_cloud.points.append(points[i])
            o3d_cloud.colors.append(DebugUtil.sigmoid_color(value[i]))

        filename = '{}.png'.format(name)
        if save_folder_path:
            if not os.path.isdir(save_folder_path):
                os.makedirs(save_folder_path)
            filename = os.path.join(save_folder_path, filename)

        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        vis.add_geometry(o3d_cloud)
        vis.update_geometry(o3d_cloud)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(filename)
        vis.destroy_window()

    @staticmethod
    def get_now_str():
        """現在日時の文字列取得

        Returns:
            str: 現在日時(YYYYmmdd_HHMMSS)の文字列
        """
        now = datetime.datetime.now()
        str_now = now.strftime('%Y%m%d_%H%M%S_%f')
        return str_now

    @staticmethod
    def draw_lines(line_lists: list[list[geo.LineString]],
                   swapxy=True, title: str = None,
                   save=False, save_folder_path: str = None):
        """線の描画

        Args:
            line_lists (list[list[geo.LineString]]): 線群
            swapxy (bool, optional): \
                xy座標を反転して描画するか否か. Defaults to True.
            title (str, optional): タイトル. Defaults to None.
            save (bool, optional): \
                True=画像保存, False=Window表示. Defaults to False.
            save_folder_path (str, optional): \
                保存先フォルダパス. Defaults to None.
        """
        if len(line_lists) == 0:
            return

        name = 'lines' if title is None else title

        offset = 10
        step = 10
        fig = plt.figure(figsize=WIN_SIZE, dpi=90)
        plt.subplots_adjust(wspace=0.6, hspace=0.6)
        fig.set_frameon(True)
        ax = fig.add_subplot(111)
        ax.set_title(name)

        # 線の描画
        init_flag = True
        for i in range(len(line_lists)):
            color_code = DebugUtil.get_color_code(i)

            for line in line_lists[i]:
                points = np.array(line.coords)
                if swapxy:
                    x = points[:, 0:1]
                    y = points[:, 1:2]
                    points = np.hstack([y, x])
                    line = geo.LineString(points)

                x = points[:, 0:1]
                y = points[:, 1:2]
                ax.plot(x, y, color=color_code, linewidth=2.0,
                        solid_capstyle='round', zorder=2, alpha=0.7)
                if init_flag:
                    xmin, ymin, xmax, ymax = line.bounds
                    init_flag = False
                else:
                    tmp_xmin, tmp_ymin, tmp_xmax, tmp_ymax = line.bounds
                    
                    xmin = tmp_xmin if xmin > tmp_xmin else xmin
                    ymin = tmp_ymin if ymin > tmp_ymin else ymin
                    xmax = tmp_xmax if xmax < tmp_xmax else xmax
                    ymax = tmp_ymax if ymax < tmp_ymax else ymax

        xmin = int(np.floor(xmin) - offset)
        ymin = int(np.floor(ymin) - offset)
        xmax = int(np.floor(xmax) + offset)
        ymax = int(np.floor(ymax) + offset)
        ax.set_xlim(xmin, xmax)
        ax.set_xticks(np.arange(xmin, xmax + 1, step))
        ax.set_ylim(ymin, ymax)
        ax.set_yticks(np.arange(ymin, ymax + 1, step))
        ax.set_aspect('equal')

        if swapxy:
            ax.set_xlabel('y')
            ax.set_ylabel('x')
        else:
            ax.set_xlabel('x')
            ax.set_ylabel('y')

        if save:
            # 画像保存
            filename = '{}.png'.format(name)
            if save_folder_path:
                if not os.path.isdir(save_folder_path):
                    os.makedirs(save_folder_path)
                filename = os.path.join(save_folder_path, filename)
            plt.savefig(filename)
        else:
            # window表示
            plt.show(block=True)

        plt.clf()
        plt.close(fig=fig)

    @staticmethod
    def draw_color_corner(
            points: NDArray, swapxy=True, title: str = None,
            save=False, save_folder_path: str = None):
        """角度による色つきコーナー点の描画

        Args:
            points (NDArray): 輪郭線の頂点列
            swapxy (bool, optional): \
                xy座標を反転して描画するか否か. Defaults to True.
            title (str, optional): タイトル. Defaults to None.
            save (bool, optional): \
                True=画像保存, False=Window表示. Defaults to False.
            save_folder_path (str, optional): \
                保存先フォルダパス. Defaults to None.
        """
        if len(points) == 0:
            return

        name = 'corners' if title is None else title

        offset = 10
        step = 10
        fig = plt.figure(figsize=WIN_SIZE, dpi=90)
        plt.subplots_adjust(wspace=0.6, hspace=0.6)
        fig.set_frameon(True)
        ax = fig.add_subplot(111)
        ax.set_title(name)

        # 線の描画
        if swapxy:
            x = points[:, 0:1]
            y = points[:, 1:2]
            points = np.hstack([y, x])

        color_code = '#000000'  # black
        x = points[:, 0:1]
        y = points[:, 1:2]
        ax.plot(x, y, color=color_code, linewidth=2.0,
                solid_capstyle='round', zorder=2, alpha=0.7)
        xmin = x.min()
        xmax = x.max()
        ymin = y.min()
        ymax = y.max()

        # コーナーの描画
        angles = np.zeros(len(points), dtype=np.float32)
        for i in range(len(points)):
            prev = i - 1
            next = i + 1 if i < len(points) - 1 else 0
            vec1 = points[prev] - points[i]
            vec2 = points[next] - points[i]
            angles[i] = GeoUtil.angle(vec1, vec2)

        x = points[:, 0:1]
        y = points[:, 1:2]
        sc = ax.scatter(x, y, c=angles, marker='.', cmap='jet', zorder=3)
        clb = plt.colorbar(sc)
        clb.set_label('degree')
        
        xmin = int(np.floor(xmin) - offset)
        ymin = int(np.floor(ymin) - offset)
        xmax = int(np.floor(xmax) + offset)
        ymax = int(np.floor(ymax) + offset)
        ax.set_xlim(xmin, xmax)
        ax.set_xticks(np.arange(xmin, xmax + 1, step))
        ax.set_ylim(ymin, ymax)
        ax.set_yticks(np.arange(ymin, ymax + 1, step))
        ax.set_aspect('equal')

        if swapxy:
            ax.set_xlabel('y')
            ax.set_ylabel('x')
        else:
            ax.set_xlabel('x')
            ax.set_ylabel('y')

        if save:
            # 画像保存
            filename = '{}.png'.format(name)
            if save_folder_path:
                if not os.path.isdir(save_folder_path):
                    os.makedirs(save_folder_path)
                filename = os.path.join(save_folder_path, + filename)
            plt.savefig(filename)
        else:
            # window表示
            plt.show(block=True)

        plt.clf()
        plt.close(fig=fig)
