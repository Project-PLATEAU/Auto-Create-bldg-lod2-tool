# -*- coding:utf-8 -*-
import numpy as np
import cv2
import os
import math
import pathlib

from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.geometry import LineString
from enum import Enum
from ..util.objinfo import BldElementType, ObjInfo
from ..util.faceinfo import MaterialInfo
from ..util.cvsupportjp import Cv2Japanese
from ..util.log import Log, ModuleType, LogLevel
from .photoimage import PhotoImage


class VerticalObject():
    """建物情報クラス
    """
    photonum = 0
    photolist = None

    def __init__(
            self, obj_path: str, photonum: int,
            photolist: list[PhotoImage]) -> None:
        """コンストラクタ

        Args:
            obj_path (str): OBJフォルダパス
            photonum (int): 写真枚数
            photolist (list[PhotoImage]): 写真のリスト
        """
        self._photonum = photonum          # 写真数
        self._photolist = photolist        # 写真リスト

        self._texcollection = None         # 出力テクスチャ
        self._rooftexture = list()         # 屋根面テクスチャ情報
        self._walltexture = list()         # 壁面テクスチャ情報

        self._vertexroof = list()
        self._vertexwall = list()

        # OBJファイル読み出し
        self._obj_info = ObjInfo()
        self._obj_info.read_file(obj_path)
        self._obj_filename = os.path.splitext(os.path.basename(obj_path))[0]

        # 屋根面の座標配列読み込み
        polygon_list_r = self._obj_info.get_polygon_list(BldElementType.ROOF)
        if polygon_list_r:
            for poly in polygon_list_r:
                ar = np.zeros((0, 3))
                for pos in poly:
                    ar = np.append(
                        ar, np.array([[pos.x, pos.y, pos.z]]), axis=0)
                self._vertexroof.append(ar)

        # 壁面の座標配列読み込み
        polygon_list_w = self._obj_info.get_polygon_list(BldElementType.WALL)
        if polygon_list_w:
            for poly in polygon_list_w:
                ar = np.zeros((0, 3))
                for pos in poly:
                    ar = np.append(
                        ar, np.array([[pos.x, pos.y, pos.z]]), axis=0)
                self._vertexwall.append(ar)

    def select_rooftexture(self):
        """屋根面テクスチャ画像を検索してセットする
        """
        # テクスチャ画像出力オブジェクト作成
        self._texcollection = DstTextureFile()
        tex_collection = self._texcollection

        # 屋根用テクスチャ情報オブジェクトを確保
        self._rooftexture = [
            TextureInfo() for i in range(len(self._vertexroof))]

        # 屋根面のテクスチャ画像として使用する写真を決定する
        # 屋根面枚数分のテクスチャを全画像から検索
        for r_idx, verroof in enumerate(self._vertexroof):
            tex_coord = np.zeros((len(verroof), 2))    # 画像上の座標
            roof_coord = np.zeros((len(verroof), 2))
            max_area, area = 0.0, 0.0
            roof_valid = [0] * len(verroof)
            set_idx = 0             # 写真のインデックス

            for i in range(self._photonum):
                # 屋根面一枚分の座標の、画像範囲内の頂点位置数を求める
                in_count = 0
                for j, ver in enumerate(verroof):
                    roof_valid[j] = self._photolist[i].get_imagepos(ver,
                                                                    tex_coord[j])
                    if roof_valid[j]:
                        in_count += 1

                if in_count < len(verroof):
                    continue

                # 全点が画像上の点であれば面積最大の画像を選択する
                point = [0] * len(tex_coord)
                for num, tex in enumerate(tex_coord):
                    point[num] = Point(tex)
                area = Polygon(point).area
                if area < max_area:
                    continue
                max_area = area
                set_idx = i
                roof_coord = tex_coord.copy()

            if np.all(roof_coord == 0):
                # テクスチャ画像が見つかったか
                continue

            # テクスチャ画像が未登録の場合は追加
            if set_idx not in \
               (srcTex.ref_image_index for srcTex in tex_collection.src_texture):

                # 参照テクスチャ画像オブジェクトを作成
                newsrctex = tex_collection.get_newsrc_texture()
                newsrctex.ref_image_index = set_idx
                newsrctex.ref_image = self._photolist[set_idx]
  
            # 屋根面の座標配列を追加
            for tex in tex_collection.src_texture:
                if tex.ref_image_index == set_idx:
                    self._rooftexture[r_idx].texver = tex.append_texturecoord(
                        roof_coord)
                    self._rooftexture[r_idx].texinfo = tex
                    self._rooftexture[r_idx].polarea = max_area

                    refimg = self._photolist[set_idx].filename
                    Log.output_log_write(
                        LogLevel.DEBUG,
                        ModuleType.PASTE_TEXTURE,
                        'roof refImage:' + refimg)

    def select_walltexture(self):
        """壁面テクスチャ画像を検索してセットする
        """
        # テクスチャ画像出力オブジェクト
        tex_collection = self._texcollection

        # 壁面用テクスチャ情報オブジェクトを確保
        self._walltexture = [
            TextureInfo() for i in range(len(self._vertexwall))]

        for w_idx, wall in enumerate(self._vertexwall):
            set_idx = 0             # 写真のインデックス
            r_common = list()
            r_tmp = list()
            max_area, area = 0.0, 0.0
            roof_idx = 0
            imgpos_chk = True
            set_wall_imgpos = np.zeros((len(wall), 2))

            # 接地する屋根面の検索/屋根面と一致する頂点の検索
            for i in range(len(self._vertexroof)):
                r_tmp = [r for r in self._vertexroof[i] if r in wall]

                # 2頂点以上接地している屋根面が対象
                if 1 < len(r_tmp):
                    if len(r_common) != 0:
                        # すでに入っている屋根面の方が高い位置にある場合
                        if r_tmp[0][2] < r_common[0][2]:
                            continue
                    r_common = r_tmp.copy()
                    roof_idx = i

            # 全写真から壁面に最適なテクスチャを検索
            for i in range(self._photonum):
                roof_imgPos = np.zeros((len(self._vertexroof[roof_idx]), 2))
                wall_imgPos = np.zeros((len(wall), 2))

                imgpos_chk = True

                # 対象の屋根面が画像の範囲内にあるか
                for num, r_ver in enumerate(self._vertexroof[roof_idx]):
                    if not self._photolist[i].get_imagepos(r_ver,
                                                           roof_imgPos[num]):
                        # 指定範囲は画像外
                        imgpos_chk = False
                        break
                                     
                # 対象の壁面が画像の範囲内にあるか
                for num, w_ver in enumerate(wall):
                    if not self._photolist[i].get_imagepos(w_ver,
                                                           wall_imgPos[num]):
                        # 指定範囲は画像外
                        imgpos_chk = False
                        break

                if imgpos_chk is False:
                    # 対象の屋根面と壁面が画像の範囲内にない
                    continue

                # 陰面判定
                if self._judge_hiddensurface(
                        wall, wall_imgPos, roof_imgPos, r_common):
                    # 陰面あり
                    continue

                # 壁面用テクスチャ情報保持
                # より最適なテクスチャが見つかった場合は上書き
                # 全点が画像上の点であれば面積最大の画像を選択する
                point = [0] * len(wall_imgPos)
                for num, tex in enumerate(wall_imgPos):
                    point[num] = Point(tex)
                area = Polygon(point).area
                if area < max_area:
                    continue
                max_area = area
                set_idx = i
                set_wall_imgpos = wall_imgPos.copy()

            if np.all(set_wall_imgpos == 0):
                # テクスチャ画像が見つかったか
                continue

            # テクスチャ画像が未登録の場合は追加
            if set_idx not in\
               (srcTex.ref_image_index for srcTex in tex_collection.src_texture):
                # 参照テクスチャ画像オブジェクトを作成
                newsrctex = tex_collection.get_newsrc_texture()
                newsrctex.ref_image_index = set_idx
                newsrctex.ref_image = self._photolist[set_idx]

            # 壁面毎のテクスチャ情報を追加
            for tex in tex_collection.src_texture:
                if tex.ref_image_index == set_idx:
                    self._walltexture[w_idx].texver = tex.append_texturecoord(
                        set_wall_imgpos)
                    self._walltexture[w_idx].texinfo = tex
                    self._walltexture[w_idx].polarea = max_area

                    refimg = self._photolist[set_idx].filename
                    Log.output_log_write(
                        LogLevel.DEBUG,
                        ModuleType.PASTE_TEXTURE,
                        'wall refImage:' + refimg)

    def _judge_hiddensurface(self, wall, wall_imgpos, roof_imgpos, r_common):
        """壁面の陰面判定

        Args:
            wall (float[]): 壁面の座標(絶対座標)
            wall_imgpos (float[]): 壁面の座標(画像座標)
            roof_imgpos (float[]): 屋根面の座標(画像座標)
            r_common (float[]): 壁面座標のうち、屋根面と重複している座標(絶対座標)

        Returns:
            bool: 陰面あり(True)/陰面なし(False)
        """
        prev_flag = FaceInfo.NONE
        next_flag = FaceInfo.NONE
        prev_ver = [0 for i in range(2)]
        imgpos_chk = True
        # 壁面の最初の頂点を最後に追加する(一周確認を行う)
        wall_around = np.append(wall, [wall[0]], axis=0)
        for num, w_ver in enumerate(wall_around):
            imgpos_chk = False

            if any(np.array_equal(ary, w_ver) for ary in np.array(r_common)):
                next_flag = FaceInfo.ROOF
            else:
                next_flag = FaceInfo.NOT_ROOF

            w_num = num
            if num == len(wall_imgpos):
                w_num = 0

            # 屋根面の頂点を含む(陰面判定対象)
            if prev_flag == FaceInfo.NOT_ROOF and next_flag == FaceInfo.ROOF:
                # 屋根面→屋根面以外
                tuple_equal = np.where(np.array_equal(
                    ary, wall_imgpos[w_num]) for ary in np.array(roof_imgpos))
                imgpos_chk = self._is_occluded_byself(
                    tuple_equal[0][0], prev_ver, roof_imgpos)

            elif prev_flag == FaceInfo.ROOF and next_flag == FaceInfo.NOT_ROOF:
                # 屋根面以外→屋根面
                tuple_equal = np.where(np.array_equal(
                    ary, prev_ver) for ary in np.array(roof_imgpos))
                imgpos_chk = self._is_occluded_byself(
                    tuple_equal[0][0], wall_imgpos[w_num], roof_imgpos)

            if imgpos_chk is True:
                # 陰面である
                return imgpos_chk

            if num == len(wall_imgpos):
                # 追加した最後の頂点まで検索したら抜ける
                break

            prev_flag = next_flag
            prev_ver = wall_imgpos[num]

        return imgpos_chk

    def _is_occluded_byself(self, index, img_base_pt, ver_img_pt):
        """ある頂点の底面(地盤)位置が自身の陰面かどうかを判定する

        Args:
            index (int): 交差判定対象の面のインデックス
            img_base_pt (float[]): 判定対象の頂点
            ver_img_pt (float[]): 交差判定を行う面の座標情報

        Returns:
            bool: 陰面(True)/陰面ではない(False)
        """
        num_ver = len(ver_img_pt)

        a = []
        for img_pt in ver_img_pt:
            a.append((img_pt[0], img_pt[1]))

        polygon = Polygon(a)
        if polygon.contains(Point(img_base_pt[0], img_base_pt[1])):
            return True

        # 頂点と写真中心を結ぶ直線と、頂点に隣接しない線分が交差するかどうかで判定する
        for i in range(num_ver):
            ie = (i + 1) % num_ver
            # 頂点と隣接する線分は検証対象外
            if (i == index) or (ie == index):
                continue

            line1 = LineString([(img_base_pt[0], img_base_pt[1]),
                                (ver_img_pt[index][0], ver_img_pt[index][1])])
            line2 = LineString([(ver_img_pt[i][0], ver_img_pt[i][1]),
                                (ver_img_pt[ie][0], ver_img_pt[ie][1])])
            if line1.touches(line2):
                return True
            # if Geometry::isCrossing(img_base_pt, ver_img_pt[index], ver_img_pt[i], ver_img_pt[ie]):
                # 交差する場合は陰面
            #    return TRUE

        return False

    def output_texture(self, objdir, outputdir, mtl_file_name):
        """テクスチャ画像・情報の出力

        Args:
            objdir (string): OBJファイル出力先フォルダパス
            outputdir (string): テクスチャ情報出力先のフォルダパス
            mtl_file_name (string): マテリアルファイルパス

        Returns:
            bool: テクスチャ画像出力結果
        """
        # テクスチャ画像の出力
        ret = self._texcollection.output_texture(
            os.path.join(outputdir, self._obj_filename))
        
        if ret:
            for num, roof in enumerate(self._rooftexture):
                if roof.polarea != 0:
                    out_texver = roof.texinfo.outputcoord[roof.texver]
                    p = list()
                    for ver in out_texver:
                        p.append(Point(ver[0], ver[1]))
                    self._obj_info.append_texture(BldElementType.ROOF, num, p)

            for num, wall in enumerate(self._walltexture):
                if wall.polarea != 0:
                    out_texver = wall.texinfo.outputcoord[wall.texver]
                    p = list()
                    for ver in out_texver:
                        p.append(Point(ver[0], ver[1]))
                    self._obj_info.append_texture(BldElementType.WALL, num, p)

            # マテリアル情報設定
            mtl_info = MaterialInfo(self._obj_filename)
            jpg_path = os.path.join(
                pathlib.Path(outputdir).name, self._obj_filename + ".jpg")
            # テクスチャ画像パスの区切り文字は/固定とする
            mtl_info.map_kd = jpg_path.replace(os.path.sep, '/')

            self._obj_info.mtl_file_name = mtl_file_name
            self._obj_info.set_mtl_info(mtl_info)

            output_path = os.path.join(objdir, self._obj_filename + ".obj")
            self._obj_info.write_file(output_path)

        return ret

    def output_optional_obj(
            self, objdir: str, texture_dir: str, mtl_file_name: str):
        """オプションのOBJ出力処理(最終結果にOBJファイルを出力する場合の処理)

        Args:
            objdir (str): OBJファイル出力先フォルダパス
            texture_dir (str): テクスチャ画像フォルダパス
            mtl_file_name (str): マテリアルファイルパス

        Note:
            output_texture()を先に呼び出す必要がある\n
            オプション出力のOBJファイルはCityGMLのテクスチャ画像を参照するため、
            この処理ではテクスチャ画像の出力はせず、output_texture()に任せる
        """
        # obj出力フォルダ基点のテクスチャ画像の相対パス
        relpath = os.path.relpath(texture_dir, objdir)

        for num, roof in enumerate(self._rooftexture):
            if roof.polarea != 0:
                out_texver = roof.texinfo.outputcoord[roof.texver]
                p = list()
                for ver in out_texver:
                    p.append(Point(ver[0], ver[1]))
                self._obj_info.append_texture(BldElementType.ROOF, num, p)

        for num, wall in enumerate(self._walltexture):
            if wall.polarea != 0:
                out_texver = wall.texinfo.outputcoord[wall.texver]
                p = list()
                for ver in out_texver:
                    p.append(Point(ver[0], ver[1]))
                self._obj_info.append_texture(BldElementType.WALL, num, p)

        # マテリアル情報設定
        mtl_info = MaterialInfo(self._obj_filename)
        jpg_path = os.path.join(relpath, self._obj_filename + ".jpg")
        # テクスチャ画像パスの区切り文字は/固定とする
        mtl_info.map_kd = jpg_path.replace(os.path.sep, '/')

        self._obj_info.mtl_file_name = mtl_file_name
        self._obj_info.set_mtl_info(mtl_info)

        output_path = os.path.join(objdir, self._obj_filename + ".obj")
        self._obj_info.write_file(output_path, swap_xy=False)


class FaceInfo(Enum):
    """面情報定義クラス
    """
    NONE = 0
    ROOF = 1
    NOT_ROOF = 2


class SrcTexture():
    """テクスチャ画像クラス
    """
    def __init__(self):
        """コンストラクタ
        """
        # 参照する写真
        self.ref_image = None
        # 参照する写真のphotoListインデックス
        self.ref_image_index = 0
        # オリジナルのテクスチャ座標配列
        self.texcoord = list()
        # 貼り付け先のテクスチャ座標配列
        self.outputcoord = list()

        # 割り当て済みテクスチャ座標数
        self.num_texcoord = -1
        # 参照フラグ
        self.refflag = 0

    def append_texturecoord(self, point):
        """テクスチャ座標を追加する

        Args:
            point (float[]): 追加する座標

        Returns:
            int: 追加された座標の合計数
        """
        self.num_texcoord += 1
        self.texcoord.append(point)
        return self.num_texcoord


class TextureInfo():
    """テクスチャ情報クラス
    """
    def __init__(self):
        """コンストラクタ
        """
        # 参照テクスチャ
        self.texinfo = None
        # オリジナル画像での各頂点のテクスチャ座標インデックス
        self.texver = 0
        # 設定された参照テクスチャにおける画像座標系での面積
        self.polarea = 0


class DstTextureFile():
    """出力テクスチャファイルクラス
    """
    OUTPUT_MAX_W = 4096
    OUTPUT_MAX_H = 4096

    def __init__(self):
        """コンストラクタ
        """
        # 入力テクスチャ画像
        self.src_texture = list()
        # 入力テクスチャ画像数
        self.num_srctex = 0
        self._outputmargin = 2

    def get_newsrc_texture(self):
        """入力テクスチャオブジェクトを作成する

        Returns:
            tex_info: 入力テクスチャオブジェクト
        """
        tex_info = SrcTexture()
        self.src_texture.append(tex_info)
        self.num_srctex += 1
        return tex_info
    
    def output_texture(self, outputpath):
        """テクスチャ画像出力

        Args:
            outputpath (string): テクスチャ情報出力先のフォルダパス

        Returns:
            bool: テクスチャ出力成功(True)/テクスチャ出力画像なし(False)
        """

        def get_texpoly_bbox(texver, img_width, img_height):
            min_ver = np.floor(np.minimum.reduce(texver)).astype(np.int32)
            max_ver = np.ceil(np.maximum.reduce(texver)).astype(np.int32)
            if (0 <= min_ver[0] - self._outputmargin) \
                and (0 <= min_ver[1] - self._outputmargin) \
                and (max_ver[0] + self._outputmargin < img_width) \
                and (max_ver[1] + self._outputmargin < img_height):
                # 元画像をはみ出さない場合、マージンをつける
                min_ver = min_ver - self._outputmargin
                max_ver = max_ver + self._outputmargin
                output_margin = self._outputmargin
            else:
                output_margin = 0

            polygon_w = max_ver[0] - min_ver[0] + 1
            polygon_h = max_ver[1] - min_ver[1] + 1

            return min_ver[0], min_ver[1], polygon_w, polygon_h, output_margin


        if self.num_srctex < 1:
            return False

        # 出力画像サイズの計算 => output_h, output_w
        origin_w = 0
        origin_h = 0
        linemax_h = 0
        linemax_w = 0
        for srcTex in self.src_texture:
            img = Cv2Japanese.imread(os.path.join(srcTex.ref_image.photodir, srcTex.ref_image.filename))
            for texver in srcTex.texcoord:
                _, _, polygon_w, polygon_h, _ = get_texpoly_bbox(texver, img.shape[1], img.shape[0])

                if self.OUTPUT_MAX_W < origin_w + polygon_w:
                    # 出力幅を超えたら次の行へ移動
                    origin_w = polygon_w
                    origin_h += linemax_h
                    linemax_h = polygon_h
                else:
                    origin_w += polygon_w
                    # 同じ行で最大の高さ
                    linemax_h = linemax_h if linemax_h > polygon_h else polygon_h
                    
                # 最大行長さ
                linemax_w = linemax_w if linemax_w  > origin_w else origin_w

        output_h = origin_h + linemax_h
        output_w = linemax_w

        # 出力画像サイズを2^nに補正
        val = 1
        while val < output_h:
            val *= 2
        output_h = val

        val = 1
        while val < output_w:
            val *= 2
        output_w = val

        # 白紙の出力画像を作成
        output = np.full((output_h, output_w, 3), 255, dtype='uint8')

        # テクスチャ貼り付け
        origin_h = 0
        origin_w = 0
        linemax_h = 0
        for srcTex in self.src_texture:
            # オリジナル画像のオープン
            img = Cv2Japanese.imread(os.path.join(srcTex.ref_image.photodir, srcTex.ref_image.filename))
            for texver in srcTex.texcoord:
                min_x, min_y, polygon_w, polygon_h, output_margin = get_texpoly_bbox(texver, img.shape[1], img.shape[0])

                # 背景画像（白画像）
                back = np.full((polygon_h, polygon_w, 3), 255, dtype='uint8')

                # マスク画像
                mask = np.full((polygon_h, polygon_w), 0, dtype='uint8')

                # 前景画像（テクスチャ）
                dst = img[min_y:min_y + polygon_h, min_x:min_x + polygon_w]

                # テクスチャポリコン座標の原点を(min_x, min_y)にする
                poly_ver = texver - [min_x, min_y]

                # 前景（テクスチャポリコン+マージン）のマスクを生成
                cv2.fillPoly(mask, [poly_ver.astype(np.int64)], color=(255, 255, 255))
                cv2.polylines(mask, [poly_ver.astype(np.int64)], isClosed=True, color=(255, 255, 255), thickness=output_margin*2)

                # 前景画像+背景画像
                polygon = np.where(mask[:, :, np.newaxis] == 0, back, dst)

                # 出力幅を超えたら次の行へ移動                
                if output_w < origin_w + polygon_w:
                    origin_w = 0
                    origin_h += linemax_h
                    linemax_h = 0                   

                # テクスチャ貼付け
                output[origin_h:origin_h + polygon_h, origin_w:origin_w + polygon_w] = polygon

                # テクスチャポリコンの座標を貼り付け先の座標に変換               
                xy_set = texver - [min_x, min_y] + [origin_w, origin_h]

                # XY座標→UV左上原点→UV左下原点に変換
                uv_set = np.array([output_w, output_h])
                srcTex.outputcoord.append(abs(xy_set / uv_set - [0, 1]))

                # 更新
                origin_w += polygon_w
                linemax_h = linemax_h if linemax_h > polygon_h else polygon_h

        # テクスチャ貼付け画像出力
        if self.OUTPUT_MAX_H < output_h:
            # 高さが指定サイズを超えた場合、最大の高さに合わせて縮小する
            h, w = output.shape[:2]
            output_w = round(w * (self.OUTPUT_MAX_H / h))
            output_rs = cv2.resize(output, dsize=(output_w, self.OUTPUT_MAX_H))
            ret = Cv2Japanese.imwrite(outputpath + '.jpg', output_rs)
        else:
            ret = Cv2Japanese.imwrite(outputpath + '.jpg', output)

        return ret
