# -*- coding:utf-8 -*-
import sys
import shutil
import os
import cv2
import lxml
import math
from lxml import etree
import numpy as np
import datetime
import time
from tqdm import tqdm

from .layoutRect import LayoutTexture
from .util.parammanager import ParamManager
from .thirdparty import plateaupy as plapy
from .util.config import Config
from .util.cvsupportjp import Cv2Japanese

def main():
    """メイン関数
    """
    # 引数入力暫定
    args = ["Atlas_Prot.py", os.path.join(".", "param.json")]

    if len(args) != 2:
        print('usage: python AutoCreateLod2.py param.json')
        sys.exit()

    try:
        param_manager = ParamManager()
        param_manager.read(args[1])
    except Exception as e:
        # param_manager.debug_log_output = False
        # log = Log(param_manager, args[1])
        # log.output_log_write(LogLevel.ERROR, ModuleType.NONE, e)
        # log.log_footer()
        print(e)
        sys.exit()

    try:
        time = datetime.datetime.now()
        print(time)

        citygml_manager = CityGmlManager(
            param_manager.input_gml_folder_path,
            param_manager.output_gml_folder_path,
            param_manager.output_width,
            param_manager.output_height,
            param_manager.extent_pixel)

        # CityGML読み込み
        citygml_infoList = citygml_manager.input_citygml()

        # アトラス化・画像出力
        layout_Rect = LayoutTexture(
            param_manager.input_gml_folder_path,
            param_manager.output_gml_folder_path,
            param_manager.output_width,
            param_manager.output_height,
            param_manager.background_color,
            param_manager.extent_pixel)
        layout_Rect.layout_texture_main(citygml_infoList)

        # CityGML出力
        citygml_manager.output_citygml()

        time = datetime.datetime.now()
        print(time)

    except Exception as e:
        print(e)
        sys.exit()


class CityGmlManager():
    """CityGML処理クラス
    """

    def __init__(self, inputpath, outputpath, outputwidth, outputheight, extentpix) -> None:
        """コンストラクタ

        Args:
            inputpath(String):CityGML入力フォルダパス
            outputpath(String):CityGML出力フォルダパス
            outputwidth(int):出力画像横サイズ
            outputheight(int):出力画像縦サイズ
            extentpix(int):ポリゴン余白
        """
        self.inputpath = inputpath
        self.outputpath = outputpath
        self.output_w = outputwidth
        self.output_h = outputheight
        self.extent_pixel = extentpix
        self.citygml_infoList = list()
        self.citygml_namelist = list()

    def input_citygml(self):
        """CityGMLファイル情報入力
        """
        for file in os.listdir(self.inputpath):
            base, ext = os.path.splitext(file)
            if ext == '.gml':
                self.citygml_namelist.append(base)

            if os.path.isdir(os.path.join(self.inputpath, file)):
                try:
                    if not os.path.isdir(os.path.join(self.outputpath, file)):
                        os.makedirs(os.path.join(self.outputpath, file))
                except Exception:
                    raise Exception(
                        os.path.join(self.outputpath, file) + ' cannot make.')

        # CityGMLファイル毎に処理
        for gml_file in tqdm(self.citygml_namelist, desc='loading gml files'):
            # print(self.inputpath + "\\" + gml_file + ".gml")

            plbld = plapy.plbldg(self.inputpath + "\\" + gml_file + ".gml")
            mesh_list = list()

            for bldg in plbld.buildings:

                if any(bldg.lod2ground) or any(bldg.lod2roof) or any(bldg.lod2wall):
                    mesh = SetyMesh()
                    for lod2ground in bldg.lod2ground:
                        mesh.ids.append(lod2ground)
                    for lod2roof in bldg.lod2roof:
                        mesh.ids.append(lod2roof)
                    for lod2wall in bldg.lod2wall:
                        mesh.ids.append(lod2wall)

                    if bldg.lod0RoofEdge:
                        lat = bldg.lod0RoofEdge[0][0][0]
                        lon = bldg.lod0RoofEdge[0][0][1]
                        mesh.code = self.get_mesh(lat, lon)

                    elif bldg.lod0FootPrint:
                        lat = bldg.lod0FootPrint[0][0][0]
                        lon = bldg.lod0FootPrint[0][0][1]
                        mesh.code = self.get_mesh(lat, lon)
                    mesh_list.append(mesh)

            citygmlInfo = CityGmlInfo()
            self.citygml_infoList.append(citygmlInfo)

            citygmlInfo.in_cityGmlName = (
                self.inputpath + "\\" + gml_file + ".gml")
            citygmlInfo.out_cityGmlName = (
                self.outputpath + "\\" + gml_file + ".gml")

            parser = etree.XMLParser(remove_blank_text=True)
            tree = etree.parse(
                self.inputpath + "\\" + gml_file + ".gml", parser)
            root = tree.getroot()
            self._nsmap = self.removeNoneKeyFromDic(root.nsmap)

            apps = tree.xpath(
                '/core:CityModel/app:appearanceMember/app:Appearance/ \
                    app:surfaceDataMember/app:ParameterizedTexture',
                namespaces=self._nsmap)

            for elem1 in apps:
                # 建物毎に処理
                build = BuildingInfo()
                citygmlInfo.add_buildingInfo(build)

                imageURI = elem1.xpath('app:imageURI', namespaces=self._nsmap)
                build.in_imgpath = imageURI[0].text

                # 解像度取得
                build.in_imgsize = (
                    os.path.getsize(self.inputpath + "/" + build.in_imgpath)) \
                    / 1024
                
                im = Cv2Japanese.imread(self.inputpath + "/" + build.in_imgpath)
                build.img_h = im.shape[0]
                build.img_w = im.shape[1]

                mimeType = elem1.xpath('app:mimeType', namespaces=self._nsmap)
                build.mimeType = mimeType[0].text

                # ポリゴン毎に処理
                target = elem1.xpath('app:target', namespaces=self._nsmap)
                for elem2 in target:
                    uri = elem2.get("uri")

                    # 4・5次メッシュの検索
                    if build.meshCode == 0:
                        for mesh_elem in mesh_list:
                            if uri in mesh_elem.ids:
                                build.meshCode = mesh_elem.code

                    texCoord = elem2.xpath(
                        'app:TexCoordList/app:textureCoordinates',
                        namespaces=self._nsmap)
                    ring = texCoord[0].get("ring")
                    clist = texCoord[0].text.split(" ")
                    
                    # ポリゴンの座標をコピーする
                    if ((self.output_w == build.img_w) and (self.output_h == build.img_h)) or ((self.output_w < build.img_w) or (self.output_h < build.img_h)):
                        #  一定サイズ以上の場合はそのままの座標値を入力
                        coord_f = []
                        clistiter = iter(clist)
                        for u, v in zip(clistiter, clistiter):
                            coord_f.append(float(u))
                            coord_f.append(float(v))
                        arrays_r = np.reshape(np.array(coord_f), (-1, 2))
                    else:
                        coord_f = []
                        clistiter = iter(clist)
                        for u, v in zip(clistiter, clistiter):
                            coord_f.append(float(u) * build.img_w - 0.5)
                            coord_f.append((1.0 - float(v)) * build.img_h - 0.5)
                        arrays_r = np.reshape(np.array(coord_f), (-1, 2))

                    build.add_polygonInfo(
                        uri, ring, arrays_r, [build.img_w, build.img_h], self.extent_pixel)

                    # # ポリゴンを高さ順にソートする
                    # if (build.img_w < self.output_w) and (build.img_h < self.output_h):
                    #     build.polygon = sorted(
                    #         build.polygon,
                    #         key=lambda PolygonInfo: PolygonInfo.useH,
                    #         reverse=True)
                
            citygmlInfo.builds = sorted(
                citygmlInfo.builds,
                key=lambda builds: builds.meshCode)
            
        return self.citygml_infoList
            
    def output_citygml(self):
        """CityGMLファイル情報出力
        """

        if self.outputpath:
            # 出力フォルダの作成
            if not os.path.isdir(self.outputpath):
                os.makedirs(self.outputpath)
        
        # CityGMLファイル毎に処理
        for info in self.citygml_infoList:

            shutil.copy(info.in_cityGmlName,
                        info.out_cityGmlName)
            
            parser = etree.XMLParser(remove_blank_text=True)
            tree = etree.parse(info.out_cityGmlName, parser)
            root = tree.getroot()

            print(self._nsmap['app'])

            # 既存のテクスチャ記述部分削除
            for app in root.findall(
                    "{" + self._nsmap['app'] + "}" + 'appearanceMember'):
                root.remove(app)

            temp_imgpath = None

            tex_appmem_elem = lxml.etree.Element(
                "{" + self._nsmap['app'] + "}" + "appearanceMember")
            tex_app_elem = lxml.etree.SubElement(
                tex_appmem_elem,
                "{" + self._nsmap['app'] + "}" + "Appearance")
            tex_theme_elem = lxml.etree.SubElement(
                tex_app_elem, "{" + self._nsmap['app'] + "}" + "theme")
            tex_theme_elem.text = "rgbTexture"

            for build in info.builds:
                if build is not None:
                    elem5 = None
                    if temp_imgpath != build.out_imgpath:                  
                        elem1 = lxml.etree.SubElement(
                            tex_app_elem,
                            "{" + self._nsmap['app'] + "}"
                                + "surfaceDataMember")
                        elem2 = lxml.etree.SubElement(
                            elem1,
                            "{" + self._nsmap['app'] + "}"
                                + "ParameterizedTexture")
                        elem3 = lxml.etree.SubElement(
                            elem2,
                            "{" + self._nsmap['app'] + "}"
                                + "imageURI")
                        elem3.text = build.out_imgpath
                        temp_imgpath = build.out_imgpath
                        elem4 = lxml.etree.SubElement(
                            elem2,
                            "{" + self._nsmap['app'] + "}"
                                + "mimeType")
                        elem4.text = build.mimeType

                    for poly in build.polygon:
                        str_list = []
                        elem5 = lxml.etree.SubElement(
                            elem2,
                            "{" + self._nsmap['app'] + "}" + "target",
                            {"uri": poly.target_uri})
                        elem6 = lxml.etree.SubElement(
                            elem5,
                            "{" + self._nsmap['app'] + "}" + "TexCoordList")
                        for coord in poly.out_texcoord:
                            str_list.append(str(coord[0]))
                            str_list.append(str(coord[1]))
                        elem7 = lxml.etree.SubElement(
                            elem6,
                            "{" + self._nsmap['app'] + "}"
                                + "textureCoordinates",
                            {"ring": poly.coord_ring})
                        elem7.text = ' '.join(str_list)

            root.append(tex_appmem_elem)

            # LoD2 CityGML書き出し
            lxml.etree.indent(root, space="\t")  # tab区切り
            print(info.out_cityGmlName)
            tree.write(info.out_cityGmlName,
                       pretty_print=True,
                       xml_declaration=True,
                       encoding="utf-8")

    def removeNoneKeyFromDic(self, nsmap):
        """namespase取得
        """
        newnsmap = dict()
        for k, v in nsmap.items():
            if k is not None:
                newnsmap[k] = v
        return newnsmap

    def get_mesh(self, lat, lon):
        """メッシュ取得
        """
        code4 = (int(math.floor(lat * 240)) % 2 * 2
                 + int(math.floor((lon - 100) * 160)) % 2 + 1)
        code5 = (int(math.floor(lat * 480)) % 2 * 2
                 + int(math.floor((lon - 100) * 320)) % 2 + 1)
        #return (code4 * 10 + code5)
        return (code4)
    

class CityGmlInfo():
    """CityGML情報クラス
    """
    def __init__(self) -> None:
        """コンストラクタ
        """
        self.builds = list()            # 建物情報リスト
        self.in_cityGmlName = None      # 入力CityGMLファイル名
        self.out_cityGmlName = None     # 出力CityGMLファイル名
        self.imgFolderName = None       # 画像フォルダ名

    def add_buildingInfo(self, build):
        """建物オブジェクトの作成
        """
        self.builds.append(build)
        return self.builds


class BuildingInfo():
    """建物情報クラス
    """
    def __init__(self) -> None:
        """コンストラクタ
        """
        self.mimeType = None        # 拡張子タイプ

        self.in_imgpath = None      # 入力画像パス
        self.in_imgsize = 0         # 入力画像サイズ
        self.img_w = 0
        self.img_h = 0
        self.out_img_w = 0
        self.out_img_h = 0
        self.meshCode = 0     # 4次+5次メッシュ

        self.out_imgpath = None     # 出力画像パス
        self.out_imgsize = 0        # 出力画像サイズ

        self.polygon = list()         # ポリゴン情報リスト

    def add_polygonInfo(self, uri, ring, coords, imgSize, extentPixel):

        poly = PolygonInfo()
        poly.target_uri = uri
        poly.coord_ring = ring
        poly.in_texcoord = coords
        
        if (int(coords.max(axis=0)[0]) - int(coords.min(axis=0)[0])) == 0 or (int(coords.max(axis=0)[1]) - int(coords.min(axis=0)[1])) == 0 :
        	# 面積なし(Line)の場合は幅2pixポリゴンとして切り出す
            extentPixel  = 2
        
        poly.marginX = extentPixel
        poly.marginY = extentPixel
        poly.minX = coords.min(axis=0)[0] - extentPixel
        poly.minY = coords.min(axis=0)[1] - extentPixel
        poly.maxX = coords.max(axis=0)[0] + extentPixel
        poly.maxY = coords.max(axis=0)[1] + extentPixel
        
        
        if poly.minX < 0:
            poly.minX = 0
            poly.marginX = 0
        if poly.minY < 0:
            poly.minY = 0
            poly.marginY = 0
        if poly.maxX > imgSize[0]:
            poly.maxX = imgSize[0]
        if poly.maxY > imgSize[1]:
            poly.maxY = imgSize[1]
        
        poly.useW = poly.maxX - poly.minX
        poly.useH = poly.maxY - poly.minY
        poly.flag = False
        
        self.polygon.append(poly)


class PolygonInfo():
    """ポリゴン情報クラス
    """
    def __init__(self) -> None:
        """コンストラクタ
        """
        self.target_uri = None
        self.coord_ring = None
        self.in_texcoord = list()   # 入力テクスチャポリゴン座標
        self.out_texcoord = list()  # 出力テクスチャポリゴン座標

        self.minX = 0
        self.minY = 0
        self.maxX = 0
        self.maxY = 0
        self.useW = 0
        self.useH = 0
        self.useX = 0
        self.useY = 0
        self.flag = False
        self.marginX = 0
        self.marginY = 0


class SetyMesh():
    """4次メッシュ情報クラス
    """
    def __init__(self) -> None:
        """コンストラクタ
        """
        self.ids = list()
        self.code = 0

