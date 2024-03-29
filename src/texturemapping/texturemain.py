# -*- coding:utf-8 -*-
import os
import csv
import shutil
import re
from datetime import datetime
from pathlib import Path
from .photoimage import PhotoImage
from .verticalobject import VerticalObject
from ..util.parammanager import ParamManager
from ..util.config import Config
from ..util.log import Log, ModuleType, LogLevel
from ..util.resulttype import ResultType, ProcessResult


class TextureMain():
    """テクスチャ貼付けメインクラス
    """
    def __init__(self, param_manager: ParamManager) -> None:
        """コンストラクタ

        Args:
            param_manager (ParamManager): パラメータ情報
        """
        self.input_objdir = Config.OUTPUT_PHASE_OBJDIR  # 入力OBJフォルダパス
        self.output_objdir = Config.OUTPUT_TEX_OBJDIR   # 出力OBJフォルダパス
        self.param_manager = param_manager              # パラメータ情報
        # オプション出力のOBJフォルダパス
        self.optional_output_objdir = ''

    def texture_main(self, buildings, file_name: str) -> None:
        """テクスチャ張付け開始

        Args:
            buildings (list[CityGmlManager.BuildInfo]): 建物外形情報リスト
            file_name (str): 入力CityGMLファイル名(拡張子付き)

        Raises:
            FileNotFoundError: OBJファイル入力先フォルダなし
            FileNotFoundError: 入力写真フォルダなし
            Exception: 内部標定要素情報パラメータエラー
            Exception: 入力写真なし
        """
        photolist = list()
        photonum = 0
        excalib = list()
        caminfo = list()
        calibflag = False
        restype = ResultType.SUCCESS

        try:
            # 中間フォルダの確認
            # 入力写真フォルダ確認でエラーした際に入力objをそのまま
            # 中間フォルダにコピーするため先に中間フォルダの作成を行う
            if os.path.isdir(self.output_objdir):
                shutil.rmtree(self.output_objdir)   # 既存フォルダは削除
            os.mkdir(self.output_objdir)

            # 最終出力にOBJファイルを出力する場合
            if self.param_manager.output_obj:
                # 出力フォルダの作成
                self.optional_output_objdir = os.path.join(
                    self.param_manager.output_folder_path, 'obj',
                    os.path.splitext(file_name)[0])
                if not os.path.isdir(self.optional_output_objdir):
                    os.makedirs(self.optional_output_objdir)

            if not os.path.isdir(self.input_objdir):
                # OBJファイル入力先フォルダなし
                raise FileNotFoundError('Folder not found (OBJfolder)')

            if not os.path.isdir(self.param_manager.texture_folder_path):
                # 入力写真フォルダなし
                raise FileNotFoundError('Folder not found (Texturefolder)')

            if not os.path.exists(self.param_manager.output_folder_path):
                # テクスチャ画像出力先フォルダなし
                os.makedirs(self.param_manager.output_folder_path)

            # 外部標定要素ファイル読み込み
            with open(self.param_manager.ex_calib_element_path) as f:
                reader = csv.reader(f, delimiter='\t')
                excalib = [row for row in reader]

            # カメラ情報ファイル読み込み
            with open(self.param_manager.camera_info_path) as f:
                reader = csv.reader(f, delimiter='\t')
                caminfo = [row for row in reader]

            # カメラ情報チェック
            calib_count = 0
            for idx, info in enumerate(caminfo[1]):
                if idx < 7:
                    if info == '':
                        raise Exception('caminfo data is insufficient')
                else:
                    if info != '':
                        calib_count += 1

            if calib_count == 3:
                # キャリブレーションデータが五つ揃っている時は有効
                calibflag = True
                Log.output_log_write(
                    LogLevel.DEBUG,
                    ModuleType.PASTE_TEXTURE,
                    'calib ON')

            # 写真情報読み込み
            for data in excalib[1:]:
                ret = True
                # 外部標定要素チェック
                if len(data) != 7:  # ファイル名,x,y,z,omega,phi,kappa
                    ret = False
                    Log.output_log_write(
                        LogLevel.WARN,
                        ModuleType.PASTE_TEXTURE,
                        'excalib data is insufficient')

                for idx, info in enumerate(data):  # 値が入っていない場合
                    if info == '':
                        ret = False
                        Log.output_log_write(
                            LogLevel.WARN,
                            ModuleType.PASTE_TEXTURE,
                            'excalib data including empty')
                if ret:
                    photo = PhotoImage()
                    ret = photo.set_photo_param(
                        self.param_manager.texture_folder_path,
                        data, caminfo[1], calibflag,
                        self.param_manager.rotate_matrix_mode)
                    if ret:
                        photolist.append(photo)
                        photonum += 1
                    else:
                        Log.output_log_write(
                            LogLevel.WARN,
                            ModuleType.PASTE_TEXTURE,
                            'PhotoFile Not Found ' + data[0])

            if photonum < 1:
                raise Exception('Photo not found')

            # テクスチャ画像出力フォルダ作成
            # [メッシュコード]_[地物型]_[CRS]_[オプション]_appearance
            base_name = os.path.splitext(file_name)[0]
            texturedir = os.path.join(
                self.param_manager.output_folder_path,
                base_name.split('_op')[0] + "_appearance")
            if not os.path.isdir(texturedir):
                os.mkdir(texturedir)

            # マテリアルファイル名
            # YYYYMMDD_HHMMSS.mtl
            date = datetime.now().strftime("%Y%m%d_%H%M%S")
            mtl_file_name = date + ".mtl"

            file_list = os.listdir(self.input_objdir)

            building_list = [i for i in buildings
                             if i.build_id + '.obj' in file_list]
            self._obj_num = len(building_list)

            if self._obj_num == 0:
                # フォルダ内にファイルが存在しない場合
                Log.output_log_write(LogLevel.ERROR,
                                     ModuleType.PASTE_TEXTURE,
                                     f'{self.input_objdir}: '
                                     'obj folder do not have obj file.')
                restype = ResultType.WARN
            else:
                for build in building_list:
                    # 建造物分テクスチャ貼付け処理
                    try:
                        id = build.build_id
                        build.paste_texture = ProcessResult.SKIP
                        path = os.path.join(
                            self.input_objdir, f'{id}.obj')

                        Log.output_log_write(
                            LogLevel.DEBUG,
                            ModuleType.PASTE_TEXTURE,
                            f'bldid:{id}')

                        ver = VerticalObject(path, photonum, photolist)
                        ver.select_rooftexture()
                        ver.select_walltexture()
                        ret = ver.output_texture(
                            self.output_objdir, texturedir, mtl_file_name)
                        if self.param_manager.output_obj:
                            # マテリアルファイル名はCityGMLファイル名とする
                            ver.output_optional_obj(
                                objdir=self.optional_output_objdir,
                                texture_dir=texturedir,
                                mtl_file_name=f'{base_name}.mtl')

                        if not ret:
                            shutil.copyfile(
                                path,
                                os.path.join(
                                    self.output_objdir,
                                    os.path.basename(path)))
                            Log.output_log_write(
                                LogLevel.WARN,
                                ModuleType.PASTE_TEXTURE,
                                f'Texture not found id:{id}')
                            restype = ResultType.WARN
                            build.paste_texture = ProcessResult.ERROR
                        else:
                            build.paste_texture = ProcessResult.SUCCESS

                    except Exception as e:
                        shutil.copyfile(
                            path,
                            os.path.join(
                                self.output_objdir,
                                os.path.basename(path)))
                        Log.output_log_write(LogLevel.WARN,
                                             ModuleType.PASTE_TEXTURE,
                                             str(e) + ' ' + path)
                        restype = ResultType.WARN
                        build.paste_texture = ProcessResult.ERROR
            return restype

        except FileNotFoundError as e:
            self._copy_folder()
            Log.output_log_write(LogLevel.MODEL_ERROR,
                                 ModuleType.PASTE_TEXTURE, e)
            return ResultType.WARN

        except Exception as e:
            self._copy_folder()
            Log.output_log_write(LogLevel.MODEL_ERROR,
                                 ModuleType.PASTE_TEXTURE, e)
            return ResultType.WARN

    def _copy_folder(self):
        """OBJ入力フォルダの中身を出力フォルダにコピー
            モジュール処理を中断した場合
        """
        if os.path.isdir(self.input_objdir):
            pathlist = sorted(
                [p for p in Path(self.input_objdir).glob('**/*')
                    if re.search(r'/*\.obj', str(p))])
            for path in pathlist:
                shutil.copyfile(
                    path,
                    os.path.join(self.output_objdir, os.path.basename(path)))

                if self.param_manager.output_obj:
                    # 最終出力にOBJファイルを出力する場合
                    shutil.copyfile(
                        path,
                        os.path.join(self.optional_output_objdir,
                                     os.path.basename(path)))
