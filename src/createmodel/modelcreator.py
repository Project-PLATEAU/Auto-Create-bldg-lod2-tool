# -*- coding:utf-8 -*-
import os
import sys
import glob
import shutil
import traceback
from .building import Building
from .message import CreateModelMessage
from .createmodelexception import CreateModelException
from ..util.parammanager import ParamManager
from ..util.resulttype import ResultType, ProcessResult
from ..util.log import Log, LogLevel, ModuleType
from ..util.config import Config
from ..util.citygmlinfo import CityGmlManager
from ..util.coordinateconverter import CoordinateConverter
from ..util.coordinateconverter import CoordinateConverterException


class ModelCreator:
    """モデル要素生成モジュール
    """

    def __init__(self, param: ParamManager):
        """コンストラクタ

        Args:
            param (ParamManager): パラメータクラス

        Raises:
            FileNotFoundError: DSMフォルダが存在しない場合
            FileNotFoundError: LASファイルが存在しない場合
            FileNotFoundError: オルソ画像フォルダが存在しない場合
            FileNotFoundError: TIFFファイルが存在しない場合
            FileNotFoundError: TFWファイルが存在しない場合
            CoordinateConverterException: 座標系コンバーターのエラー
        """
        class_name = self.__class__.__name__
        func_name = sys._getframe().f_code.co_name
        self._img_target = None

        try:
            self._param_mng = param

            # 座標変換用のコンバータの作成
            self._converter = CoordinateConverter(
                coordinate_id=self._param_mng.las_coordinate_system)

            # 入力DSM点群フォルダパスの確認
            if not os.path.isdir(self._param_mng.dsm_folder_path):
                # 入力DSM点群フォルダが存在しない
                raise CreateModelException(
                    CreateModelMessage.ERR_MSG_LAS_FOLDER_NOT_FOUND)

            # lasファイルの存在確認
            las_path = os.path.join(self._param_mng.dsm_folder_path, '*.las')
            files = glob.glob(las_path)
            if len(files) == 0:
                # lasファイルが存在しない場合
                raise CreateModelException(
                    CreateModelMessage.ERR_MSG_LAS_FILE_NOT_FOUND)

            # obj出力フォルダの作成
            self._output_folder = Config.OUTPUT_MODEL_OBJDIR
            if os.path.isdir(self._output_folder):
                # 既にフォルダが存在する場合は削除
                shutil.rmtree(self._output_folder)
            os.makedirs(self._output_folder)

        except CoordinateConverterException:
            # 座標変換用のコンバータのエラー
            msg = '{}.{}, {}'.format(
                class_name, func_name,
                CreateModelMessage.ERR_MSG_LAS_COORDINATE_SYSTEM)
            Log.output_log_write(
                LogLevel.ERROR, ModuleType.MODEL_ELEMENT_GENERATION,
                msg)
            raise CoordinateConverterException(msg)
        except CreateModelException as e:
            # モデル作成のエラー
            msg = '{}.{}, {}'.format(class_name, func_name, e)
            Log.output_log_write(
                LogLevel.ERROR, ModuleType.MODEL_ELEMENT_GENERATION,
                msg)
            raise   # 呼び出し側にも通知
        except Exception:
            # 予期せぬエラー
            Log.output_log_write(
                LogLevel.ERROR, ModuleType.MODEL_ELEMENT_GENERATION,
                traceback.format_exc())
            raise   # 呼び出し側にも通知

    def create(self, gmls: list[CityGmlManager.BuildInfo]) -> ResultType:
        """モデル生成

        Args:
            gmls (list[CityGmlManager.BuildInfo]): 建物外形情報リスト

        Returns:
            ResultType: 動作結果
        """
        class_name = self.__class__.__name__
        func_name = sys._getframe().f_code.co_name

        if gmls is None or len(gmls) == 0:
            msg = '{}.{}, {}'.format(
                class_name, func_name,
                CreateModelMessage.ERR_MSG_CITY_GML_DATA)
            Log.output_log_write(
                LogLevel.ERROR, ModuleType.MODEL_ELEMENT_GENERATION, msg)
            return ResultType.ERROR

        warn_flag = False
        for gml in gmls:
            try:
                if gml.read_lod0_model is ProcessResult.ERROR:
                    # lod0データがない場合skip
                    gml.create_lod2_model = ProcessResult.SKIP
                    continue

                gml.create_lod2_model = ProcessResult.ERROR     # 初期値
                # 経緯度を平面直角座標に変換
                shape = []
                for pos in gml.lod0_poslist:
                    if len(pos) > 1:
                        x, y = self._converter.to_cartesian(
                            latitude=pos[0], longitude=pos[1])
                        shape.append([x, y])

                info = Building(
                    gml.build_id, shape,
                    self._param_mng.dsm_folder_path,
                    0.25,   # grid_size
                    self._output_folder,
                )

                # モデル生成
                info.create(self._param_mng.las_swap_xy)

                # 動作結果の保存
                gml.create_lod2_model = ProcessResult.SUCCESS

            except CreateModelException as e:
                # モデル作成のエラー(想定エラー)
                msg = '{}, {}'.format(gml.build_id, e)
                Log.output_log_write(
                    LogLevel.MODEL_ERROR, ModuleType.MODEL_ELEMENT_GENERATION,
                    msg)
                warn_flag = True
                
            except Exception as e:
                # 予期せぬエラー
                msg = '{}\n{}'.format(gml.build_id, traceback.format_exc())
                Log.output_log_write(
                    LogLevel.MODEL_ERROR,
                    ModuleType.MODEL_ELEMENT_GENERATION,
                    msg)
                warn_flag = True
        
        if warn_flag:
            # 未作成のモデルがある場合
            Log.output_log_write(
                LogLevel.WARN,
                ModuleType.MODEL_ELEMENT_GENERATION,
                CreateModelMessage.WARN_MSG_COULD_NOT_CREATE_MODEL)
            return ResultType.WARN
        else:
            # 全てのモデルを生成した場合
            return ResultType.SUCCESS
