import sys
import shutil
import os
import glob
import csv
import re
from pathlib import Path

# import before shapely (https://github.com/shapely/shapely/issues/1435)
import torch

from .texturemapping.texturemain import TextureMain
from .createmodel.modelcreator import ModelCreator
from .util.parammanager import ParamManager
from .util.citygmlinfo import CityGmlManager
from .phaseconsistensy.mainmanager import MainManager
from .util.log import Log, ModuleType, LogLevel
from .util.config import Config
from .util.resulttype import ResultType


def _delete_module_tmp_folder() -> None:
    """中間フォルダの削除(モジュールごと)
    """
    if os.path.isdir(Config.OUTPUT_MODEL_OBJDIR):
        shutil.rmtree(Config.OUTPUT_MODEL_OBJDIR)
    if os.path.isdir(Config.OUTPUT_PHASE_OBJDIR):
        shutil.rmtree(Config.OUTPUT_PHASE_OBJDIR)
    if os.path.isdir(Config.OUTPUT_PHASE_OBJDIR):
        shutil.rmtree(Config.OUTPUT_PHASE_OBJDIR)


def main():
    """メイン関数
    """
    args = sys.argv

    if len(args) != 2:
        print('usage: python AutoCreateLod2.py param.json')
        sys.exit()

    # 中間フォルダがある場合は削除
    if os.path.isdir(Config.OUTPUT_OBJDIR):
        shutil.rmtree(Config.OUTPUT_OBJDIR)

    # 中間フォルダの作成
    os.makedirs(Config.OUTPUT_OBJDIR)

    try:
        param_manager = ParamManager()
        change_params = param_manager.read(args[1])

    except Exception as e:
        param_manager.debug_log_output = False
        log = Log(param_manager, args[1])
        log.output_log_write(LogLevel.ERROR, ModuleType.NONE, e)
        log.log_footer()
        sys.exit()

    try:
        ret_citygml_read = ResultType.ERROR     # CityGML入力結果初期化

        # ログクラスインスタンス化
        log = Log(param_manager, args[1])

        # パラメータがデフォルトに変更された場合
        for change_param in change_params:
            message = f'{change_param.name} Value change to '
            message += f'{change_param.value}'
            log.output_log_write(LogLevel.WARN, ModuleType.NONE,
                                 message)

        # CityGMLファイル一覧の取得
        citygml_files = glob.glob(
            os.path.join(param_manager.citygml_folder_path, '*.gml'))

        if len(citygml_files) == 0:
            # 入力CityGMLファイルがない場合
            log.output_log_write(LogLevel.ERROR, ModuleType.NONE,
                                 'CityGML file not found')

        buildings_for_summary = []
        for citygml_file_path in citygml_files:
            # 入力ファイル名
            file_name = os.path.basename(citygml_file_path)

            # 処理対象のファイル名のログを出力
            log.process_start_log(file_name)

            # CityGML入力
            log.module_start_log(ModuleType.INPUT_CITYGML, file_name)

            citygml = CityGmlManager(param_manager)
            # CityGML読み込み
            ret_citygml_read, buildings = citygml.read_file(
                file_name=file_name)

            log.module_result_log(
                ModuleType.INPUT_CITYGML, ret_citygml_read)

            if ret_citygml_read is not ResultType.ERROR:
                # OBJの処理

                # モデル要素生成
                log.module_start_log(
                    ModuleType.MODEL_ELEMENT_GENERATION, file_name)

                create_model = ModelCreator(param_manager)
                ret_model_element_generation = create_model.create(
                    buildings)

                log.module_result_log(ModuleType.MODEL_ELEMENT_GENERATION,
                                      ret_model_element_generation)

                # モデル要素生成中間出力フォルダ確認
                files = glob.glob(
                    os.path.join(Config.OUTPUT_MODEL_OBJDIR, '*.obj'))

                if not files:
                    log.output_log_write(
                        LogLevel.ERROR, ModuleType.NONE,
                        "ModelElementGeneration Module Not Output Obj File")
                    buildings_for_summary.extend(buildings)
                    _delete_module_tmp_folder()     # 中間フォルダの削除
                    continue

                # 位相一貫性補正
                log.module_start_log(
                    ModuleType.CHECK_PHASE_CONSISTENSY, file_name)

                main_manager = MainManager(param_manager)
                ret_check_phaseconsistensy = \
                    main_manager.check_and_correction(buildings)

                log.module_result_log(ModuleType.CHECK_PHASE_CONSISTENSY,
                                      ret_check_phaseconsistensy)

                # 位相一貫性補正中間出力フォルダ確認
                files = glob.glob(
                    os.path.join(Config.OUTPUT_PHASE_OBJDIR, '*.obj'))

                if not files:
                    log.output_log_write(
                        LogLevel.ERROR, ModuleType.NONE,
                        "CheckPhaseConsistensy Module Not Output Obj File")
                    buildings_for_summary.extend(buildings)
                    _delete_module_tmp_folder()     # 中間フォルダの削除
                    continue

                # テクスチャ自動張付け
                if param_manager.output_texture:
                    log.module_start_log(ModuleType.PASTE_TEXTURE, file_name)

                    texture_main = TextureMain(param_manager)
                    ret_paste_texture = texture_main.texture_main(
                        buildings=buildings, file_name=file_name)

                    log.module_result_log(ModuleType.PASTE_TEXTURE,
                                          ret_paste_texture)
                else:
                    input_objdir = Config.OUTPUT_PHASE_OBJDIR
                    output_objdir = Config.OUTPUT_TEX_OBJDIR
                    if os.path.isdir(output_objdir):
                        shutil.rmtree(output_objdir)   # 既存フォルダは削除
                    os.mkdir(output_objdir)

                    pathlist = sorted(
                        [p for p in Path(input_objdir).glob('**/*')
                            if re.search(r'/*\.obj', str(p))])
                    for path in pathlist:
                        shutil.copyfile(
                            path,
                            os.path.join(output_objdir, os.path.basename(path)))

                    # 最終出力にOBJファイルを出力する場合
                    if param_manager.output_obj:
                        # 出力フォルダの作成
                        optional_output_objdir = os.path.join(
                            param_manager.output_folder_path, 'obj',
                            os.path.splitext(file_name)[0])
                        if not os.path.isdir(optional_output_objdir):
                            os.makedirs(optional_output_objdir)

                        pathlist = sorted(
                            [p for p in Path(input_objdir).glob('**/*')
                                if re.search(r'/*\.obj', str(p))])
                        for path in pathlist:
                            shutil.copyfile(
                                path,
                                os.path.join(optional_output_objdir,
                                                os.path.basename(path)))

                # CityGML出力
                log.module_start_log(ModuleType.OUTPUT_CITYGML, file_name)
                # CityGML書き込み
                ret_citygml_write = citygml.write_file(file_name=file_name)

                log.module_result_log(
                    ModuleType.OUTPUT_CITYGML, ret_citygml_write)

                # summary用にモデル化結果を保存
                buildings_for_summary.extend(buildings)

            # 中間フォルダの削除(モジュールごと)
            _delete_module_tmp_folder()

        # 中間フォルダの削除(temp)
        if os.path.isdir(Config.OUTPUT_OBJDIR):
            shutil.rmtree(Config.OUTPUT_OBJDIR)

    except Exception as e:
        log.output_log_write(LogLevel.ERROR, ModuleType.NONE, e)
        buildings_for_summary.extend(buildings)

    finally:
        if ret_citygml_read is not ResultType.ERROR:
            # モデル化結果サマリー出力
            log.output_summary(buildings_for_summary)

        # 実行ログファイルと標準出力にフッタ出力
        log.log_footer()
