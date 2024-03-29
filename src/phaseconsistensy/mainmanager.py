import os
import sys
import shutil
import logging

from ..util.objinfo import ObjInfos, ObjInfo, BldElementType
from ..util.parammanager import ParamManager
from ..util.config import Config
from ..util.log import Log, ModuleType, LogLevel
from ..util.resulttype import ProcessResult, ResultType
from .resultinfo import ResultInfo, StatusType
from .resultinfo import ErrorType
from .checkface import CheckFace, CheckFaces, TestResultType

# LOG_LEVEL = logging.DEBUG
LOG_LEVEL = logging.WARNING

logging.basicConfig(level=LOG_LEVEL, format='%(levelname)s: %(message)s')


class MainManager:
    """位相一貫性検査/補正全体制御クラス
    """

    _module_name = 'CheckPhaseConsistensy Module'
    _module_version = '0.0.1'

    def __init__(self, param_manager: ParamManager):
        """コンストラクタ

        Args:
            param_manager (ParamManager): パラメータ情報
        """
        self._obj_infos = ObjInfos()                    # 建物情報群
        self._results = []                              # 検査結果情報リスト
        self._param_manager = param_manager             # パラメータ情報
        self._input_folder = Config.OUTPUT_MODEL_OBJDIR     # 入力フォルダパス
        self._output_folder = Config.OUTPUT_PHASE_OBJDIR    # 出力フォルダパス
        self._obj_num = 0                               # OBJ ファイル数
        self._file_obj = sys.stdout                     # ログファイル出力先

        self._summary = dict()
        self._summary[StatusType.NO_ERROR] = 0
        self._summary[StatusType.AUTO_CORRECTED] = 0
        self._summary[StatusType.DELETED] = 0
        self._summary[StatusType.ERROR] = 0

    def check_and_correction(self, buildings):
        """位相一貫性検査/補正処理実行

        Args:
            buildings (list[CityGmlManager.BuildInfo]): 建物外形情報リスト

        Returns:
            ResultType: 実行結果情報
        """
        Log.output_log_write(LogLevel.INFO, ModuleType.CHECK_PHASE_CONSISTENSY,
                             'DeleteErrorObject : '
                             + f'{self._param_manager.delete_error_flag}')

        if not os.path.exists(self._input_folder):
            Log.output_log_write(LogLevel.ERROR,
                                 ModuleType.CHECK_PHASE_CONSISTENSY,
                                 f'{self._input_folder}: '
                                 'obj folder does not exist.')
            return ResultType.ERROR

        file_list = os.listdir(self._input_folder)
        
        building_list = [i for i in buildings
                         if i.build_id + '.obj' in file_list]
        self._obj_num = len(building_list)

        if self._obj_num == 0:
            # フォルダ内にファイルが存在しない場合
            Log.output_log_write(LogLevel.ERROR,
                                 ModuleType.CHECK_PHASE_CONSISTENSY,
                                 f'{self._input_folder}: '
                                 'obj folder do not have obj file.')
            return ResultType.ERROR

        result_type = ResultType.SUCCESS

        # 出力フォルダ作成
        if os.path.isdir(self._output_folder):
            shutil.rmtree(self._output_folder)
        os.makedirs(self._output_folder)

        # 各 OBJ ファイル対して検査/補正処理
        for build in building_list:
            err_message = ''
            result_info = ResultInfo()
            result_info.obj_name = os.path.join(
                self._input_folder, f'{build.build_id}.obj')
            obj_info = ObjInfo()
            except_flag = False
            try:
                # OBJ ファイル入力
                obj_info.read_file(result_info.obj_name, err_message)

                # 連続頂点重複検査/補正
                self._check_double_point(obj_info, result_info, build)
                
                # ソリッド閉合検査/補正
                if result_info.status != StatusType.ERROR:
                    self._check_solid(obj_info, result_info, build)
                
                # 非平面検査/三角形分割
                if result_info.status != StatusType.ERROR:
                    self._check_non_plane(obj_info, result_info, build)

                # 面積 0 ポリゴン検査
                if result_info.status != StatusType.ERROR:
                    self._check_zero_area(obj_info, result_info, build)

                # 自己交差・自己接触検査
                if result_info.status != StatusType.ERROR:
                    self._check_intersection(obj_info, result_info, build)

                # 地物内面同士交差検査
                if result_info.status != StatusType.ERROR:
                    self._check_face_intersection(obj_info, result_info, build)

                output_flag = True
                if result_info.status != StatusType.NO_ERROR:
                    if result_info.status == StatusType.ERROR:
                        if self._param_manager.delete_error_flag:
                            # 削除
                            output_flag = False
                            result_info.status = StatusType.DELETED
                    self._results.append(result_info)

                    # ログファイル出力
                    Log.output_log_write(LogLevel.MODEL_ERROR,
                                         ModuleType.CHECK_PHASE_CONSISTENSY,
                                         result_info.get_str())
                    result_type = ResultType.WARN
                
                # OBJ ファイル出力
                if output_flag:
                    file_path = os.path.join(
                        self._output_folder,
                        os.path.basename(obj_info.file_name))
                    obj_info.write_file(file_path)

            except (FileNotFoundError, SyntaxError, ValueError) as e:
                # 入力ファイルエラー
                result_info.add_err(ErrorType.INVALID_INPUT, None, str(e))
                except_flag = True

            except Exception as e:
                # その他のエラー
                result_info.add_err(ErrorType.OTHERS, None, str(e))
                except_flag = True

            if except_flag:
                result_info.status = StatusType.ERROR
                if not self._param_manager.delete_error_flag:
                    # 入力 OBJ を出力
                    out_path = os.path.join(
                        self._output_folder,
                        os.path.basename(result_info.obj_name))
                    shutil.copy(result_info.obj_name, out_path)
                else:
                    result_info.status = StatusType.DELETED
                Log.output_log_write(LogLevel.MODEL_ERROR,
                                     ModuleType.CHECK_PHASE_CONSISTENSY,
                                     result_info.get_str())
                result_type = ResultType.WARN
            self._summary[result_info.status] += 1

        # ログファイル出力(サマリ部)
        self._output_log_file_summary()

        return result_type
    
    def _output_log_file_summary(self):
        """ログファイル サマリー部出力
        """
        message = 'Summary\n'
        no_error = self._obj_num \
            - (self._summary[StatusType.AUTO_CORRECTED]
                + self._summary[StatusType.DELETED]
                + self._summary[StatusType.ERROR])
        message += f'\t\tNumber of files      : {self._obj_num}\n'
        message += f'\t\tNo Error files       : {no_error}\n'
        tmp_str = 'Auto corrected files : ' \
            + str(self._summary[StatusType.AUTO_CORRECTED])
        message += f'\t\t{tmp_str}\n'
        message += '\t\tDeleted files        : '
        message += f'{self._summary[StatusType.DELETED]}\n'
        message += '\t\tError files          : '
        message += f'{self._summary[StatusType.ERROR]}'

        Log.output_log_write(LogLevel.INFO, ModuleType.CHECK_PHASE_CONSISTENSY,
                             message)
    
    def _check_double_point(self, obj_info: ObjInfo, result_info: ResultInfo,
                            build):
        """連続頂点重複検査/補正

        Args:
            obj_info (ObjInfo): 建物情報
            result_info (ResultInfo): 検査結果格納先
            build (CityGmlManager.BuildInfo): 建物外形情報
        """

        build.double_point = ProcessResult.SUCCESS
        for f_key, f_value in obj_info.faces_list.items():
            for i, face in enumerate(f_value.faces):
                check_face = CheckFace(obj_info, face,
                                       self._param_manager)
                ret = check_face.check_double_point()
                if ret is TestResultType.AUTO_CORRECTED:
                    # エラーあり、自動補正済み
                    logging.debug('error occured')
                    result_info.add_err(ErrorType.DOUBLE_POINT,
                                        check_face.err_list)
                    result_info.status = StatusType.AUTO_CORRECTED
                    #print("_check_double_point AUTO_CORRECTED")
                elif ret is TestResultType.AUTO_CORRECTION_FAILURE:
                    # 自動補正失敗
                    result_info.status = StatusType.ERROR
                    build.double_point = ProcessResult.ERROR
                    #print("_check_double_point ERROR")
                    return
    
    def _check_intersection(self, obj_info: ObjInfo, result_info: ResultInfo,
                            build):
        """自己交差・自己接触検査

        Args:
            obj_info (ObjInfo): 建物情報
            result_info (ResultInfo): 検査結果格納先
            build (CityGmlManager.BuildInfo): 建物外形情報
        """
        build.intersection = ProcessResult.SUCCESS
        for f_key, f_value in obj_info.faces_list.items():
            for i, face in enumerate(f_value.faces):
                check_face = CheckFace(obj_info, face,
                                       self._param_manager)
                if not check_face.check_intersection():
                    # エラーあり
                    result_info.add_err(ErrorType.SELF_INTERSECTION,
                                        check_face.err_list)
                    result_info.status = StatusType.ERROR
                    build.intersection = ProcessResult.ERROR
                    #print("_check_intersection ERROR")

    def _check_face_intersection(self, obj_info: ObjInfo,
                                 result_info: ResultInfo,
                                 build):
        """地物内面同士交差検査

        Args:
            obj_info (ObjInfo): 建物情報
            result_info (ResultInfo): 検査結果格納先
            build (CityGmlManager.BuildInfo): 建物外形情報
        """
        build.face_intersection = ProcessResult.SUCCESS
        check_faces = CheckFaces(obj_info, self._param_manager)
        if not check_faces.check_face_intersection():
            # エラーあり
            result_info.add_err(ErrorType.FACE_INTERSECTION,
                                check_faces.err_list)
            result_info.status = StatusType.ERROR
            build.face_intersection = ProcessResult.ERROR
            #print("_check_face_intersection ERROR")

    def _check_non_plane(self, obj_info: ObjInfo, result_info: ResultInfo,
                         build):
        """非平面検査/三角形分割補正

        Args:
            obj_info (ObjInfo): 建物情報
            result_info (ResultInfo): 検査結果格納先
            build (CityGmlManager.BuildInfo): 建物外形情報
        """
        build.non_plane = ProcessResult.SUCCESS
        
        for f_key, f_value in obj_info.faces_list.items():
            for i, face in enumerate(f_value.faces):
                check_face = CheckFace(obj_info, face,
                                       self._param_manager)
                ret = check_face.check_non_plane()
                if ret is TestResultType.AUTO_CORRECTED:
                    # エラーあり、自動補正済み
                    result_info.add_err(ErrorType.NON_PLANE,
                                        check_face.err_list)
                    result_info.status = StatusType.AUTO_CORRECTED
                    #print("_check_non_plane AUTO_CORRECTED")
                elif ret is TestResultType.AUTO_CORRECTION_FAILURE:
                    # 自動補正失敗
                    retult_info.status = StatusType.ERROR
                    build.non_plane = ProcessResult.ERROR
                    #print("_check_non_plane ERROR")
                    return

    def _check_zero_area(self, obj_info: ObjInfo, result_info: ResultInfo,
                         build):
        """面積 0 ポリゴン検査/補正

        Args:
            obj_info (ObjInfo): 建物情報
            result_info (ResultInfo): 検査結果格納先
            build (CityGmlManager.BuildInfo): 建物外形情報
        """
        for f_key, f_value in obj_info.faces_list.items():
            remove_face_list = []
            for face in f_value.faces:
                check_face = CheckFace(obj_info, face,
                                       self._param_manager)
                if not check_face.check_zero_area():
                    remove_face_list.append(face)
                    # エラーあり
                    result_info.add_err(ErrorType.ZERO_AREA,
                                        check_face.err_list)
            if len(remove_face_list) > 0:
                try:
                    # 補正処理
                    for face in remove_face_list:
                        obj_info.remove_face(f_key, face)
                    result_info.status = StatusType.AUTO_CORRECTED
                    #print("_check_zero_area AUTO_CORRECTED")
                except Exception:
                    # 予期せぬエラーが発生して、補正処理が失敗
                    retult_info.status = StatusType.ERROR
                    build.zero_area = ProcessResult.ERROR
                    #print("_check_zero_area ERROR")
                    return

        build.zero_area = ProcessResult.SUCCESS

    def _check_solid(self, obj_info: ObjInfo,
                     result_info: ResultInfo,
                     build):
        """ソリッド閉合検査/補正

        Args:
            obj_info (ObjInfo): 建物情報
            result_info (ResultInfo): 検査結果格納先
            build (CityGmlManager.BuildInfo): 建物外形情報
        """
        build.solid = ProcessResult.SUCCESS
        check_faces = CheckFaces(obj_info, self._param_manager)
        ret = check_faces.check_solid()
        if ret is TestResultType.AUTO_CORRECTED:
            # エラーあり、自動補正済み
            result_info.add_err(ErrorType.OPEN_SOLID,
                                check_faces.err_list)
            result_info.status = StatusType.AUTO_CORRECTED
            #print("check_solid: AUTO_CORRECTED")
        elif ret is TestResultType.AUTO_CORRECTION_FAILURE:
            # 自動補正失敗
            retult_info.status = StatusType.ERROR
            build.solid = ProcessResult.ERROR
            #print("check_solid: ERROR")
