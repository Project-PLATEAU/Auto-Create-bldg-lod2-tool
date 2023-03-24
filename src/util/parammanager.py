import json
import os
import datetime
from enum import IntEnum


class ParamManager:
    """ パラメータファイル管理クラス
    """

    class ChangeParam:
        """エラー値が入力された場合にデフォルト値に変更したパラメータ通知用クラス

        Returns:
            _type_: _description_
        """
        @property
        def name(self):
            """パラメータ名

            Returns:
                string: パラメータ名
            """
            return self._name

        @property
        def value(self):
            """デフォルト設定値

            Returns:
                Any: デフォルト設定値(パラメータによって値が異なる)
            """
            return self._value

        def __init__(self, name: str, value) -> None:
            """コンストラクタ

            Args:
                name (string): パラメータ名
                value (Any): 設定値
            """
            self._name = name
            self._value = value

    class RotateMatrixMode(IntEnum):
        """外部標定要素から算出する回転行列のモード
        """
        XYZ = 0
        """R = Rx(ω)Ry(Φ)Rz(κ)
        """
        ZYX = 1
        """R = Rz(κ)Ry(Φ)Rx(ω)
        """

    # クラス変数
    # jsonファイルのキー
    KEY_LAS_COORDINATE_SYS = 'LasCoordinateSystem'
    KEY_LAS_SWAP_XY = 'LasSwapXY'
    KEY_DSM_FOLDER_PATH = 'DsmFolderPath'
    KEY_CITYGML_FOLDER_PATH = 'CityGMLFolderPath'
    KEY_TEXTURE_FOLDER_PATH = 'TextureFolderPath'
    KEY_EX_CALIB_ELEMENT_PATH = 'ExternalCalibElementPath'
    KEY_ROTATE_MATRIX_MODE = 'RotateMatrixMode'
    KEY_CAMERA_INFO_PATH = 'CameraInfoPath'
    KEY_OUTPUT_FOLDER_PATH = 'OutputFolderPath'
    KEY_OUTPUT_OBJ = 'OutputOBJ'
    KEY_OUTPUT_LOG_FOLDER_PATH = 'OutputLogFolderPath'
    KEY_DEBUG_LOG_OUTPUT = 'DebugLogOutput'
    KEY_PHASE_CONSISTENCY = 'PhaseConsistency'
    KEY_DELETE_ERROR_OBJECT = 'DeleteErrorObject'
    KEY_NON_PLANE_THICKNESS = 'NonPlaneThickness'
    KEY_NON_PLANE_ANGLE = 'NonPlaneAngle'

    # jsonファイルキーリスト
    KEYS = [
        KEY_LAS_COORDINATE_SYS,
        KEY_LAS_SWAP_XY,
        KEY_DSM_FOLDER_PATH,
        KEY_CITYGML_FOLDER_PATH,
        KEY_TEXTURE_FOLDER_PATH,
        KEY_EX_CALIB_ELEMENT_PATH,
        KEY_ROTATE_MATRIX_MODE,
        KEY_CAMERA_INFO_PATH,
        KEY_OUTPUT_FOLDER_PATH,
        KEY_OUTPUT_OBJ,
        KEY_OUTPUT_LOG_FOLDER_PATH,
        KEY_DEBUG_LOG_OUTPUT,
        KEY_PHASE_CONSISTENCY]
    KEYS_PHASECONSISTENCY = [
        KEY_DELETE_ERROR_OBJECT,
        KEY_NON_PLANE_THICKNESS,
        KEY_NON_PLANE_ANGLE]

    # デバッグログ設定のデフォルト値
    DEFALT_DEBUG_LOG_OUTPUT = False
    # 位相一貫補正のデフォルト値
    DEFALT_PHASE_CONSISTENCY_DELETE_ERROR_OBJECT = False
    DEFALT_PHASE_CONSISTENCY_NON_PLANE_THICKNESS = 0.03
    DEFALT_PHASE_CONSISTENCY_NON_PLANE_ANGLE = 20
    # objファイル出力のデフォルト値
    DEFALT_OUTPUT_OBJ = False
    # LASのXY座標swapフラグ
    DEFALT_LAS_SWAP_XY = False

    def __init__(self) -> None:
        """ コンストラクタ
        """
        self.las_coordinate_system = 9      # LASの平面直角座標系
        self.dsm_folder_path = ''           # 点群フォルダパス
        self.citygml_folder_path = ''       # CityGML入力フォルダパス
        self.texture_folder_path = ''       # テクスチャフォルダパス
        self.ex_calib_element_path = ''     # 外部標定要素ファイルパス
        self.camera_info_path = ''          # 内部標定要素ファイルパス
        self.output_folder_path = ''        # CityGML出力フォルダパス
        self.output_log_folder_path = ''    # ログ出力先
        # デバッグログ出力フラグ
        self.debug_log_output \
            = ParamManager.DEFALT_DEBUG_LOG_OUTPUT
        # 位相一貫性検査エラー時OBJ削除フラグ
        self.delete_error_flag \
            = ParamManager.DEFALT_PHASE_CONSISTENCY_DELETE_ERROR_OBJECT
        # 位相一貫性非平面厚み検査閾値
        self.non_plane_thickness \
            = ParamManager.DEFALT_PHASE_CONSISTENCY_NON_PLANE_THICKNESS
        # 位相一貫性非平面法線検査閾値
        self.non_plane_angle \
            = ParamManager.DEFALT_PHASE_CONSISTENCY_NON_PLANE_ANGLE
        self.output_obj = ParamManager.DEFALT_OUTPUT_OBJ    # obj出力フラグ
        # lasのxy座標のswapフラグ
        self.las_swap_xy = ParamManager.DEFALT_LAS_SWAP_XY
        # 外部標定要素から算出する回転行列のモード
        self.rotate_matrix_mode = ParamManager.RotateMatrixMode.XYZ

        # 作業用パラメータ
        self.time = datetime.datetime.now()     # 処理開始時刻

    def read(self, file_path) -> list[ChangeParam]:
        """jsonファイル読み込み関数

        Args:
            file_path (string): jsonファイルパス

        Raises:
            FileNotFoundError: filePathで指定されたファイルが存在しない
            ValueError: LASファイルの座標系が範囲外
            Exception: ファイル/フォルダパスが文字列ではない場合,または空文字の場合

        Returns:
            list[ChangeParam]: 入力エラーによりデフォルト値を採用したパラメータリスト
        """

        change_params = []  # デフォルト値に変更したパラメータリスト

        if not os.path.isfile(file_path):
            # ファイルが存在しない場合
            raise FileNotFoundError('parameter file does not exist.')

        # ファイルが存在する場合
        with open(file_path, encoding='utf-8', mode='r') as jsonOpen:
            try:
                jsonLoad = json.load(jsonOpen)
            except json.decoder.JSONDecodeError as e:
                # 未記入項目がある場合にデコードエラーが発生する
                r = e.lineno
                c = e.colno
                raise(Exception(
                    f'json file decoding error: {e.msg} line {r} column {c}.'))

            # キーの確認
            for key in ParamManager.KEYS:
                if key not in jsonLoad:
                    # キーがない場合エラーとする
                    raise ValueError(f'{key} key does not exist in json file.')
                if key is ParamManager.KEY_PHASE_CONSISTENCY:
                    phase_param = jsonLoad.get(self.KEY_PHASE_CONSISTENCY)
                    for phase_key in ParamManager.KEYS_PHASECONSISTENCY:
                        if phase_key not in phase_param:
                            # 位相一貫性補正用のキーがない場合エラーとする
                            raise ValueError(
                                f'{phase_key} key does not exist in json file.')

            # 値の取得
            self.las_coordinate_system = jsonLoad[self.KEY_LAS_COORDINATE_SYS]
            self.dsm_folder_path = jsonLoad[self.KEY_DSM_FOLDER_PATH]
            self.citygml_folder_path = jsonLoad[self.KEY_CITYGML_FOLDER_PATH]
            self.texture_folder_path = jsonLoad[self.KEY_TEXTURE_FOLDER_PATH]
            self.ex_calib_element_path = jsonLoad[
                self.KEY_EX_CALIB_ELEMENT_PATH]
            self.camera_info_path = jsonLoad[self.KEY_CAMERA_INFO_PATH]
            self.output_folder_path = jsonLoad[self.KEY_OUTPUT_FOLDER_PATH]
            self.output_obj = jsonLoad[self.KEY_OUTPUT_OBJ]
            self.output_log_folder_path = jsonLoad[
                self.KEY_OUTPUT_LOG_FOLDER_PATH]
            self.debug_log_output = jsonLoad[self.KEY_DEBUG_LOG_OUTPUT]
            self.las_swap_xy = jsonLoad[self.KEY_LAS_SWAP_XY]
            self.rotate_matrix_mode = jsonLoad[self.KEY_ROTATE_MATRIX_MODE]
            self.delete_error_flag \
                = (jsonLoad[self.KEY_PHASE_CONSISTENCY]
                           [self.KEY_DELETE_ERROR_OBJECT])
            self.non_plane_thickness \
                = (jsonLoad[self.KEY_PHASE_CONSISTENCY]
                           [self.KEY_NON_PLANE_THICKNESS])
            self.non_plane_angle \
                = (jsonLoad[self.KEY_PHASE_CONSISTENCY]
                           [self.KEY_NON_PLANE_ANGLE])
        
        if (type(self.las_coordinate_system) is not int
                or not (0 < self.las_coordinate_system < 20)):
            raise ValueError(ParamManager.KEY_LAS_COORDINATE_SYS
                             + ' is invalid. (input value range : 1 - 19)')

        if (type(self.dsm_folder_path) is not str
                or not self.dsm_folder_path):
            # 文字列ではない or 空文字の場合
            raise Exception(ParamManager.KEY_DSM_FOLDER_PATH + ' is invalid.')
        
        if (type(self.citygml_folder_path) is not str
                or not self.citygml_folder_path):
            # 文字列ではない or 空文字の場合
            raise Exception(
                ParamManager.KEY_CITYGML_FOLDER_PATH + ' is invalid.')
        if not os.path.isdir(self.citygml_folder_path):
            # 入力CityGMLフォルダが存在しない場合
            raise Exception(
                ParamManager.KEY_CITYGML_FOLDER_PATH + ' not found.')

        if (type(self.texture_folder_path) is not str
                or not self.texture_folder_path):
            # 文字列ではない or 空文字の場合
            raise Exception(
                ParamManager.KEY_TEXTURE_FOLDER_PATH + ' is invalid.')

        if (type(self.ex_calib_element_path) is not str
                or not self.ex_calib_element_path):
            # 文字列ではない or 空文字の場合
            raise Exception(
                ParamManager.KEY_EX_CALIB_ELEMENT_PATH + ' is invalid.')

        if (type(self.camera_info_path) is not str
                or not self.camera_info_path):
            # 文字列ではない or 空文字の場合
            raise Exception(
                ParamManager.KEY_CAMERA_INFO_PATH + ' is invalid.')

        if (type(self.output_folder_path) is not str
                or not self.output_folder_path):
            # 文字列ではない or 空文字の場合
            raise Exception(
                ParamManager.KEY_OUTPUT_FOLDER_PATH + ' is invalid.')

        input_folder_name = os.path.basename(self.citygml_folder_path)
        time_str = self.time.strftime('%Y%m%d_%H%M')
        output_folder_path = os.path.join(
            self.output_folder_path, f'{input_folder_name}_{time_str}')
        if not os.path.exists(output_folder_path):
            # CityGML出力ファイルの格納フォルダがない場合
            try:
                os.makedirs(output_folder_path)
                self.output_folder_path = output_folder_path    # 更新
            except Exception:
                raise Exception(
                    output_folder_path + ' cannot make.')

        if (type(self.output_log_folder_path) is not str
                or not self.output_log_folder_path):
            # 文字列ではない or 空文字の場合
            self.output_log_folder_path = None
            # エラー対応(途中終了)に変更
            raise Exception(
                ParamManager.KEY_OUTPUT_LOG_FOLDER_PATH + ' is invalid.')
        else:
            try:
                # 作成不能なパスであるかの確認
                os.makedirs(self.output_log_folder_path, exist_ok=True)
            except Exception:
                self.output_log_folder_path = None
                # エラー対応(途中終了)に変更
                raise Exception(
                    ParamManager.KEY_OUTPUT_LOG_FOLDER_PATH + ' is invalid.')

        if (type(self.debug_log_output) is not bool):
            # self.debug_log_output = ParamManager.DEFALT_DEBUG_LOG_OUTPUT
            # change_params.append(
            #     ParamManager.ChangeParam(
            #         ParamManager.KEY_DEBUG_LOG_OUTPUT,
            #         ParamManager.DEFALT_DEBUG_LOG_OUTPUT))
            # エラー対応(途中終了)に変更
            raise Exception(
                ParamManager.KEY_DEBUG_LOG_OUTPUT + ' is invalid.')

        if (type(self.delete_error_flag) is not bool):
            # self.delete_error_flag \
            #     = ParamManager.DEFALT_PHASE_CONSISTENCY_DELETE_ERROR_OBJECT
            # change_params.append(
            #     ParamManager.ChangeParam(
            #         ParamManager.KEY_DELETE_ERROR_OBJECT,
            #         ParamManager.DEFALT_PHASE_CONSISTENCY_DELETE_ERROR_OBJECT))
            # エラー対応(途中終了)に変更
            raise Exception(
                ParamManager.KEY_DELETE_ERROR_OBJECT + ' is invalid.')

        if ((type(self.non_plane_thickness) is not int
                and type(self.non_plane_thickness) is not float)
                or self.non_plane_thickness < 0.0):
            # self.non_plane_thickness \
            #     = ParamManager.DEFALT_PHASE_CONSISTENCY_NON_PLANE_THICKNESS
            # change_params.append(
            #     ParamManager.ChangeParam(
            #         ParamManager.KEY_NON_PLANE_THICKNESS,
            #         ParamManager.DEFALT_PHASE_CONSISTENCY_NON_PLANE_THICKNESS))
            # エラー対応(途中終了)に変更
            raise Exception(
                ParamManager.KEY_NON_PLANE_THICKNESS + ' is invalid.')

        if ((type(self.non_plane_angle) is not int
                and type(self.non_plane_angle) is not float)
                or not (0 < self.non_plane_angle < 90)):
            # self.non_plane_angle \
            #     = ParamManager.DEFALT_PHASE_CONSISTENCY_NON_PLANE_ANGLE
            # change_params.append(
            #     ParamManager.ChangeParam(
            #         ParamManager.KEY_NON_PLANE_ANGLE,
            #         ParamManager.DEFALT_PHASE_CONSISTENCY_NON_PLANE_ANGLE))
            # エラー対応(途中終了)に変更
            raise Exception(
                ParamManager.KEY_NON_PLANE_ANGLE + ' is invalid.')

        if (type(self.output_obj) is not bool):
            # self.output_obj = ParamManager.DEFALT_OUTPUT_OBJ
            # change_params.append(
            #     ParamManager.ChangeParam(
            #         ParamManager.KEY_OUTPUT_OBJ,
            #         ParamManager.DEFALT_OUTPUT_OBJ))
            # エラー対応(途中終了)に変更
            raise Exception(
                ParamManager.KEY_OUTPUT_OBJ + ' is invalid.')

        if (type(self.las_swap_xy) is not bool):
            # self.las_swap_xy = ParamManager.DEFALT_LAS_SWAP_XY
            # change_params.append(
            #     ParamManager.ChangeParam(
            #         ParamManager.KEY_LAS_SWAP_XY,
            #         ParamManager.DEFALT_LAS_SWAP_XY))
            # エラー対応(途中終了)に変更
            raise Exception(
                ParamManager.KEY_LAS_SWAP_XY + ' is invalid.')

        if (type(self.rotate_matrix_mode) is not int
                or not (self.rotate_matrix_mode
                        is int(ParamManager.RotateMatrixMode.XYZ)
                        or self.rotate_matrix_mode
                        is int(ParamManager.RotateMatrixMode.ZYX))):
            # self.rotate_matrix_mode = ParamManager.RotateMatrixMode.XYZ
            # change_params.append(
            #     ParamManager.ChangeParam(
            #         ParamManager.KEY_ROTATE_MATRIX_MODE,
            #         ParamManager.RotateMatrixMode.XYZ))
            # エラー対応(途中終了)に変更
            raise Exception(
                ParamManager.KEY_ROTATE_MATRIX_MODE + ' is invalid.')
        else:
            if (self.rotate_matrix_mode
                    is int(ParamManager.RotateMatrixMode.XYZ)):
                self.rotate_matrix_mode = ParamManager.RotateMatrixMode.XYZ
            elif (self.rotate_matrix_mode
                    is int(ParamManager.RotateMatrixMode.ZYX)):
                self.rotate_matrix_mode = ParamManager.RotateMatrixMode.ZYX

        return change_params
