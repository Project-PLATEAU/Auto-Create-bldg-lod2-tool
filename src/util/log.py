import inspect
import os
import shutil
import datetime
import json
import logging

from .parammanager import ParamManager
from .config import Config
from .resulttype import ResultType, ProcessResult
from enum import IntEnum
from logging import getLogger, config


class ModuleType(IntEnum):
    """モジュール情報
    """
    INPUT_CITYGML = 0               # CityGML入力
    MODEL_ELEMENT_GENERATION = 1    # モデル要素生成
    CHECK_PHASE_CONSISTENSY = 2     # 位相一貫性
    PASTE_TEXTURE = 3               # テクスチャ貼付け
    OUTPUT_CITYGML = 4              # CityGML出力
    NONE = 5                        # モジュール不明


class LogLevel(IntEnum):
    """ログレベル
    """
    ERROR = 50              # エラー
    MODEL_ERROR = 40        # モデルエラー
    WARN = 30               # 警告
    INFO = 20               # お知らせ
    DEBUG = 10              # デバッグ


class Singleton(object):
    def __new__(cls, *args, **kargs):
        if not hasattr(cls, "_instance"):
            cls._instance = super(Singleton, cls).__new__(cls)
        return cls._instance


class Log(Singleton):
    """ログクラス
        シングルトン
    """
    log_conf = []
    _MAIN_LOG_FILE_PATH = 'main_log.txt'
    _main_log_file = []           # 実行ログ
    _module_log_file = ['', '', '', '', '']     # モジュールログ
    _standard_log = []              # 標準出力
    FORMAT_ON = 0                   # フォーマット設定あり
    FORMAT_OFF = 1                  # フォーマット設定なし
    # モジュール情報
    MODULE_LIST = {ModuleType.INPUT_CITYGML:
                   ["InputCityGML", "input_citygml_log.txt"],
                   ModuleType.MODEL_ELEMENT_GENERATION:
                   ["ModelElementGeneration",
                    "model_element_generation_log.txt"],
                   ModuleType.CHECK_PHASE_CONSISTENSY:
                   ["CheckPhaseConsistensy",
                    "check_phase_consistensy_log.txt"],
                   ModuleType.PASTE_TEXTURE:
                   ["PasteTexture", "paste_texture_log.txt"],
                   ModuleType.OUTPUT_CITYGML:
                   ["OutputCityGML", "output_citygml_log.txt"]
                   }
    RESULT_MESSAGE = ['SUCCESS', 'WARNING', 'ERROR']    # モジュール実行結果メッセージ
    debug_flag = False          # DEBUGログを出力するかのフラグ

    def __init__(self, param: ParamManager, param_file):
        """ログクラスコンストラクタ
            ロガー作成、ヘッダ部分出力
            モジュールごとのログファイルに開始ログ出力

        Args:
            param (ParamManager): パラメータ情報
            param_file: パラメータファイルパス
        """
        # パラメータ情報からの読み込み情報
        log_folder_path = param.output_log_folder_path    # ログ出力先
        Log.debug_flag = param.debug_log_output     # デバッグログ出力フラグ
        Log.delete_flag = param.delete_error_flag   # エラーデータ削除フラグ
        Log.output_obj_flag = param.output_obj      # OBJファイル出力フラグ
        Log._las_swap_xy = param.las_swap_xy        # las座標のxy入れ替えフラグ
        # 外部標定要素から算出する回転行列のモード
        Log._rotate_matrix_mode = param.rotate_matrix_mode

        # 出力フォルダ等に記載する時刻を揃えるためParamManagerで取得した時刻を使用する
        create_time = param.time.strftime("%Y%m%d_%H%M%S")
        time_log_folder = f'outputlog_{create_time}'
        
        # 環境設定ファイルパス取得
        config_file = os.path.join(
            os.path.dirname(__file__), 'log_config.json')
        # 環境設定用の辞書作成
        with open(config_file, 'r') as f:
            Log.log_conf = json.load(f)
        
        # ログレベル設定
        logging.addLevelName(LogLevel.MODEL_ERROR, 'MODEL_ERROR')
        logging.addLevelName(LogLevel.ERROR, 'ERROR')

        path_err = False
        try:
            if not os.path.isdir(log_folder_path):
                os.makedirs(log_folder_path)
            
            output_log_folder_path = os.path.join(
                log_folder_path, time_log_folder)

        except Exception:
            path_err = True
            output_log_folder_path = os.path.join(
                'output_log', time_log_folder)

        Log.output_log_folder_path = output_log_folder_path

        # フォルダが存在する場合は削除
        if os.path.isdir(Log.output_log_folder_path):
            shutil.rmtree(Log.output_log_folder_path)

        os.makedirs(Log.output_log_folder_path)

        # 実行ログファイルパス作成
        main_log_path = os.path.join(
            Log.output_log_folder_path, Log._MAIN_LOG_FILE_PATH)

        # 既に出力ファイルがあったら削除
        if os.path.isfile(main_log_path):
            os.remove(main_log_path)

        # 実行ログファイル出力先設定
        handlers = Log.log_conf['handlers']
        handlers['MainLogFile']['filename'] = main_log_path
        handlers['MainLogFileNoForm']['filename'] = main_log_path

        # logger環境設定
        config.dictConfig(Log.log_conf)

        # 実行ログと標準出力のロガー作成
        Log._standard_log.append(getLogger('Console'))
        Log._standard_log.append(getLogger('ConsoleNoForm'))
        Log._main_log_file.append(getLogger('MainLogFile'))
        Log._main_log_file.append(getLogger('MainLogFileNoForm'))

        # ヘッダ出力
        self.__log_header(Log._main_log_file[Log.FORMAT_OFF], param_file)
        self.__log_header(Log._standard_log[Log.FORMAT_OFF], param_file)

        if path_err:
            message = f'OutputLogFolderPath Value change "{log_folder_path}"'
            message += ' to "output_log"'
            self.output_log_write(LogLevel.WARN, ModuleType.NONE, message)
 
    def __log_header(self, logger, param_file):
        """実行ログファイルヘッダ出力

        Args:
            logger: ログ出力先
            param_file: パラメータファイルパス
        """
        self._start_time = datetime.datetime.now()      # 実行開始時間

        # 実行ログファイルへのヘッダ出力
        logger.info('AutoCreateLod2')
        logger.info(f'Version : {Config.SYSYTEM_VERSION}')
        logger.info(f'Start Time : {self._start_time}\n')
        logger.info('Module Information List')

        for module_type in Log.MODULE_LIST:
            # モジュール情報出力
            module = Log.MODULE_LIST[module_type]
            logger.info(f'{module[0]} Module')
            logger.info(f'LogFileName : {module[1]}')

        logger.info(f'\nInput Parameter File Path : {param_file}')
        logger.info(f'DebugFlag : {Log.debug_flag}')
        logger.info(f'OutputOBJ : {Log.output_obj_flag}')
        logger.info(f'LasSwapXY : {Log._las_swap_xy}')
        logger.info(f'RotateMatrixMode : {Log._rotate_matrix_mode}\n')

    def log_footer(self):
        """実行ログファイルフッタ出力
        """
        # 終了日時
        end_time = datetime.datetime.now()
        # 実行時間
        process_time = end_time - self._start_time

        # 実行ログファイル出力
        Log._main_log_file[Log.FORMAT_OFF].info(
            f'\nEnd Time : {end_time}')
        Log._main_log_file[Log.FORMAT_OFF].info(
            f'Process Time: {process_time}')

        # 標準出力
        Log._standard_log[Log.FORMAT_OFF].info(
            f'\nEnd Time : {end_time}')
        Log._standard_log[Log.FORMAT_OFF].info(
            f'Process Time: {process_time}')

        # ログファイル操作終了
        logging.shutdown()

        # モジュールロガーリストのリセット
        Log._module_log_file = ['', '', '', '', '']

    @classmethod
    def module_result_log(self, module: ModuleType, result: ResultType):
        """モジュールの実行結果ログ出力
        
        Args:
            module (ModuleType): モジュール情報
            result (ResultType): モジュール実行結果
        """
        # モジュールの実行結果メッセージ取得
        message = Log.RESULT_MESSAGE[result]

        # 実行結果出力メッセージ作成
        module_name = f'{Log.MODULE_LIST[module][0]} Module'
        message = f'{module_name} : Result : {message}'
        Log._main_log_file[Log.FORMAT_ON].info(message)

        # モジュール実行終了ログ出力
        Log._main_log_file[Log.FORMAT_ON].info(f'{module_name} End\n')
        Log._module_log_file[module].info(f'{module_name} End')

        # 標準出力にログ出力
        Log._standard_log[Log.FORMAT_ON].info(message)
        Log._standard_log[Log.FORMAT_ON].info(f'{module_name} End\n')
        
    def __create_logger(module: ModuleType):
        """ロガー作成

        Args:
            module (ModuleType): モジュール情報
        """
        # モジュールタイプがNONE以外はログファイル作成
        if module != ModuleType.NONE and Log._module_log_file[module] == "":
            # モジュールログ出力用のロガー作成
            Log._module_log_file[module] = getLogger(
                f'{Log.MODULE_LIST[module][0]}Log')

            # モジュールログファイルパス作成
            module_log_file_path = os.path.join(
                Log.output_log_folder_path, Log.MODULE_LIST[module][1])

            # 既に出力ファイルが存在していたら削除
            if os.path.isfile(module_log_file_path):
                os.remove(module_log_file_path)

            # ロガーの出力先設定
            fh = logging.FileHandler(module_log_file_path, encoding='utf-8')

            if not Log.debug_flag:
                # 出力ログレベル設定
                fh.setLevel(logging.INFO)

            # ロガーフォーマット設定
            fmt = logging.Formatter(Log.log_conf
                                    ['formatters']['Versatility']['format'])
            fh.setFormatter(fmt)
            Log._module_log_file[module].addHandler(fh)

    @classmethod
    def output_log_write(self, level: LogLevel, module: ModuleType,
                         message=None, standard_flag=False):
        """ログ出力
            モジュールごとのログファイルに出力

        Args:
            module (ModuleType): モジュール情報
            level : ログレベル情報
            message: ログメッセージ
            standard_flag: 標準出力するかのフラグ情報
        """
        if module is not ModuleType.NONE:
            # 出力ログメッセージ作成
            module_name = f'{Log.MODULE_LIST[module][0]} Module'
            message = f'{module_name} : {message}'
            if Log.debug_flag and level >= LogLevel.WARN:
                caller = '\n     [DEBUG] : Caller : relative path = '
                caller += f'{os.path.relpath(inspect.stack()[1].filename)}, '
                caller += f'function = {inspect.stack()[1].function}, '
                caller += f'line = {inspect.stack()[1].lineno}'
                message += caller

            # 標準出力にログ出力
            if standard_flag:
                Log._standard_log[Log.FORMAT_ON].log(level, message)

            Log._module_log_file[module].log(level, message)
        else:
            # 実行ログファイルと標準出力にログ出力
            Log._main_log_file[Log.FORMAT_ON].log(level, message)
            Log._standard_log[Log.FORMAT_ON].log(level, message)

    @classmethod
    def module_start_log(self, module: ModuleType, citygml_filename: str = ''):
        """実行ログ、標準出力、モジュールログへのモジュール実行開始のログ出力
            モジュールログ出力用のロガーを作成
        
        Args:
            module (ModuleType): モジュール情報
            citygml_filename (str, optional): 処理対象ファイル名. Defaults to ''.
        """
        # モジュール名取得
        module_name = f'{Log.MODULE_LIST[module][0]} Module'

        # 実行ログファイルログ出力
        Log._main_log_file[Log.FORMAT_ON].info(f'{module_name} Run')
    
        # 標準出力にログ出力
        Log._standard_log[Log.FORMAT_ON].info(f'{module_name} Run')

        # ロガー作成
        self.__create_logger(module)

        # モジュールログファイルに開始ログ出力
        Log._module_log_file[module].info(
            '--------------------------------------')
        Log._module_log_file[module].info(
            f'start processing {citygml_filename}')
        Log._module_log_file[module].info(
            f'{Log.MODULE_LIST[module][0]} Module Run')

    @classmethod
    def process_start_log(self, citygml_filename: str = ''):
        """実行ログ、標準出力に処理対象のCityGMLファイル名のログを出力する

        Args:
            citygml_filename (str, optional): 処理対象ファイル名. Defaults to ''.
        """
        # 実行ログファイルログ出力
        Log._main_log_file[Log.FORMAT_OFF].info(
            '--------------------------------------')
        Log._main_log_file[Log.FORMAT_ON].info(
            f'{citygml_filename} processing')

        # 標準出力にログ出力
        print('--------------------------------------')
        Log._standard_log[Log.FORMAT_ON].info(
            f'start processing {citygml_filename}')

    def output_summary(self, buildings):
        """モデル作成結果csv出力

        Args:
            buildings (list[CityGmlManager.BuildInfo]): 建物外形情報リスト
        """
        now_time = datetime.datetime.now()
        create_result = getLogger('Summary')

        # モジュールログファイルパス作成
        file_path = os.path.join(
            Log.output_log_folder_path, 'model_create_result.csv')

        # 既に出力ファイルが存在していたら削除
        if os.path.isfile(file_path):
            os.remove(file_path)

        # ロガーの出力先設定
        fh = logging.FileHandler(file_path, encoding='utf_8_sig')

        if not Log.debug_flag:
            # 出力ログレベル設定
            fh.setLevel(logging.INFO)

        # ロガーフォーマット設定
        fmt = logging.Formatter('%(message)s')
        fh.setFormatter(fmt)
        create_result.addHandler(fh)
        
        # ヘッダ部分出力
        time = f'{now_time.year}/{now_time.month}/{now_time.day} '
        time += f'{now_time.hour}:{now_time.minute}:{now_time.second}'

        create_result.info(f'{time}')
        create_result.info('\n[最終結果]')
        create_result.info('SUCCESS: Lod2モデルの建物')
        create_result.info('WARNING: 問題のあるLod2モデルの建物')
        create_result.info('ERROR: 入力したままの建物')

        create_result.info('\n[詳細項目]')
        # 項目説明用の出力フォーマット
        RESULT_ITEMS = [['CityGML読み込み', '〇: Lod0モデルの取得に成功',
                         '×: Lod0モデルの取得に失敗'],
                        ['\nLod2モデルの作成', '〇: Lod2モデルの作成に成功',
                         '×: Lod2モデルの作成に失敗',
                         '-: 作成対象外'],
                        ['\n連続頂点重複検査',
                         '〇: モデル面の頂点が連続して重複していない場合、'
                         'または、重複を検知し自動補正した場合',
                         '×: モデル面の重複頂点を自動補正できなかった場合',
                         '-: 検査対象外'],
                        ['\nソリッド閉合検査',
                         '〇: モデルが閉じた空間である場合、または、'
                         'エラーを検知して自動補正した場合',
                         '×: モデルが閉じた空間となるように自動補正できなかった場合',
                         '-: 検査対象外'],
                        ['\n非平面検査',
                         '〇: モデル面が平面である場合、または、'
                         '非平面を検知して自動補正した場合',
                         '×: 非平面を自動補正できなかった場合',
                         '-: 検査対象外'],
                        ['\n面積0ポリゴン検査',
                         '〇: 面積が0のモデル面が無い場合、または、'
                         '面積が0のモデル面を検知して自動補正した場合',
                         '×: 面積が0のモデル面を自動補正できなかった場合',
                         '-: 検査対象外'],
                        ['\n自己交差/自己接触検査',
                         '〇: モデル面が始終点以外で交差や接触していない場合',
                         '×: モデル面が始終点以外で交差や接触している場合',
                         '-: 検査対象外'],
                        ['\n地物内面同士交差検査',
                         '〇: 異なる面同士が交差していない場合',
                         '×: 異なる面同士が交差している場合',
                         '-: 検査対象外'],
                        ['\nテクスチャ貼付け',
                         '〇: テクスチャ画像の貼付けに成功',
                         '×: モデル面にテクスチャ画像が貼付けられなかった場合',
                         '-: 貼付け対象外']]

        # 項目説明出力
        for i in range(len(RESULT_ITEMS)):
            for j in RESULT_ITEMS[i]:
                create_result.info(j)
        
        # 項目結果のメッセージ
        PROCESS_RESULT_MESSAGE = [',〇', ',×', ',-']
        
        # モデル化結果項目出力
        create_result.info('\nNo,ファイル名,建物ID,最終結果,'
                           'CityGML読み込み,LOD2モデルの作成,'
                           '連続頂点重複検査,ソリッド閉合検査,非平面検査,'
                           '面積0ポリゴン検査,自己交差/自己接触検査,'
                           '地物内面同士交差検査,テクスチャ貼付け')

        row_count = 0       # 行番号

        # モデル化結果サマリー出力
        for build in buildings:
            # モデル化結果の最終結果判定
            if build.paste_texture == ProcessResult.SKIP:
                build.create_result = ResultType.ERROR
            elif (build.intersection == ProcessResult.ERROR
                  or build.face_intersection == ProcessResult.ERROR
                  or build.paste_texture == ProcessResult.ERROR):
                build.create_result = ResultType.WARN
            else:
                build.create_result = ResultType.SUCCESS

            row_count += 1
            message = f'{row_count},{build.citygml_filename},{build.build_id}'
            message += f',{Log.RESULT_MESSAGE[build.create_result]}'
            message += PROCESS_RESULT_MESSAGE[build.read_lod0_model]
            message += PROCESS_RESULT_MESSAGE[build.create_lod2_model]
            message += PROCESS_RESULT_MESSAGE[build.double_point]
            message += PROCESS_RESULT_MESSAGE[build.solid]
            message += PROCESS_RESULT_MESSAGE[build.non_plane]
            message += PROCESS_RESULT_MESSAGE[build.zero_area]
            message += PROCESS_RESULT_MESSAGE[build.intersection]
            message += PROCESS_RESULT_MESSAGE[build.face_intersection]
            message += PROCESS_RESULT_MESSAGE[build.paste_texture]
            create_result.info(message)
