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

    # クラス変数
    # jsonファイルのキー
    KEY_FILE_PATH = 'FilePath'
    KEY_INPUT_GML_FOLDER_PATH = 'InputGMLFolderPath'
    KEY_OUTPUT_GML_FOLDER_PATH = 'OutputGMLFolderPath'
    KEY_OUTPUT_W = 'OutputWidth'
    KEY_OUTPUT_H = 'OutputHeight'
    KEY_BACKGROUND_COLOR = 'BackGroundColor'
    KEY_EXTENTPIXEL = 'Extentpixel'

    # jsonファイルキーリスト
    KEYS = [
        KEY_FILE_PATH,
        KEY_INPUT_GML_FOLDER_PATH,
        KEY_OUTPUT_GML_FOLDER_PATH,
        KEY_OUTPUT_W,
        KEY_OUTPUT_H,
        KEY_BACKGROUND_COLOR,
        KEY_EXTENTPIXEL]

    def __init__(self) -> None:
        """ コンストラクタ
        """
        self.input_gml_folder_path = ''     # CityGML出力フォルダパス
        self.output_gml_folder_path = ''    # ログ出力先
        self.output_width = 0               # 出力画像幅
        self.output_height = 0              # 出力画像高さ
        self.background_color = 0           # 背景色
        self.extent_pixel = 0               # ポリゴン余白

        # 作業用パラメータ
        self.time = datetime.datetime.now()     # 処理開始時刻

    def read(self, file_path) -> list[ChangeParam]:
        """jsonファイル読み込み関数

        Args:
            file_path (string): jsonファイルパス

        Raises:
            FileNotFoundError: filePathで指定されたファイルが存在しない
            Exception: ファイル/フォルダパスが文字列ではない場合,または空文字の場合

        Returns:
            list[ChangeParam]: 入力エラーによりデフォルト値を採用したパラメータリスト
        """

        # change_params = []  # デフォルト値に変更したパラメータリスト

        if not os.path.isfile(file_path):
            # ファイルが存在しない場合
            raise FileNotFoundError('parameter file does not exist.')

        # ファイルが存在する場合
        try:
            print(file_path)
            jsonLoad = json.load(open(file_path, encoding='Shift-JIS', mode='r'))
        except json.decoder.JSONDecodeError as e:
            # 未記入項目がある場合にデコードエラーが発生する
            r = e.lineno
            c = e.colno
            raise(Exception(
                f'json file decoding error: {e.msg} line {r} column {c}.'))
                
        # 値の取得
        self.input_gml_folder_path = jsonLoad[self.KEY_FILE_PATH][
            self.KEY_INPUT_GML_FOLDER_PATH]
        self.output_gml_folder_path = jsonLoad[self.KEY_FILE_PATH][
            self.KEY_OUTPUT_GML_FOLDER_PATH]
        self.output_width = jsonLoad[self.KEY_OUTPUT_W]
        self.output_height = jsonLoad[self.KEY_OUTPUT_H]
        self.background_color = jsonLoad[self.KEY_BACKGROUND_COLOR]
        self.extent_pixel = jsonLoad[self.KEY_EXTENTPIXEL]
        
        if (type(self.input_gml_folder_path) is not str
                or not self.input_gml_folder_path):
            # 文字列ではない or 空文字の場合
            raise Exception(
                ParamManager.KEY_FILE_PATH.KEY_INPUT_GML_FOLDER_PATH + ' is invalid.')
        if not os.path.isdir(self.input_gml_folder_path):
            # 入力CityGMLフォルダが存在しない場合
            raise Exception(
                ParamManager.KEY_FILE_PATH.KEY_INPUT_GML_FOLDER_PATH + ' not found.')

        if (type(self.output_gml_folder_path) is not str
                or not self.output_gml_folder_path):
            # 文字列ではない or 空文字の場合
            raise Exception(
                ParamManager.KEY_FILE_PATH.KEY_OUTPUT_GML_FOLDER_PATH + ' is invalid.')
        
        input_folder_name = os.path.basename(self.input_gml_folder_path)
        time_str = self.time.strftime('%Y%m%d_%H%M')
        output_folder_path = os.path.join(
            self.output_gml_folder_path, f'{input_folder_name}_{time_str}')
        if not os.path.exists(self.output_gml_folder_path):
            # CityGML出力ファイルの格納フォルダがない場合
            try:
                os.makedirs(output_folder_path)
                self.output_gml_folder_path = output_folder_path    # 更新
            except Exception:
                raise Exception(
                    output_folder_path + ' cannot make.')

        if type(self.output_width) is not int:
            raise Exception(
                ParamManager.KEY_FILE_PATH.KEY_OUTPUT_W + ' is invalid.')

        if type(self.output_height) is not int:
            raise Exception(
                ParamManager.KEY_FILE_PATH.KEY_OUTPUT_H + ' is invalid.')

        if (type(self.background_color) is not int
                or not (0 <= self.background_color <= 255)):
            raise Exception(
                ParamManager.KEY_FILE_PATH.KEY_BACKGROUND_COLOR + ' is invalid.')

        if type(self.extent_pixel) is not int:
            raise Exception(
                ParamManager.KEY_FILE_PATH.KEY_EXTENTPIXEL + ' is invalid.')
