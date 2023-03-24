import os
import string
from logging import getLogger
from enum import IntEnum
from numpy import empty

from .geoutil import GeoUtil
from ..util.objinfo import BldElementType, ObjInfo

logger = getLogger(__name__)


class StatusType(IntEnum):
    """エラーステータス種別
    """
    NO_ERROR = 0            # エラーなし
    AUTO_CORRECTED = 1      # 自動補正済
    ERROR = 2               # エラーあり
    DELETED = 3             # 削除済


class ErrorType(IntEnum):
    """エラー種別
    """
    DOUBLE_POINT = 0        # 重複点
    SELF_INTERSECTION = 1   # 自己交差
    FACE_INTERSECTION = 2   # 面交差
    NON_PLANE = 3           # 非平面
    ZERO_AREA = 4           # 面積 0
    OPEN_SOLID = 5          # ソリッド開口
    INVALID_INPUT = 6       # 入力エラー
    OTHERS = 7              # その他エラー


class ErrorInfo:
    """エラー情報クラス
    """

    _error_str_dict = {ErrorType.DOUBLE_POINT: 'Double Points is detected.',
                       ErrorType.SELF_INTERSECTION: 'Self intersecting '
                                                    'polygon is detected.',
                       ErrorType.FACE_INTERSECTION: 'Self intersecting '
                                                    'faces are detected.',
                       ErrorType.NON_PLANE: 'Non-Plane face is detected',
                       ErrorType.ZERO_AREA: 'Zero area face is detected.',
                       ErrorType.OPEN_SOLID: 'Model is not a solid model.',
                       ErrorType.INVALID_INPUT: 'Input File is invalid.',
                       ErrorType.OTHERS: 'Unknown error occurred.'}

    def __init__(self, error_type: ErrorType, pos_list: list, message=None):
        """コンストラクタ

        Args:
            error_type (ErrorType): エラー種別
            pos_list (list<geo.Point>): エラー位置座標列
            message (string): エラーメッセージ
        """
        self._error = error_type      # エラー種別
        self._pos = pos_list          # エラー位置座標列
        self._message = message       # エラーメッセージ

    @property
    def pos_list(self) -> list:
        return self._pos
    
    def get_str(self) -> list:
        """エラー出力文字列リスト取得

        Returns:
            list<string>: エラー出力文字列リスト
        """
        str_list = []
        error_str = 'Error    : ' + self._error_str_dict[self._error]
        str_list.append(error_str)

        if self._message is not None and self._message is not empty:
            str_list.append(self._message)
        
        if self._pos is not None:
            pos_str = 'Vertices : '
            for i, point in enumerate(self._pos):
                if i != 0:
                    pos_str += ' - '
                pos_str += GeoUtil.get_point_str(point)
            str_list.append(pos_str)
        
        return str_list
    

class ResultInfo:
    """検査結果管理クラス
    """

    _status_str_dict = {StatusType.NO_ERROR: 'No Error.',
                        StatusType.AUTO_CORRECTED: 'Auto corrected.',
                        StatusType.DELETED: 'Deleted.',
                        StatusType.ERROR: 'Error.'}

    def __init__(self):
        """コンストラクタ_
        """
        self._obj_name = ''                 # Obj ファイル名
        self._err_list = []                 # エラー情報リスト
        self._status = StatusType.NO_ERROR  # エラーステータス
    
    @property
    def obj_name(self):
        return self._obj_name
    
    @obj_name.setter
    def obj_name(self, value: string):
        self._obj_name = value
    
    @property
    def status(self):
        return self._status
    
    @status.setter
    def status(self, value: StatusType):
        if value > self._status:
            self._status = value
    
    def add_err(self, err_type: ErrorType, pos_list: list, message=None):
        """エラー情報追加

        Args:
            err_type (ErrorType): エラー種別
            pos_list (list<geo.Point>): エラー対象位置の座標列
            message (string): エラーメッセージ
        """

        err_info = ErrorInfo(err_type, pos_list, message)
        self._err_list.append(err_info)
    
    def get_str(self) -> list:
        """エラー出力文字列取得

        Returns:
            list<string>: エラー出力文字列
        """

        str_list = []

        # ファイル名
        file_str = 'File     : ' + self.obj_name
        str_list.append(file_str)

        # エラー内容
        for err_info in self._err_list:
            str_list.extend(err_info.get_str())

        # エラーステータス
        logger.debug(f'self.status = {self.status}')
        status_str = 'Status   : ' + self._status_str_dict[self.status]
        str_list.append(status_str)

        return str_list
    
    def output_objfile(self, dir_path: string):
        """エラー検出部分 OBJ ファイル出力 (デバッグ用)

        Args:
            dir_path (string): 出力フォルダ
        """
        
        basename = os.path.basename(self.obj_name)
        for i, err_info in enumerate(self._err_list):
            if len(err_info.pos_list) > 2:
                obj_info = ObjInfo()
                obj_info.append_point_list(BldElementType.ROOF,
                                           err_info.pos_list)
                obj_path = os.path.join(dir_path, f'{basename}_{str(i)}.obj')
                obj_info.write_file(obj_path)
