import sys
import os
import glob
import string
from logging import getLogger
from enum import Enum

from shapely.geometry import Point
# from shapely.geometry import MultiPoint
# from shapely.geometry import LineString

from .faceinfo import FaceInfos, FaceInfo, IndexInfo, MaterialInfo

logger = getLogger(__name__)


class BldElementType(Enum):
    """部材タイプ
    """
    NONE = 0        # 未定義
    ROOF = 1        # 屋根
    WALL = 2        # 壁
    GROUND = 3      # 地面


class BldElement:
    """部材情報管理クラス
    """
    _element_dict = {'Roof': BldElementType.ROOF,
                     'Wall': BldElementType.WALL,
                     'Ground': BldElementType.GROUND}

    @classmethod
    def get_element_type(self, str) -> BldElementType:
        if str in self._element_dict:
            return self._element_dict[str]
        else:
            return BldElementType.NONE
    
    @classmethod
    def get_element_str(self, element_type) -> string:
        r_str = ''
        for key, value in self._element_dict.items():
            if (value == element_type):
                r_str = key
        
        return r_str


class ObjInfo:
    """建物情報クラス
    """

    def __init__(self):
        """コンストラクタ
        """
        self._faces_list = dict()   # 部材毎の面情報リスト
        self._file_name = ''        # ファイル名
        self._mtl_file_name = ''    # マテリアルファイル名
        self._mtl_list = dict()     # マテリアル情報リスト

        self._v_list = PointManager()
        self._vt_list = PointManager()
        self._vn_list = PointManager()

        # Obj ファイル内のインデックス値が 1 から始まるため、ダミーの座標を追加
        self._v_list.append_pos(Point(0.0, 0.0, 0.0))
        self._vt_list.append_pos(Point(0.0, 0.0, 0.0))
        self._vn_list.append_pos(Point(0.0, 0.0, 0.0))

        # 重複した座標値が含まれるケースがあるため、一旦 OBJ ファイル内の各座標を
        # そのままリストで保持
        self._tmp_v_list = []
        self._tmp_vt_list = []
        self._tmp_vn_list = []

    @property
    def v_list(self) -> list:
        return self._v_list.get_list()
    
    @property
    def v_list_manger(self):
        return self._v_list

    @property
    def faces_list(self) -> dict:
        return self._faces_list

    @property
    def file_name(self) -> string:
        return self._file_name
    
    @file_name.setter
    def file_name(self, value):
        self._file_name = value

    @property
    def mtl_file_name(self) -> string:
        return self._mtl_file_name
    
    @mtl_file_name.setter
    def mtl_file_name(self, value):
        self._mtl_file_name = value

    def set_mtl_info(self, mtl_info: MaterialInfo):
        self._mtl_list[mtl_info.name] = mtl_info

    def get_mtl_info(self):
        if any(self._mtl_list):
            return self._mtl_list
        else:
            return None

    def reset_faces(self):
        """面情報リストリセット
        """
        self._faces_list.clear()

    # def set_vlist(self, point_list):
    #     """座標値の設定

    #     Args:
    #         point_list (list(Point)): 座標点列
    #     """
    #     self._v_list = point_list.copy()
    
    # def set_vt_list(self, point_list):
    #     """テクスチャ座標値の設定

    #     Args:
    #         point_list (list(Point)): テクスチャ座標点列
    #     """
    #     self._vt_list = point_list.copy()

    def get_polygon_list(self, element_type) -> list:
        """指定した部材の面(Polygon)のリストを取得

        Args:
            element_type (BldElementType): 部材タイプ

        Returns:
            Point[][]: 座標列のリスト
                       座標は、self._v_list 内の座標値の参照
        """
        if not (element_type in self._faces_list):
            return None
        
        faces = self._faces_list[element_type]
        multi_point_list = []
        for face in faces.faces:
            point_list = []
            for index in face.indx:
                pos = self._v_list.get_pos(index.pos)
                if pos is not None:
                    point_list.append(pos)
            multi_point_list.append(point_list)
        
        return multi_point_list

    def get_texture_list(self, element_type) -> list:
        """指定した部材の面(Polygon)のテクスチャ座標リストを取得

        Args:
            element_type (BldElementType): 部材タイプ

        Returns:
            Point[][]: テクスチャ座標列のリスト
                       座標は、self._vt_list 内の座標値の参照
        """
        if not (element_type in self._faces_list):
            return None
        
        faces = self._faces_list[element_type]
        multi_point_list = []
        for face in faces.faces:
            point_list = []
            for index in face.indx:
                tex = self._vt_list.get_pos(index.tex)
                point_list.append(tex)
            multi_point_list.append(point_list)
        
        return multi_point_list

    def append_face(self, element_type, face):
        """指定した部材に面情報を追加

        Args:
            element_type (BldElementType): 部材タイプ
            face (FaceInfo): 面情報
        """
        if not (element_type in self._faces_list):
            self._faces_list[element_type] = FaceInfos()
        self._faces_list[element_type].append(face)
            
    def append_faces(self, element_type: BldElementType,
                     polygon_list: list):
        """指定した部材に面情報群を追加

        Args:
            element_type (BldElementType): 部材タイプ
            polygon_list (list[list[NDArray]]): 面情報群リスト
        """

        for point_list in polygon_list:
            face = FaceInfo()
            for pos in point_list:
                p = Point(pos)
                index = self._v_list.append_pos(p)  # 座標値リストに追加
                face.append(IndexInfo(index))
            self.append_face(element_type, face)    # 面情報を追加
        logger.debug(self._v_list)

    def append_point_list(self, element_type: BldElementType,
                          point_list: list):
        """指定した部材に面情報を追加

        Args:
            element_type (BldElementType):  部材タイプ
            point_list (list<Point>): 面の座標列
        """
        face = FaceInfo()
        for pos in point_list:
            index = self._v_list.append_pos(pos)    # 座標値リストに追加
            face.append(IndexInfo(index))
        self.append_face(element_type, face)        # 面情報を追加

    def append_textures(self, element_type: BldElementType,
                        multi_point_list: list):
        """指定した部材にテクスチャ座標情報を付加

        Args:
            element_type (BldElementType): 部材タイプ
            multi_point_list (Point[][]): テクスチャ座標リスト
        """

        faces = self._faces_list[element_type]

        for i, multi_point in enumerate(multi_point_list):
            tex_list = []
            for tex_pos in multi_point:
                index = self._vt_list.append_pos(tex_pos)       # テクスチャ座標リストに追加
                tex_list.append(index)
            faces.append_texture(i, tex_list)       # 面情報更新

    def append_texture(self, element_type: BldElementType, face_no: int,
                       point_list: list):
        """指定した部材の指定した面にテクスチャ座標を付加

        Args:
            element_type (BldElementType): 部材タイプ
            face_no (int): 面番号
            point_list (list): テクスチャ座標リスト
        """
        faces = self._faces_list[element_type]

        if face_no >= len(faces.faces):
            return

        tex_list = []
        for tex_pos in point_list:
            index = self._vt_list.append_pos(tex_pos)   # テクスチャ座標リストに追加
            tex_list.append(index)
        faces.append_texture(face_no, tex_list)     # 面情報更新

    def remove_face(self, element_type, face):
        """指定した部材の指定した面情報を削除
        
        Args:
            element_type(BldElementType): 部材タイプ
            face (FaceInfo): 面情報
        """
        self._faces_list[element_type].remove_face(face)    # 指定した面情報を削除

    def read_file(self, file_path: string, err_message=None):
        """OBJファイル入力

        Args:
            file_path (string): OBJファイルパス
        """

        self._file_name = file_path

        if not os.path.isfile(file_path):
            # ファイルが存在しない場合
            raise FileNotFoundError(f'{file_path} : obj file does not exist.')

        # obj ファイル入力
        cur_parts = BldElementType.NONE
        line_ct = 0
        with open(file_path, mode='r') as f:
            for line in f:
                line_ct += 1
                s_list = line.strip().split(' ')
                if len(s_list) == 0 or not s_list[0]:
                    continue

                try:
                    if s_list[0] == 'v':                    # 頂点座標
                        p = self._get_v(s_list)
                        if p is not None:
                            self._tmp_v_list.append(p)

                    elif s_list[0] == 'vt':                 # テクスチャ座標
                        tp = self._get_vt(s_list)
                        if tp is not None:
                            self._tmp_vt_list.append(tp)
                    
                    elif s_list[0] == 'vn':                 # 法線
                        continue
                    
                    elif s_list[0] == 'mtllib':             # マテリアルファイル名
                        self._set_mtllib(s_list)
                    
                    elif s_list[0] == 'usemtl':             # マテリアル名
                        self._set_usemtl(s_list)

                    elif s_list[0] == 'f':                  # インデックス情報
                        if cur_parts in self._faces_list:
                            index_list, tex_list \
                                = self._faces_list[cur_parts].append_by_str(
                                    s_list)
                            for index in index_list:
                                if index > len(self._tmp_v_list):
                                    raise ValueError(
                                        f'{index} :index value exceeds'
                                        ' coordinates num.')
                            for tex in tex_list:
                                if tex > len(self._tmp_vt_list):
                                    raise ValueError(
                                        f'{tex}: tex index value '
                                        'exceeds coordinates num.')
                    
                    elif s_list[0][0] == '#':
                        if len(s_list) > 1:
                            cur_parts = BldElement.get_element_type(s_list[1])
                            if cur_parts != BldElementType.NONE:
                                # 部材情報
                                self._faces_list[cur_parts] = FaceInfos()
                    
                except (SyntaxError, ValueError) as e:
                    message = 'Line ' + str(line_ct) + ', ' + e.msg
                    if err_message is None:
                        print(message)
                        e.msg = ''
                    else:
                        # err_message = message
                        e.msg = message
                    raise(e)

        if BldElementType.ROOF not in self._faces_list \
                or BldElementType.WALL not in self._faces_list \
                or BldElementType.GROUND not in self._faces_list:
            message = 'element information (ex. # ROOF) does not exist.'
            if err_message is None:
                print(message)
            else:
                err_message += message
            raise(SyntaxError)

        # 座標値を PointManager に登録
        self._restruct_points()

        # mtl ファイル入力
        if self._mtl_file_name:
            line_ct = 0
            folder_path = os.path.dirname(file_path)
            mtl_file = os.path.join(folder_path, self._mtl_file_name)

            if not os.path.exists(mtl_file):
                message = mtl_file + ': mtl file does not exist.'
                if err_message is None:
                    print(message)
                else:
                    err_message += message
                raise FileNotFoundError()

            cur_mtl = None
            with open(mtl_file, mode='r') as f:
                for line in f:
                    line_ct += 1
                    s_list = line.strip().split(' ')
                    if len(s_list) == 0 or not s_list[0]:
                        continue

                    try:
                        if s_list[0] == 'newmtl':
                            mtl_name = ''
                            if len(s_list) > 1:
                                mtl_name = s_list[1]
                            else:
                                err_str = 'newmtl: material name required.'
                                raise SyntaxError(err_str)
                            if mtl_name in self._mtl_list:
                                cur_mtl = self._mtl_list[mtl_name]
                        else:
                            if cur_mtl is not None:
                                cur_mtl.set_by_str(s_list)
                                cur_mtl = None
                    except (SyntaxError, ValueError) as e:
                        message = 'Line ' + str(line_ct) + ', ' + e.msg
                        if err_message is None:
                            print(message)
                        else:
                            err_message += message
                        raise(e)

    def write_file(self, file_path='', swap_xy=False):
        """OBJファイル出力

        Args:
            file_path (string): OBJファイルパス
            swap_xy (bool, optional):\
                True:xy座標を入れ替える, False:xy座標を入れ替えない.\
                Defaults to False.
        """
        
        path = file_path
        if not path:
            path = self._file_name
        
        logger.debug(f'out path = {path}')

        # obj ファイル出力
        try:
            with open(path, mode='w', encoding='UTF-8') as f:
                # マテリアルファイル名
                if self._mtl_file_name:
                    f.write(f'mtllib {self._mtl_file_name}\n\n')
                
                # 頂点座標出力
                f.write('#Vertex\n')
                f.writelines(self._get_v_str(swap_xy))

                # テクスチャ座標出力
                f.writelines(self._get_vt_str())

                # usemtl 出力
                for mtl_info in self._mtl_list.values():
                    f.write(f'usemtl {mtl_info.name}\n\n')

                # 部材毎の面情報出力
                f.write('#Face Index\n')
                for f_key, f_value in self._faces_list.items():
                    f.write(f'# {BldElement.get_element_str(f_key)}\n')
                    faces_str = f_value.get_str(swap_xy)
                    f.writelines(faces_str)
                    f.write('\n')
            
            # mtl ファイル出力
            if self._mtl_file_name:
                path = os.path.join(os.path.dirname(path),
                                    os.path.basename(self._mtl_file_name))
                
                with open(path, mode='a', encoding='UTF-8') as f:
                    for mtl_value in self._mtl_list.values():
                        f.writelines(mtl_value.get_str())
        except Exception as e:
            raise(e)

    def _get_v(self, s_list) -> Point:
        """座標値情報読み込み

        座標値記述フォーマット: v x y z

        Args:
            s_list (string[]): 座標値文字列リスト

        Returns:
            Point: 座標値
        """
        list_len = len(s_list)

        if list_len < 4:
            join_str = ' '.join(s_list)
            raise SyntaxError(f'{join_str} : x y z values required.')
        
        float_list = []
        for s in s_list[1:]:
            try:
                float_list.append(float(s))
            except ValueError:
                join_str = ' '.join(s_list)
                raise SyntaxError(f'{join_str} : purse error.')
        
        return Point(float_list[0], float_list[1], float_list[2])

    def _get_v_str(self, swap_xy=False) -> list:
        """Objファイル出力用座標値文字列を作成

        Args:
            swap_xy (bool, optional):\
                True:xy座標を入れ替える, False:xy座標を入れ替えない.\
                Defaults to False.

        Returns:
            list: Objファイル出力用座標値文字列リスト
        """
        r_list = []

        if self._v_list.get_point_num() < 2:
            return r_list
        
        v_list = self._v_list.get_list()

        for v_point in v_list[1:]:
            if swap_xy:
                #r_str = 'v ' + str(v_point.y) + ' ' + str(v_point.x)
                r_str = 'v {:.03f} {:.03f}'.format(v_point.y, v_point.x)
            else:
                #r_str = 'v ' + str(v_point.x) + ' ' + str(v_point.y)
                r_str = 'v {:.03f} {:.03f}'.format(v_point.x, v_point.y)

            #r_str = r_str + ' ' + str(v_point.z) + '\n'
            r_str = r_str + ' ' + '{:.03f}'.format(v_point.z) + '\n'
            r_list.append(r_str)
        
        r_list.append('\n')

        return r_list

    def _get_vt(self, s_list) -> Point:
        """テクスチャ座標値情報読み込み

        テクスチャ座標値記述フォーマット: vt x y

        Args:
            s_list (string[]): 座標値文字列リスト

        Returns:
            Point: 座標値 (x, y)
        """
        list_len = len(s_list)

        if list_len < 3:
            join_str = ' '.join(s_list)
            raise SyntaxError(f'{join_str} : x y values required.')
        
        float_list = []
        for s in s_list[1:]:
            try:
                float_list.append(float(s))
            except ValueError:
                join_str = ' '.join(s_list)
                raise SyntaxError(f'{join_str} : purse error.')
        
        return Point(float_list[0], float_list[1])

    def _get_vt_str(self) -> list:
        """Objファイル出力用テクスチャ座標値文字列を作成

        Returns:
            list: Objファイル出力用テクスチャ座標値文字列リスト
        """
        r_list = []

        if self._vt_list.get_point_num() < 2:
            return r_list

        vt_list = self._vt_list.get_list()

        for vt_point in vt_list[1:]:
            r_str = 'vt ' + str(vt_point.x) + ' ' + str(vt_point.y) + '\n'
            #r_str = 'vt {:.03f} {:.03f}\n'.format(vt_point.x, vt_point.y)
            r_list.append(r_str)
        
        r_list.append('\n')

        return r_list

    def _set_mtllib(self, s_list):
        """テクスチャファイル名設定

        Args:
            s_list (string[]): mtllib 文字列リスト
        """
        if len(s_list) > 1:
            self._mtl_file_name = s_list[1]
        else:
            raise SyntaxError('mtllib : texture file name required.')

    def _set_usemtl(self, s_list):
        """テクスチャ名設定

        Args:
            s_list (string[]): usemtl 文字列リスト

        """
        if len(s_list) > 1:
            mtl_name = s_list[1]
            if mtl_name:
                self._mtl_list[mtl_name] = MaterialInfo(mtl_name)
        else:
            raise SyntaxError('usemtl : texture name required.')

    def _restruct_points(self):
        """座標情報を PointManager に登録し、重複頂点を除外
        """
        v_index_dict = dict()
        for i, pos in enumerate(self._tmp_v_list):
            index = self._v_list.append_pos(pos)
            if i + 1 != index:
                v_index_dict[i + 1] = index

        vt_index_dict = dict()
        for i, pos in enumerate(self._tmp_vt_list):
            index = self._vt_list.append_pos(pos)
            if i + 1 != index:
                vt_index_dict[i + 1] = index

        # 新しいインデックス値に置き換え
        for key in self._faces_list.keys():
            faces = self._faces_list[key]
            for face in faces.faces:
                for index in face.indx:
                    if index.pos in v_index_dict:
                        index.pos = v_index_dict[index.pos]
                    if index.tex in vt_index_dict:
                        index.tex = vt_index_dict[index.tex]

        self._tmp_v_list.clear()
        self._tmp_vt_list.clear()


class ObjInfos:
    """建物情報群クラス
    """

    def __init__(self):
        """コンストラクタ
        """
        self._obj_list = []         # 建物情報リスト

    @property
    def obj_list(self):
        return self._obj_list

    def get_obj_num(self):
        return len(self._obj_list)

    def read_files(self, folder_path):
        """OBJファイル群入力

        Args:
            folder_path (string): OBJファイル格納フォルダパス
        """
        if not os.path.exists(folder_path):
            print(f'{folder_path}: obj folder does not exist.')
            raise FileNotFoundError()

        path_list = glob.glob(os.path.join(folder_path, '*.obj'))
        if len(path_list) == 0:
            # フォルダ内にファイルが存在しない場合
            raise FileNotFoundError(
                f'{folder_path} : obj folder do not have obj file.')

        for path in path_list:
            obj_info = ObjInfo()
            logger.debug(f'path = {path}')
            obj_info.read_file(path)
            self._obj_list.append(obj_info)

    def write_files(self, folder_path):
        """OBJファイル群出力

        Args:
            folder_path (string): OBJファイル格納フォルダパス
        """
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f'{folder_path}:\
                 obj folder does not exist.')
        
        logger.debug(f'out_folder = {folder_path}')
        logger.debug(f'obj len = {len(self._obj_list)}')

        for obj in self._obj_list:
            file_path = os.path.join(
                folder_path, os.path.basename(obj.file_name))
            logger.debug(f'file_path = {file_path}')
            obj.write_file(file_path)


class CompPoint(object):
    """座標値クラス (辞書キー対応版)
    """
    def __init__(self, x: float, y: float, z: float):
        """コンストラクタ

        Args:
            x (float): x 座標値
            y (float): y 座標値
            z (float): z 座標値
        """
        self.x = x
        self.y = y
        self.z = z

    def __eq__(self, other):
        """比較関数
        """
        if not isinstance(other, CompPoint):
            return False
        return (abs(self.x - other.x) < sys.float_info.epsilon) \
            and (abs(self.y - other.y) < sys.float_info.epsilon) \
            and (abs(self.z - other.z) < sys.float_info.epsilon)
    
    def __hash__(self):
        """ハッシュ関数
        """
        return hash(self.x + self.y + self.z)


class PointManager:
    """座標値管理クラス
    """

    def __init__(self):
        """コンストラクタ
        """
        self._dict = dict()         # dict<CompPoint, int>
        self._list = []             # list<Point>
    
    def get_pos(self, index: int) -> Point:
        """インデックス値から座標値を取得

        Args:
            index (int): インデックス値

        Returns:
            Point: 座標値
                   インデックス値が無効の場合は　None を返却
        """
        if index >= 0 and index < len(self._list):
            return self._list[index]
        return None
    
    def append_pos(self, pos: Point) -> int:
        """座標値追加

        Args:
            pos (Point): 座標値

        Returns:
            int: インデックス値
        """

        if pos.has_z:
            c_pos = CompPoint(pos.x, pos.y, pos.z)
        else:
            c_pos = CompPoint(pos.x, pos.y, 0.0)
        
        if c_pos in self._dict:
            return self._dict[c_pos]
        else:
            index_no = len(self._list)
            self._list.append(pos)
            self._dict[c_pos] = index_no
            return index_no

    def get_list(self) -> list:
        """座標値リストを取得

        Returns:
            list<Point>: 座標値リスト
        """
        return self._list
    
    def get_point_num(self) -> int:
        return len(self._list)
