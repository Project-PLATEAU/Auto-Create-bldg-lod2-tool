import string
from logging import getLogger

logger = getLogger(__name__)


class IndexInfo:
    """インデックス情報クラス
    """
    
    def __init__(self, pos=-1, tex=-1, norm=-1):
        """コンストラクタ

        Args:
            pos (int): 座標番号
            tex (int): テクスチャ座標番号
            norm (int): 法線番号
        
        """
        self._pos = pos
        self._tex = tex
        self._norm = norm
    
    def set(self, str):
        """文字列からインデックス値設定
        
        Obj ファイル内のインデックス情報文字列(座標値[/テクスチャ座標値[/法線値]])
        
        Args:
            str (string): Obj ファイル内のインデックス情報文字列
        
        raise:
            ValueError  文字列から番号へのパース失敗時
            SyntaxError インデックス情報に誤りがある場合

        """
        
        s_list = str.split('/')
        list_len = len(s_list)
        if list_len == 1:
            self._pos = int(s_list[0])
        elif list_len == 2:
            self._pos = int(s_list[0])
            if s_list[1]:
                self._tex = int(s_list[1])
        elif list_len == 3:
            self._pos = int(s_list[0])
            if s_list[1]:
                self._tex = int(s_list[1])
            if s_list[2]:
                self._norm = int(s_list[2])
        elif list_len == 0:
            raise SyntaxError(f'{str} : index values required.')
        else:
            raise SyntaxError(f'{str} : too many index values.')
    
    def get_str(self) -> string:
        """Objファイル出力用インデックス値文字列作成

        Returns:
            string: Objファイル出力用インデックス値文字列
        """

        r_str = ''
        if self._pos != -1:
            r_str = str(self._pos)
        
        if self._tex != -1:
            r_str += '/' + str(self._tex)

        if self._norm != -1:
            r_str += '/' + str(self._norm)

        return r_str

    @property
    def pos(self):
        return self._pos
    
    @pos.setter
    def pos(self, value):
        self._pos = value

    @property
    def tex(self):
        return self._tex

    @tex.setter
    def tex(self, value):
        self._tex = value

    @property
    def norm(self):
        return self._norm

    @norm.setter
    def norm(self, value):
        self._norm = value


class RGBInfo:
    """RGB情報クラス
    
    R, G, B 値(0.0～1.0)
    """

    def __init__(self, r_value=0.0, g_value=0.0, b_value=0.0):
        """コンストラクタ
        """
        self.set(r_value, g_value, b_value)
    
    def set(self, r_value=0.0, g_value=0.0, b_value=0.0):
        """R, G, B 値設定

        Args:
            r_value (float, optional): Red 値. Defaults to 0.0.
            g_value (float, optional): Greeg 値. Defaults to 0.0.
            b_value (float, optional): Blue 値. Defaults to 0.0.
        """
        if (0.0 <= r_value <= 1.0):
            self._r = r_value
        else:
            self._r = 0.0
        if (0.0 <= g_value <= 1.0):
            self._g = g_value
        else:
            self._g = 0.0
        if (0.0 <= b_value <= 1.0):
            self._b = b_value
        else:
            self._b = 0.0
    
    def set_by_str(self, s_list):
        """R, G, B 値設定 (mtl ファイル内文字列から)

        Args:
            s_list (string[]): mtl ファイル内文字列リスト (R G B)
        
        raise:
            ValueError  文字列から番号へのパース失敗時
            SyntaxError R G B 値の指定が無い場合

        """

        if len(s_list) == 4:
            self._r = float(s_list[1])
            self._g = float(s_list[2])
            self._b = float(s_list[3])
        else:
            raise SyntaxError(f'{s_list[0]} : R G B values required.')
    
    def get_str(self) -> string:
        """R G B 値の文字列取得

        Returns:
            string: R G B 値の文字列
        """
        return str(self._r) + ' ' + str(self._g) + ' ' + str(self._b)


class UVInfo:
    """UV座標クラス
    """
    def __init__(self, u_value=0, v_value=0):
        """コンストラクタ

        Args:
            u_value (int, optional): U 座標値. Defaults to 0.
            v_value (int, optional): V 座標値. Defaults to 0.
        """
        self._u = u_value
        self._v = v_value

    def get_str(self) -> string:
        """U V 値の文字列取得

        Returns:
            string: U V 値の文字列
        """

        return str(self._u) + ' ' + str(self._v)


class MaterialInfo:
    """マテリアル情報クラス
    """

    def __init__(self, name):
        """コンストラクタ

        Args:
            name (string): マテリアル名
        """
        self._name = name       # マテリアル名
        self._ka = None         # アンビエントカラー
        self._kd = None         # ディフューズカラー
        self._map_ka = ''       # テクスチャ画像ファイル(アンビエントカラー)
        self._map_kd = ''       # テクスチャ画像ファイル(ディフューズカラー)
    
    @property
    def name(self) -> string:
        return self._name
    
    @name.setter
    def name(self, value: string):
        self._name = value
    
    @property
    def ka(self) -> RGBInfo:
        return self._ka
    
    @ka.setter
    def ka(self, value: RGBInfo):
        self._ka = value

    @property
    def kd(self) -> RGBInfo:
        return self._kd
    
    @kd.setter
    def kd(self, value: RGBInfo):
        self._kd = value
    
    @property
    def map_ka(self) -> string:
        return self._map_ka
    
    @map_ka.setter
    def map_ka(self, value: string):
        self._map_ka = value

    @property
    def map_kd(self) -> string:
        return self._map_kd
    
    @map_kd.setter
    def map_kd(self, value: string):
        self._map_kd = value

    def set_by_str(self, s_list):
        """値セット (mtl ファイル内文字列から)

        Args:
            s_list (string[]): mtl ファイル内 1 行の文字列リスト
        """

        list_len = len(s_list)
        if list_len == 0:
            return
        if s_list[0] == 'ka':
            self._ka = RGBInfo()
            self._ka.set_by_str(s_list)
        elif s_list[0] == 'kd':
            self._kd = RGBInfo()
            self._kd.set_by_str(s_list)
        elif s_list[0] == 'map_ka':
            self._map_ka = self._get_texture_name(s_list)
        elif s_list[0] == 'map_kd':
            self._map_kd = self._get_texture_name(s_list)

    def _get_texture_name(self, s_list):
        """テクスチャ画像ファイル名取得

        Args:
            s_list (string[]): mtl ファイル内 1 行の文字列リスト
        """
        if len(s_list) == 2:
            return s_list[1]
        else:
            raise SyntaxError(f'{s_list[0]} : texture filename required.')

    def get_str(self) -> list:
        """mtl ファイル出力用文字列リストを作成

        Returns:
            string[]: mtl ファイル出力用文字列リスト
        """
        r_list = []
        newmtl_str = 'newmtl ' + self._name + '\n\n'
        r_list.append(newmtl_str)
        if self._ka is not None:
            ka_str = 'ka ' + self._ka.get_str() + '\n'
            r_list.append(ka_str)
        if self._kd is not None:
            kd_str = 'kd ' + self._kd.get_str() + '\n'
            r_list.append(kd_str)
        if self._map_ka:
            map_ka_str = 'map_ka ' + self._map_ka + '\n'
            r_list.append(map_ka_str)
        if self._map_kd:
            map_kd_str = 'map_kd ' + self._map_kd + '\n'
            r_list.append(map_kd_str)
        r_list.append('\n')
        
        logger.debug(r_list)

        return r_list


class FaceInfo:
    """面情報クラス
    """

    def __init__(self):
        """コンストラクタ
        """
        self._indx = []          # インデックス情報リスト

    @property
    def indx(self) -> list:
        return self._indx

    def append(self, index_info):
        """インデックス情報追加

        Args:
            index_info (IndexInfo): インデックス情報
        """
        self._indx.append(index_info)

    def append_texture(self, texture_list):
        """テクスチャ情報付加

        Args:
            texture_list (int[]): テクスチャ番号リスト
        """
        if len(self._indx) != len(texture_list):
            return
        
        for i, index_info in enumerate(self._indx):
            index_info.tex = texture_list[i]

    def modify(self, index_no, index_info):
        """インデックス情報更新

        Args:
            index_no (int): 更新対象データのリスト内位置
            index_info (IndexInfo): インデックス情報
        """
        if index_no >= len(self._indx):
            return

        self._indx[index_no] = index_info

    def delete(self, index_no):
        """インデックス情報削除

        Args:
            index_no (int): 削除対象データのリスト内位置
        """
        if index_no >= len(self._indx):
            return
        
        del self._indx[index_no]

    def set_by_str(self, s_list) -> tuple:
        """値セット (Obj ファイル内の f 行文字列から)
        
        Args:
            s_list (string[]): Obj ファイル内 f 行文字列
        
        returns:
            list<int>, list<int>: 座標インデックスリスト, テクスチャ座標インデックスリスト
        """
        
        pos_list = []
        tex_list = []
        for str in s_list[1:]:
            index = IndexInfo()
            index.set(str)
            self._indx.append(index)
            pos_list.append(index.pos)
            tex_list.append(index.tex)
        
        return pos_list, tex_list
    
    def get_str(self, swap_xy=False) -> string:
        """Objファイル出力用インデックス文字列を作成

        Args:
            swap_xy (bool, optional):\
                True:xy座標を入れ替える, False:xy座標を入れ替えない.\
                Defaults to False.

        Returns:
            string: Objファイル出力用インデックス文字列
        """

        target = self._indx
        if swap_xy:
            # 頂点のxy座標を入れ替える場合
            target = reversed(self._indx)

        r_str = ''
        for index in target:
            if r_str:
                r_str += ' '
            else:
                r_str = 'f '
            r_str += index.get_str()
        
        logger.debug(f'r_str = {r_str}')
        return r_str


class FaceInfos:
    """部材毎の面情報クラス
    """

    def __init__(self):
        """コンストラクタ_
        """
        self._faces = []    # 面情報リスト

    @property
    def faces(self):
        return self._faces

    def append(self, face):
        """面情報追加

        Args:
            face (FaceInfo): 追加する面情報
        """
        self._faces.append(face)
    
    def append_texture(self, index_no, texture_index_list):
        """テクスチャ情報追加

        Args:
            index_no (int): 追加対象の面番号
            texture_index_list (int[]): テクスチャインデックス番号リスト
        """
        if index_no >= len(self._faces):
            return
        
        self._faces[index_no].append_texture(texture_index_list)

    def append_by_str(self, s_list) -> tuple:
        """面情報追加

        Args:
            s_list (string[]): Obj ファイル内 f 行文字列
        
        returns:
            list<int>, list<int>: 座標インデックスリスト, テクスチャ座標インデックスリスト
        """

        face = FaceInfo()
        index, tex = face.set_by_str(s_list)
        self._faces.append(face)

        return index, tex

    def get_str(self, swap_xy=False) -> list:
        """Objファイル出力用インデックス文字列リストを作成

        Args:
            swap_xy (bool, optional):\
                True:xy座標を入れ替える, False:xy座標を入れ替えない.\
                Defaults to False.

        Returns:
            list: objファイル出力用インデックス文字列リスト
        """

        r_list = []
        for face in self._faces:
            str = face.get_str(swap_xy) + '\n'
            r_list.append(str)
        
        return r_list

    def remove_face(self, face):
        """面情報削除
        
        Args:
            face(FaceInfo): 削除する面情報
        """
        self._faces.remove(face)
