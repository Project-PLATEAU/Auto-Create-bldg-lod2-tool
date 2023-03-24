from pyproj import Transformer
from typing import Tuple


class CoordinateConverter:
    """座標変換クラス
    """

    # 平面直角座標系 系番号とEPSGコードの変換
    _EPSG = {1: "epsg:6669", 2: "epsg:6670", 3: "epsg:6671",
             4: "epsg:6672", 5: "epsg:6673", 6: "epsg:6674",
             7: "epsg:6675", 8: "epsg:6676", 9: "epsg:6677",
             10: "epsg:6678", 11: "epsg:6679", 12: "epsg:6680",
             13: "epsg:6681", 14: "epsg:6682", 15: "epsg:6683",
             16: "epsg:6684", 17: "epsg:6685", 18: "epsg:6686",
             19: "epsg:6687"}

    """経緯度座標系
    """
    _EPSG_6668 = 'epsg:6668'

    def __init__(self, coordinate_id: int) -> None:
        """コンストラクタ

        Args:
            coordinate_id (int): 平面直角座標系の系番号(1-19)

        Raises:
            CoordinateConverterException: 平面直角座標系の系番号が範囲外の場合
        """
        self._polar_trans = None
        self._cartesian_trans = None

        if coordinate_id in CoordinateConverter._EPSG:
            self._polar_trans = Transformer.from_crs(
                CoordinateConverter._EPSG[coordinate_id],
                CoordinateConverter._EPSG_6668, always_xy=True)

            self._cartesian_trans = Transformer.from_crs(
                CoordinateConverter._EPSG_6668,
                CoordinateConverter._EPSG[coordinate_id], always_xy=True)
        else:
            raise CoordinateConverterException(
                'coordinate_id is out of range. Values range from 1 to 19.')

    def to_polar(self, x: float, y: float) -> Tuple[float, float]:
        """平面直角座標系から経緯度座標系に変換する

        Args:
            x (float): x座標(東方向の値)
            y (float): y座標(北方向の値)

        Raises:
            CoordinateConverterException: 初期化が行われていない場合

        Returns:
            Tuple[float, float]: 緯度(latitude), 経度(longitude)
        """
        if self._polar_trans is not None:
            lon, lat = self._polar_trans.transform(x, y)
            return lat, lon

        else:
            raise CoordinateConverterException(
                'CoordinateConverter class is not initialized.')

    def to_cartesian(
            self, latitude: float, longitude: float) -> Tuple[float, float]:
        """経緯度座標系から平面直角座標系に変換する

        Args:
            latitude (float): 緯度
            longitude (float): 経度

        Raises:
            CoordinateConverterException: 初期化が行われていない場合

        Returns:
            Tuple[float, float]: x座標, y座標(x:東方向の値, y:北方向の値)
        """
        if self._cartesian_trans is not None:
            x, y = self._cartesian_trans.transform(longitude, latitude)
            return x, y

        else:
            raise CoordinateConverterException(
                'CoordinateConverter class is not initialized.')


class CoordinateConverterException(Exception):
    """座標変換に関するExceptionクラス
    """
    pass
