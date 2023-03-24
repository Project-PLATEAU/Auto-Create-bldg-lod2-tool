from typing import Final


class CoordinatesConverter:
    """画像座標と地理座標を変換するクラス
    """

    _grid_size: Final[float]
    _geocoords_upper_left: Final[tuple[float, float]]

    def __init__(
        self,
        grid_size: float,
        geocoords_upper_left: tuple[float, float],
    ) -> None:
        """コンストラクタ

        Args:
            grid_size(float): 点群の間隔(meter)
            geocoords_upper_left(tuple[float,float]) 画像左上位置の地理座標(x, yの順)
        """
        self._grid_size = grid_size
        self._geocoords_upper_left = geocoords_upper_left

    def geocoords_to_imagecoords(self, geo_x: float, geo_y: float) -> tuple[int, int]:
        """画像座標から地理座標へ変換する
        Args:
            geo_x(float): 地理座標のx座標
            geo_y(float): 地理座標のy座標

        Returns:
            int: 画像のx座標(左を0とする)
            int: 画像のy座標(上を0とする)
        """

        left, upper = self._geocoords_upper_left
        return (
            round((geo_x - left) / self._grid_size),
            round((upper - geo_y) / self._grid_size)
        )

    def imagecoords_to_geocoords(self, image_x: float, image_y: float) -> tuple[float, float]:
        """画像座標から地理座標へ変換する
        Args:
            image_x(float): 画像のx座標(左を0とする)
            image_y(float): 画像のy座標(上を0とする)

        Returns:
            float: 地理座標のx座標
            float: 地理座標のy座標
        """
        left, upper = self._geocoords_upper_left
        return (
            left + image_x * self._grid_size,
            upper - image_y * self._grid_size,
        )
