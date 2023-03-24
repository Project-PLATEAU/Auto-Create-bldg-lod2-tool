import shapely.geometry as geo

from .classifier import BuildingClass, Classifier
from ..lasmanager import PointCloud
from .preprocess import Preprocess


def ClassifyBuilding(
    cloud: PointCloud,
    shape: geo.Polygon,
    classifier_checkpoint_path: str,
    use_gpu: bool = False,
    grid_size: float = 0.25,
    expand_rate_for_house_model: float = 1,
) -> BuildingClass:
    """建物の分類を行う

    Args:
        cloud(PointCloud): 建物点群
        shape(Polygon): 建物外形ポリゴン
        classifier_checkpoint_path(str): 建物分類の学習済みモデルファイルパス
        use_gpu(bool, optional): 推論時のGPU使用の有無 (Default: False) 
        grid_size(float,optional): 点群の間隔(meter) (Default: 0.25),
        expand_rate_for_house_model(float, optional): 家屋モデル作成時のの画像の拡大率 (Default: 1),

    Returns:
        BuildingClass: 建物クラス
    """
    # 建物が大きい場合は陸屋根とする (TODO: 要対応)
    points = cloud.get_points().copy()
    min_x, min_y = points[:, :2].min(axis=0)
    max_x, max_y = points[:, :2].max(axis=0)
    height = round((max_y - min_y) / grid_size) + 1
    width = round((max_x - min_x) / grid_size) + 1

    if round(max(height, width) * expand_rate_for_house_model) > 256:
        return BuildingClass.FLAT

    # 判定用データの作成
    preprocess = Preprocess(grid_size)
    building_img = preprocess.preprocess(
        cloud,
        shape,
    )

    classifier = Classifier(
        classifier_checkpoint_path,
        use_gpu
    )
    building_class = classifier.classify(building_img)

    return building_class


if __name__ == '__main__':
    pass
