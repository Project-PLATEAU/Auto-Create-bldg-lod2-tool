class CreateModelMessage:
    """メッセージ定数クラス
    """

    # ##### 入力データに関するメッセージ #####
    ERR_MSG_CITY_GML_DATA = 'No LoD0 polygon data.'

    ERR_MSG_CITY_GML_POLYGON_DATA = 'LoD0 polygon has less than 4 vertices.'
    """建物外形形状の頂点列が4点未満
    """

    ERR_MSG_CITY_GML_POLYGON_NO_AREA = 'LoD0 polygon has no area.'
    """建物外形形状の面線が無い
    """

    ERR_MSG_LAS_COORDINATE_SYSTEM = 'Las coordinate system error.'
    """LASファイルの平面直角座標系の系番号が不正
    """

    ERR_MSG_LAS_FOLDER_NOT_FOUND = 'DSM folder not found.'
    """DSMフォルダが存在しない
    """

    ERR_MSG_LAS_FILE_NOT_FOUND = 'LAS file (*.las) not found.'
    """LASファイルが存在しない
    """

    ERR_MSG_FAILED_TO_READ_LAS_FILE = 'Failed to read Las file.'
    """LASファイルの読み込みに失敗
    """

    # ##### ModelCreator用メッセージ #####
    ERR_MSG_MODEL_CREATOR_UNINITIALIZE = 'ModelCreator is uninitialized.'

    # ##### モデル生成結果のメッセージ #####
    WARN_MSG_COULD_NOT_CREATE_MODEL = 'Could not create models for some data.'
    """一部のデータにおいてモデルが未作成
    """

    # ##### ImageManager用のメッセージ #####
    ERR_MSG_IMG_MNG_FOLDER_NOT_FOUND = 'TIFF folder not found.'
    """TIFFフォルダが存在しない
    """
    ERR_MSG_IMG_MNG_IMAGE_NOT_FOUND = 'TIFF image not found.'
    """Tiffファイルが存在しない
    """
    ERR_MSG_IMG_MNG_TIFF_READ = 'TIFF file read error.'
    """TIFFファイル読み込みに失敗
    """
    ERR_MSG_IMG_MNG_TFW_READ = 'TFW file read error.'
    """TFWファイルの読み込みに失敗
    """
    ERR_MSG_IMG_MNG_READ = 'Failed to read TIFF file.'
    """フォルダ内の全TIFFファイルの読み込みに失敗
    """
    WARN_MSG_IMG_MNG_READ = 'Failed to read some TIFF files.'
    """一部のTIFFファイルの読み込みに失敗
    """

    # ##### LasManager用のメッセージ #####
    ERR_MSG_LAS_MNG_LAS_FOLDER_NOT_FOUND = 'LAS folder not found.'
    """Lasフォルダが存在しない
    """

    ERR_MSG_LAS_MNG_LAS_NOT_FOUND = 'LAS file does not exist.'
    """Lasファイルが存在しない
    """

    ERR_MSG_LAS_MNG_NO_LAS_FILE = 'No LAS file within the read range.'
    """読み込み範囲内に重畳するlasファイルが存在しない
    """

    ERR_MSG_LAS_MNG_NO_POINTS = 'No point cloud data within the read range.'
    """読み込み範囲内に点群データが存在しない
    """

    ERR_MSG_LAS_MNG_NO_GROUOND_POINTS = 'No ground point cloud data.'
    """地面点群データが存在しない
    """

    # ##### PreProcess用メッセージ #####
    ERR_MSG_PREPROC_NO_ROOF_CLUSTER = 'No roof cluster.'
    """屋根クラスタが存在しない
    """

    ERR_MSG_GRAPHCUT_NO_ROOF_POINTS = 'No roof point cloud.'
    """地面点と屋根点の分割で屋根点群が取得できなかった
    """

    ERR_MSG_GRAPHCUT_NO_MERGED_ROOF_POINTS = 'No merged roof points.'
    """高さでマージし、水平連続性で分割後の屋根点群が取得できなかった
    """

    # ##### MBR用メッセージ #####
    ERR_MSG_MBR_NO_CLUSTER = 'No clusters to enter in MBR.'
    """入力点群クラスタが存在しない
    """

    # ##### Model用メッセージ #####
    ERR_MSG_MODEL_NUM_TH = 'Enter a value greater than zero for num_th.'
    """ユークリッド距離によるクラスタリングの最小クラスタ点数閾値のエラー
    """

    ERR_MSG_MODEL_DIST_TH = 'Enter a value greater than ' \
        'or equal to zero for dist_th.'
    """ユークリッド距離によるクラスタリングの距離閾値が範囲外
    """
