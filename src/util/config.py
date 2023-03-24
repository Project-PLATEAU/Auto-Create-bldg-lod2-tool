import os


class Config:
    """static変数設定
    """
    # システムバージョン
    SYSYTEM_VERSION = "1.0.0"

    # OBJファイル出力先(中間出力物)
    OUTPUT_OBJDIR = os.path.join('.', 'temp')  # 中間出力フォルダ
    # モデル要素生成
    OUTPUT_MODEL_OBJDIR = os.path.join(OUTPUT_OBJDIR, 'createmodel')
    # 位相一貫性補正後
    OUTPUT_PHASE_OBJDIR = os.path.join(OUTPUT_OBJDIR, 'phaseconsistensy')
    # テクスチャ自動貼付け後
    OUTPUT_TEX_OBJDIR = os.path.join(OUTPUT_OBJDIR, 'texturemapping')
