
# LOD2建築物モデル自動生成ツール

LOD2建築物モデル自動生成ツール（以下、「本ツール」）は、国土交通省が進めるProject PLATEAUの一環として2022年度に開発されました。\
本ツールは、DSM点群や建物外形データ等を入力データとしてLOD2建築物モデルの作成を自動的に行い、CityGML 形式で出力するシステムです。

<br />

# 動作環境

## ハードウェア、OS環境

本ツールの推奨環境、および必要環境を以下に示します。

- 推奨環境

 <table>
    <tr>
      <td>OS</td>
      <td>Microsoft Windows 10 / 11</td>
    </tr>
    <tr>
      <td>CPU</td>
      <td>Intel Core i7以上</td>
    </tr>
        <tr>
      <td>Memory</td>
      <td>16GByte以上</td>
    </tr>
        <tr>
      <td>GPU</td>
      <td>NVIDIA RTX 2080以上</td>
    </tr>
        <tr>
      <td>GPU Memory</td>
      <td>8GByte以上</td>
    </tr>
 </table>

<br />

- 必要環境

 <table>
    <tr>
      <td>OS</td>
      <td>Microsoft Windows 10 / 11</td>
    </tr>
    <tr>
      <td>CPU</td>
      <td>Intel Core i5以上</td>
    </tr>
        <tr>
      <td>Memory</td>
      <td>8GByte以上</td>
    </tr>
        <tr>
      <td>GPU</td>
      <td>NVIDIA Quadro P620以上</td>
    </tr>
        <tr>
      <td>GPU Memory</td>
      <td>2GByte以上</td>
    </tr>
 </table>

 <br />

## ソフトウェア環境

本ツールは、Python(バージョン3.9以上)のインストールが必要です。\
本ツールが必要とするPythonライブラリを以下に示します。

<br />

<ライブラリ一覧>  
|<center>ライブラリ名</center>|<center>ライセンス</center>|<center>説明</center>|
| - | - | - |
|alphashape|MIT License|点群外形形状作成ライブラリ|
|anytree|Apache 2.0|木構造ライブラリ|
|autopep8|MIT License|コーディング規約(PEP)準拠にソースコードを自動修正するフォーマッターライブラリ|
|coverage|Apache 2.0|カバレッジ取得ライブラリ|
|einops|MIT License|数値計算ライブラリ|
|flake8|MIT License|静的解析ライブラリ|
|jakteristics|BSD License|点群の幾何学的特徴量計算ライブラリ|
|laspy|BSD 2-Clause License|LASファイル処理ライブラリ|
|lxml|BSD 3-Clause License|xml処理ライブラリ|
|matplotlib|Python Software Foundation License|グラフ描画ライブラリ|
|MLCollections|Apache 2.0|機械学習ライブラリ|
|MultiScaleDeformableAttention|Apache 2.0|物体検出ライブラリ|
|NumPy|BSD 3-Clause License|数値計算ライブラリ|
|Open3D|MIT License|点群処理ライブラリ|
|opencv-python|MIT License|画像処理ライブラリ|
|opencv-contrib-python|MIT License|画像処理ライブラリ|
|Pytorch|BSD 3-Clause License|機械学習ライブラリ|
|plateaupy|MIT License|CityGML読み込みライブラリ|
|PyMaxflow|GNU General Public License version 3.0|GraphCut処理ライブラリ|
|pyproj|MIT License|地理座標系変換ライブラリ|
|PuLP|BSD License|数理最適化ライブラリ|
|scikit-learn|BSD 3-Clause License|機械学習ライブラリ|
|scipy|BSD 3-Clause License|統計や線形代数、信号・画像処理などのライブラリ|
|Shapely|BSD 3-Clause License|図形処理ライブラリ|
|Torchvision|BSD 3-Clause Lisence|機械学習ライブラリ|

<br />

# ダウンロード

## リポジトリのクローン

以下のコマンドで本ツールのリポジトリの最新版をクローンします。

`> git clone https://github.com/AAS-BasicSystemsDevelopmentDept/Auto-Create-bldg-lod2-tool.git AutoCreateLod2`

<br />

## AIモデルパラメータのダウンロード

本ツールに搭載されているAIモデルのパラメータをダウンロードします。

（1） 建物分類用モデル

以下より、建物分類用モデル（ファイル名：classifier_parameter.pkl）をダウンロードします。
<br />
<https://nikken-jp.box.com/s/cbme96vyjxfj7a1pnnqc0ryl1dmdvkm7>

<br />

（2） 屋根線検出用モデル

以下より、屋根線検出用モデル（ファイル名：roof_edge_detection_parameter.pth）をダウンロードします。
<br />
<https://nikken-jp.box.com/s/ymnqcgj7azgkfm2rhx9x2hy2eq0762w6>

<br />

（3） バルコニー検出用モデル

以下より、バルコニー検出用モデル（ファイル名：balcony_segmentation_parameter.pkl）をダウンロードします。
<br />
<https://nikken-jp.box.com/s/epu8ihytupko12lnp001ro5462xkuczy>

<br />

ダウンロードしたファイル（classifier_parameter.pkl、roof_edge_detection_parameter.pth、balcony_segmentation_parameter.pkl）をAutoCreateLod2/src/createmodel/data/フォルダに保存します。\
(AutoCreateLod2/srcは本ツールのsrcフォルダまでのパス)

<br />

# 利用手順

本ツールの利用方法についてはチュートリアルを参照してください。

<br />

# ライセンス

- ソースコードおよび関連ドキュメントの著作権は国土交通省に帰属します。
- 本ツールはGNU General Public License v3.0を適用します。
- 本ツールは開発者の許可を得てHEAT: Holistic Edge Attention Transformer for Structured Reconstructionを利用させて頂いております。HEATは2023年1月29日より商用不可とライセンスを変更されましたが、本ソフトウェアはそれより前のバージョンを使用しております。
- 本ドキュメントはProject PLATEAUのサイトポリシー（CCBY4.0および政府標準利用規約2.0）に従い提供されています。
  
<br />

# 注意事項

- 本レポジトリは参考資料として提供しているものです。動作保証は行っておりません。
- 予告なく変更・削除する可能性があります。
- 本レポジトリの利用により生じた損失及び損害等について、国土交通省はいかなる責任も負わないものとします。

<br />

# 参考資料

- LOD2建築物モデル自動生成ツールチュートリアル:  
<https://github.com/AAS-BasicSystemsDevelopmentDept/LOD2Creator/blob/main/docs/TUTORIAL.md>
- LOD2建築物モデル自動生成ツールユーザマニュアル:  
<https://github.com/AAS-BasicSystemsDevelopmentDept/LOD2Creator/blob/main/docs/USER_MANUAL.md>
