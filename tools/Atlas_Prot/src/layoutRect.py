import cv2
import math
import lxml
from lxml import etree
import numpy as np
import shutil

from .util.cvsupportjp import Cv2Japanese
from .util.config import Config


class LayoutTexture():

    def __init__(self, inputpath, outputpath, outputwidth, outputheight, bgcolor, extentpix) -> None:
        """コンストラクタ

        Args:
            inputpath(String):CityGML入力フォルダパス
            outputpath(String):CityGML出力フォルダパス
            outputwidth(int):出力画像横サイズ
            outputheight(int):出力画像縦サイズ
            bgcolor(int):出力画像背景色
            extentpix(int):ポリゴン余白
        """
        self.inputpath = inputpath
        self.outputpath = outputpath
        self.output_w = outputwidth
        self.output_h = outputheight
        self.bgcolor = bgcolor
        self.extentpix = extentpix
        self.aw = 0
        self.ah = 0
        self.maxRect_W = 0
        self.maxRect_H = 0
        self.lastRect_W = 0
        self.lastRect_H = 0
        self.rectNumUnfiniAry = []
        self.rectNumBuildUnfiniAry = []
        self.rectNumCompleAry = []
        self.inputRectAry = []
        self.freeSpaceAry = []
        self.buildCount = 0
        self.newRangeList = []          # 出力画像に入る建物数のリスト(アトラス化後)
        self.tempSizeAry = []           # 出力画像サイズのリスト(アトラス化後)

    def layout_texture_main(self, citygml_infoList):
        """アトラス化メイン

        Args:
            citygml_infoList (CityGmlInfo): CityGml情報リスト
        """
        self.citygml_infoList = citygml_infoList

        # CityGMLファイル毎に処理
        for gml_file in self.citygml_infoList:
            tempRectAry = []
            setRangeList = []
            initSizeW = 0
            initSizeH = 0
            setRange = 0
            setCount = 0
            bldCount = 1
            self.newRangeList.clear()
            self.tempSizeAry.clear()
            self.rectNumUnfiniAry.clear()
            self.inputRectAry.clear()
            
            setRangeList = self.set_range(gml_file)

            for build in gml_file.builds:
                # 指定棟数ごとに処理
                if ((self.output_w == build.img_w) and (self.output_h == build.img_h)) or ((self.output_w < build.img_w) or (self.output_h < build.img_h)):
                    # 入力画像サイズが出力画像サイズより大きい場合、アトラス化対象外とする
                    shutil.copyfile(self.inputpath + "\\" + build.in_imgpath, self.outputpath + "\\" + build.in_imgpath)
                    build.out_imgpath = build.in_imgpath
                    build.out_img_w = build.img_w
                    build.out_img_h = build.img_h
                    for poly in build.polygon:
                        poly.out_texcoord = poly.in_texcoord.copy()
                    continue

                for poly in build.polygon:
                    inputRect = self.RectInfo()
                    inputRect.w = poly.useW 
                    inputRect.h = poly.useH
                    inputRect.imgPath = build.in_imgpath
                    inputRect.rectID = poly.coord_ring
                    self.inputRectAry.append(inputRect)
                
                    # 初期サイズ更新
                    if initSizeW < poly.useW:
                        initSizeW = poly.useW
                    if initSizeH < poly.useH:
                        initSizeH = poly.useH

                if setRange < len(setRangeList):
                    if setRangeList[setRange] == bldCount - setCount:
                        # アトラス化
                        self.pack_blf(self.output_w, self.output_h)
                        # outImgW = self.get_areaW()
                        # outImgH = self.get_areaH()
                        tempRectAry.extend(self.inputRectAry)

                        # 初期化
                        self.inputRectAry.clear()
                        setCount += setRangeList[setRange]
                        setRange += 1
                        initSizeW = 0
                        initSizeH = 0
                bldCount += 1

            # データのコピーと画像出力
            setRange = 0
            setCount = 0
            iter1 = iter(tempRectAry)
            iter2 = iter(self.tempSizeAry)
            out_w = 0
            out_h = 0
            out_path = None
            bldCount = 1

            for build in gml_file.builds:
                if ((self.output_w == build.img_w) and (self.output_h == build.img_h)) or ((self.output_w < build.img_w) or (self.output_h < build.img_h)):
                    continue

                # 白紙の出力画像を作成
                if setRange < len(self.newRangeList):
                    if 1 == bldCount - setCount:
                        temp = next(iter2)
                        out_w = int(temp[0])
                        out_h = int(temp[1])
                        val_h = 1
                        val_w = 1
                        while val_h < out_h:
                            val_h *= 2
                        out_h = val_h
                        while val_w < out_w:
                            val_w *= 2
                        out_w = val_w
                        output = np.full((out_h, out_w, 3), self.bgcolor, dtype='uint8')
                        
                        out_path = build.in_imgpath  # 先頭画像のパスを使用

                # オリジナル画像のオープン
                img = Cv2Japanese.imread(
                    self.inputpath + "\\" + build.in_imgpath)

                build.out_imgpath = out_path
                build.out_img_w = out_w
                build.out_img_h = out_h

                # 指定棟数ごとに処理
                for poly in build.polygon:
                    temp = next(iter1)
                    if poly.coord_ring == temp.rectID:
                        poly.useX = temp.x
                        poly.useY = temp.y
                        
                        arrays_r = poly.in_texcoord

                        min_ver = [poly.minX, poly.minY]
                        # max_ver = [poly.maxX, poly.maxY]
                        polygon_w = int(poly.maxX) - int(poly.minX)
                        polygon_h = int(poly.maxY) - int(poly.minY)

                        back = np.full(
                            (polygon_h, polygon_w, 3), self.bgcolor, dtype='uint8')
                        mask = np.full(
                            (polygon_h, polygon_w), 0, dtype='uint8')

                        dst = img[int(min_ver[1]):
                                int(min_ver[1]) + polygon_h,
                                int(min_ver[0]):
                                int(min_ver[0]) + polygon_w]
                        poly_ver = arrays_r - [min_ver[0], min_ver[1]]

                        cv2.fillPoly(mask,
                                    [poly_ver.astype(np.int64)],
                                    color=(255, 255, 255),
                                    )
                        cv2.polylines(mask, [poly_ver.astype(np.int64)],
                                    isClosed=True, color=(255, 255, 255))

                        polygon = np.where(
                            mask[:, :, np.newaxis] == 0, back, dst)

                        #テクスチャ貼付け
                        output[int(poly.useY):int(poly.useY) + polygon.shape[0],
                            int(poly.useX): int(poly.useX) + polygon.shape[1]] \
                           = polygon


                        # 四角形で貼り付ける場合
                        #img_temp = img[int(poly.minY):
                        #               int(poly.minY) + int(poly.useH),
                        #               int(poly.minX):
                        #               int(poly.minX) + int(poly.useW)]
                        #output[int(poly.useY):
                        #       int(poly.useY) + int(poly.useH),
                        #       int(poly.useX):
                        #       int(poly.useX) + int(poly.useW)] = img_temp

                        # 貼り付け先の座標に変換
                        uv_set = np.array([build.out_img_w, build.out_img_h])
                        #xy_set = (arrays_r
                        #        - np.minimum.reduce(arrays_r
                        #            - [poly.useX, poly.useY]))
                        xy_set = (arrays_r
                                - (np.minimum.reduce(arrays_r
                                    - [poly.useX, poly.useY])
                                    - [poly.marginX, poly.marginY]))
                        # XY座標→UV左上原点→UV左下原点に変換
                        poly.out_texcoord = (
                            abs((xy_set + 0.5) / uv_set - [0, 1]))
                    
                if setRange < len(self.newRangeList):
                    if self.newRangeList[setRange] == bldCount - setCount:
                        # 画像出力
                        # cv2.imwrite(self.outputpath + "\\"
                        #                 z  + build.out_imgpath, output)
                        Cv2Japanese.imwrite(self.outputpath
                                            + "\\" + build.out_imgpath, output)
                        # 初期化
                        setCount += self.newRangeList[setRange]
                        setRange += 1
                bldCount += 1
            tempRectAry.clear()

    def set_range(self, gml_file):
        """処理する棟数の区分け

        Args:
            gml_file (CityGmlInfo): CityGml情報リスト

        Return;
            setRangeList(List):メッシュ区分毎の建物数のリスト
        """
        setRangeList = []
        bldCount = 0
        Mesh = 0

        for i, build in enumerate(gml_file.builds, 1):
            if ((self.output_w == build.img_w) and (self.output_h == build.img_h)) or ((self.output_w < build.img_w) or (self.output_h < build.img_h)):
                continue
            
            if bldCount == 0:
                Mesh = build.meshCode

            if Mesh != build.meshCode:
                setRangeList.append(bldCount)
                bldCount = 0
                Mesh = build.meshCode
            bldCount += 1

        if bldCount != 0:
            setRangeList.append(bldCount) 

        return setRangeList

    def pack_blf(self, areaW, areaH):
        """BLF法(Bottom-Left Algorithm)で矩形を配置

        Args:
            areaW (int): 矩形配置枠の横幅初期値
            areaH (int): 矩形配置枠の高さ幅初期値
        """
        # 初期サイズ(1つ目の矩形が入るサイズ)
        self.aw = areaW
        self.ah = areaH

        bRet = False

        while True:
            # 前処理
            self.pack_blf1()

            # 本処理(未配置がなくなるまで続ける)
            unFiniCnt = len(self.rectNumUnfiniAry)
            while unFiniCnt:
                bRet = self.pack_blf2()
                if bRet is False:
                    # 次の画像に移る
                    print("NextImage")
                    break
                unFiniCnt = len(self.rectNumUnfiniAry)
            if self.buildCount != 0:
                self.newRangeList.append(self.buildCount)
                self.tempSizeAry.append([self.maxRect_W, self.maxRect_H])
                self.maxRect_W = 0
                self.maxRect_H = 0
                self.lastRect_W = 0
                self.lastRect_W = 0
                self.aw = areaW
                self.ah = areaH
            if bRet:
                print("AtlasComplete")
                break

    def pack_blf1(self):
        """BLF法前処理
        """
        # 空き領域を初期化
        self.clear_flag_rect()

        # 並べる
        # self.align_rect()

        self.freeSpaceAry[0].w = 0
        self.freeSpaceAry[0].h = 0
    
    def pack_blf2(self):
        """BLF法本処理
        """
        # 矩形を配置
        # NMAX = self.aw * 2
        NMAX = self.aw
        NMAY = self.ah

        # 矩形を配置可能な座標
        minX = NMAX
        minY = NMAY

        # 配置対象
        trgIdx = self.rectNumUnfiniAry[0]
        trgRect = self.inputRectAry[trgIdx]

        # BL安定点候補の個数分ループ
        foundIdx = -1
        freeSpCnt = len(self.freeSpaceAry)
        for ic in range(freeSpCnt):
            space = self.freeSpaceAry[ic]
            if trgRect.w >= space.w and trgRect.h >= space.h:
                x1 = space.x
                y1 = space.y
                x2 = space.x + trgRect.w
                y2 = space.y + trgRect.h
                if y1 < minY or (y1 == minY and x1 < minX):
                    # 配置済み矩形との衝突をチェック
                    if self.is_recthit(x1, y1, x2, y2) is False:
                        # 衝突している矩形は無い
                        # 配置予定座標を記憶
                        minX = x1
                        minY = y1
                        foundIdx = ic

        if minX >= NMAX or minY >= NMAY:
            # 入る場所が無いので別画像にする
            if self.buildCount == 0:
                # 建物一棟分で範囲超過した場合エリアを拡大
                self.expand_area_size()
            else:
                # ひとつ前の最大縦横幅に戻す
                self.maxRect_W = self.lastRect_W
                self.maxRect_H = self.lastRect_H

            # 建物の途中で別画像になった場合、処理中のポリゴンをやり直す
            self.rectNumUnfiniAry = self.rectNumBuildUnfiniAry.copy()
            return False
        else:
            # 入る場所があったので配置
            trgRect.x = minX
            trgRect.y = minY

            if self.maxRect_W < (trgRect.x + trgRect.w):
                self.maxRect_W = trgRect.x + trgRect.w
            if self.maxRect_H < (trgRect.y + trgRect.h):
                self.maxRect_H = trgRect.y + trgRect.h

            trgRect.flag = True
            self.rectNumCompleAry.append(trgIdx)
            self.rectNumUnfiniAry[0:1] = []

            if (len(self.inputRectAry) == trgIdx + 1) or (self.inputRectAry[trgIdx + 1].imgPath != trgRect.imgPath):
                # 建物一棟分の最後
                self.rectNumBuildUnfiniAry = self.rectNumUnfiniAry.copy()
                self.buildCount += 1
                self.lastRect_W = self.maxRect_W
                self.lastRect_H = self.maxRect_H

            # 配置に使ったBL安定点はもう使えないため除去する
            if foundIdx >= 0:
                self.freeSpaceAry[foundIdx:foundIdx + 1] = []

            blPonitAry = []

            # エリア枠と矩形で作れるBL安定点候補を追加登録
            cx1 = trgRect.x
            cx2 = trgRect.x + trgRect.w
            cy1 = trgRect.y
            cy2 = trgRect.y + trgRect.h

            newRect1 = self.RectInfo()
            newRect1.x = cx2
            newRect1.y = 0
            newRect1.w = 0
            newRect1.h = cy1
            blPonitAry.append(newRect1)

            newRect2 = self.RectInfo()
            newRect2.x = 0
            newRect2.y = cy2
            newRect2.w = cx1
            newRect2.h = 0
            blPonitAry.append(newRect2)

            # 現矩形と配置済み矩形との間で作れるBL安定点候補を追加登録
            compCnt = len(self.rectNumCompleAry)
            for rt in range(compCnt):
                compIdx = self.rectNumCompleAry[rt]
                compRect = self.inputRectAry[compIdx]

                px1 = compRect.x
                px2 = compRect.x + compRect.w
                py1 = compRect.y
                py2 = compRect.y + compRect.h

                # 現矩形が配置済み矩形の左側にある場合
                if cx2 <= px1 and cy2 > py2:
                    newRect = self.RectInfo()
                    newRect.x = cx2
                    newRect.y = py2
                    newRect.w = px1 - cx2
                    newRect.h = cy1 - py2 if cy1 > py2 else 0
                    blPonitAry.append(newRect)

                # 現矩形が配置済み矩形の右側にある場合
                if px2 <= cx1 and py2 > cy2:
                    newRect = self.RectInfo()
                    newRect.x = px2
                    newRect.y = cy2
                    newRect.w = cx1 - px2
                    newRect.h = py1 - cy2 if py1 > cy2 else 0
                    blPonitAry.append(newRect)

                # 現矩形が配置済み矩形の上側にある場合
                if cy2 <= py1 and cx2 > px2:
                    newRect = self.RectInfo()
                    newRect.x = px2
                    newRect.y = cy2
                    newRect.w = cx1 - px2 if cx1 > px2 else 0
                    newRect.h = py1 - cy2
                    blPonitAry.append(newRect)

                # 現矩形が配置済み矩形の下側にある場合
                if py2 <= cy1 and px2 > cx2:
                    newRect = self.RectInfo()
                    newRect.x = cx2
                    newRect.y = py2
                    newRect.w = px1 - cx2 if px1 > cx2 else 0
                    newRect.h = cy1 - py2
                    blPonitAry.append(newRect)

            # 得られたBL安定点候補を登録する
            for ic in range(len(blPonitAry)):
                bl = blPonitAry[ic]
                    
                if bl.x < 0 or bl.x >= self.aw or bl.y < 0 or bl.y >= self.ah:
                    continue
                    
                isHit = False
                for compRt in range(compCnt):
                    compIdx = self.rectNumCompleAry[compRt]
                    compRect = self.inputRectAry[compIdx]

                    if compRect.x <= bl.x \
                            and bl.x < compRect.x + compRect.w \
                            and compRect.y <= bl.y \
                            and bl.y < compRect.y + compRect.h:
                        # 配置済み矩形の中にBL安定点候補が入っている
                        isHit = True
                        break

                if isHit:
                    continue

                # 空き領域を追加登録
                space = self.RectInfo()
                space.x = bl.x
                space.y = bl.y
                space.w = bl.w
                space.h = bl.h
                self.freeSpaceAry.insert(0, space)
            
        return True

    def clear_flag_rect(self):
        """全矩形の配置済みフラグを初期化
        """
        if len(self.rectNumUnfiniAry) != 0:
            self.rectNumCompleAry.clear()
            #self.rectNumBuildUnfiniAry = self.rectNumUnfiniAry.copy()

        else:
            self.rectNumUnfiniAry.clear()
            self.rectNumBuildUnfiniAry.clear()
            self.rectNumCompleAry.clear()

            rectCnt = len(self.inputRectAry)
            for ic in range(rectCnt):
                self.rectNumUnfiniAry.append(ic)
                self.rectNumBuildUnfiniAry.append(ic)

        # 空き領域を初期化
        space = self.RectInfo()
        space.x = 0
        space.y = 0
        space.w = self.aw
        space.h = self.ah
        self.freeSpaceAry.clear()
        self.freeSpaceAry.append(space)
        self.buildCount = 0
        self.maxRect_W = 0
        self.maxRect_H = 0
        
    def expand_area_size(self):
        """配置先エリアのサイズを拡大
        """
        # 縦横サイズを別々に拡大していく場合
        if self.aw > self.ah:
            # 幅が高さより大きいので、高さを拡大
            self.ah *= 2
        else:
            # 幅を拡大
            self.aw *= 2

    def align_rect(self):
        """矩形を横一列に並べる
        """
        # 配置先エリアの少し下に並べる
        lx = 0
        ly = self.ah + 16
        lh = 0
        rectCnt = len(self.rectNumUnfiniAry)
        for ic in range(rectCnt):  # 未配置
            idx = self.rectNumUnfiniAry[ic]
            rect = self.inputRectAry[idx]
            if lh < rect.h:
                # その段の高さの最大値を更新
                lh = rect.h
            
            if lx + rect.w >= self.aw:
                # 一定幅を超えてしまったので、段を変える
                lx = 0
                ly += lh
                lh = rect.h

            rect.x = lx
            rect.y = ly
            lx += rect.w

    def is_recthit(self, ax1, ay1, ax2, ay2):
        """配置済みの全矩形と、与えられた矩形領域が重なるかどうかチェック
            重なってたら true を、重なってなければ false を返す

        Args:
            ax1:対象矩形の左上x座標
            ay1:対象矩形の左上y座標
            ax2:対象矩形の右上x座標
            ay2:対象矩形の右上y座標
        """
        if ax2 > self.aw or ay2 > self.ah:
            # 配置先エリアをオーバーしてる
            return True
        
        compCnt = len(self.rectNumCompleAry)
        for ic in range(compCnt):  # 配置済み矩形
            rect = self.inputRectAry[self.rectNumCompleAry[ic]]
            if (ax1 < (rect.x + rect.w)
                and rect.x < ax2
                and ay1 < (rect.y + rect.h)
                    and rect.y < ay2):
                return True
        return False

    def get_areaW(self):
        """配置領域幅取得
        """
        return self.aw
    
    def get_areaH(self):
        """配置領域高さ取得
        """
        return self.ah

    class RectInfo():
        def __init__(self) -> None:
            """コンストラクタ
            """
            self.rectID = None  # ID
            self.imgPath = None  # 元画像のパス
            self.x = 0		    # 配置位置　左下X
            self.y = 0		    # 配置位置　左下Y
            self.w = 0		    # 矩形幅
            self.h = 0		    # 矩形高さ
            self.flag = False   # 配置フラグ
