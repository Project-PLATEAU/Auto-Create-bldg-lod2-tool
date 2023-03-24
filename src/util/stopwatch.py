import time


class StopWatch:
    """ストップウォッチクラス
    """
    
    def __init__(self) -> None:
        """コンストラクタ
        """
        self._m_start = 0
        self._m_stop = 0

    def start(self):
        """スタート関数
        """
        if self._m_start == 0:
            self._m_start = time.time()

    def stop(self):
        """ストップ関数
        """
        self._m_stop = time.time()
    
    def restart(self):
        """リスタート関数(開始時刻をリセット)
        """
        self._m_start = time.time()
    
    def getSec(self):
        """経過時間[sec]の取得

        Returns:
            float: 経過時間[sec]
        """
        return self._m_stop - self._m_start
