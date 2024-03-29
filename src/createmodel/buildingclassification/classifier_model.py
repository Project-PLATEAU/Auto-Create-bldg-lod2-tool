import torch
import torchvision.models as models


class ClassifierModel(torch.nn.Module):
    """建物分類モデルクラス
    """

    model: torch.nn.Module

    def __init__(self, in_channels=3, n_classes=10) -> None:
        """コンストラクタ

        Args:
            in_channels(int): 入力チャネル数
            n_classes(int): 分類するクラス数
        """

        super(ClassifierModel, self).__init__()
        #self.model = models.resnet34(weights=None, num_classes=n_classes)
        self.model = models.resnet50(weights=None, num_classes=n_classes)
        #self.model.conv1 = torch.nn.Conv2d(
        #    in_channels,
        #    self.model.conv1.out_channels,
        #    kernel_size=7,
        #    stride=2,
        #    padding=3,
        #    bias=False
        #)

    def forward(self, x):
        """順伝搬
        """
        return self.model(x)