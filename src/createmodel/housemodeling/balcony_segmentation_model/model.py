from typing import Final, cast
import ml_collections
import torch

from .networks.vit_seg_configs import get_r50_b16_config
from .networks.vit_seg_modeling import DecoderCup, SegmentationHead, Transformer


class SegmentationModel(torch.nn.Module):
    """セマンティックセグメンテーションを行うモデル
    """

    config: Final[ml_collections.ConfigDict]
    num_classes: Final[int]
    transformer: Final[torch.nn.Module]
    decoder: Final[torch.nn.Module]
    segmentation_head: Final[torch.nn.Module]

    def __init__(
        self,
        num_classes: int,
        image_size: int,
        in_channels: int,
    ) -> None:
        """コンストラクタ

        Args:
            num_classes(int): クラス数
            image_size(int): 画像サイズ
            in_channels(int): 画像のチャンネル数
        """

        super(SegmentationModel, self).__init__()
        self.config = get_r50_b16_config()

        self.num_classes = num_classes
        self.transformer = Transformer(
            self.config, image_size, in_channels, vis=False)
        self.decoder = DecoderCup(self.config)
        self.segmentation_head = SegmentationHead(
            in_channels=cast(tuple[int, int, int, int],
                             self.config['decoder_channels'])[-1],
            out_channels=self.config['n_classes'],
            kernel_size=3,
        )

    def forward(self, x):
        """順伝搬
        """
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x, _, features = self.transformer(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        return logits
