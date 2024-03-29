import os
import torch
from typing import Dict, Any, Optional
import lightning.pytorch as pl

from mmedit.apis import delete_cfg
from mmedit.registry import MODELS
from mmedit.utils import register_all_modules

from mmengine import Config
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from mmengine.utils import import_modules_from_strings


class SRModule(pl.LightningModule):
    def __init__(self, output_root, config, checkpoint, device) -> None:
        super(SRModule, self).__init__()
        self.save_hyperparameters()

        cfg_py = os.path.join(str(output_root), "config.py")
        with open(cfg_py, 'w', encoding='utf-8') as file:
            file.write(config)
 
        if isinstance(config, str):
            config = Config.fromfile(cfg_py)
            os.remove(cfg_py)

        delete_cfg(config.model, 'init_cfg')
        register_all_modules()
        self.model = MODELS.build(config.model)

        self.model.cfg = config
        self.model.to(device)
        self.model.eval()

        self.pred_results = []
    
    def forward(self, batch) -> torch.Tensor:
        data = self.model.data_preprocessor(batch, False)
        return self.model(**data, mode="predict")
