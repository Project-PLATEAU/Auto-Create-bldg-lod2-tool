import torch
from torchvision.io import write_png

from mmcv.transforms import LoadImageFromFile
from mmedit.datasets.transforms import PackEditInputs

from .model import SRModule
 

class Inferencer:
    def __init__(self, data_root, output_root, log_root, checkpoint, device="cuda", debug="false"):

        ckpt = torch.load(checkpoint, map_location='cpu')

        self.model = SRModule(output_root, ckpt['meta']['cfg'], checkpoint, device)
        self.model.load_state_dict(ckpt["state_dict"])

        self.load_image = LoadImageFromFile()
        self.pack = PackEditInputs(keys=['img'])
        self.device = device
        

    def inference(self, img_path):
        input = self.load_image({'img_path': str(img_path)})
        input = self.pack(input)
        input = {
            "inputs": [input['inputs']['img'].to(self.device)],
            "data_samples": [input['data_samples']],
        }

        with torch.no_grad():
            result = self.model(input)[0]
        result = result.output.pred_img.data.to(torch.uint8)

        return result.to('cpu').detach().numpy().copy()

        # write_png(result.to(torch.uint8), "aaa.png")