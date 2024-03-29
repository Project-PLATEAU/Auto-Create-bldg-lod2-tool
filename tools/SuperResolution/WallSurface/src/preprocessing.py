import json
import pathlib

from .tools.cut import Cut
from .tools.transform import seitaika_main

class PreProcessing:
    def __init__(self, logger, overlap=0.0, size=256, z_threshold=0.02, lower_limit=32, upper_limit=1024):
        self.logger = logger
        self.overlap = overlap
        self.size = size
        self.z_threshold = z_threshold
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit


    def main_step(self, obj_path: pathlib.Path, output_dir: pathlib.Path):
        assert 0 <= self.overlap and self.overlap < 1
        assert self.size > 0
        assert self.z_threshold >= 0

        output_figs_all = []
        cut_logs = []
        cut_log_paths = []
        seitaika_info, seitaika_figs, roof_info = seitaika_main(self.logger, obj_path, output_dir, self.z_threshold)
        seitaika_logs = seitaika_info['log']
        seitaika_log_paths = seitaika_info['path']
        roof_logs = roof_info['log']
        roof_log_paths = roof_info['path']


        if self.logger is not None:
            self.logger.info(f"n_images = {len(seitaika_figs)}")

        for i, seitaika_fig in enumerate(seitaika_figs, 1):
            if self.logger is not None:
                self.logger.info(f"Enter Cut {i}")
                self.logger.info(f"im_path = {seitaika_fig['path'].name}")
                
            cut_class = Cut(self.logger, seitaika_fig, output_dir, overlap=self.overlap, size=self.size)
            if self.lower_limit <= cut_class.height <= self.upper_limit and self.lower_limit <= cut_class.width <= self.upper_limit:
                cut_class.calc_nH()
                cut_class.calc_nW()
                cut_class.cut()
                output_figs = cut_class.save(i)
                output_figs_all.append(output_figs)

                cut_log, cut_log_path = cut_class.output_log(f"{seitaika_fig['path'].stem}.json")
                cut_logs.append(cut_log)
                cut_log_paths.append(cut_log_path)

            else:
                output_figs_all.append([[]])
                cut_logs.append('')
        
        preprocess_log = {"output_images": output_figs_all,
                          "seitaika_figs": seitaika_figs,
                          "cut_logs": cut_logs,
                          "seitaika_logs": seitaika_logs,
                          "roof_logs": roof_logs}

        if self.logger is not None:
            save_log = {"output_images": [[[str(pathlib.Path(fig['path']).name) for fig in output_fig]
                                         if len(output_fig) > 0 else output_fig for output_fig in output_figs]
                                         for output_figs in output_figs_all],
                        "im_paths": [str(seitaika_fig['path'].name) for seitaika_fig in seitaika_figs],
                        "cut_logs": cut_log_paths,
                        "seitaika_logs": seitaika_log_paths,
                        "roof": roof_log_paths}
            preprocess_log_path = output_dir.joinpath('preprocess.json')
            with open(preprocess_log_path, "w") as f:
                json.dump(save_log, f, indent=4)

        return preprocess_log