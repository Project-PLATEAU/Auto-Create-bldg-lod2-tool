import os
import cv2
import time
import json
import yaml
import shutil
import logging
import argparse
from tqdm import tqdm
from pathlib import Path

from src.preprocessing import PreProcessing
from src.postprocessing import PostProcessing
from cyclegan.dataset import DatasetDataLoader
from cyclegan.model.cyclegan_model import CycleGANModel
from cyclegan.util import util


def setup_logging(log_filename="debug.log", log_flag=False):

    # Create a logger
    logger = None

    # If DebugLogOutput is set to 'true', configure the logging
    if log_flag:
        logger = logging.getLogger(__name__)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(log_filename)
        file_handler.setFormatter(formatter)

        # Set the logging level
        logger.setLevel(logging.DEBUG)
        # Add the file handler to the logger
        logger.addHandler(file_handler)

    return logger


def write_log(log_root, action, filename=None):
    """
    Write a log entry to a log file.

    Parameters:
    - log_root: Path to the log file.
    - action: The action to log.
    - filename: Optional filename for additional information.
    """
    log_file = open(log_root, 'a')

    if filename is not None:
        log_file.write(f"\n{time.strftime('%Y-%m-%d %H:%M:%S')} : Execution of {filename}\n")

    log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} : {action}\n")
    log_file.close()


def check_path(path, cfg, logger):
    """
    Check if a path exists and raise an error if it doesn't.

    Parameters:
    - path: Path to check.
    - cfg: Configuration information.
    - logger: Logger for logging messages.
    """
    try:
        if not os.path.exists(path):
            raise ValueError(f"Error : {path} not found")
    except ValueError as ve:
        handle_error(ve, cfg, logger)


def handle_error(logger, error, log_flag):
    """
    Handle errors, log the message, and exit the program.

    Parameters:
    - error: The error that occurred.
    - param: Parameter Information.
    - logger: Logger for logging messages.
    """
    if log_flag:
        logger.error(f"{error}")
    print(f"Error: {error}")
    raise SystemExit(1)


def format_elapsed_time(process_time):
    """
    Format elapsed time into a human-readable string.

    Parameters:
    - process_time: Elapsed time in seconds.

    Returns:
    - Formatted elapsed time string.
    """
    hours, remainder = divmod(process_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    return f"{int(hours)} hours {int(minutes)} minutes {int(seconds)} seconds"


def check_error(param):
    """
    Check the validity of configuration information, log errors, 
    and exit the program if there are any.

    Parameters:
    - param: Parameter Information.
    """
    try:
        log_root = None
        if not param.get('OutputLogDir'):
            log_root = Path("main_log.txt")
            bug_root = Path("debug.log")
        else:
            log_dir = os.path.join(param['OutputLogDir'], f"outputlog_{time.strftime('%Y%m%d_%H%M%S')}")
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            log_root = Path(os.path.join(log_dir, "main_log.txt"))
            bug_root = Path(os.path.join(log_dir, "debug.log"))
        
        if not param.get('DebugLogOutput'):
            param['DebugLogOutput'] = 'false'
        elif param.get('DebugLogOutput') not in ['true', 'false']:
            param['DebugLogOutput'] = 'false'

        # Initialize the logger conditionally based on DebugLogOutput
        logger = setup_logging(bug_root, (param['DebugLogOutput'] == 'true'))

        if not param.get('InputDir') or not param.get('OutputDir'):
            raise ValueError("'InputDir' and 'OutputDir' must be specified in the JSON file.")
        
        if not param.get('InputDir'):
            param["Device"] = 'cuda'
        elif param.get('Device') not in ['cuda', 'cpu']:
            param["Device"] = 'cuda'

    except ValueError as ve:
        handle_error(logger, ve, (param['DebugLogOutput'] == 'true'))

    except Exception as e:
        handle_error(logger, e, (param['DebugLogOutput'] == 'true'))

    return log_root, logger


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("param_file", type=Path)
    parser.add_argument("--cfg_file", type=Path, default="config.yml")
    args = parser.parse_args()

    # Load parameter information from JSON file
    with args.param_file.open("rt") as pf:
        param = json.load(pf)

    cfg_path = Path("src", args.cfg_file)
    with cfg_path.open("rt") as cf:
        cfg = yaml.safe_load(cf)

    # Check required fields in the configuration
    log_root, logger = check_error(param)

    # Create execution log
    start_time = time.time()
    with open(log_root, 'w') as log_file:
        log_file.write(f"処理開始時刻 : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n")
        log_file.write(f"指定パラメータ内容 : {json.dumps(param)}\n")
        log_file.write(f"各処理の詳細情報 : {json.dumps(cfg)}\n")

    cfg_process = cfg['processing']
    processA_dir = Path(os.path.join(param['OutputDir'], "processA"))
    preprocessing = PreProcessing(logger, cfg_process['overlap'], cfg_process['size'], cfg_process['z_threshold'],
                                  cfg_process['lower_limit'], cfg_process['upper_limit'])
    
    cfg_cyclegan = cfg['cyclegan']
    processB_dir = Path(os.path.join(param['OutputDir'], "processB"))
    dataset = DatasetDataLoader(cfg_cyclegan)
    model = CycleGANModel(cfg_cyclegan, param['Device'])
    model.setup(cfg_cyclegan)
    AtoB = cfg_cyclegan['direction'] == 'AtoB'

    processC_dir = Path(os.path.join(param['OutputDir'], "processC"))
    postprocessing = PostProcessing(logger, processC_dir, cfg_process['overlap'], cfg_process['size'], cfg_process['z_threshold'])


    input_files = Path(param['InputDir']).iterdir()
    for input_file in input_files:
        # Check cityGML
        if input_file.suffix.lower() == ".gml":
            input_obj_dir = Path(param['InputDir']).joinpath(Path("obj", input_file.stem))
            output_obj_dir = Path(param['OutputDir']).joinpath(Path("obj", input_file.stem))

            # Copy cityGML and Object Directories
            if output_obj_dir.is_dir():
                shutil.rmtree(output_obj_dir)
            shutil.copytree(input_obj_dir, output_obj_dir)
            shutil.copy(input_file, Path(param['OutputDir']).joinpath(Path(input_file.name)))

            progress_bar = tqdm(input_obj_dir.iterdir(), desc=f"{input_file.stem}", position=0, 
                                leave=True, total=len(list(input_obj_dir.iterdir())))

            for obj_file in progress_bar:
                # Check object files
                if obj_file.suffix.lower() == ".obj":
                    # if 'bldg-0b39fb58-a0a5-48d0-aba4-14ba7ae53bed' not in str(obj_file):
                    #     continue
                    index = obj_file.name.replace('.', '_')
                    # Setting Sub Directories Paths
                    sub_processA_dir = processA_dir.joinpath(Path(input_file.stem, index))
                    sub_processB_dir = processB_dir.joinpath(Path(input_file.stem, index))
                    sub_processC_dir = processC_dir.joinpath(Path(input_file.stem, index))
                    # Creating Sub Directories Paths
                    if logger is not None:
                        sub_processA_dir.mkdir(exist_ok=True, parents=True)
                        sub_processB_dir.mkdir(exist_ok=True, parents=True)
                        sub_processC_dir.mkdir(exist_ok=True, parents=True)
                    
                    # Check if the file is present
                    check_path(obj_file, param, logger)
                    
                    # pre-processing
                    write_log(log_root, "変換対象壁面の抽出および正対化開始", obj_file)
                    preprocess_log = preprocessing.main_step(obj_file, sub_processA_dir)

                    # cyclegan processing
                    write_log(log_root, "壁面画像生成開始")
                    for num_iw, img_iw in enumerate(preprocess_log['output_images']):
                        for num_ih, img_ih in enumerate(img_iw):
                            for num, img in enumerate(img_ih):
                                img = dataset.read_img(img['img'], img['path'])

                                model.set_input(img)
                                model.test()
                                visuals = model.get_current_visuals()  # get image results
                                result = util.tensor2im(visuals['fake_B' if AtoB else 'fake_A'])
                                if logger is not None:
                                    img_path = model.get_image_paths()     # get image paths
                                    util.save_image(result, os.path.join(sub_processB_dir, os.path.basename(str(img_path))))

                                result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                                preprocess_log['output_images'][num_iw][num_ih][num]['img'] = result

                    # post-processing
                    write_log(log_root, "アトラス化画像再構成開始")
                    result = postprocessing.main_step(preprocess_log, sub_processA_dir, sub_processB_dir, sub_processC_dir)

                    # Saving output results
                    resolve_path_img = Path(preprocess_log['seitaika_logs'][0]['texture_file_path'])
                    relative_path_img = resolve_path_img.relative_to(Path(param['InputDir']).resolve())
                    output_path = Path(param['OutputDir']).joinpath(relative_path_img)
                    output_path.parent.mkdir(exist_ok=True, parents=True)
                    cv2.imwrite(str(output_path), result)

    end_time = time.time()
    process_time = end_time - start_time
    with open(log_root, 'a') as log_file:
        log_file.write(f"\n処理終了時刻 : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}\n")
        log_file.write(f"トータル処理時間 : {format_elapsed_time(process_time)}\n")

    # debug log
    if logger is not None:
        logger.info(f"Total processing time: {format_elapsed_time(process_time)}")
