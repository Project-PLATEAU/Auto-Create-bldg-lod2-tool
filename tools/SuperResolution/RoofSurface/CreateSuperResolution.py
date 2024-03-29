import os
import cv2
import time
import json
import shutil
import logging
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path


from tools.inference import Inferencer
from tools.image_processing import *


def setup_logging(log_filename='debug.log', log_flag=False):
    """
    Set up logging configurations.

    Parameters:
    - log_filename: Name of the log file.
    - log_flag: Flag to enable logging.
    """
    # Create a logger
    logger = logging.getLogger(__name__)

    # If DebugLogOutput is set to 'true', configure the logging
    if log_flag:
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


def handle_error(error, cfg, logger):
    """
    Handle errors, log the message, and exit the program.

    Parameters:
    - error: The error that occurred.
    - cfg: Configuration information.
    - logger: Logger for logging messages.
    """
    if cfg.get('DebugLogOutput') == 'true':
        logger.error(f"Error: {error}")
    print(f"Error: {error}")
    raise SystemExit(1)


def handle_warning(warning, cfg, logger):
    """
    Handle warning, log the message.

    Parameters:
    - warning: The warning that occurred.
    - cfg: Configuration information.
    - logger: Logger for logging messages.
    """
    if cfg.get('DebugLogOutput') == 'true':
        logger.warning(f"UserWarning: {warning}")
    print(f"UserWarning: {warning}")


def check_error(cfg):
    """
    Check the validity of configuration information, log errors, and exit the program if there are any.

    Parameters:
    - cfg: Configuration information.

    Returns:
    - log_root: Path to the log file.
    - logger: Logger for logging messages.
    """
    try:
        log_root, logger = None, None
        if not cfg.get('OutputLogDir'):
            log_root = Path("main_log.txt")
            bug_root = Path("debug.log")
        else:
            log_dir = os.path.join(cfg['OutputLogDir'], f"outputlog_{time.strftime('%Y%m%d_%H%M%S')}")
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            log_root = Path(os.path.join(log_dir, "main_log.txt"))
            bug_root = Path(os.path.join(log_dir, "debug.log"))

        if not cfg.get('DebugLogOutput'):
            cfg['DebugLogOutput'] = 'false'
        if cfg.get('DebugLogOutput') not in ['true', 'false']:
            cfg['DebugLogOutput'] = 'false'

        # Initialize the logger conditionally based on DebugLogOutput
        logger = setup_logging(bug_root, (cfg['DebugLogOutput'] == 'true'))

        if not cfg.get('InputDir') or not cfg.get('OutputDir'):
            raise ValueError("'InputDir' and 'OutputDir' must be specified in the JSON file.")

        Path(cfg['OutputDir']).mkdir(parents=True, exist_ok=True)

        if not cfg.get('GSD'):
            cfg['GSD'] = '0.25'
            warning_text = "'GSD' is not defined in the JSON file. Set the ground resolution to 0.25[m]."
            handle_warning(warning_text, cfg, logger)
        elif float(cfg['GSD']) < 0.1 or 0.25 < float(cfg['GSD']):
            raise ValueError(f"'GSD':{cfg['GSD']} is out of the target range. (0.1[m] ~ 0.25[m])")

        if not cfg.get('Device'):
            cfg["Device"] = 'cuda'
        elif cfg.get('Device') not in ['cuda', 'cpu']:
            cfg["Device"] = 'cuda'

    except ValueError as ve:
        handle_error(ve, cfg, logger)

    except Exception as e:
        handle_error(e, cfg, logger)

    return log_root, logger


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_file", type=Path)
    parser.add_argument("--checkpoint", type=Path, default="iter_280000_conv.pth")
    args = parser.parse_args()

    # Load configuration from JSON file
    with args.cfg_file.open("rt") as f:
        cfg = json.load(f)

    checkpoint_path = os.path.join("checkpoint", args.checkpoint)

    # Check required fields in the configuration
    log_root, logger = check_error(cfg)

    # Create execution log
    start_time = time.time()
    with open(log_root, 'w') as log_file:
        log_file.write(f"処理開始時刻 : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n")
        log_file.write(f"指定パラメータ内容 : {json.dumps(cfg)}\n")

    # Load a trained model
    inferencer = Inferencer(Path(cfg['InputDir']), Path(cfg['OutputDir']), log_root,
                            checkpoint_path, cfg['Device'], cfg['DebugLogOutput'])

    # Process all images in the specified folder
    for filename in tqdm(os.listdir(cfg['InputDir']), desc='Status of Processing', position=0):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
            input_path = os.path.join(cfg['InputDir'], filename)
            # Check if the file is present
            check_path(input_path, cfg, logger)

            # Image Segmentation and Storage
            if cfg.get('DebugLogOutput') == 'true':
                logger.info(f"Split processing (filename): {filename}")
            write_log(log_root, "分割処理開始", filename)
            ori_size, images = split_image(input_path, float(cfg['GSD']), size=120)
            split_list = save_images(images, Path(cfg['OutputDir']), filename)

            if cfg.get('DebugLogOutput') == 'true':
                logger.debug(f"Number of images after splitting: {len(images)}")

            # Process each segmented image
            write_log(log_root, "高解像度化・統合処理開始")
            merged_image = np.zeros((ori_size[0]*4, ori_size[1]*4, ori_size[2]), dtype=np.uint8)
            for num, split_path in tqdm(enumerate(split_list), desc=f'FileName({filename})', position=1, leave=False, total=len(split_list)):
                # Check if the file is present
                check_path(split_path, cfg, logger)
                if cfg.get('DebugLogOutput') == 'true':
                    logger.info(f"Super-resolution processing (filename): {os.path.basename(split_path)}")

                # Super-Resolution of segmented images
                result = inferencer.inference(split_path)
                result = np.transpose(result, (1, 2, 0))

                # Integration of Super-Resolution images
                if cfg.get('DebugLogOutput') == 'true':
                    logger.info(f"Merged image processing (filename): {os.path.basename(split_path)}")
                merged_image = merge_images(merged_image, result, num, size=480)

            # Check if the folder is present
            check_path(cfg['OutputDir'], cfg, logger)

            # Save merged images
            output_path = os.path.join(Path(cfg['OutputDir']), filename)
            cv2.imwrite(output_path, merged_image)
            shutil.rmtree(os.path.join(Path(cfg['OutputDir']), 'split_images'))


    end_time = time.time()
    process_time = end_time - start_time
    with open(log_root, 'a') as log_file:
        log_file.write(f"\n処理終了時刻 : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}\n")
        log_file.write(f"トータル処理時間 : {format_elapsed_time(process_time)}\n")

    # debug log
    if cfg.get('DebugLogOutput') == 'true':
        logger.info(f"Total processing time: {format_elapsed_time(process_time)}")
 