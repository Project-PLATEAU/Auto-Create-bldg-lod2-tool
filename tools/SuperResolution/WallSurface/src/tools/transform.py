from typing import Optional
import cv2
import pathlib
import json
import math
import numpy as np
import os
from ..mtl.classes import Material
from ..mtl.parser import parse_mtl_from_file

import codecs


def read_mtl(mtl_path: pathlib.Path):
    mtl = parse_mtl_from_file(mtl_path)
    return mtl.materials


def seitaika_main(logger, obj_path: pathlib.Path, output_dir: pathlib.Path, z_threshold=0.02):
    if logger is not None:
        logger.info(f"input_obj_path = {obj_path}")

    seitaika_info, seitaika_figs, roof_info = process_obj_file(logger, obj_path, output_dir, z_threshold)

    return seitaika_info, seitaika_figs, roof_info


def rotateToXZ(vs):
    if len(vs) < 3:
        raise Exception("Invalid array size")

    normal = np.zeros(3)
    for i in range(len(vs)):
        a, b, c = vs[i], vs[(i+1) % len(vs)], vs[(i+2) % len(vs)]
        normal += np.cross(b-a, c-b)
    normal = normal / np.linalg.norm(normal)

    normal_save = normal.copy()

    normal_xy = np.array([normal[0], normal[1], 0])
    if np.linalg.norm(normal_xy) > 0:
        normal_xy = normal_xy / np.linalg.norm(normal_xy)
    if normal[0] == 0 and normal[1] == 0:
        normal_xy[1] = 1

    inner_product = np.clip(np.inner(normal_xy, np.array([0.0, 1.0, 0.0])), -1, 1)

    theta = np.arccos(inner_product)

    if normal[0] < 0:
        theta *= -1

    c = np.cos(theta)
    s = np.sin(theta)

    Rz = np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])

    normal = Rz @ normal

    inner_productX = np.clip(np.inner(normal, np.array([0.0, 1.0, 0.0])), -1, 1)
    thetaX = np.arccos(inner_productX)

    if normal[2] > 0.0:
        thetaX *= -1.0

    cX = np.cos(thetaX)
    sX = np.sin(thetaX)
    Rx = np.array([
        [1,  0,  0],
        [0, cX, -sX],
        [0, sX, cX]
    ])

    normal = Rx @ normal

    new_vs = np.empty(0)

    for i in range(len(vs)):
        new_vs = np.append(new_vs, Rx @ Rz @ vs[i])

    new_vs = new_vs.reshape(len(vs), 3)

    return new_vs, normal_save


def process_obj_file(logger, obj_path: pathlib.Path, output_dir: pathlib.Path, z_threshold: float):
    if logger is not None:
        logger.info(f"filepath: {obj_path}")

    geo_vs = []
    tex_vs = []

    texture_filepath: Optional[pathlib.Path] = None
    face_index = 1
    roof_index = 1

    seitaika_logs = []
    seitaika_log_paths = []
    seitaika_figs = []
    roof_logs = []
    roof_log_paths = []

    with codecs.open(str(obj_path), "r", "utf-8", "ignore") as f:
        mtl: Optional[dict[str, Material]] = None
        for s_line in f:
            src_points = None
            dst_points = None
            new_vs = None

            if s_line == '\n' or s_line == "":
                continue
            elems = s_line.split()

            if len(elems) == 0:
                continue

            command = elems.pop(0)

            if len(command) >= 1 and command[0] == "#":
                continue
            elif command == "f":
                if len(elems) < 4 or len(elems[0].split("/")) < 2:
                    continue

                vs = np.empty(0)
                us = np.empty(0)

                for i in range(len(elems)):
                    arg = elems[i].split("/")
                    vi, ui = arg[0], arg[1]
                    vs = np.append(vs, np.array(geo_vs[int(vi)-1]))
                    us = np.append(us, np.array(tex_vs[int(ui)-1]))
                vs = vs.reshape(len(elems), 3)
                us = us.reshape(len(elems), 2)
                height_max = np.abs(vs[:, 2].max() - vs[:, 2].min())
                texture_height = np.abs(us[:, 1].max() - us[:, 1].min())
                texture_width = np.abs(us[:, 0].max() - us[:, 0].min())

                min_x = vs[:, 0].min()
                min_y = vs[:, 1].min()
                min_z = vs[:, 2].min()
                vs -= np.array([min_x, min_y, min_z])
                new_vs, normal = rotateToXZ(vs)

                if np.abs(normal[2]) > z_threshold or np.isnan(normal[2]):
                    assert texture_filepath is not None
                    roof_log = {
                        "obj_filename": str(obj_path.resolve()),
                        "texture_file_path": str(texture_filepath.resolve()),
                        "normal": normal.tolist(),
                        "texture": us.tolist()
                    }

                    roof_logs.append(roof_log)
                    if logger is not None:
                        roof_name = f"roof_{roof_index}.json"
                        roof_path = output_dir.joinpath(roof_name)
                        with open(roof_path, "w") as f:
                            json.dump(roof_log, f, indent=4)
                            roof_log_paths.append(roof_path.name)

                    roof_index += 1
                    continue
                else:
                    face_index += 1

                min_x = new_vs[:, 0].min()
                min_y = new_vs[:, 1].min()
                min_z = new_vs[:, 2].min()
                new_vs -= np.array([min_x, min_y, min_z])

                output_face_name = obj_path.stem + f"_{face_index-1}.jpg"
                assert texture_filepath is not None
                image = cv2.imread(str(texture_filepath))

                h, w, _ = image.shape
                min_x = new_vs[:, 0].min()
                min_y = new_vs[:, 2].min()
                max_x = new_vs[:, 0].max()
                max_y = new_vs[:, 2].max()

                pixel_per_meter = math.ceil(texture_width * w / height_max)
                new_w = math.ceil((max_x - min_x) * pixel_per_meter)
                new_h = math.ceil((max_y - min_y) * pixel_per_meter)

                src_points = np.empty(0)
                reverse_x, reverse_y = False, False
                for i in range(len(new_vs)):
                    src_x = us[i][0] * w
                    src_y = (1 - us[i][1]) * h
                    src_points = np.append(src_points, [src_x, src_y])
                src_points = src_points.reshape(len(vs), 2)

                for i in range(len(us)):
                    ni = (i+1) % len(us)
                    if (src_points[i][0] - src_points[ni][0]) * (new_vs[i][0] - new_vs[ni][0]) < 0:
                        reverse_x = True
                    if (src_points[i][1] - src_points[ni][1]) * (new_vs[i][1] - new_vs[ni][1]) < 0:
                        reverse_y = True
                
                dst_points = np.empty(0)
                for i in range(len(new_vs)):
                    if reverse_x:
                        dx = ((max_x - min_x) - new_vs[i][0]) * pixel_per_meter
                    else:
                        dx = new_vs[i][0] * pixel_per_meter
                    if reverse_x:
                        dy = ((max_y - min_y) - new_vs[i][2]) * pixel_per_meter
                    else:
                        dy = new_vs[i][2] * pixel_per_meter
                    dst_points = np.append(dst_points, [dx, dy])
                dst_points = dst_points.reshape(len(vs), 2)

                homo, _ = cv2.findHomography(src_points, dst_points)
                dst_image = cv2.warpPerspective(image, homo, (new_w, new_h))
                mask = np.zeros_like(dst_image)
                cv2.fillPoly(mask, [dst_points.reshape((-1, 1, 2)).astype(np.int32)], (255, 255, 255))
                dst_image = cv2.bitwise_and(dst_image, mask)
                h_dst, w_dst, _ = dst_image.shape

                output_face_path = output_dir.joinpath(output_face_name)
                seitaika_figs.append({"img":dst_image, "path":output_face_path})
                if logger is not None:
                    cv2.imwrite(str(output_face_path), dst_image)

                d = {"obj_filename": str(obj_path.resolve()),
                     "texture_file_path": str(texture_filepath.resolve()),
                     "pixel_per_meter": pixel_per_meter,
                     "face_index": s_line,
                     "normal": normal.tolist(),
                     "homo": homo.tolist(),
                     "texture": us.tolist(),
                     "src": src_points.tolist(),
                     "dst": dst_points.tolist(),
                     "h": h,
                     "w": w,
                     "h_dst": h_dst,
                     "w_dst": w_dst
                    }

                seitaika_logs.append(d)
                json_filepath = output_face_path.with_suffix(".log")
                if logger is not None:
                    with open(json_filepath, "w") as f:
                        json.dump(d, f, indent=4)
                    json_filepath = json_filepath.name
                seitaika_log_paths.append(json_filepath)
                
            elif command == "g":
                if logger is not None:
                    logger.debug("group_name %s", elems[0])
            elif command == "mtllib":
                if logger is not None:
                    logger.debug("material_filename %s", elems[0])
                mtl = read_mtl(obj_path.parent.joinpath(elems[0]))
            elif command == "s":
                if logger is not None:
                    logger.debug("smooth_shading %s", elems[0])
            elif command == "usemtl":
                obj_texture = elems[0]
                assert mtl is not None
                texture_path_raw = mtl[obj_texture].texture_name
                if texture_path_raw is not None:
                    texture_filepath = obj_path.parent.joinpath(texture_path_raw.replace('\\', os.path.sep))
            elif command == "v":
                geo_vs.append([float(elems[0]), float(elems[1]), float(elems[2])])
            elif command == "vt":
                tex_vs.append([float(elems[0]), float(elems[1])])
            else:
                if logger is not None:
                    logger.warning("unknown command %s", command)

    seitaika_info = {'log': seitaika_logs, 'path': seitaika_log_paths}
    roof_info = {'log': roof_logs, 'path': roof_log_paths}
                    
    return seitaika_info, seitaika_figs, roof_info
