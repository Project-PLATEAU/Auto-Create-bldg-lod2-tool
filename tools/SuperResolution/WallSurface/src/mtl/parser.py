from pathlib import Path
from typing import List, Optional

from .classes import Material, MaterialLib

# # parse関数群
#
# 返り値
#   None以外の値: パース成功
#   None: パースには失敗したが、他の関数で成功する可能性あり
#   例外: パースに失敗


def parse_newmtl(tokens: List[str]) -> Optional[str]:
    if tokens[0].lower() != "newmtl":
        return None

    if len(tokens) == 2:
        return tokens[1]
    else:
        raise ValueError()


def parse_map_kd(tokens: List[str]) -> Optional[str]:
    if tokens[0].lower() != "map_kd":
        return None

    if len(tokens) == 2:
        return tokens[1]
    else:
        raise ValueError()

"""
def parse_mtl_from_file(path: Path) -> MaterialLib:
    mtl = MaterialLib()
    current_material: Optional[Material] = None

    with open(path, "r") as f:
        for line in f:
            tokens = line.split()

            if len(tokens) == 0:
                continue

            if material_name := parse_newmtl(tokens):
                if current_material is not None:
                    mtl.materials[current_material.name] = current_material
                current_material = Material(material_name)
            elif map_kd := parse_map_kd(tokens):
                if current_material is None:
                    raise ValueError()
                current_material.texture_name = map_kd

    if current_material is not None:
        mtl.materials[current_material.name] = current_material

    return mtl
"""
def parse_mtl_from_file(path: Path) -> MaterialLib:
    mtl = MaterialLib()
    current_material: Optional[Material] = None
 
    with open(path, "r") as f:
        for line in f:
            tokens = line.split()
 
            if len(tokens) == 0:
                continue
 
            material_name = parse_newmtl(tokens)
            map_kd = parse_map_kd(tokens)
            
            if material_name:
                if current_material is not None:
                    mtl.materials[current_material.name] = current_material
                current_material = Material(material_name)

            elif map_kd:
                if current_material is None:
                    raise ValueError()
                current_material.texture_name = map_kd
 
    if current_material is not None:
        mtl.materials[current_material.name] = current_material
 
    return mtl

