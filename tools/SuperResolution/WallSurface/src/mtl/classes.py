from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class Material:
    name: str
    texture_name: Optional[str] = None


@dataclass
class MaterialLib:
    materials: Dict[str, Material] = field(default_factory=dict)
