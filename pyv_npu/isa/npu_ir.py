
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List

@dataclass
class NPUOp:
    opcode: str
    args: Dict[str, Any]

@dataclass
class Program:
    ops: List[NPUOp]

    def to_json(self) -> Dict[str, Any]:
        return {"ops": [dict(opcode=o.opcode, args=o.args) for o in self.ops]}
