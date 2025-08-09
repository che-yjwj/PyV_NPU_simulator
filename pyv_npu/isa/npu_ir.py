
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple
import numpy as np

@dataclass
class Tensor:
    name: str
    shape: Tuple[int, ...]
    dtype: np.dtype
    # producer: NPUOp = None # The op that generates this tensor

    def num_elements(self) -> int:
        return np.prod(self.shape).item()

@dataclass
class NPUOp:
    opcode: str
    args: Dict[str, Any]
    inputs: List[Tensor] = field(default_factory=list)
    outputs: List[Tensor] = field(default_factory=list)
    name: str = ""

@dataclass
class Program:
    ops: List[NPUOp]

    def to_json(self) -> Dict[str, Any]:
        return {"ops": [
            {
                "opcode": o.opcode, 
                "name": o.name,
                "args": o.args,
                "inputs": [t.name for t in o.inputs],
                "outputs": [t.name for t in o.outputs],
            } 
            for o in self.ops
        ]}
