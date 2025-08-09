
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple

# ONNX uses numpy-like dtypes
import numpy as np

@dataclass
class Tensor:
    name: str
    shape: Tuple[int, ...]
    dtype: np.dtype

@dataclass
class Node:
    name: str
    op_type: str
    inputs: List[str]
    outputs: List[str]
    attrs: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Graph:
    nodes: List[Node]
    inputs: List[str]
    outputs: List[str]
    tensors: Dict[str, Tensor] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "inputs": self.inputs,
            "outputs": self.outputs,
            "nodes": [vars(n) for n in self.nodes],
            "tensors": {k: dict(name=v.name, shape=v.shape, dtype=str(v.dtype)) for k, v in self.tensors.items()}
        }
