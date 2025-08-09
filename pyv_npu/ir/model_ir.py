
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any

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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "inputs": self.inputs,
            "outputs": self.outputs,
            "nodes": [vars(n) for n in self.nodes],
        }
