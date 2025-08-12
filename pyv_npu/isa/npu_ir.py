from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple
from functools import reduce
import operator

# Forward declaration for type hint
class Tensor:
    pass

@dataclass
class NPUOp:
    """A single operation in the NPU instruction stream."""
    opcode: str
    name: str = ""
    inputs: List[Tensor] = field(default_factory=list)
    outputs: List[Tensor] = field(default_factory=list)
    args: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Tensor:
    """Represents a tensor in the NPU program."""
    name: str
    shape: Tuple[int, ...]
    dtype: str # e.g., "float16", "int8"

    @property
    def num_elements(self) -> int:
        """Calculates the total number of elements in the tensor."""
        if not self.shape or len(self.shape) == 0:
            return 0
        return reduce(operator.mul, self.shape)

    @classmethod
    def from_model_ir_tensor(cls, model_tensor) -> Tensor:
        """Creates an NpuTensor from a Model IR Tensor."""
        dtype_str = str(model_tensor.dtype)
        return cls(name=model_tensor.name, shape=model_tensor.shape, dtype=dtype_str)

@dataclass
class Program:
    """Represents a full NPU program, which is a sequence of NPUOps."""
    ops: List[NPUOp]
    inputs: List[Tensor] = field(default_factory=list)
    outputs: List[Tensor] = field(default_factory=list)
    initializers: List[Tensor] = field(default_factory=list)

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