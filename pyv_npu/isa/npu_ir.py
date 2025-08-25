from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple
from functools import reduce
import operator

# A map to get byte size from dtype string
DTYPE_MAP = {
    "float32": 4,
    "float16": 2,
    "int8": 1,
    "uint8": 1,
}

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
    address: int | None = None

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

    def clone_with_suffix(self, suffix: str) -> Tensor:
        """Creates a new Tensor instance with a suffix added to its name."""
        # Address is not copied as the new tensor will have a different location (e.g., in SPM)
        return Tensor(
            name=f"{self.name}{suffix}",
            shape=self.shape,
            dtype=self.dtype,
            address=None
        )

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