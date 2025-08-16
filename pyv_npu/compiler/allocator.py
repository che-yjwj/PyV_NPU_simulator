from __future__ import annotations
from ..isa.npu_ir import Program, Tensor, DTYPE_MAP

class Allocator:
    """A simple memory allocator for assigning addresses to tensors."""

    def __init__(self, dram_base_address: int = 0x0, page_size: int = 4096):
        self.next_address = dram_base_address
        self.page_size = page_size

    def _align_to_page(self, address: int) -> int:
        """Aligns an address to the next page boundary."""
        return (address + self.page_size - 1) & ~(self.page_size - 1)

    def allocate(self, program: Program):
        """
        Assigns addresses to all unique tensors in the program.
        This is a simple bump allocator.
        """
        # This set tracks tensors that have already been processed to avoid re-allocating
        # shared tensors (like model weights or intermediate results).
        processed_tensor_names = set()

        # Create a list of all tensors, ensuring unique tensors are processed once.
        all_tensors: list[Tensor] = []
        tensor_map = {}

        # Collect unique tensors from the entire program
        tensor_sources = program.inputs + program.initializers + program.outputs
        for op in program.ops:
            tensor_sources.extend(op.inputs)
            tensor_sources.extend(op.outputs)

        for t in tensor_sources:
            if t.name not in tensor_map:
                tensor_map[t.name] = t
                all_tensors.append(t)

        for tensor in all_tensors:
            if tensor.address is None:
                tensor.address = self.next_address

                # Calculate size and update next_address
                byte_size = DTYPE_MAP.get(tensor.dtype, 1) # Default to 1 byte
                tensor_size = tensor.num_elements * byte_size
                
                # Align the next address to a page boundary for the next tensor
                self.next_address = self._align_to_page(tensor.address + tensor_size)
