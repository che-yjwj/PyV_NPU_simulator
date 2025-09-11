from __future__ import annotations
from ...ir.model_ir import Graph, Node
import logging
from typing import List, Dict

def apply_fusion_pass(g: Graph) -> Graph:
    """Applies a simple fusion pass for MatMul -> Add -> GELU patterns."""
    # Build a map of tensor consumers to quickly find connections
    tensor_consumers: Dict[str, List[str]] = {t: [] for t in g.tensors}
    for node in g.nodes:
        for t_in in node.inputs:
            if t_in in tensor_consumers:
                tensor_consumers[t_in].append(node.name)

    # Build a map of node names to Node objects for quick lookups
    node_map: Dict[str, Node] = {n.name: n for n in g.nodes}

    nodes_to_remove = set()
    new_nodes = []

    for node in g.nodes:
        if node.name in nodes_to_remove:
            continue

        # Pattern matching: MatMul -> Add -> GELU
        if node.op_type != 'MatMul' or len(node.outputs) != 1:
            continue

        # 1. Check MatMul's consumer
        matmul_out_tensor = node.outputs[0]
        consumers = tensor_consumers.get(matmul_out_tensor, [])
        if len(consumers) != 1:
            continue
        
        add_node = node_map.get(consumers[0])
        if not add_node or add_node.op_type != 'Add' or len(add_node.outputs) != 1 or len(add_node.inputs) != 2:
            continue

        # 2. Check Add's consumer
        add_out_tensor = add_node.outputs[0]
        consumers = tensor_consumers.get(add_out_tensor, [])
        if len(consumers) != 1:
            continue

        gelu_node = node_map.get(consumers[0])
        if not gelu_node or gelu_node.op_type != 'Erf':
            continue

        # --- Fusion pattern found! --- 
        logging.info(f"Fusing nodes: {node.name}, {add_node.name}, {gelu_node.name}")

        # Identify the bias tensor for the Add operation
        bias_tensor_name = next((t for t in add_node.inputs if t != matmul_out_tensor), None)
        if not bias_tensor_name:
            continue

        # Create the new fused node
        fused_node = Node(
            name=f"fused_{node.name}_{add_node.name}_{gelu_node.name}",
            op_type='MatMulAddGelu',
            inputs=node.inputs + [bias_tensor_name],
            outputs=gelu_node.outputs,
            attrs={}
        )
        new_nodes.append(fused_node)

        # Mark old nodes for removal
        nodes_to_remove.add(node.name)
        nodes_to_remove.add(add_node.name)
        nodes_to_remove.add(gelu_node.name)

    if not nodes_to_remove:
        return g # No changes

    # Rebuild the node list
    final_nodes = [n for n in g.nodes if n.name not in nodes_to_remove] + new_nodes
    g.nodes = final_nodes

    # It's safer to re-run the pass until no more fusions can be found,
    # but for this simple case, one pass is often enough. For now, we just return.
    # For a more robust implementation, you might wrap this in a while loop.
    return g