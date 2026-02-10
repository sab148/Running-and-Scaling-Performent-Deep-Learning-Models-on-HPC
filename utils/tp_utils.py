import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor, distribute_tensor, Replicate, Shard

# Tensor Parallelism Functions
def shard_linear(layer, device_mesh, dim=0):
    """Shard a linear layer along specified dimension"""
    weight_dtensor = distribute_tensor(_to_mesh_device(layer.weight, device_mesh), device_mesh, [Shard(dim)])
    layer.weight = nn.Parameter(weight_dtensor)
    
    if layer.bias is not None:
        if dim == 0:
            bias_dtensor = distribute_tensor(_to_mesh_device(layer.bias, device_mesh), device_mesh, [Shard(0)])
        else:
            bias_dtensor = distribute_tensor(_to_mesh_device(layer.bias, device_mesh), device_mesh, [Replicate()])
        layer.bias = nn.Parameter(bias_dtensor)


def _to_mesh_device(tensor, device_mesh):
    if device_mesh.device_type == "cuda":
        local_device = torch.device("cuda", torch.cuda.current_device())
    else:
        local_device = torch.device(device_mesh.device_type)
    if tensor.device == local_device:
        return tensor
    return tensor.to(local_device)


def _replicate_tensor(tensor, device_mesh):
    if isinstance(tensor, DTensor):
        return tensor
    return distribute_tensor(_to_mesh_device(tensor, device_mesh), device_mesh, [Replicate()])


def _replicate_param(param, device_mesh):
    if param is None:
        return None
    dtensor = _replicate_tensor(param, device_mesh)
    return nn.Parameter(dtensor, requires_grad=param.requires_grad)


def apply_tensor_parallelism(model, device_mesh, rank):
    """Apply tensor parallelism to TransformerLM"""
    
    # Shard embedding
    embed_weight = distribute_tensor(_to_mesh_device(model.embed.weight, device_mesh), device_mesh, [Shard(1)])
    model.embed.weight = nn.Parameter(embed_weight)
    model.pos_encoder.pe = _replicate_tensor(model.pos_encoder.pe, device_mesh)
    
    # Shard each transformer layer
    for i, layer in enumerate(model.layers):
        
        # Attention: column-parallel for QKV, row-parallel for output
        shard_linear(layer.self_attn.linear_Q, device_mesh, dim=0)
        shard_linear(layer.self_attn.linear_K, device_mesh, dim=0)
        shard_linear(layer.self_attn.linear_V, device_mesh, dim=0)
        shard_linear(layer.self_attn.out_proj, device_mesh, dim=1)
        
        # FFN: column-parallel for first, row-parallel for second
        shard_linear(layer.linear1, device_mesh, dim=0)
        shard_linear(layer.linear2, device_mesh, dim=1)

        layer.norm1.weight = _replicate_param(layer.norm1.weight, device_mesh)
        layer.norm1.bias = _replicate_param(layer.norm1.bias, device_mesh)
        layer.norm2.weight = _replicate_param(layer.norm2.weight, device_mesh)
        layer.norm2.bias = _replicate_param(layer.norm2.bias, device_mesh)
    
    # Shard output layer
    shard_linear(model.fc, device_mesh, dim=1)

    model.norm.weight = _replicate_param(model.norm.weight, device_mesh)
    model.norm.bias = _replicate_param(model.norm.bias, device_mesh)
    
    return model