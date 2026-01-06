from typing import Any
from torch import Tensor, nn
import torch
import torch.distributed as dist
from torch.autograd import Function

class ColumnParallelLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        # Adjust our weight matrix to only allocate a shard of the full shape - do we just change out_featues here?
        # same for bias
        world_size = dist.get_world_size()
        self.world_size = world_size
        assert out_features % world_size == 0
        out_features_per_partition = out_features//world_size
        super().__init__(in_features, out_features_per_partition, bias, device, dtype)

    def forward(self, input: Tensor) -> Tensor:
        input = _split_input.apply(input) # type: ignore
        # replicate input to all devices? Or do we assume its replicated already?
        Y = input @ self.weight.T + self.bias
        # if no reduction op here, return sharded Y (for combination with Rowise)
        if False:
            # we would need a different backwards pass here (wihtout comm on backwards), but I really dont want to bother implementing that...
            # cant use all_gather_into_tensor, that gathers across axis 0, so cat manuially along axis -1
            gathered = [torch.empty_like(Y) for _ in range(self.world_size)]
            dist.all_gather(gathered, Y)
            Y_full = torch.cat(gathered, dim=-1)  # [batch, out_features]
            return Y_full

        return Y # sharded
    
class RowwiseParallelLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        # Adjust in features here as above
        world_size = dist.get_world_size()
        assert in_features % world_size == 0
        in_features_per_partition = in_features//world_size

        super().__init__(in_features_per_partition, out_features, bias, device, dtype)

    def forward(self, input: Tensor) -> Tensor:
        # assumes sharded input
        Y = input @ self.weight.T
        # run all reduce, just assume we have some default process group initialized
        Y: Tensor = _gather_input.apply(Y) # type: ignore
        # Bias after to not sum up the bias terms (replicated) across devices
        Y = Y + self.bias
        return Y

class _split_input(Function):
    # for column parallel: identity (copy) forward, need reduction backwards fromm sharded output to reverse information flow

    # autograd forward interface: ctx, args, return same count of args
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Any:
        dist.all_reduce(grad_output, dist.ReduceOp.SUM)
        return grad_output
    
class _gather_input(Function):
    # for rowwise parallel: all reduce forward, identity backward

    @staticmethod
    def forward(ctx, input):
        output = input.clone()
        dist.all_reduce(output, dist.ReduceOp.SUM)
        return output
    
    @staticmethod
    def backward(ctx: Any, grad_outputs: Any) -> Any:
        return grad_outputs
