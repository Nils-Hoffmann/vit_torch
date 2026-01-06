import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from linear import ColumnParallelLinear, RowwiseParallelLinear


def test_column_parallel(rank: int, world_size: int):
    """Test ColumnParallelLinear forward and backward match reference."""
    device = f"cuda:{rank}"

    # Dimensions
    batch_size = 4
    in_features = 8
    out_features = 16  # must be divisible by world_size

    # Seed for reproducibility (same on all ranks)
    torch.manual_seed(42)

    # Reference full linear layer
    ref = nn.Linear(in_features, out_features).to(device)

    # Create parallel layer
    col = ColumnParallelLinear(in_features, out_features).to(device)

    # Copy sharded weights from reference
    shard_size = out_features // world_size
    with torch.no_grad():
        col.weight.copy_(ref.weight[rank * shard_size : (rank + 1) * shard_size])
        col.bias.copy_(ref.bias[rank * shard_size : (rank + 1) * shard_size])

    # Same input on all ranks (replicated)
    torch.manual_seed(123)
    x = torch.randn(batch_size, in_features, device=device, requires_grad=True)
    x_ref = x.clone().detach().requires_grad_(True)

    # Forward pass
    y_ref = ref(x_ref)
    y_col = col(x)  # sharded output

    # Gather sharded outputs to compare
    gathered = [torch.empty_like(y_col) for _ in range(world_size)]
    dist.all_gather(gathered, y_col)
    y_col_full = torch.cat(gathered, dim=-1)

    forward_match = torch.allclose(y_ref, y_col_full, atol=1e-6)

    # Backward pass
    # Use same loss: sum of outputs
    loss_ref = y_ref.sum()
    loss_col = y_col.sum()  # sum of local shard

    loss_ref.backward()
    loss_col.backward()

    # Check input gradients match
    grad_x_match = torch.allclose(x_ref.grad, x.grad, atol=1e-6)

    # Check weight gradients match (compare shard to corresponding slice of ref)
    grad_w_match = torch.allclose(
        ref.weight.grad[rank * shard_size : (rank + 1) * shard_size],
        col.weight.grad,
        atol=1e-6
    )
    grad_b_match = torch.allclose(
        ref.bias.grad[rank * shard_size : (rank + 1) * shard_size],
        col.bias.grad,
        atol=1e-6
    )

    if rank == 0:
        print(f"=== ColumnParallelLinear Test ===")
        print(f"  Forward match: {forward_match}")
        print(f"  Grad input match: {grad_x_match}")
        print(f"  Grad weight match: {grad_w_match}")
        print(f"  Grad bias match: {grad_b_match}")

    return forward_match and grad_x_match and grad_w_match and grad_b_match


def test_row_parallel(rank: int, world_size: int):
    """Test RowwiseParallelLinear forward and backward match reference."""
    device = f"cuda:{rank}"

    # Dimensions
    batch_size = 4
    in_features = 16  # must be divisible by world_size
    out_features = 8

    torch.manual_seed(42)

    # Reference full linear layer
    ref = nn.Linear(in_features, out_features).to(device)

    # Create parallel layer
    row = RowwiseParallelLinear(in_features, out_features).to(device)

    # Copy sharded weights from reference
    shard_size = in_features // world_size
    with torch.no_grad():
        row.weight.copy_(ref.weight[:, rank * shard_size : (rank + 1) * shard_size])
        row.bias.copy_(ref.bias)  # bias is full (not sharded)

    # Input: each rank gets its shard of the input
    torch.manual_seed(123)
    x_full = torch.randn(batch_size, in_features, device=device, requires_grad=True)
    x_shard = x_full[:, rank * shard_size : (rank + 1) * shard_size].clone().detach().requires_grad_(True)

    # Forward pass
    y_ref = ref(x_full)
    y_row = row(x_shard)  # should be replicated output after all_reduce

    forward_match = torch.allclose(y_ref, y_row, atol=1e-6)

    # Backward pass
    loss_ref = y_ref.sum()
    loss_row = y_row.sum()

    loss_ref.backward()
    loss_row.backward()

    # Check input gradients (shard should match corresponding slice of ref)
    grad_x_match = torch.allclose(
        x_full.grad[:, rank * shard_size : (rank + 1) * shard_size],
        x_shard.grad,
        atol=1e-6
    )

    # Check weight gradients
    grad_w_match = torch.allclose(
        ref.weight.grad[:, rank * shard_size : (rank + 1) * shard_size],
        row.weight.grad,
        atol=1e-6
    )

    # Bias gradient: row parallel has full bias, but gradient computation differs
    # In ref: grad_bias = sum over batch of grad_output
    # In row: same, since output is replicated after all_reduce
    grad_b_match = torch.allclose(ref.bias.grad, row.bias.grad, atol=1e-6)

    if rank == 0:
        print(f"=== RowwiseParallelLinear Test ===")
        print(f"  Forward match: {forward_match}")
        print(f"  Grad input match: {grad_x_match}")
        print(f"  Grad weight match: {grad_w_match}")
        print(f"  Grad bias match: {grad_b_match}")

    return forward_match and grad_x_match and grad_w_match and grad_b_match


def test_column_row_mlp(rank: int, world_size: int):
    """Test Column -> Row pairing (like MLP) matches reference."""
    device = f"cuda:{rank}"

    batch_size = 4
    in_features = 8
    hidden_features = 16  # divisible by world_size
    out_features = 8

    torch.manual_seed(42)

    # Reference MLP: two linear layers
    ref_fc1 = nn.Linear(in_features, hidden_features).to(device)
    ref_fc2 = nn.Linear(hidden_features, out_features).to(device)

    # Parallel MLP: Column -> Row
    col = ColumnParallelLinear(in_features, hidden_features).to(device)
    row = RowwiseParallelLinear(hidden_features, out_features).to(device)

    # Copy weights
    shard_size = hidden_features // world_size
    with torch.no_grad():
        # Column parallel: shard output features
        col.weight.copy_(ref_fc1.weight[rank * shard_size : (rank + 1) * shard_size])
        col.bias.copy_(ref_fc1.bias[rank * shard_size : (rank + 1) * shard_size])

        # Row parallel: shard input features
        row.weight.copy_(ref_fc2.weight[:, rank * shard_size : (rank + 1) * shard_size])
        row.bias.copy_(ref_fc2.bias)

    # Input (replicated)
    torch.manual_seed(123)
    x = torch.randn(batch_size, in_features, device=device, requires_grad=True)
    x_ref = x.clone().detach().requires_grad_(True)

    # Forward: reference
    h_ref = ref_fc1(x_ref)
    y_ref = ref_fc2(h_ref)

    # Forward: parallel (Column output is sharded, Row input expects sharded)
    h_col = col(x)  # sharded
    y_par = row(h_col)  # replicated after all_reduce

    forward_match = torch.allclose(y_ref, y_par, atol=1e-5)

    # Backward
    loss_ref = y_ref.sum()
    loss_par = y_par.sum()

    loss_ref.backward()
    loss_par.backward()

    # Check input gradient
    grad_x_match = torch.allclose(x_ref.grad, x.grad, atol=1e-5)

    # Check fc1 weight gradients (column parallel shards out_features)
    grad_fc1_w_match = torch.allclose(
        ref_fc1.weight.grad[rank * shard_size : (rank + 1) * shard_size],
        col.weight.grad,
        atol=1e-5
    )

    # Check fc2 weight gradients (row parallel shards in_features)
    grad_fc2_w_match = torch.allclose(
        ref_fc2.weight.grad[:, rank * shard_size : (rank + 1) * shard_size],
        row.weight.grad,
        atol=1e-5
    )

    if rank == 0:
        print(f"=== Column -> Row MLP Test ===")
        print(f"  Forward match: {forward_match}")
        print(f"  Grad input match: {grad_x_match}")
        print(f"  Grad fc1 weight match: {grad_fc1_w_match}")
        print(f"  Grad fc2 weight match: {grad_fc2_w_match}")

    return forward_match and grad_x_match and grad_fc1_w_match and grad_fc2_w_match


def main(rank: int, world_size: int):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    if rank == 0:
        print(f"Running tensor parallel tests with world_size={world_size}\n")

    dist.barrier()

    col_pass = test_column_parallel(rank, world_size)
    dist.barrier()

    row_pass = test_row_parallel(rank, world_size)
    dist.barrier()

    mlp_pass = test_column_row_mlp(rank, world_size)
    dist.barrier()

    if rank == 0:
        print(f"\n=== Summary ===")
        print(f"  ColumnParallel: {'PASS' if col_pass else 'FAIL'}")
        print(f"  RowParallel: {'PASS' if row_pass else 'FAIL'}")
        print(f"  Column->Row MLP: {'PASS' if mlp_pass else 'FAIL'}")

        all_pass = col_pass and row_pass and mlp_pass
        print(f"\nOverall: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = 2
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"  # different port from main.py
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
