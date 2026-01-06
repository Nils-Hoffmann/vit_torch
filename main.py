import datetime
import os
import time
import contextlib
import dataloaderv2
import data
from model import ModelConfig, ViT
import torch
import torch.profiler
from torch.optim import Adam
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.multiprocessing as mp

def main(rank: int = 1, world_size: int = 1):
    use_v2_dataloader = True
    enable_profiling = False
    run_eval = False
    use_data_iter_prefetch = False

    profile_wait = 1
    profile_warmup = 1
    profile_active = 5
    profile_repeat = 1
    enable_eval_profiling = False
    eval_profile_wait = 1
    eval_profile_warmup = 1
    eval_profile_active = 3
    eval_profile_repeat = 1

    train_batch_size = 256
    eval_batch_size = 256

    # Training configuration
    num_epochs = 10
    log_interval = 50  # Log stats every N steps

    copy_stream = torch.cuda.Stream()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    print(f"Using matmul precision: {torch.get_float32_matmul_precision()}")

    config = ModelConfig()
    device = f"cuda:{rank}"

    if world_size == 1:
        model = ViT(config).to(device)
        model_compiled = torch.compile(model)
        optimizer = Adam(model.parameters(), lr=3e-4)
    else:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        model = ViT(config).to(device)
        model = DistributedDataParallel(model, device_ids=[rank])
        model_compiled = torch.compile(model)
        optimizer = Adam(model.parameters(), lr=3e-4)


    def train_step(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        model.train()
        assert x.is_cuda and y.is_cuda
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits = model_compiled(x)
            loss = F.cross_entropy(logits.float(), y)

        loss.backward()
        optimizer.step()
    
        with torch.no_grad():
            acc = (logits.argmax(-1) == y).float().mean()

        return loss.detach(), acc


    @torch.no_grad()
    def eval_step(x, y):
        model.eval()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits = model(x.to(device, non_blocking=True))
        y = y.to(device, non_blocking=True)
        loss = F.cross_entropy(logits.float(), y)
        acc = (logits.argmax(-1) == y).float().mean()
        return loss, acc
    

    if use_v2_dataloader:
        pipeline = dataloaderv2.build_cifar10_pipeline(
            batch_size=train_batch_size,
            augment=True,
            use_bf16=True,
            num_workers=4,
            to_torch=True,
            rank=rank,
            world_size=world_size,
        )
        eval_pipeline = dataloaderv2.build_cifar10_pipeline(
            batch_size=eval_batch_size,
            split="test",
            shuffle=False,
            augment=False,
            use_bf16=True,
            num_workers=4,
            to_torch=True,
            infinite=False,
            rank=rank,
            world_size=world_size,
        )
    else:
        pipeline = data.build_cifar10_pipeline(
            batch_size=train_batch_size,
            augment=True,
            to_torch=True,
            rank=rank,
            world_size=world_size,
        )
        eval_pipeline = data.build_cifar10_pipeline(
            batch_size=eval_batch_size,
            split="test",
            shuffle=False,
            augment=False,
            to_torch=True,
            infinite=False,
            rank=rank,
            world_size=world_size,
        )

    steps_per_epoch = pipeline.steps_per_epoch


    assert steps_per_epoch is not None, "Could not determine steps_per_epoch from pipeline"
    total_steps = num_epochs * steps_per_epoch

    print(f"Training for {num_epochs} epochs, {steps_per_epoch} steps per epoch")
    print(f"Total steps: {total_steps}\n")

    # Stats aggregation
    running_loss = torch.zeros((), device=device, dtype=torch.float32)
    running_acc = torch.zeros((), device=device, dtype=torch.float32)
    running_steps = 0
    running_samples = 0

    def eval_model(epoch_idx: int):
        eval_losses = []
        eval_accs = []
        eval_samples = 0
        eval_start = time.time()
        eval_profiler_context: contextlib.AbstractContextManager
        if enable_eval_profiling:
            eval_profiler_context = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    wait=eval_profile_wait,
                    warmup=eval_profile_warmup,
                    active=eval_profile_active,
                    repeat=eval_profile_repeat,
                ),
                record_shapes=True,
                profile_memory=True,
                with_flops=True,
            )
        else:
            eval_profiler_context = contextlib.nullcontext()

        with eval_profiler_context as eval_prof:
            for x, y in eval_pipeline:
                loss, acc = eval_step(x, y)
                eval_losses.append(loss.to("cpu"))
                eval_accs.append(acc.to("cpu"))
                eval_samples += x.shape[0]
                if enable_eval_profiling:
                    eval_prof.step()
                    prof.export_chrome_trace("./torch_prof")

        eval_duration = time.time() - eval_start

        avg_eval_loss = sum(eval_losses) / len(eval_losses)
        avg_eval_acc = sum(eval_accs) / len(eval_accs)
        eval_throughput = eval_samples / eval_duration if eval_duration > 0 else float("inf")
        print(
            f"[Eval][Epoch {epoch_idx}] loss: {avg_eval_loss} | acc: {avg_eval_acc} | "
            f"time: {eval_duration:.2f}s | throughput: {eval_throughput:.2f} samples/s"
        )

    # CPU-side prefetch helps ensure the next batch is ready early enough that
    data_iter = iter(pipeline.prefetch(buffer_size=2) if use_data_iter_prefetch else pipeline)
    sample_counts: list[int] = []

    next_x, next_y = next(data_iter)
    device_next_x, device_next_y = torch.empty_like(next_x, device=device), torch.empty_like(next_y, device=device)
    device_x, device_y = torch.empty_like(next_x, device=device), torch.empty_like(next_y, device=device)

    if enable_profiling:
        profiler_context = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=profile_wait,
                warmup=profile_warmup,
                active=profile_active,
                repeat=profile_repeat,
            ),
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
        )


        with profiler_context as prof:
            if not next_y.is_pinned():
                next_y = next_y.pin_memory()
            if not next_x.is_pinned():
                next_x = next_x.pin_memory()

            with torch.cuda.stream(copy_stream):
                device_next_x.copy_(next_x, non_blocking=True)
                device_next_y.copy_(next_y, non_blocking=True)

            for step in range(20):

                # wait for copy stream (async memocpy, establish wait relationship)
                torch.cuda.current_stream().wait_stream(copy_stream)
                device_x, device_y = device_next_x, device_next_y
                # establish ownership relationship for GC
                device_x.record_stream(torch.cuda.current_stream())
                device_y.record_stream(torch.cuda.current_stream())

                loss, acc = train_step(device_x, device_y)
                
                next_x, next_y = next(data_iter)
                if not next_y.is_pinned():
                    next_y = next_y.pin_memory()
                if not next_x.is_pinned():
                    next_x = next_x.pin_memory()

                with torch.cuda.stream(copy_stream):
                    device_next_x.copy_(next_x, non_blocking=True)
                    device_next_y.copy_(next_y, non_blocking=True)

                sample_counts.append(device_x.shape[0])
                prof.step()

            filename = f"./profiles/vit-main-rank-{rank}-{datetime.datetime.now().isoformat()}"
            print(f"profiling part complete, writing to {filename}")
            prof.export_chrome_trace(filename)
        training_loop_start_step = 100
    else:
        training_loop_start_step = 0

    next_x, next_y = next(data_iter)
    if not next_y.is_pinned():
        next_y = next_y.pin_memory()
    if not next_x.is_pinned():
        next_x = next_x.pin_memory()

    with torch.cuda.stream(copy_stream):
        device_next_x.copy_(next_x, non_blocking=True)
        device_next_y.copy_(next_y, non_blocking=True)

    logging_start = time.time()
    for step in range(training_loop_start_step, total_steps):

        # wait for copy stream (async memocpy, establish wait relationship)
        torch.cuda.current_stream().wait_stream(copy_stream)
        device_x, device_y = device_next_x, device_next_y
        # establish ownership relationship for GC
        device_x.record_stream(torch.cuda.current_stream())
        device_y.record_stream(torch.cuda.current_stream())

        loss, acc = train_step(device_x, device_y)

        next_x, next_y = next(data_iter)
        if not next_y.is_pinned():
            next_y = next_y.pin_memory()
        if not next_x.is_pinned():
            next_x = next_x.pin_memory()

        with torch.cuda.stream(copy_stream):
            device_next_x.copy_(next_x, non_blocking=True)
            device_next_y.copy_(next_y, non_blocking=True)

        running_loss += loss.float()
        running_acc += acc.float()
        running_steps += 1
        running_samples += device_x.shape[0]
        if (step + 1) % log_interval == 0:
            time_taken = time.time() - logging_start
            # Sync once per log interval instead of every step.
            if world_size == 1:
                avg_loss = (running_loss / running_steps).item()
                avg_acc = (running_acc / running_steps).item()
                avg_time = time_taken / log_interval
                throughput = running_samples / time_taken
            else:
                running_samples_t = torch.tensor(running_samples, device=device)
                dist.all_reduce(running_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(running_acc, op=dist.ReduceOp.SUM)
                dist.all_reduce(running_samples_t, op=dist.ReduceOp.SUM)
                avg_loss = (running_loss / running_steps).item() / world_size
                avg_acc = (running_acc / running_steps).item() / world_size

                # use rank 0 time instead of reducing, this is a sync barrier anyway.
                avg_time = time_taken / log_interval
                throughput = running_samples_t.item() / time_taken

            current_epoch = (step + 1 + steps_per_epoch - 1) // steps_per_epoch
            step_in_epoch = (step % steps_per_epoch) + 1

            if rank == 0:
                print(
                    f"Step {step + 1} (Epoch {current_epoch}/{num_epochs}, "
                    f"Step {step_in_epoch}/{steps_per_epoch})"
                )
                print(f"avg loss: {avg_loss} | avg accuracy: {avg_acc}")
                print(f"avg time/step: {avg_time} | throughput: {throughput:.2f} samples/s")

            logging_start = time.time()
            running_loss.zero_()
            running_acc.zero_()
            running_steps = 0
            running_samples = 0

        if (step + 1) % steps_per_epoch and run_eval:
            eval_model((step + 1 + steps_per_epoch - 1) // steps_per_epoch)

    time_taken = time.time() - logging_start
    # Sync once per log interval instead of every step.
    if world_size == 1:
        avg_loss = (running_loss / running_steps).item()
        avg_acc = (running_acc / running_steps).item()
        avg_time = time_taken / log_interval
        throughput = running_samples / time_taken
    else:
        running_samples_t = torch.tensor(running_samples, device=device)
        dist.all_reduce(running_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(running_acc, op=dist.ReduceOp.SUM)
        dist.all_reduce(running_samples_t, op=dist.ReduceOp.SUM)
        avg_loss = (running_loss / running_steps).item() / world_size
        avg_acc = (running_acc / running_steps).item() / world_size
        avg_time = time_taken / log_interval
        throughput = running_samples_t.item() / time_taken

    current_epoch = (total_steps + 1 + steps_per_epoch - 1) // steps_per_epoch
    step_in_epoch = (total_steps % steps_per_epoch) + 1

    if rank == 0:
        print(
            f"Step {total_steps + 1} (Epoch {current_epoch}/{num_epochs}, "
            f"Step {step_in_epoch}/{steps_per_epoch})"
        )
        print(f"avg loss: {avg_loss} | avg accuracy: {avg_acc}")
        print(f"avg time/step: {avg_time} | throughput: {throughput:.2f} samples/s")

    # Cleanup
    if world_size > 1:
        dist.barrier()  # Ensure all ranks finish before cleanup
        dist.destroy_process_group()


if __name__ == "__main__":
    world_size = 2
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    mp.spawn(main,
        args=(world_size,),
        nprocs=world_size,
        join=True)
