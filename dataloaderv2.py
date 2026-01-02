import math
import multiprocessing
import random
import os
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Sequence
import datasets
from datasets.distributed import split_dataset_by_node
import numpy as np
from PIL import Image
import threading
import torch


def get_cifar_10_iter() -> tuple[datasets.IterableDataset, datasets.IterableDataset]:
    ds = datasets.load_dataset("cifar10", streaming=True)
    return ds["train"], ds["test"]


@dataclass
class AugmentConfig:
    flip: float = 0.5


def _random_resized_crop_pil(
    image: Image.Image,
    size: tuple[int, int],
    *,
    scale: tuple[float, float] = (0.08, 1.0),
    ratio: tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
    num_attempts: int = 10,
    interpolation: int = Image.Resampling.BICUBIC,
    rng: random.Random | int | None = None,
) -> Image.Image:
    """Random resized crop using PIL with rejection sampling.

    Args:
        image: Input PIL image.
        size: Output (width, height).
        scale: Area range (fraction of original area).
        ratio: Aspect ratio range.
        num_attempts: Attempts before falling back to center crop.
        interpolation: PIL interpolation mode for resizing.
        rng: Optional random generator or seed for reproducibility.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("random_resized_crop_pil expects a PIL.Image.Image input.")

    if not (0 < scale[0] <= scale[1]):
        raise ValueError("scale must satisfy 0 < scale[0] <= scale[1].")
    if not (0 < ratio[0] <= ratio[1]):
        raise ValueError("ratio must satisfy 0 < ratio[0] <= ratio[1].")

    generator = rng
    if generator is None:
        generator = random
    elif isinstance(generator, int):
        generator = random.Random(generator)

    width, height = image.size
    area = width * height

    log_scale_min = math.log(scale[0])
    log_scale_max = math.log(scale[1])

    log_ratio_min = math.log(ratio[0])
    log_ratio_max = math.log(ratio[1])

    for _ in range(num_attempts):
        target_area = (
            math.exp(
                log_scale_min + generator.random() * (log_scale_max - log_scale_min)
            )
            * area
        )
        aspect_ratio = math.exp(
            log_ratio_min + generator.random() * (log_ratio_max - log_ratio_min)
        )

        new_w = int(round(math.sqrt(target_area * aspect_ratio)))
        new_h = int(round(math.sqrt(target_area / aspect_ratio)))

        if 0 < new_w <= width and 0 < new_h <= height:
            x0 = generator.randint(0, width - new_w)
            y0 = generator.randint(0, height - new_h)
            crop = image.crop((x0, y0, x0 + new_w, y0 + new_h))
            return crop.resize(size, interpolation)

    in_ratio = width / height
    if in_ratio < ratio[0]:
        new_w = width
        new_h = int(round(new_w / ratio[0]))
    elif in_ratio > ratio[1]:
        new_h = height
        new_w = int(round(new_h * ratio[1]))
    else:
        new_w, new_h = width, height

    x0 = max((width - new_w) // 2, 0)
    y0 = max((height - new_h) // 2, 0)
    crop = image.crop((x0, y0, x0 + new_w, y0 + new_h))
    return crop.resize(size, interpolation)


def _apply_transforms_worker(
    input_q: multiprocessing.Queue,
    output_q: multiprocessing.Queue,
    transforms: list[Callable[[Any], Any]],
):
    while True:
        sample = input_q.get()

        if sample is None:
            output_q.put(None)
            return

        result = sample
        try: 
            for transform in transforms:
                result = transform(result)

            output_q.put(result)

        except Exception as e:
            # just skip this sample
            print("Encountered error processing sample!")
            print(e)



class DataPipeline:
    """Lightweight dataset wrapper supporting chainable CPU-side transforms."""

    def __init__(
        self,
        dataset: datasets.IterableDataset,
        *,
        transforms: Sequence[Callable[[Any], Any]] | None = None,
        batch_size: int | None = None,
        drop_last: bool = False,
        num_examples: int | None = None,
        num_workers: int = 1,
        infinite: bool = False,
        to_torch: bool = False,
    ) -> None:
        self.dataset = dataset
        self.transforms = list(transforms) if transforms is not None else []
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_examples = num_examples
        self.num_workers = num_workers
        self.infinite = infinite
        self.to_torch = to_torch

    def with_transform(self, transform: Callable[[Any], Any]) -> "DataPipeline":
        """Return a new pipeline with an additional transform appended."""
        return DataPipeline(
            self.dataset,
            transforms=self.transforms + [transform],
            batch_size=self.batch_size,
            drop_last=self.drop_last,
            num_examples=self.num_examples,
            num_workers=self.num_workers,
            infinite=self.infinite,
            to_torch=self.to_torch,
        )

    def map(self, transform: Callable[[Any], Any]) -> "DataPipeline":
        """Alias for with_transform to mirror datasets API."""
        return self.with_transform(transform)

    def batch(self, batch_size: int, *, drop_last: bool = False) -> "DataPipeline":
        """Return a new pipeline that yields batches of the given size."""
        return DataPipeline(
            self.dataset,
            transforms=self.transforms,
            batch_size=batch_size,
            drop_last=drop_last,
            num_examples=self.num_examples,
            num_workers=self.num_workers,
            infinite=self.infinite,
            to_torch=self.to_torch,
        )

    @property
    def steps_per_epoch(self) -> int | None:
        """Estimate number of batches per epoch."""
        if self.num_examples is None or self.batch_size is None:
            return None
        if self.drop_last:
            return self.num_examples // self.batch_size
        return (self.num_examples + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Any]:
        
        if self.num_workers == 1:
            print("Using serial dataloading in process")

            # Setup dataset iterator (infinite or single-pass)
            if self.infinite:
                import itertools
                dataset_iter = itertools.cycle(self.dataset)
            else:
                dataset_iter = iter(self.dataset)

            if self.batch_size is None:
                for example in dataset_iter:
                    yield self._apply_transforms(example)
            else:
                buffer: list[Any] = []
                for example in dataset_iter:
                    buffer.append(self._apply_transforms(example))
                    if len(buffer) == self.batch_size:
                        yield self._collate(buffer)
                        buffer.clear()
                if buffer and not self.drop_last:
                    yield self._collate(buffer)

            return

        print(f"Using {self.num_workers} workers")
        ctx = multiprocessing.get_context("spawn")

        input_q = ctx.Queue(maxsize=4 * self.num_workers)
        output_q = ctx.Queue(maxsize=4 * self.num_workers)

        def enqueue_thread():
            try:
                if self.infinite:
                    # Infinite loop mode - keep cycling through dataset
                    import itertools
                    for e in itertools.cycle(self.dataset):
                        input_q.put(e, block=True)
                else:
                    # Single epoch mode
                    for e in self.dataset:
                        input_q.put(e, block=True)
            except Exception as e:
                print("Error during enqueue", e)
            finally:
                # Only send sentinels if not in infinite mode (shouldn't reach here in infinite mode)
                if not self.infinite:
                    for _ in range(self.num_workers):
                        input_q.put(None)

        producer_thread = threading.Thread(target=enqueue_thread, daemon=True)
        producer_thread.start()

        workers: list[multiprocessing.Process] = []
        for _ in range(self.num_workers):
            p = ctx.Process(
                target=_apply_transforms_worker,
                args=(input_q, output_q, self.transforms),
            )
            p.start()
            workers.append(p) #type: ignore


        workers_done = 0
        try:
            if self.batch_size is None:
                while workers_done < self.num_workers:
                    example = output_q.get()
                    if example is None:
                        workers_done += 1
                        continue
                    yield example
            else:
                buffer: list[Any] = []
                while workers_done < self.num_workers:
                    example = output_q.get()
                    if example is None:
                        workers_done += 1
                        continue

                    buffer.append(example)
                    if len(buffer) == self.batch_size:
                        yield self._collate(buffer)
                        buffer.clear()

                if buffer and not self.drop_last:
                    yield self._collate(buffer)
        finally:
            if workers_done != self.num_workers:
                print("Closing workers without all samples being processed!")
                
            for w in workers:
                w.terminate()
                w.join(timeout=1)
                if w.is_alive():
                    w.kill()
            producer_thread.join(timeout=1)

    def _apply_transforms(self, example: Any) -> Any:
        result = example
        for transform in self.transforms:
            result = transform(result)
        return result

    def _collate(self, batch: Sequence[Any]) -> Any:
        # Stack arrays and optionally convert to torch tensors
        first = batch[0]
        if isinstance(first, dict):
            stacked = {k: np.stack([item[k] for item in batch]) for k in first}
            if self.to_torch:
                return {k: torch.from_numpy(v) for k, v in stacked.items()}
            return stacked
        if isinstance(first, (tuple, list)):
            transposed = list(zip(*batch, strict=True))
            return type(first)(self._collate(list(items)) for items in transposed)  # type: ignore[arg-type]
        stacked = np.stack(batch)
        if self.to_torch:
            return torch.from_numpy(stacked)
        return stacked


def _convert_to_np(example: dict[str, Any]) -> dict[str, Any]:
    image = np.asarray(example["img"], dtype=np.float32) / 255.0
    label = np.int64(example["label"])  # PyTorch cross_entropy requires int64
    return {"img": image, "label": label}


def _dict_to_tuple(example: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Convert dict format to tuple (image, label) for unpacking."""
    return (example["img"], example["label"])


@dataclass
class HorizontalFlip:
    prob: float
    seed: int | None = None

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        image = example["img"]
        if self._rng.random() < self.prob:
            image = np.flip(image, axis=1).copy()
        return {"img": image, "label": example["label"]}

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("_rng", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._rng = random.Random(self.seed)


@dataclass
class RandomResizedCrop:
    output_size: tuple[int, int]

    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        image = _random_resized_crop_pil(example["img"], self.output_size)
        return {"img": image, "label": example["label"]}


@dataclass
class Resize:
    output_size: tuple[int, int]

    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        image = example["img"].resize(self.output_size, Image.Resampling.BICUBIC)
        return {"img": image, "label": example["label"]}


def build_cifar10_pipeline(
    batch_size: int,
    *,
    split: str = "train",
    shuffle: bool = True,
    seed: int | None = 0,
    augment: bool = False,
    cache_dir: str | None = None,
    streaming: bool = True,
    output_size=(224, 224),
    use_bf16: bool = False,
    num_workers=1,
    infinite: bool = True,
    to_torch: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> DataPipeline:
    """Construct a CIFAR-10 training pipeline yielding batched (image, label) pairs.

    Args:
        batch_size: Number of examples per batch.
        split: Dataset split to load, e.g. ``\"train\"`` or ``\"test\"``.
        shuffle: Whether to shuffle the dataset before iteration.
        seed: PRNG seed used for shuffling and augmentations.
        augment: If ``True`` applies basic CPU-side augmentations.
        cache_dir: Optional cache directory override when ``streaming`` is False.
        streaming: When True uses the Hugging Face streaming API.
        use_bf16: If ``True`` converts images to bfloat16 on CPU (reduces memory transfer).
        rank: Process rank for distributed training (0 to world_size-1).
        world_size: Total number of processes for distributed training.
    """
    cache_path = cache_dir or os.path.join(os.getcwd(), ".hf_cache")
    os.makedirs(cache_path, exist_ok=True)

    # CIFAR-10 dataset sizes (per-worker count for DDP)
    total_examples = 50_000 if split == "train" else 10_000
    num_examples = total_examples // world_size

    if streaming:
        dataset_dict = datasets.load_dataset(
            "cifar10", streaming=True, cache_dir=cache_path
        )
        iterable = dataset_dict[split]
        if shuffle:
            shuffle_seed = 0 if seed is None else seed
            iterable = iterable.shuffle(buffer_size=10_000, seed=shuffle_seed)
        # Shard dataset for distributed training
        if world_size > 1:
            iterable = split_dataset_by_node(iterable, rank=rank, world_size=world_size)
    else:
        dataset = datasets.load_dataset("cifar10", split=split, cache_dir=cache_path)
        if shuffle:
            shuffle_seed = 0 if seed is None else seed
            dataset = dataset.shuffle(seed=shuffle_seed)
        iterable = dataset.to_iterable_dataset()
        # Shard dataset for distributed training
        if world_size > 1:
            iterable = split_dataset_by_node(iterable, rank=rank, world_size=world_size)

    pipeline = DataPipeline(
        iterable,
        num_examples=num_examples,
        num_workers=num_workers,
        infinite=infinite,
        to_torch=to_torch,
    )

    if augment:
        aug_seed = 0 if seed is None else seed
        pipeline = (
            pipeline.with_transform(RandomResizedCrop(output_size=output_size))
            .with_transform(_convert_to_np)
            .with_transform(HorizontalFlip(prob=0.5, seed=aug_seed))
        )
    else:
        pipeline = pipeline.with_transform(
            Resize(output_size=(224, 224))
        ).with_transform(_convert_to_np)

    # Note: use_bf16 parameter is kept for backward compatibility but ignored.
    # PyTorch's autocast handles dtype conversion on GPU more efficiently.

    # Convert dict to tuple for unpacking as (x, y)
    pipeline = pipeline.with_transform(_dict_to_tuple)

    # Use drop_last=True to prevent recompilation from variable batch sizes
    return pipeline.batch(batch_size, drop_last=True)
