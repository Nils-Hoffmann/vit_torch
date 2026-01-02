import math
import random
import os
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Sequence
import datasets
import numpy as np
from PIL import Image
import threading
import queue
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
        target_area = math.exp(
            log_scale_min + generator.random() * (log_scale_max - log_scale_min)
        ) * area
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
        to_torch: bool = False,
        infinite: bool = False,
    ) -> None:
        self.dataset = dataset
        self.transforms = list(transforms) if transforms is not None else []
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_examples = num_examples
        self.to_torch = to_torch
        self.infinite = infinite

    def with_transform(self, transform: Callable[[Any], Any]) -> "DataPipeline":
        """Return a new pipeline with an additional transform appended."""
        return DataPipeline(
            self.dataset,
            transforms=self.transforms + [transform],
            batch_size=self.batch_size,
            drop_last=self.drop_last,
            num_examples=self.num_examples,
            to_torch=self.to_torch,
            infinite=self.infinite,
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
            to_torch=self.to_torch,
            infinite=self.infinite,
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
        while True:
            if self.batch_size is None:
                for example in self.dataset:
                    yield self._apply_transforms(example)
            else:
                buffer: list[Any] = []
                for example in self.dataset:
                    buffer.append(self._apply_transforms(example))
                    if len(buffer) == self.batch_size:
                        yield self._collate(buffer)
                        buffer.clear()
                if buffer and not self.drop_last:
                    yield self._collate(buffer)

            if not self.infinite:
                break

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
    
    def prefetch(self, buffer_size: int = 2) -> "PrefetchIterator":
        """Return an iterator that prefetches batches in a background thread."""
        return PrefetchIterator(self, buffer_size=buffer_size)


class PrefetchIterator:
    """Iterator that prefetches batches in a background thread to overlap data loading with compute."""

    def __init__(self, iterable: Iterable[Any], buffer_size: int = 2):
        """
        Args:
            iterable: The iterator to prefetch from.
            buffer_size: Number of batches to prefetch (default 2: current + next).
        """
        self.iterable = iterable
        self.buffer_size = buffer_size
        self._queue: queue.Queue = queue.Queue(maxsize=buffer_size)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def _producer(self):
        """Background thread that loads batches into the queue."""
        try:
            for item in self.iterable:
                if self._stop_event.is_set():
                    break
                # Block if queue is full (wait for consumer to take items)
                self._queue.put(item)
        except Exception as e:
            # Put exception in queue so it can be raised in main thread
            self._queue.put(e)
        finally:
            # Signal end of iteration
            self._queue.put(None)

    def __iter__(self):
        # Start background thread
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._producer, daemon=True)
        self._thread.start()
        return self

    def __next__(self):
        # Get next item from queue
        item = self._queue.get()

        if item is None:
            # End of iteration
            self._cleanup()
            raise StopIteration

        if isinstance(item, Exception):
            # Exception from producer thread
            self._cleanup()
            raise item

        return item

    def _cleanup(self):
        """Stop the background thread and clean up."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def __del__(self):
        self._cleanup()


def _convert_to_np(example: dict[str, Any]) -> dict[str, Any]:
    image = np.asarray(example["img"], dtype=np.float32) / 255.0
    label = np.int64(example["label"])
    return {"img": image, "label": label}


def _dict_to_tuple(example: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Convert dict format to tuple (image, label) for unpacking."""
    return (example["img"], example["label"])


def _horizontal_flip_transform(
    prob: float, *, seed: int | None = None
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)

    def _transform(example: dict[str, Any]) -> dict[str, Any]:
        image = example["img"]
        if rng.random() < prob:
            image = np.flip(image, axis=1).copy()
        return {"img": image, "label": example["label"]}

    return _transform

def _random_crop(output_size: tuple[int, int]) -> Callable[[dict[str, Any]], dict[str, Any]]:

    def _transform(example: dict[str, Any]) -> dict[str, Any]:
        image = example["img"]
        image = _random_resized_crop_pil(image, output_size)
        return {"img": image, "label": example["label"]}

    return _transform

def _resize(output_size: tuple[int, int]) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def _transform(example: dict[str, Any]) -> dict[str, Any]:
        image = example["img"]
        image = image.resize(output_size, Image.Resampling.BICUBIC)
        return {"img": image, "label": example["label"]}

    return _transform


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
    to_torch: bool = True,
    infinite: bool = True,
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
    """
    cache_path = cache_dir or os.path.join(os.getcwd(), ".hf_cache")
    os.makedirs(cache_path, exist_ok=True)

    # CIFAR-10 dataset sizes
    num_examples = 50_000 if split == "train" else 10_000

    if streaming:
        dataset_dict = datasets.load_dataset("cifar10", streaming=True, cache_dir=cache_path)
        iterable = dataset_dict[split]
        if shuffle:
            shuffle_seed = 0 if seed is None else seed
            iterable = iterable.shuffle(buffer_size=10_000, seed=shuffle_seed)
    else:
        dataset = datasets.load_dataset("cifar10", split=split, cache_dir=cache_path)
        if shuffle:
            shuffle_seed = 0 if seed is None else seed
            dataset = dataset.shuffle(seed=shuffle_seed)
        iterable = dataset.to_iterable_dataset()

    pipeline = DataPipeline(
        iterable,
        num_examples=num_examples,
        to_torch=to_torch,
        infinite=infinite,
    )

    if augment:
        aug_seed = 0 if seed is None else seed
        pipeline = pipeline \
                        .with_transform(_random_crop(output_size=output_size)) \
                        .with_transform(_convert_to_np) \
                        .with_transform(_horizontal_flip_transform(0.5, seed=aug_seed))
    else:
        pipeline = pipeline.with_transform(_resize((224, 224))).with_transform(_convert_to_np)

    # convert dict samples to tuples so batches unpack as (img, label)
    pipeline = pipeline.with_transform(_dict_to_tuple)


    # Use drop_last=True to prevent recompilation from variable batch sizes
    return pipeline.batch(batch_size, drop_last=True)
