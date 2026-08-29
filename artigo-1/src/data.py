"""Pipeline reprodutível do BloodMNIST sem vazamento entre splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import medmnist
import numpy as np
import torch
from medmnist import INFO
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import DATA_CONFIG, DataConfig

Split = Literal["train", "val", "test"]


class TargetAdapter(Dataset):
    """Converte os alvos `(1,)` do MedMNIST para escalares `torch.long`."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, target = self.dataset[index]
        target_array = np.asarray(target).reshape(-1)
        if target_array.size != 1:
            raise ValueError(
                f"Era esperado um rótulo por imagem; recebido formato {target_array.shape}."
            )
        return image, torch.tensor(int(target_array[0]), dtype=torch.long)


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("A seed deve ser um número inteiro.")
    if seed < 0 or seed >= 2**32:
        raise ValueError("A seed deve estar no intervalo [0, 2**32).")


def _seed_worker(_: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(seed: int) -> torch.Generator:
    """Configura as fontes de aleatoriedade e devolve um gerador do loader."""

    _validate_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _dataset_class(config: DataConfig) -> type[Dataset]:
    try:
        python_class = INFO[config.dataset_flag]["python_class"]
    except KeyError as error:
        raise ValueError(
            f"Dataset MedMNIST desconhecido: {config.dataset_flag!r}."
        ) from error
    return getattr(medmnist, python_class)


def _make_dataset(
    split: Split,
    transform: Callable[[Any], torch.Tensor] | None,
    config: DataConfig,
) -> Dataset:
    config.data_root.mkdir(parents=True, exist_ok=True)
    dataset_class = _dataset_class(config)
    dataset = dataset_class(
        split=split,
        root=str(config.data_root),
        size=config.dataset_size,
        transform=transform,
        download=True,
    )
    expected_size = config.expected_split_sizes[split]
    if len(dataset) != expected_size:
        raise RuntimeError(
            f"Split {split!r} contém {len(dataset)} imagens; "
            f"eram esperadas {expected_size}."
        )
    return dataset


@lru_cache(maxsize=None)
def compute_train_stats(
    config: DataConfig = DATA_CONFIG,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Calcula média e desvio RGB usando exclusivamente o split de treino."""

    dataset = TargetAdapter(
        _make_dataset("train", transforms.ToTensor(), config)
    )
    loader = DataLoader(
        dataset,
        batch_size=config.stats_batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    channel_sum = torch.zeros(config.channels, dtype=torch.float64)
    channel_square_sum = torch.zeros(config.channels, dtype=torch.float64)
    pixels_per_channel = 0

    for images, _ in loader:
        if images.ndim != 4 or images.shape[1:] != (
            config.channels,
            config.dataset_size,
            config.dataset_size,
        ):
            raise RuntimeError(f"Formato inesperado no treino: {tuple(images.shape)}.")
        images = images.to(torch.float64)
        channel_sum += images.sum(dim=(0, 2, 3))
        channel_square_sum += images.square().sum(dim=(0, 2, 3))
        pixels_per_channel += images.shape[0] * images.shape[2] * images.shape[3]

    if pixels_per_channel == 0:
        raise RuntimeError("O split de treino está vazio.")

    mean = channel_sum / pixels_per_channel
    variance = channel_square_sum / pixels_per_channel - mean.square()
    std = variance.clamp_min(0).sqrt()
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise RuntimeError("As estatísticas calculadas contêm valores não finitos.")
    if (std <= 0).any():
        raise RuntimeError("Todos os canais devem possuir desvio padrão positivo.")

    return tuple(mean.tolist()), tuple(std.tolist())


def build_transforms(
    mean: Sequence[float],
    std: Sequence[float],
    *,
    train: bool,
    config: DataConfig = DATA_CONFIG,
) -> transforms.Compose:
    """Monta transformações distintas para treino e avaliação."""

    if len(mean) != config.channels or len(std) != config.channels:
        raise ValueError(f"Média e desvio devem conter {config.channels} canais.")

    operations: list[Callable[[Any], Any]] = [transforms.ToTensor()]
    if train:
        operations.extend(
            [
                transforms.RandomHorizontalFlip(
                    p=config.horizontal_flip_probability
                ),
                transforms.RandomVerticalFlip(p=config.vertical_flip_probability),
                transforms.RandomRotation(degrees=config.rotation_degrees),
            ]
        )
    operations.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(operations)


def get_dataloaders(
    seed: int,
    config: DataConfig = DATA_CONFIG,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Cria loaders dos splits oficiais; a seed não altera sua composição."""

    generator = set_seed(seed)
    mean, std = compute_train_stats(config)
    train_transform = build_transforms(mean, std, train=True, config=config)
    eval_transform = build_transforms(mean, std, train=False, config=config)

    train_dataset = TargetAdapter(_make_dataset("train", train_transform, config))
    val_dataset = TargetAdapter(_make_dataset("val", eval_transform, config))
    test_dataset = TargetAdapter(_make_dataset("test", eval_transform, config))

    common_options = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
        "worker_init_fn": _seed_worker,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        drop_last=config.train_drop_last,
        **common_options,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        **common_options,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        drop_last=False,
        **common_options,
    )

    return train_loader, val_loader, test_loader


def _array_digest(array: np.ndarray, digest: Any) -> None:
    contiguous = np.ascontiguousarray(array)
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(memoryview(contiguous).cast("B"))


def _split_fingerprint(dataset: Dataset) -> str:
    images = getattr(dataset, "imgs", None)
    labels = getattr(dataset, "labels", None)
    if images is None or labels is None:
        raise RuntimeError("A versão instalada do MedMNIST não expõe imgs/labels.")
    digest = hashlib.sha256()
    _array_digest(np.asarray(images), digest)
    _array_digest(np.asarray(labels), digest)
    return digest.hexdigest()


def _class_counts(dataset: Dataset, config: DataConfig) -> list[int]:
    labels = getattr(dataset, "labels", None)
    if labels is None:
        raise RuntimeError("A versão instalada do MedMNIST não expõe labels.")
    counts = Counter(int(value) for value in np.asarray(labels).reshape(-1))
    unknown = set(counts) - set(range(config.num_classes))
    if unknown:
        raise RuntimeError(f"Rótulos desconhecidos encontrados: {sorted(unknown)}.")
    return [counts.get(index, 0) for index in range(config.num_classes)]


@lru_cache(maxsize=None)
def inspect_dataset(seed: int, config: DataConfig = DATA_CONFIG) -> dict[str, Any]:
    """Produz metadados auditáveis, sem resultados de modelos."""

    set_seed(seed)
    mean, std = compute_train_stats(config)
    info = INFO[config.dataset_flag]
    labels = info["label"]
    class_names_en = [labels[str(index)] for index in range(config.num_classes)]

    splits: dict[str, dict[str, Any]] = {}
    for split in ("train", "val", "test"):
        dataset = _make_dataset(split, transforms.ToTensor(), config)
        sample_image, sample_target = TargetAdapter(dataset)[0]
        splits[split] = {
            "size": len(dataset),
            "class_counts": _class_counts(dataset, config),
            "fingerprint_sha256": _split_fingerprint(dataset),
            "sample_image_shape": list(sample_image.shape),
            "sample_image_dtype": str(sample_image.dtype),
            "sample_target_dtype": str(sample_target.dtype),
        }

    return {
        "dataset": config.dataset_flag,
        "size": config.dataset_size,
        "doi": config.dataset_doi,
        "medmnist_version": medmnist.__version__,
        "seed_used_for_inspection": seed,
        "canonical_seeds": list(config.canonical_seeds),
        "class_names_en": class_names_en,
        "class_names_pt": list(config.class_names_pt),
        "normalization": {
            "source_split": "train",
            "mean_rgb": list(mean),
            "std_rgb": list(std),
        },
        "augmentation": {
            "train_only": True,
            "rotation_degrees": config.rotation_degrees,
            "horizontal_flip_probability": config.horizontal_flip_probability,
            "vertical_flip_probability": config.vertical_flip_probability,
            "color_jitter": False,
        },
        "splits": splits,
    }


def write_dataset_summary(
    summary: dict[str, Any],
    path: Path = DATA_CONFIG.summary_path,
) -> None:
    """Grava o relatório em JSON com ordenação estável."""

    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _print_summary(summary: dict[str, Any]) -> None:
    print(
        f"{summary['dataset']} {summary['size']}x{summary['size']} "
        f"(MedMNIST {summary['medmnist_version']})"
    )
    print("split  total  contagens por classe")
    for split, metadata in summary["splits"].items():
        counts = ", ".join(str(value) for value in metadata["class_counts"])
        print(f"{split:<5}  {metadata['size']:>5}  {counts}")
    normalization = summary["normalization"]
    print(f"média RGB (treino): {normalization['mean_rgb']}")
    print(f"desvio RGB (treino): {normalization['std_rgb']}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect", action="store_true", help="inspeciona os splits")
    parser.add_argument("--seed", type=int, default=DATA_CONFIG.canonical_seeds[0])
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_CONFIG.summary_path,
        help="caminho do resumo JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.inspect:
        raise SystemExit("Use --inspect para validar e documentar o dataset.")
    summary = inspect_dataset(args.seed)
    write_dataset_summary(summary, args.output)
    _print_summary(summary)
    print(f"resumo gravado em: {args.output}")


if __name__ == "__main__":
    main()
