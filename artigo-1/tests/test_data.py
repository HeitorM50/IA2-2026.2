from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import RandomSampler, SequentialSampler
from torchvision import transforms

from src.config import DATA_CONFIG
from src.data import (
    build_transforms,
    compute_train_stats,
    get_dataloaders,
    inspect_dataset,
    set_seed,
)


def test_config_matches_closed_protocol() -> None:
    config = DATA_CONFIG
    assert config.dataset_flag == "bloodmnist"
    assert config.dataset_size == 64
    assert config.dataset_doi == "10.5281/zenodo.10519652"
    assert config.expected_split_sizes == {
        "train": 11_959,
        "val": 1_712,
        "test": 3_421,
    }
    assert sum(config.expected_split_sizes.values()) == 17_092
    assert config.num_classes == len(config.class_names_pt) == 8
    assert config.canonical_seeds == (42, 1337, 2026)


@pytest.mark.parametrize("invalid_seed", [-1, 2**32, 1.5, "42", True])
def test_set_seed_rejects_invalid_values(invalid_seed: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        set_seed(invalid_seed)  # type: ignore[arg-type]


def test_set_seed_controls_python_numpy_and_torch() -> None:
    set_seed(42)
    first = (random.random(), np.random.random(), torch.rand(1).item())
    set_seed(42)
    second = (random.random(), np.random.random(), torch.rand(1).item())
    assert first == second


def test_eval_transform_is_deterministic_and_has_no_augmentation() -> None:
    transform = build_transforms((0.5,) * 3, (0.25,) * 3, train=False)
    operation_types = tuple(type(operation) for operation in transform.transforms)
    assert operation_types == (transforms.ToTensor, transforms.Normalize)

    image = Image.fromarray(np.full((64, 64, 3), 127, dtype=np.uint8))
    assert torch.equal(transform(image), transform(image))


def test_train_transform_contains_only_geometric_augmentation() -> None:
    transform = build_transforms((0.5,) * 3, (0.25,) * 3, train=True)
    operation_types = tuple(type(operation) for operation in transform.transforms)
    assert operation_types == (
        transforms.ToTensor,
        transforms.RandomHorizontalFlip,
        transforms.RandomVerticalFlip,
        transforms.RandomRotation,
        transforms.Normalize,
    )


@pytest.fixture(scope="session")
def loaders_seed_42():
    return get_dataloaders(42)


@pytest.mark.integration
def test_real_loaders_obey_contract(loaders_seed_42) -> None:
    train_loader, val_loader, test_loader = loaders_seed_42
    expected_sizes = DATA_CONFIG.expected_split_sizes
    assert len(train_loader.dataset) == expected_sizes["train"]
    assert len(val_loader.dataset) == expected_sizes["val"]
    assert len(test_loader.dataset) == expected_sizes["test"]
    assert isinstance(train_loader.sampler, RandomSampler)
    assert isinstance(val_loader.sampler, SequentialSampler)
    assert isinstance(test_loader.sampler, SequentialSampler)

    for loader in loaders_seed_42:
        images, targets = next(iter(loader))
        assert images.dtype == torch.float32
        assert images.shape[1:] == (3, 64, 64)
        assert targets.dtype == torch.long
        assert targets.ndim == 1
        assert targets.min().item() >= 0
        assert targets.max().item() < DATA_CONFIG.num_classes


@pytest.mark.integration
def test_train_statistics_are_finite_and_reproducible() -> None:
    compute_train_stats.cache_clear()
    first_mean, first_std = compute_train_stats()
    compute_train_stats.cache_clear()
    second_mean, second_std = compute_train_stats()
    assert np.allclose(first_mean, second_mean, rtol=0, atol=1e-12)
    assert np.allclose(first_std, second_std, rtol=0, atol=1e-12)
    assert len(first_mean) == len(first_std) == DATA_CONFIG.channels
    assert np.isfinite(first_mean).all()
    assert np.isfinite(first_std).all()
    assert (np.asarray(first_std) > 0).all()


@pytest.mark.integration
def test_split_fingerprints_do_not_depend_on_seed() -> None:
    summary_42 = inspect_dataset(42)
    summary_1337 = inspect_dataset(1337)
    for split in ("train", "val", "test"):
        assert (
            summary_42["splits"][split]["fingerprint_sha256"]
            == summary_1337["splits"][split]["fingerprint_sha256"]
        )


@pytest.mark.integration
def test_same_seed_repeats_train_order_and_other_seed_changes_it() -> None:
    train_a, _, _ = get_dataloaders(42)
    train_b, _, _ = get_dataloaders(42)
    train_c, _, _ = get_dataloaders(1337)

    order_a = list(iter(train_a.sampler))[:256]
    order_b = list(iter(train_b.sampler))[:256]
    order_c = list(iter(train_c.sampler))[:256]
    assert order_a == order_b
    assert order_a != order_c


@pytest.mark.integration
def test_eval_tensors_repeat_exactly() -> None:
    _, val_a, test_a = get_dataloaders(42)
    _, val_b, test_b = get_dataloaders(1337)
    for first, second in ((val_a, val_b), (test_a, test_b)):
        first_images, first_targets = next(iter(first))
        second_images, second_targets = next(iter(second))
        assert torch.equal(first_images, second_images)
        assert torch.equal(first_targets, second_targets)
