"""Tests for the RNG seeding helper."""
from __future__ import annotations

import random

import numpy as np

from src.utils.seeding import DEFAULT_SEED, seed_everything


def test_returns_seed():
    assert seed_everything(123) == 123
    assert seed_everything() == DEFAULT_SEED


def test_python_rng_reproducible():
    seed_everything(7)
    a = [random.random() for _ in range(5)]
    seed_everything(7)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_numpy_rng_reproducible():
    seed_everything(7)
    a = np.random.rand(5)
    seed_everything(7)
    b = np.random.rand(5)
    assert np.array_equal(a, b)


def test_different_seeds_differ():
    seed_everything(1)
    a = np.random.rand(5)
    seed_everything(2)
    b = np.random.rand(5)
    assert not np.array_equal(a, b)
