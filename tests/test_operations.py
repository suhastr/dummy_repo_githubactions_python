"""
test_operations.py — Unit tests for arithmetic/operations.py.

Uses pytest fixtures and parameterized tests to ensure correctness,
readability, and maintainability in a production-ready manner.
"""

from __future__ import annotations

import numpy as np
import pytest
from src.arithmetic import operations


# === Fixtures ================================================================


@pytest.fixture
def sample_arrays() -> tuple[np.ndarray, np.ndarray]:
    """Return a pair of sample arrays for testing."""
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([5, 4, 3, 2, 1])
    return a, b


@pytest.fixture
def random_array() -> np.ndarray:
    """Return a random array for normalization and mean tests."""
    rng = np.random.default_rng(seed=42)
    return rng.integers(low=1, high=100, size=10)


# === Tests for basic arithmetic =============================================


def test_add_arrays(sample_arrays: tuple[np.ndarray, np.ndarray]) -> None:
    a, b = sample_arrays
    expected = np.array([6, 6, 6, 6, 6])
    np.testing.assert_array_equal(operations.add_arrays(a, b), expected)


def test_multiply_arrays(sample_arrays: tuple[np.ndarray, np.ndarray]) -> None:
    a, b = sample_arrays
    expected = np.array([5, 8, 9, 8, 5])
    np.testing.assert_array_equal(operations.multiply_arrays(a, b), expected)


@pytest.mark.parametrize(
    "a, b",
    [
        (np.array([1, 2]), np.array([1, 2, 3])),
        (np.array([[1, 2], [3, 4]]), np.array([1, 2])),
    ],
)
def test_add_arrays_shape_mismatch(a: np.ndarray, b: np.ndarray) -> None:
    with pytest.raises(ValueError):
        operations.add_arrays(a, b)


# === Tests for statistics and normalization ==================================


def test_mean_of_array(random_array: np.ndarray) -> None:
    result = operations.mean_of_array(random_array)
    expected = float(np.mean(random_array))
    assert np.isclose(result, expected)


def test_mean_of_empty_array() -> None:
    with pytest.raises(ValueError):
        operations.mean_of_array(np.array([]))


def test_normalize_array_range(random_array: np.ndarray) -> None:
    norm = operations.normalize_array(random_array)
    assert np.min(norm) == 0.0
    assert np.max(norm) == 1.0


def test_normalize_constant_array() -> None:
    arr = np.ones(5)
    result = operations.normalize_array(arr)
    np.testing.assert_array_equal(result, np.zeros_like(arr))


# === Tests for generator utilities ===========================================


def test_generate_batches() -> None:
    data = np.arange(10)
    batches = list(operations.generate_batches(data, batch_size=3))
    assert len(batches) == 4
    np.testing.assert_array_equal(batches[0], np.array([0, 1, 2]))
    np.testing.assert_array_equal(batches[-1], np.array([9]))


def test_rolling_average() -> None:
    data = [1, 2, 3, 4, 5]
    result = list(operations.rolling_average(data, window_size=3))
    expected = [1.0, 1.5, 2.0, 3.0, 4.0]
    np.testing.assert_allclose(result, expected)


def test_rolling_average_with_short_data() -> None:
    data = [10, 20]
    result = list(operations.rolling_average(data, window_size=5))
    expected = [10.0, 15.0]  # average of available data
    np.testing.assert_allclose(result, expected)


# === Smoke test for main() ===================================================


def test_main_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure main() runs without exceptions."""
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: None)
    operations.main()
