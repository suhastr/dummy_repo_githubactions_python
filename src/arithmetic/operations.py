"""
operations.py — Core arithmetic utilities using NumPy.

This module demonstrates production-level Python practices:
- Type hints for clarity and tooling support.
- NumPy vectorized operations for performance.
- Generator-based streaming computations.
- Docstrings for maintainability.
"""

from __future__ import annotations
from typing import Generator, Iterable
import numpy as np


def add_arrays(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Elementwise addition of two NumPy arrays.

    Args:
        a: First input array.
        b: Second input array.

    Returns:
        A NumPy array containing the elementwise sum of `a` and `b`.
    """
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape=} and {b.shape=}")
    return np.add(a, b)


def multiply_arrays(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Elementwise multiplication of two arrays.

    Args:
        a: First array.
        b: Second array.

    Returns:
        A NumPy array containing the elementwise product.
    """
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape=} and {b.shape=}")
    return np.multiply(a, b)


def mean_of_array(a: np.ndarray) -> float:
    """
    Compute the mean of a NumPy array.

    Args:
        a: Input NumPy array.

    Returns:
        The mean of all elements as a float.
    """
    if a.size == 0:
        raise ValueError("Cannot compute mean of an empty array")
    return float(np.mean(a))


def normalize_array(a: np.ndarray) -> np.ndarray:
    """
    Normalize an array to have values between 0 and 1.

    Args:
        a: Input array.

    Returns:
        Normalized array.
    """
    min_val = np.min(a)
    max_val = np.max(a)
    if np.isclose(max_val, min_val):
        return np.zeros_like(a)
    return (a - min_val) / (max_val - min_val)


def generate_batches(
    data: np.ndarray, batch_size: int
) -> Generator[np.ndarray, None, None]:
    """
    Yield data in batches using a generator pattern.

    Args:
        data: NumPy array to split into batches.
        batch_size: Number of elements per batch.

    Yields:
        NumPy array batches of size `batch_size`.
    """
    num_samples = data.shape[0]
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        yield data[start:end]


def rolling_average(
    data: Iterable[float], window_size: int
) -> Generator[float, None, None]:
    """
    Compute a rolling average (moving average) using a generator.

    Args:
        data: Iterable of numeric values.
        window_size: The number of elements in each average window.

    Yields:
        The average of the last `window_size` elements as a float.
    """
    window: list[float] = []
    for value in data:
        window.append(value)
        if len(window) > window_size:
            window.pop(0)
        yield float(np.mean(window))


def main() -> None:
    """Example usage when running this file directly."""
    arr1 = np.array([1, 2, 3, 4, 5])
    arr2 = np.array([5, 4, 3, 2, 1])

    print("Sum:", add_arrays(arr1, arr2))
    print("Product:", multiply_arrays(arr1, arr2))
    print("Mean:", mean_of_array(arr1))
    print("Normalized:", normalize_array(arr1))

    print("\nGenerated batches:")
    for batch in generate_batches(arr1, batch_size=2):
        print(batch)

    print("\nRolling average:")
    for avg in rolling_average(arr1, window_size=3):
        print(round(avg, 2))


if __name__ == "__main__":
    main()
