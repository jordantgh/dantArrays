import threading
import functools
from contextlib import contextmanager
import numpy as np

# Thread-local storage
_internal_access = threading.local()


class UnsafeDataAccessWarning(Warning):
    """Warning for direct access to the underlying array data"""

    pass


@contextmanager
def internal_data_access():
    """
    Context manager that suppresses _data_unsafe access warnings
    when used within DantArray's internal methods.
    """
    # Get current state, defaulting to False if not set
    previous = getattr(_internal_access, "active", False)
    _internal_access.active = True
    try:
        yield
    finally:
        _internal_access.active = previous


def internal_method(func):
    """Decorator for methods that need to access _data_unsafe without warnings"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with internal_data_access():
            return func(*args, **kwargs)

    return wrapper


def get_array_slice(array: np.ndarray, index: int, axis: int) -> np.ndarray:
    """
    Get a slice of an array along a specified axis

    Args:
        array: The complete array
        index: Index along the specified axis
        axis: Which axis to slice along

    Returns:
        Array slice at the specified index
    """
    if axis == 0:
        return array[index].copy()
    elif axis == 1:
        return array[:, index].copy()
    else:
        # Create a tuple of slices with the index at the right position
        slices = tuple(
            index if i == axis else slice(None) for i in range(array.ndim)
        )
        return array[slices].copy()
