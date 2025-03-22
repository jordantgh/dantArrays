import pytest
import numpy as np

from dantArrays.utils import get_array_slice

def test_get_array_slice_axis0():
    arr = np.array([[1, 2], [3, 4]])
    slice0 = get_array_slice(arr, 0, 0)
    assert np.all(slice0 == [1, 2])
    # Original array remains unchanged (we do a copy)
    slice0[0] = 999
    assert arr[0, 0] == 1

def test_get_array_slice_axis1():
    arr = np.array([[1, 2], [3, 4]])
    slice1 = get_array_slice(arr, 1, 1)
    assert np.all(slice1 == [2, 4])
    slice1[0] = 999
    assert arr[0, 1] == 2

def test_get_array_slice_higher_dim():
    arr = np.arange(24).reshape(2, 3, 4)  # shape=(2,3,4)
    # axis=1 => we want e.g. index=2 => shape would be (2,4)
    slice2 = get_array_slice(arr, 2, 1)
    assert slice2.shape == (2, 4)
    # Check some known cells => arr[:,2] => top-level slices
    assert np.all(slice2[0] == arr[0, 2])
    assert np.all(slice2[1] == arr[1, 2])

def test_get_array_slice_out_of_bounds():
    arr = np.array([1, 2, 3])
    with pytest.raises(IndexError):
        _ = get_array_slice(arr, 5, 0)
