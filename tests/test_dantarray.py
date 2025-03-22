import pytest
import numpy as np
from pydantic import BaseModel, ValidationError

from dantArrays.core import DantArray
from dantArrays.utils import UnsafeDataAccessWarning
from dantArrays.computed import (
    computed,
    ComputedFieldContext,
    mean_value_field,
    max_value_field,
)

# ---------------------------------------------------------------------------
# Helpful test metadata models
# ---------------------------------------------------------------------------


def sum_of_slice(ctx: ComputedFieldContext) -> float:
    """Simple computed function that sums the slice."""
    return float(ctx.get_slice().sum())


def double_mean(ctx: ComputedFieldContext) -> float:
    """A test function that doubles the mean of a slice."""
    return float(np.mean(ctx.get_slice()) * 2.0)


class SimpleMeta(BaseModel):
    """A basic metadata model with one writable and one non-writable computed field."""

    name: str = "untitled"
    sum_val: float = computed(sum_of_slice, writable=False, default=0.0)
    double_mean_val: float = computed(double_mean, writable=True, default=0.0)


class MinimalMeta(BaseModel):
    """A minimal test metadata model."""

    pass


class MeanMeta(BaseModel):
    """Metadata model with a built-in mean_value_field from dantArrays.computed."""

    label: str = "data"
    avg: float = mean_value_field()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_array_2x3():
    """Returns a small 2x3 numpy array for testing."""
    return np.array([[1, 2, 3], [4, 5, 6]])


@pytest.fixture
def dantarr_2x3_simplemeta(small_array_2x3):
    arr = DantArray(
        data=small_array_2x3, metadata_class=SimpleMeta, major_axis=0
    )
    # Explicitly create metadata objects for each row
    arr.batch_create(indices=[0, 1], overwrite_existing=True)
    return arr


@pytest.fixture
def dantarr_2x3_meanmeta(small_array_2x3):
    arr = DantArray(data=small_array_2x3, metadata_class=MeanMeta, major_axis=0)
    # Explicitly create metadata objects for each row
    arr.batch_create(indices=[0, 1], overwrite_existing=True)
    return arr


# ---------------------------------------------------------------------------
# Tests: Initialization
# ---------------------------------------------------------------------------


def test_init_valid(small_array_2x3):
    arr = DantArray(
        data=small_array_2x3, metadata_class=SimpleMeta, major_axis=0
    )
    assert arr.shape == (2, 3)
    assert arr.major_axis == 0
    # Default metadata should auto-create with length of 2
    assert len(arr.metadata) == 2
    for meta in arr.metadata:
        # Should be of type SimpleMeta
        assert isinstance(meta, SimpleMeta)


def test_init_negative_axis(small_array_2x3):
    # major_axis = -1 should effectively select the last dimension
    arr = DantArray(
        data=small_array_2x3, metadata_class=SimpleMeta, major_axis=-1
    )
    assert arr.major_axis == 1  # Because small_array_2x3.ndim = 2
    assert arr.shape == (2, 3)


def test_init_invalid_axis(small_array_2x3):
    with pytest.raises(ValueError, match="major_axis must be an integer"):
        DantArray(
            data=small_array_2x3, metadata_class=SimpleMeta, major_axis=10
        )


def test_init_data_too_few_dims():
    data = np.array([1, 2, 3])
    # data has shape (3,); major_axis=1 is out of range
    with pytest.raises(
        ValueError, match="major_axis must be an integer in range"
    ):
        DantArray(data=data, metadata_class=SimpleMeta, major_axis=1)


def test_init_inconsistent_metadata_length(small_array_2x3):
    # shape along axis=0 is 2, but passing metadata of length 3 -> error
    dummy_meta = [SimpleMeta(), SimpleMeta(), SimpleMeta()]
    with pytest.raises(ValueError, match="Metadata length"):
        DantArray(
            data=small_array_2x3,
            metadata_class=SimpleMeta,
            major_axis=0,
            metadata=dummy_meta,
        )


def test_init_empty_data():
    data = np.array([])
    # major_axis=0 is valid for an empty 1D array, shape == (0,)
    # The resulting metadata length = 0
    arr = DantArray(data=data, metadata_class=SimpleMeta, major_axis=0)
    assert arr.shape == (0,)
    assert len(arr.metadata) == 0


# ---------------------------------------------------------------------------
# Tests: Basic Access & Properties
# ---------------------------------------------------------------------------


def test_data_property_is_read_only(dantarr_2x3_simplemeta):
    view = dantarr_2x3_simplemeta.data
    assert np.all(view == np.array([[1, 2, 3], [4, 5, 6]]))
    # Attempting to modify the view won't change the underlying data
    view[0, 0] = 999
    # Check the underlying data remains unchanged
    assert dantarr_2x3_simplemeta.data[0, 0] == 1


def test_unsafe_data_access_warning(dantarr_2x3_simplemeta):
    with pytest.warns(
        UnsafeDataAccessWarning, match="Direct access to _data_unsafe detected"
    ):
        _ = dantarr_2x3_simplemeta._data_unsafe


def test_validate_index(dantarr_2x3_simplemeta):
    with pytest.raises(IndexError, match="Index 2 out of bounds"):
        dantarr_2x3_simplemeta.get_metadata(2)

    # Valid index does not raise
    dantarr_2x3_simplemeta.get_metadata(1)


# ---------------------------------------------------------------------------
# Tests: Editing Data
# ---------------------------------------------------------------------------


def test_edit_data_without_refresh(dantarr_2x3_simplemeta):
    old_sum = dantarr_2x3_simplemeta.meta(0).sum_val
    with dantarr_2x3_simplemeta.edit_data(refresh=False) as arr_data:
        arr_data[0, 1] = 999
    # sum_val won't update because refresh=False
    assert dantarr_2x3_simplemeta.meta(0).sum_val == old_sum


def test_edit_data_with_refresh(dantarr_2x3_simplemeta):
    old_sum = dantarr_2x3_simplemeta.meta(0).sum_val
    with dantarr_2x3_simplemeta.edit_data(refresh=True) as arr_data:
        arr_data[0, 1] = 999
    # sum_val should refresh now
    new_sum = dantarr_2x3_simplemeta.meta(0).sum_val
    assert new_sum != old_sum
    # Verify correctness for slice = [1, 999, 3]
    assert new_sum == 1 + 999 + 3


# ---------------------------------------------------------------------------
# Tests: update_slice
# ---------------------------------------------------------------------------


def test_update_slice_axis0(dantarr_2x3_simplemeta):
    old_sum_1 = dantarr_2x3_simplemeta.meta(1).sum_val
    dantarr_2x3_simplemeta.update_slice(1, np.array([10, 20, 30]))
    new_sum_1 = dantarr_2x3_simplemeta.meta(1).sum_val
    assert new_sum_1 != old_sum_1
    assert new_sum_1 == 10 + 20 + 30


def test_update_slice_axis1():
    data = np.array([[1, 2, 3], [4, 5, 6]])
    arr = DantArray(data=data, metadata_class=SimpleMeta, major_axis=1)
    # Now metadata is tracked along axis=1, shape=3
    # We'll update slice idx=2 => all rows along col 2
    arr.update_slice(2, np.array([999, 1000]))
    slice_val_0 = arr.data[0, 2]
    slice_val_1 = arr.data[1, 2]
    assert slice_val_0 == 999
    assert slice_val_1 == 1000


# ---------------------------------------------------------------------------
# Tests: Metadata basic usage
# ---------------------------------------------------------------------------


def test_get_set_metadata(dantarr_2x3_simplemeta):
    # get_metadata auto-creates metadata if not present; but in fixture we already have it
    meta_0 = dantarr_2x3_simplemeta.get_metadata(0)
    assert meta_0 is not None
    assert isinstance(meta_0, SimpleMeta)

    # set_metadata
    new_meta = SimpleMeta(name="CustomRow")
    dantarr_2x3_simplemeta.set_metadata(0, new_meta)
    assert dantarr_2x3_simplemeta.meta(0).name == "CustomRow"


def test_set_metadata_wrong_type(dantarr_2x3_simplemeta):
    class OtherModel(BaseModel):
        something: int = 42

    with pytest.raises(TypeError, match="Metadata must be a SimpleMeta object"):
        dantarr_2x3_simplemeta.set_metadata(0, OtherModel())


def test_has_metadata(dantarr_2x3_simplemeta):
    assert dantarr_2x3_simplemeta.has_metadata(0) is True
    assert dantarr_2x3_simplemeta.has_metadata(1) is True


def test_get_existing_metadata(dantarr_2x3_simplemeta):
    existing = dantarr_2x3_simplemeta.get_existing_metadata()
    # Should be two items
    assert len(existing) == 2
    idx0, meta0 = existing[0]
    assert idx0 == 0
    assert isinstance(meta0, SimpleMeta)


# ---------------------------------------------------------------------------
# Tests: update_metadata
# ---------------------------------------------------------------------------


def test_update_metadata_field_ok(dantarr_2x3_simplemeta):
    dantarr_2x3_simplemeta.update_metadata(0, name="RowZero")
    assert dantarr_2x3_simplemeta.meta(0).name == "RowZero"


def test_update_metadata_invalid_field(dantarr_2x3_simplemeta):
    with pytest.raises(ValueError, match="Invalid field 'bogus_field'"):
        dantarr_2x3_simplemeta.update_metadata(0, bogus_field=123)


def test_update_metadata_non_writable_computed_field(dantarr_2x3_simplemeta):
    # sum_val is non-writable
    with pytest.raises(ValueError, match="is computed and not writable"):
        dantarr_2x3_simplemeta.update_metadata(0, sum_val=9999)


# ---------------------------------------------------------------------------
# Tests: Computed fields & overrides
# ---------------------------------------------------------------------------


def test_computed_field_automatically_calculated(dantarr_2x3_simplemeta):
    # index=0 => array slice is [1,2,3]; sum=6; double_mean=(mean=2)->4
    assert dantarr_2x3_simplemeta.meta(0).sum_val == 6.0
    assert dantarr_2x3_simplemeta.meta(0).double_mean_val == 4.0
    # index=1 => array slice is [4,5,6]; sum=15; double_mean=(mean=5)->10
    assert dantarr_2x3_simplemeta.meta(1).sum_val == 15.0
    assert dantarr_2x3_simplemeta.meta(1).double_mean_val == 10.0


def test_computed_field_writable_override(dantarr_2x3_simplemeta):
    # double_mean_val is writable
    new_val = 999.0
    dantarr_2x3_simplemeta.update_metadata(1, double_mean_val=new_val)
    # Now the user has overridden this field
    assert dantarr_2x3_simplemeta.meta(1).double_mean_val == new_val

    # If we refresh data, it should remain overridden.
    with dantarr_2x3_simplemeta.edit_data(refresh=True) as data:
        data[1, 0] = 1000
    assert dantarr_2x3_simplemeta.meta(1).double_mean_val == new_val


def test_re_enable_computed_field(dantarr_2x3_simplemeta):
    # override double_mean_val
    dantarr_2x3_simplemeta.update_metadata(0, double_mean_val=123.45)
    assert dantarr_2x3_simplemeta.meta(0).double_mean_val == 123.45

    # re-enable
    dantarr_2x3_simplemeta.re_enable_computed_field(
        idx=0, field_name="double_mean_val", recompute=True
    )
    # recomputed => for slice [1,2,3], mean=2 => double=4
    assert dantarr_2x3_simplemeta.meta(0).double_mean_val == 4.0


def test_mean_value_field_computation(dantarr_2x3_meanmeta):
    # axis=0 => row 0 has mean(1,2,3)=2, row1 has mean(4,5,6)=5
    assert dantarr_2x3_meanmeta.meta(0).avg == 2.0
    assert dantarr_2x3_meanmeta.meta(1).avg == 5.0


# ---------------------------------------------------------------------------
# Tests: Batch Create & Update
# ---------------------------------------------------------------------------


def test_batch_create(small_array_2x3):
    arr = DantArray(
        data=small_array_2x3,
        metadata_class=MinimalMeta,
        major_axis=0,
        metadata=[None, None],
    )
    # By default, the constructor already sets up a metadata list of length 2 = [None, None].
    # Let's create metadata for both indices
    arr.batch_create(indices=[0, 1], overwrite_existing=False)
    for idx, meta in arr.get_existing_metadata():
        assert isinstance(meta, MinimalMeta)


def test_batch_create_overwrite(dantarr_2x3_simplemeta):
    # We'll store old name for idx=0
    old_name = dantarr_2x3_simplemeta.meta(0).name
    # We call batch_create with overwrite => old metadata for idx=0 should be replaced
    dantarr_2x3_simplemeta.batch_create(
        [0, 1], overwrite_existing=True, name="batch_overwritten"
    )
    # Now both indices should have "batch_overwritten"
    assert dantarr_2x3_simplemeta.meta(0).name == "batch_overwritten"
    assert dantarr_2x3_simplemeta.meta(1).name == "batch_overwritten"
    assert dantarr_2x3_simplemeta.meta(0).name != old_name


def test_batch_update(dantarr_2x3_simplemeta):
    dantarr_2x3_simplemeta.batch_update([0, 1], name="Batchy")
    assert dantarr_2x3_simplemeta.meta(0).name == "Batchy"
    assert dantarr_2x3_simplemeta.meta(1).name == "Batchy"


# ---------------------------------------------------------------------------
# Tests: Copy
# ---------------------------------------------------------------------------


def test_copy_shallow(dantarr_2x3_simplemeta):
    arr_copy = dantarr_2x3_simplemeta.copy(deep_metadata=False)
    # The underlying arrays should be distinct
    assert not np.shares_memory(arr_copy.data, dantarr_2x3_simplemeta.data)
    # The metadata objects, however, for shallow copy are the same references
    assert arr_copy._metadata[0] is dantarr_2x3_simplemeta._metadata[0]
    # Changing the data in the original doesn't impact the copy's data
    with dantarr_2x3_simplemeta.edit_data(refresh=True) as data:
        data[0, 0] = 999
    # data changed in original
    assert arr_copy.data[0, 0] == 1


def test_copy_deep(dantarr_2x3_simplemeta):
    arr_copy = dantarr_2x3_simplemeta.copy(deep_metadata=True)
    # The underlying numpy arrays should be distinct
    assert not np.shares_memory(arr_copy.data, dantarr_2x3_simplemeta.data)
    # The metadata objects should also be distinct
    assert arr_copy._metadata[0] is not dantarr_2x3_simplemeta._metadata[0]
    # But they should be equivalent in content
    assert arr_copy.meta(0).dict() == dantarr_2x3_simplemeta.meta(0).dict()


# ---------------------------------------------------------------------------
# Tests: Immutable-like Updates
# ---------------------------------------------------------------------------


def test_with_updated_data(dantarr_2x3_simplemeta):
    new_data = np.array([[10, 20, 30], [40, 50, 60]], dtype=int)
    new_arr = dantarr_2x3_simplemeta.with_updated_data(new_data)
    # original remains the same
    assert np.all(
        dantarr_2x3_simplemeta.data == np.array([[1, 2, 3], [4, 5, 6]])
    )
    # new_arr has updated data
    assert np.all(new_arr.data == new_data)
    # computed fields in new_arr are correct
    assert new_arr.meta(0).sum_val == 10 + 20 + 30


def test_with_updated_slice(dantarr_2x3_simplemeta):
    new_arr = dantarr_2x3_simplemeta.with_updated_slice(
        0, np.array([100, 200, 300])
    )
    # original remains the same
    assert np.all(dantarr_2x3_simplemeta.data[0] == [1, 2, 3])
    # new array has updated row 0
    assert np.all(new_arr.data[0] == [100, 200, 300])
    # check computed for row 0 => sum=600
    assert new_arr.meta(0).sum_val == 600.0


# ---------------------------------------------------------------------------
# Tests: MetadataAccessor
# ---------------------------------------------------------------------------


def test_metadata_accessor_get_set(dantarr_2x3_simplemeta):
    accessor = dantarr_2x3_simplemeta.meta(0)
    assert accessor.name == "untitled"
    accessor.name = "NewName"
    assert dantarr_2x3_simplemeta.meta(0).name == "NewName"


def test_metadata_accessor_update(dantarr_2x3_simplemeta):
    accessor = dantarr_2x3_simplemeta.meta(1)
    accessor.update(name="updatedName")
    assert dantarr_2x3_simplemeta.meta(1).name == "updatedName"


def test_metadata_accessor_invalid_attr(dantarr_2x3_simplemeta):
    accessor = dantarr_2x3_simplemeta.meta(0)
    with pytest.raises(AttributeError, match="no attribute 'bogus'"):
        _ = accessor.bogus

    with pytest.raises(ValueError, match="computed and not writable"):
        accessor.sum_val = 999
