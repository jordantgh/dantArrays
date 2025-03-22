from typing import Optional
import pytest
import numpy as np
from pydantic import BaseModel

from dantArrays.computed import (
    computed,
    ComputedFieldContext,
    index_field,
    dimension_field,
    data_based_field,
    max_value_field,
    mean_value_field,
    IndexNameComputation,
    DimensionInfoComputation,
    DataBasedComputation,
    MaxValueComputation,
    MeanValueComputation,
)


def test_computed_factory_direct():
    def square_sum(ctx: ComputedFieldContext):
        data_slice = ctx.get_slice()
        return float(np.sum(data_slice) ** 2)

    class Model(BaseModel):
        squared_sum: float = computed(square_sum, writable=False, default=0.0)

    # Validate direct usage
    m = Model()
    assert (
        m.squared_sum == 0.0
    )  # By default, no real array context => remains 0


def test_index_field():
    class Model(BaseModel):
        index_name: str = index_field(prefix="Experiment_", suffix="_v1")

    m = Model()
    assert m.index_name == ""  # default outside real context

    # If we run the factory manually
    comp = IndexNameComputation(prefix="Foo_", suffix="_Bar")
    ctx = ComputedFieldContext(index=3, array=np.zeros((5,)), axis=0)
    assert comp(ctx) == "Foo_3_Bar"


def test_dimension_field():
    class Model(BaseModel):
        dim_info: str = dimension_field()

    m = Model()
    assert m.dim_info == ""  # out of context

    # Try a real context
    comp = DimensionInfoComputation()
    ctx = ComputedFieldContext(index=0, array=np.zeros((5, 2)), axis=1)
    info_str = comp(ctx)
    assert "Dimension 1 of size 2" in info_str


def test_data_based_field():
    class Model(BaseModel):
        mean: Optional[float] = data_based_field(
            lambda arr: float(np.mean(arr)), default=0.0
        )

    m = Model()
    assert m.mean == 0.0

    comp = DataBasedComputation(func=lambda arr: float(np.mean(arr)))
    ctx = ComputedFieldContext(
        index=0, array=np.array([[1, 2, 3], [4, 5, 6]]), axis=0
    )
    val = comp(ctx)
    # Slicing row 0 => [1,2,3], mean => 2.0
    assert val == 2.0


def test_max_value_field():
    class Model(BaseModel):
        maxv: float = max_value_field()

    m = Model()
    assert m.maxv == 0.0

    ctx = ComputedFieldContext(
        index=1, array=np.array([[1, 2, 3], [10, 50, -1]]), axis=0
    )
    comp = MaxValueComputation()
    assert comp(ctx) == 50
    val = comp(ctx)
    assert val == 50


def test_mean_value_field():
    class Model(BaseModel):
        avg: float = mean_value_field()

    arr = np.array([[10, 20, 30]])  # shape (1, 3)
    ctx = ComputedFieldContext(index=0, array=arr, axis=0)
    comp = MeanValueComputation()
    result = comp(ctx)
    assert result == 20.0
