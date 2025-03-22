from typing import TypeVar, Callable, Generic, Optional
from dataclasses import dataclass
import numpy as np
from pydantic import Field

from .utils import get_array_slice

R = TypeVar("R")

# Constants
COMPUTED_FACTORY_KEY = "_dantarrays.computed_factory"
COMPUTED_WRITABLE_KEY = "_dantarrays.computed_writable"


class ComputedFieldContext:
    """
    Context provided to computed fields, encapsulating access to array slices.
    Computed fields should rely exclusively on this interface to avoid tight coupling with the DantArray implementation.
    """

    def __init__(self, index: int, array: np.ndarray, axis: int):
        """
        Initialize array context

        Args:
            index: Index along the specified axis
            array: The complete array
            axis: Which axis this metadata belongs to
        """
        self.index = index
        self.array = array
        self.axis = axis

    @property
    def shape(self) -> tuple:
        """Get the shape of the array"""
        return self.array.shape

    @property
    def size(self) -> int:
        """Size of the array along the current axis."""
        return self.array.shape[self.axis]

    @property
    def indices(self) -> range:
        """Range object covering indices along the current axis."""
        return range(self.array.shape[self.axis])

    def get_full_array(self) -> np.ndarray:
        """Get a copy of the full array"""
        return self.array.copy()

    def get_slice(self) -> np.ndarray:
        """Get a copy of the data slice for this index"""
        return get_array_slice(self.array, self.index, self.axis)


def computed(
    factory: Callable[[ComputedFieldContext], R],
    writable: bool = False,
    default: Optional[R] = None,
) -> R:
    """
    Define a computed field with a factory function

    Args:
        factory: Callable that takes an ArrayContext and returns a value
                 (typically a computation dataclass instance)
        writable: Whether this computed field can be directly overridden
        default: Default value when no computation is possible (outside array context)

    Returns:
        A Field with metadata for computation, typed to match the factory's return type
    """
    return Field(
        default=default,
        **{
            COMPUTED_FACTORY_KEY: factory,
            COMPUTED_WRITABLE_KEY: writable,
        },
    )


@dataclass
class IndexNameComputation:
    """Computation for index-based naming"""

    prefix: str = "Item_"
    suffix: str = ""

    def __call__(self, ctx: ComputedFieldContext) -> str:
        return f"{self.prefix}{ctx.index}{self.suffix}"


@dataclass
class DimensionInfoComputation:
    """Computation for dimension information"""

    def __call__(self, ctx: ComputedFieldContext) -> str:
        return f"Dimension {ctx.axis} of size {ctx.shape[ctx.axis]}"


@dataclass
class DataBasedComputation(Generic[R]):
    """Computation based on array data"""

    func: Callable[[np.ndarray], R]

    def __call__(self, ctx: ComputedFieldContext) -> R:
        return self.func(ctx.get_slice())


@dataclass
class MaxValueComputation:
    """Computation for maximum value in slice"""

    def __call__(self, ctx: ComputedFieldContext) -> float:
        return float(np.max(ctx.get_slice()))


@dataclass
class MeanValueComputation:
    """Computation for mean value in slice"""

    def __call__(self, ctx: ComputedFieldContext) -> float:
        return float(np.mean(ctx.get_slice()))


def index_field(prefix: str = "Item_", suffix: str = "") -> str:
    """
    Create a metadata field that uses index-based naming

    Args:
        prefix: String prefix for the field
        suffix: String suffix for the field

    Returns:
        String field that generates "{prefix}{idx}{suffix}"
    """
    return computed(
        IndexNameComputation(prefix, suffix), writable=True, default=""
    )


def dimension_field() -> str:
    """
    Create a metadata field that describes the dimension size

    Returns:
        String field containing dimension information
    """
    return computed(DimensionInfoComputation(), default="")


def data_based_field(func: Callable[[np.ndarray], R]) -> R:
    """
    Create a metadata field based on the array data

    Args:
        func: Function that processes the slice data

    Returns:
        Field with the type matching the return type of the function
    """
    return computed(DataBasedComputation(func))


def max_value_field() -> float:
    """Field containing the maximum value in the slice"""
    return computed(MaxValueComputation(), default=0.0)


def mean_value_field() -> float:
    """Field containing the mean value in the slice"""
    return computed(MeanValueComputation(), default=0.0)
