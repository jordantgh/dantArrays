from .core import DantArray, MetadataAccessor
from .computed import (
    ComputedFieldContext,
    computed,
    index_field,
    dimension_field,
    data_based_field,
    max_value_field,
    mean_value_field,
)

# For backward compatibility, also expose these if they're used directly
from .computed import (
    IndexNameComputation,
    DimensionInfoComputation,
    DataBasedComputation,
    MaxValueComputation,
    MeanValueComputation,
)

# Version info
__version__ = "0.0.2"