from typing import (
    Optional,
    Type,
    TypeVar,
    Callable,
    Any,
    cast,
    ClassVar,
)
import numpy as np
from pydantic import BaseModel
from pydantic_core import core_schema
from warnings import warn
from enum import Enum, auto


class FieldType(Enum):
    FIXED = auto()
    COMPUTED = auto()


class ArrayContext:
    """Container for contextual information about an array element"""

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

    def get_slice(self) -> np.ndarray:
        """Get the data slice for this index"""
        if self.axis == 0:
            return self.array[self.index]
        elif self.axis == 1:
            return self.array[:, self.index]
        else:
            # Create a tuple of slices with the index at the right position
            slices = tuple(
                self.index if i == self.axis else slice(None)
                for i in range(len(self.shape))
            )
            return self.array[slices].copy()


class ComputedField:
    """Field that is computed based on array context"""

    field_type: ClassVar[FieldType] = FieldType.COMPUTED

    def __init__(self, factory_func: Callable[[ArrayContext], Any]):
        """
        Initialize with a factory function

        Args:
            factory_func: Function that takes an ArrayContext and returns a value
        """
        self.factory_func = factory_func

    def __call__(self, context: ArrayContext) -> Any:
        """Generate a value using the provided context"""
        return self.factory_func(context)

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: Callable[[Any], core_schema.CoreSchema],
    ) -> core_schema.CoreSchema:
        """
        Define how Pydantic should handle ComputedField instances.

        We create a schema that validates that the value is a ComputedField
        instance but doesn't transform it, since the actual value will be
        computed later when the metadata is accessed.
        """
        return core_schema.is_instance_schema(ComputedField)


# Convenience functions for common computed field patterns


def index_field(prefix: str = "Item_", suffix: str = "") -> ComputedField:
    """
    Create a metadata field that uses index-based naming

    Args:
        prefix: String prefix for the field
        suffix: String suffix for the field

    Returns:
        ComputedField that generates "{prefix}{idx}{suffix}"
    """
    return ComputedField(lambda ctx: f"{prefix}{ctx.index}{suffix}")


def dimension_field() -> ComputedField:
    """
    Create a metadata field that describes the dimension size

    Returns:
        ComputedField that generates dimension information
    """
    return ComputedField(
        lambda ctx: f"Dimension {ctx.axis} of size {ctx.shape[ctx.axis]}"
    )


def data_based_field(func: Callable[[np.ndarray], Any]) -> ComputedField:
    """
    Create a metadata field based on the array data

    Args:
        func: Function that processes the slice data

    Returns:
        ComputedField that applies the function to the data slice
    """
    return ComputedField(lambda ctx: func(ctx.get_slice()))


# Simple data-based field generators
def max_value_field() -> ComputedField:
    """Field containing the maximum value in the slice"""
    return data_based_field(lambda data: float(np.max(data)))


def mean_value_field() -> ComputedField:
    """Field containing the mean value in the slice"""
    return data_based_field(lambda data: float(np.mean(data)))


T = TypeVar("T", bound=BaseModel)


class MetadataArray:
    def __init__(
        self,
        data: np.ndarray,
        metadata_class: Type[T],
        major_axis: int = 0,
        metadata: Optional[list[Optional[T]]] = None,
    ):
        """
        Wrapper for numpy arrays with rich metadata for rows or columns.

        Args:
            data: NumPy array to wrap
            metadata_class: Pydantic model class to use for metadata
            major_axis: 0 for row metadata, 1 for column metadata
            metadata: Optional pre-existing metadata list

        Raises:
            ValueError: If major_axis is invalid or metadata length is inconsistent
        """
        self._data: np.ndarray = np.array([])  # Initialize with empty array
        self.metadata_class = metadata_class

        # Validate major_axis before assigning
        if not isinstance(major_axis, int) or major_axis < 0:
            raise ValueError(
                f"major_axis must be a non-negative integer, got {major_axis}"
            )
        self.major_axis = major_axis

        # Set data (this will initialize _metadata through the setter)
        self.data = data

        # Initialize metadata if provided
        if metadata is not None:
            if len(metadata) != self.data.shape[self.major_axis]:
                raise ValueError(
                    f"Metadata length ({len(metadata)}) must match the major dimension size ({self.data.shape[self.major_axis]})"
                )
            self._metadata = list(metadata)

    @property
    def data(self) -> np.ndarray:
        """Get the underlying numpy array"""
        return self._data

    @data.setter
    def data(self, new_data: np.ndarray) -> None:
        """
        Set the underlying numpy array, validating size consistency with metadata

        Raises:
            ValueError: If the array dimension is too small for major_axis
        """
        new_data = np.asarray(new_data)

        # Validate array has enough dimensions
        if len(new_data.shape) <= self.major_axis:
            raise ValueError(
                f"Array dimension {len(new_data.shape)} too small for major_axis={self.major_axis}"
            )

        # Handle metadata consistency if data already exists
        if hasattr(self, "_metadata"):
            old_size = (
                self._data.shape[self.major_axis] if self._data.size > 0 else 0
            )
            new_size = new_data.shape[self.major_axis]

            if old_size != new_size:
                if len(self._metadata) > new_size:
                    # Warn about truncating metadata
                    warn(
                        f"New data shape will truncate metadata from {len(self._metadata)} to {new_size} items"
                    )
                    self._metadata = self._metadata[:new_size]
                elif len(self._metadata) < new_size:
                    # Extend metadata with None for new positions
                    self._metadata.extend(
                        [None] * (new_size - len(self._metadata))
                    )
        else:
            # Initialize metadata container
            major_dim_size = new_data.shape[self.major_axis]
            self._metadata = [None] * major_dim_size

        self._data = new_data

    def _validate_index(self, idx: int) -> None:
        """
        Ensure index is within bounds of major dimension

        Raises:
            IndexError: If index is out of bounds
        """
        major_dim_size = self.data.shape[self.major_axis]
        if not 0 <= idx < major_dim_size:
            raise IndexError(
                f"Index {idx} out of bounds for axis {self.major_axis} with size {major_dim_size}"
            )

    def _is_computed_field(self, value: Any) -> bool:
        """Check if a field value is a computed field"""
        return (
            hasattr(value, "field_type")
            and value.field_type == FieldType.COMPUTED
        )

    def _resolve_computed_fields(self, instance: T, idx: int) -> None:
        """
        Resolve any ComputedField objects in the instance

        Args:
            instance: The metadata instance to process
            idx: The index for which this metadata is being created
        """
        context = ArrayContext(idx, self.data, self.major_axis)

        # Initialize storage for computed field factories if needed
        if not hasattr(instance, "_computed_field_factories"):
            instance._computed_field_factories = {}

        for field_name in instance.model_fields.keys():
            value = getattr(instance, field_name)
            if self._is_computed_field(value):
                # Store the ComputedField object for future recalculation
                instance._computed_field_factories[field_name] = value

                # Set the computed value
                setattr(instance, field_name, value(context))

    def __getitem__(self, idx: Any) -> np.ndarray:
        """Support for standard NumPy indexing"""
        return self.data[idx]

    def has_metadata(self, idx: int) -> bool:
        """
        Check if metadata exists for a specific index without creating defaults

        Args:
            idx: Index to check

        Returns:
            True if metadata exists, False otherwise

        Raises:
            IndexError: If index is out of bounds
        """
        self._validate_index(idx)
        return self._metadata[idx] is not None

    def get_metadata(
        self, idx: int, create_default: bool = True
    ) -> Optional[T]:
        """
        Get metadata for a specific index, optionally creating default if needed

        Args:
            idx: Index to retrieve metadata for
            create_default: Whether to create default metadata if none exists

        Returns:
            Metadata instance or None if no metadata exists and create_default is False

        Raises:
            IndexError: If index is out of bounds
        """
        self._validate_index(idx)

        if self._metadata[idx] is None and create_default:
            # Create default metadata
            self._metadata[idx] = self.metadata_class()
            # Resolve any computed fields
            self._resolve_computed_fields(cast(T, self._metadata[idx]), idx)

        return self._metadata[idx]

    def set_metadata(self, idx: int, metadata: T) -> None:
        """
        Set complete metadata for a specific index

        Args:
            idx: Index to set metadata for
            metadata: Metadata instance to set

        Raises:
            IndexError: If index is out of bounds
            TypeError: If metadata is not an instance of the metadata class
        """
        self._validate_index(idx)

        if not isinstance(metadata, self.metadata_class):
            raise TypeError(
                f"Metadata must be a {self.metadata_class.__name__} object"
            )
        self._metadata[idx] = metadata
        # Resolve any computed fields in the new metadata
        self._resolve_computed_fields(metadata, idx)

    def update_metadata(
        self, idx: int, create_default: bool = True, **kwargs: Any
    ) -> None:
        """
        Update specific metadata fields for the given index

        Args:
            idx: Index to update metadata for
            create_default: Whether to create default metadata if none exists
            **kwargs: Field values to update

        Raises:
            IndexError: If index is out of bounds
            ValueError: If a field in kwargs is not in the metadata model
        """
        self._validate_index(idx)

        # Get existing metadata or create default
        metadata = self.get_metadata(idx, create_default)
        if metadata is None:
            return  # No metadata and not creating default

        # Validate field names
        for field in kwargs:
            if field not in metadata.model_fields:
                raise ValueError(
                    f"Field '{field}' does not exist in {self.metadata_class.__name__}"
                )

        # Process any ComputedField values in the updates
        context = ArrayContext(idx, self.data, self.major_axis)
        for field, value in list(kwargs.items()):
            if self._is_computed_field(value):
                # Store the ComputedField object for future recalculation
                if not hasattr(metadata, "_computed_field_factories"):
                    metadata._computed_field_factories = {}
                metadata._computed_field_factories[field] = value

                # Use the computed value
                kwargs[field] = value(context)

        # Use Pydantic's model_copy with update
        self._metadata[idx] = metadata.model_copy(update=kwargs)

    def batch_create(
        self,
        indices: list[int],
        overwrite_existing: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Create metadata objects for multiple indices at once.

        Args:
            indices: List of indices to create metadata for
            overwrite_existing: Whether to overwrite existing metadata
            **kwargs: Initial field values for all created metadata objects

        Raises:
            IndexError: If any index is out of bounds
            ValueError: If any field name is invalid
        """
        # Validate all indices first
        for idx in indices:
            self._validate_index(idx)

        # Validate field names
        if kwargs:
            # Create a temporary instance to check fields
            temp = self.metadata_class()
            for field in kwargs:
                if field not in temp.model_fields:
                    valid_fields = list(temp.model_fields.keys())
                    raise ValueError(
                        f"Invalid field '{field}'. Valid fields: {valid_fields}"
                    )

        # Create metadata for each index
        for idx in indices:
            # Skip if metadata exists and we're not overwriting
            if self.has_metadata(idx) and not overwrite_existing:
                continue

            # Create metadata with the provided kwargs
            metadata = self.metadata_class(**kwargs)

            # Set the metadata (this will resolve computed fields)
            self.set_metadata(idx, metadata)

    def batch_update(
        self, indices: list[int], create_default: bool = True, **kwargs: Any
    ) -> None:
        """
        Update metadata for multiple indices at once

        Args:
            indices: list of indices to update
            create_default: Whether to create default metadata if none exists
            **kwargs: Field values to update

        Raises:
            IndexError: If any index is out of bounds
        """
        for idx in indices:
            self.update_metadata(idx, create_default, **kwargs)

    def apply_with_metadata(
        self, func: Callable[[np.ndarray, T], Any]
    ) -> list[Any]:
        """
        Apply a function to each slice along major axis with its metadata

        Args:
            func: Function taking (slice_data, metadata) and returning a result

        Returns:
            list of results from applying the function
        """
        results = []
        for i in range(self.data.shape[self.major_axis]):
            # Extract the slice
            slice_data = self.get_slice(i)

            # Get metadata, creating default if needed
            metadata = self.get_metadata(i)
            if metadata is None:
                # Skip if no metadata and we didn't create a default
                continue

            results.append(func(slice_data, metadata))
        return results

    def get_slice(self, idx: int) -> np.ndarray:
        """
        Get a slice of data along the major axis

        Args:
            idx: Index to get slice for

        Returns:
            Array slice at the specified index
        """
        self._validate_index(idx)
        context = ArrayContext(idx, self.data, self.major_axis)
        return context.get_slice()

    @property
    def metadata(self) -> list[Optional[T]]:
        """
        Return all metadata (creating defaults where needed)

        Returns:
            list of metadata instances
        """
        return [
            self.get_metadata(i, create_default=True)
            for i in range(self.data.shape[self.major_axis])
        ]

    def get_existing_metadata(self) -> list[tuple[int, T]]:
        """
        Return only indices and metadata that actually exist

        Returns:
            list of (index, metadata) tuples for indices with metadata
        """
        return [
            (i, self._metadata[i])
            for i in range(self.data.shape[self.major_axis])
            if self._metadata[i] is not None
        ]

    def refresh_computed_fields(self) -> None:
        """Refresh all computed fields in all metadata instances"""
        for idx in range(self.data.shape[self.major_axis]):
            metadata = self.get_metadata(idx, create_default=False)
            if metadata is None:
                continue

            # Skip if no computed fields are stored
            if not hasattr(metadata, "_computed_field_factories"):
                continue

            context = ArrayContext(idx, self.data, self.major_axis)
            updates = {}

            # Use the stored factory functions to recompute values
            for (
                field_name,
                factory,
            ) in metadata._computed_field_factories.items():
                updates[field_name] = factory(context)

            if updates:
                self._metadata[idx] = metadata.model_copy(update=updates)

    @property
    def shape(self) -> tuple:
        """Get the shape of the underlying array"""
        return self.data.shape

    def meta(self, idx: int) -> "MetadataAccessor":
        """
        Return a metadata accessor for easier field-by-field updates

        Args:
            idx: Index to access metadata for

        Returns:
            MetadataAccessor for the given index

        Raises:
            IndexError: If index is out of bounds
        """
        self._validate_index(idx)
        return MetadataAccessor(self, idx)

    def copy(self) -> "MetadataArray":
        """
        Create a deep copy of this MetadataArray

        Returns:
            New MetadataArray with copied data and metadata
        """
        # Copy metadata if it exists
        copied_metadata = []
        for m in self._metadata:
            if m is not None:
                copied_metadata.append(m.model_copy(deep=True))
            else:
                copied_metadata.append(None)

        return MetadataArray(
            data=self.data.copy(),
            metadata_class=self.metadata_class,
            major_axis=self.major_axis,
            metadata=copied_metadata,
        )

    # Immutable modification methods

    def with_updated_data(self, new_data: np.ndarray) -> "MetadataArray":
        """
        Create a new MetadataArray with updated data and recomputed fields

        Args:
            new_data: New array data

        Returns:
            New MetadataArray instance with updated data and refreshed computed fields
        """
        result = self.copy()
        result.data = new_data
        result.refresh_computed_fields()
        return result

    def with_updated_slice(
        self, idx: int, new_slice: np.ndarray
    ) -> "MetadataArray":
        """
        Create a new MetadataArray with an updated slice and recomputed fields

        Args:
            idx: Index of slice to update
            new_slice: New data for the slice

        Returns:
            New MetadataArray instance with updated slice and refreshed computed fields
        """
        self._validate_index(idx)

        new_data = self.data.copy()

        # Handle different axis cases
        if self.major_axis == 0:
            new_data[idx] = new_slice
        elif self.major_axis == 1:
            new_data[:, idx] = new_slice
        else:
            slices = tuple(
                idx if i == self.major_axis else slice(None)
                for i in range(len(new_data.shape))
            )
            new_data[slices] = new_slice

        result = self.copy()
        result.data = new_data

        # Only refresh computed fields for the affected index
        metadata = result.get_metadata(idx, create_default=False)
        if metadata is not None and hasattr(
            metadata, "_computed_field_factories"
        ):
            context = ArrayContext(idx, result.data, result.major_axis)
            updates = {}

            for (
                field_name,
                factory,
            ) in metadata._computed_field_factories.items():
                updates[field_name] = factory(context)

            if updates:
                result._metadata[idx] = metadata.model_copy(update=updates)

        return result


class MetadataAccessor:
    """Helper class for ergonomic access to metadata fields"""

    def __init__(self, parent: MetadataArray, idx: int):
        """
        Initialize a metadata accessor

        Args:
            parent: Parent MetadataArray
            idx: Index to access metadata for
        """
        self._parent = parent
        self._idx = idx
        # Create metadata if it doesn't exist
        if not self._parent.has_metadata(self._idx):
            self._parent.get_metadata(self._idx)

    def __getattr__(self, name: str) -> Any:
        """
        Access metadata fields as attributes

        Args:
            name: Field name to access

        Returns:
            Field value

        Raises:
            AttributeError: If field doesn't exist in metadata
        """
        if name.startswith("_"):
            return super().__getattr__(name)

        metadata = self._parent.get_metadata(self._idx)
        if metadata is None:
            raise AttributeError(f"No metadata exists for index {self._idx}")

        if name in metadata.model_fields:
            return getattr(metadata, name)
        raise AttributeError(
            f"{metadata.__class__.__name__} has no attribute '{name}'"
        )

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Set metadata fields as attributes

        Args:
            name: Field name to set
            value: Value to set

        Raises:
            ValueError: If field doesn't exist in metadata
        """
        if name in ["_parent", "_idx"]:
            super().__setattr__(name, value)
            return

        # Update the specific field
        self._parent.update_metadata(self._idx, **{name: value})

    def update(self, **kwargs: Any) -> None:
        """
        Update multiple fields at once

        Args:
            **kwargs: Field values to update

        Raises:
            ValueError: If a field doesn't exist in metadata
        """
        self._parent.update_metadata(self._idx, **kwargs)

    def get(self) -> Optional[BaseModel]:
        """
        Get the full metadata object

        Returns:
            Metadata instance or None if no metadata exists
        """
        return self._parent.get_metadata(self._idx, create_default=False)


def usage_example():
    """Demonstrate various use cases for the MetadataArray class"""
    import numpy as np
    from pydantic import BaseModel, Field
    from typing import List, Optional
    from datetime import datetime

    print("\n" + "=" * 80)
    print("MetadataArray Usage Examples".center(80))
    print("=" * 80)

    #########################################################################
    # Example 1: Financial Time Series Data
    #########################################################################
    print("\n\n" + "-" * 80)
    print("EXAMPLE 1: FINANCIAL TIME SERIES DATA".center(80))
    print("-" * 80)

    class StockMetadata(BaseModel):
        symbol: str = ""
        company_name: Optional[str] = None
        sector: Optional[str] = None
        is_benchmark: bool = False
        max_price: ComputedField = max_value_field()
        mean_price: ComputedField = mean_value_field()

    # Create sample price data for 3 stocks over 100 days
    np.random.seed(42)  # For reproducibility
    price_data = np.random.normal(
        loc=[[100], [50], [200]], scale=[[10], [5], [20]], size=(3, 100)
    )

    print(f"Stock price data shape: {price_data.shape} (3 stocks, 100 days)")

    # Create a MetadataArray with row metadata (each row is a stock)
    stocks = MetadataArray(
        data=price_data, metadata_class=StockMetadata, major_axis=0
    )

    # Set metadata for each stock
    stocks.update_metadata(
        0, symbol="AAPL", company_name="Apple Inc.", sector="Technology"
    )
    stocks.update_metadata(
        1, symbol="MSFT", company_name="Microsoft Corp.", sector="Technology"
    )
    stocks.update_metadata(
        2,
        symbol="AMZN",
        company_name="Amazon.com Inc.",
        sector="Consumer Cyclical",
    )

    # Ergonomic access to metadata
    stocks.meta(0).is_benchmark = True

    # Display metadata
    print("\nStock Metadata:")
    for i, meta in stocks.get_existing_metadata():
        print(f"  Stock {i}: {meta.symbol} ({meta.company_name})")
        print(f"    Sector: {meta.sector}")
        print(f"    Benchmark: {meta.is_benchmark}")
        print(f"    Max price: ${meta.max_price:.2f}")
        print(f"    Mean price: ${meta.mean_price:.2f}")

    # Find tech stocks with average price > 75
    tech_stocks = [
        (i, meta.symbol, meta.mean_price)
        for i, meta in stocks.get_existing_metadata()
        if meta.sector == "Technology" and meta.mean_price > 75
    ]

    print("\nTech stocks with average price > $75:")
    for idx, symbol, mean_price in tech_stocks:
        print(f"  {symbol}: ${mean_price:.2f}")

    # Apply function using both data and metadata
    volatility_by_sector = {}

    def calc_volatility(data, meta):
        vol = np.std(data)
        sector = meta.sector or "Unknown"
        if sector not in volatility_by_sector:
            volatility_by_sector[sector] = []
        volatility_by_sector[sector].append((meta.symbol, vol))
        return vol

    volatilities = stocks.apply_with_metadata(calc_volatility)

    print("\nVolatility by sector:")
    for sector, stocks_vol in volatility_by_sector.items():
        print(f"  {sector}:")
        for symbol, vol in stocks_vol:
            print(f"    {symbol}: {vol:.2f}")

    #########################################################################
    # Example 2: Image Dataset with Classification Metadata
    #########################################################################
    print("\n\n" + "-" * 80)
    print("EXAMPLE 2: IMAGE DATASET WITH CLASSIFICATION METADATA".center(80))
    print("-" * 80)

    class ImageMetadata(BaseModel):
        filename: ComputedField = index_field(prefix="img_", suffix=".jpg")
        label: Optional[str] = None
        confidence: Optional[float] = None
        brightness: ComputedField = data_based_field(
            lambda x: float(np.mean(x))
        )
        is_augmented: bool = False
        tags: List[str] = Field(default_factory=list)

    # Create a 20-image dataset with 64x64 images (grayscale)
    np.random.seed(42)
    images = np.random.rand(20, 64, 64)

    print(f"Image dataset shape: {images.shape} (20 images, 64x64 pixels)")

    # Create a MetadataArray for the image dataset
    image_dataset = MetadataArray(
        data=images, metadata_class=ImageMetadata, major_axis=0
    )

    # Create metadata for a batch of labeled images at once
    labeled_indices = [0, 1, 2, 3, 4]
    image_dataset.batch_create(
        indices=labeled_indices,
        confidence=0.80,  # Default confidence for all created metadata
        tags=["dataset_v1"],  # Default tag for all created metadata
    )

    # Now update specific fields for individual images
    image_dataset.update_metadata(
        0, label="cat", confidence=0.95, tags=["pet", "mammal", "dataset_v1"]
    )
    image_dataset.update_metadata(
        1, label="dog", confidence=0.87, tags=["pet", "mammal", "dataset_v1"]
    )
    image_dataset.update_metadata(
        2, label="car", confidence=0.92, tags=["vehicle", "dataset_v1"]
    )

    # Batch update metadata for augmented images
    augmented_indices = [5, 6, 7, 8, 9]
    image_dataset.batch_update(augmented_indices, is_augmented=True)

    # Print statistics on batch creation
    print("\nBatch Creation Statistics:")
    print(
        f"  Total images with metadata: {len(image_dataset.get_existing_metadata())}"
    )
    print(
        f"  Default confidence across batch: {image_dataset.meta(3).confidence}"
    )
    print(f"  Common tags on batch-created items: {image_dataset.meta(4).tags}")

    # Create another batch with different defaults, showing overwrite behavior
    additional_indices = [4, 5, 6, 7]  # Note: index 4 overlaps with first batch
    image_dataset.batch_create(
        indices=additional_indices,
        overwrite_existing=False,  # Won't overwrite existing metadata at index 4
        label="unknown",
        confidence=0.50,
        tags=["needs_review"],
    )

    print("\nAfter Second Batch Create:")
    print(f"  Index 4 tags (not overwritten): {image_dataset.meta(4).tags}")
    print(f"  Index 6 label (newly created): {image_dataset.meta(6).label}")
    print(
        f"  Index 6 already marked as augmented: {image_dataset.meta(6).is_augmented}"
    )

    # Display metadata for some images
    print("\nImage Metadata Examples:")
    for i in [0, 1, 2, 5]:
        meta = image_dataset.get_metadata(i)
        print(f"  Image {i} ({meta.filename}):")
        print(f"    Label: {meta.label or 'Unlabeled'}")
        if meta.confidence:
            print(f"    Confidence: {meta.confidence:.2f}")
        print(f"    Brightness: {meta.brightness:.2f}")
        print(f"    Augmented: {meta.is_augmented}")
        if meta.tags:
            print(f"    Tags: {', '.join(meta.tags)}")

    # Extract bright pet images
    bright_pets = [
        (i, meta.filename, meta.brightness)
        for i, meta in image_dataset.get_existing_metadata()
        if meta.brightness > 0.5 and meta.tags and "pet" in meta.tags
    ]

    print("\nBright pet images (brightness > 0.5):")
    for idx, filename, brightness in bright_pets:
        print(f"  {filename}: brightness = {brightness:.2f}")

    # Create a derived dataset with normalized images
    def normalize_image(img):
        return (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)

    normalized_dataset = image_dataset.with_updated_data(
        np.array([normalize_image(img) for img in images])
    )

    print("\nNormalized Images Dataset:")
    print("  Original vs. normalized brightness for first 5 images:")
    for i in range(5):
        original = image_dataset.get_metadata(i).brightness
        normalized = normalized_dataset.get_metadata(i).brightness
        print(f"  Image {i}: {original:.2f} → {normalized:.2f}")
    #########################################################################
    # Example 3: Scientific Experiment Data
    #########################################################################
    print("\n\n" + "-" * 80)
    print("EXAMPLE 3: SCIENTIFIC EXPERIMENT DATA".center(80))
    print("-" * 80)

    class ExperimentMetadata(BaseModel):
        experiment_id: str = ""
        date: datetime = datetime.now()
        researcher: str = ""
        temperature_c: Optional[float] = None
        pressure_kpa: Optional[float] = None
        peak_value: ComputedField = max_value_field()
        notes: str = ""

    # Create a 10x1000 array (10 experiments with 1000 measurements each)
    np.random.seed(42)
    experiment_data = np.random.normal(
        loc=np.linspace(0, 5, 10).reshape(-1, 1),
        scale=np.linspace(0.5, 1.5, 10).reshape(-1, 1),
        size=(10, 1000),
    )

    print(
        f"Experiment data shape: {experiment_data.shape} (10 experiments, 1000 measurements)"
    )

    # Create a MetadataArray for experiment data
    experiments = MetadataArray(
        data=experiment_data, metadata_class=ExperimentMetadata, major_axis=0
    )

    # Add metadata for experiments
    experiments.update_metadata(
        0,
        experiment_id="EXP001",
        date=datetime(2023, 1, 15),
        researcher="Alice",
        temperature_c=25.5,
        pressure_kpa=101.3,
        notes="Baseline experiment",
    )

    experiments.update_metadata(
        1,
        experiment_id="EXP002",
        date=datetime(2023, 1, 16),
        researcher="Bob",
        temperature_c=30.2,
        pressure_kpa=101.3,
    )

    experiments.update_metadata(
        2,
        experiment_id="EXP003",
        date=datetime(2023, 1, 18),
        researcher="Alice",
        temperature_c=22.8,
        pressure_kpa=100.9,
        notes="Modified parameters",
    )

    # Display experiment metadata
    print("\nExperiment Metadata:")
    for i in range(3):
        meta = experiments.get_metadata(i)
        print(f"  {meta.experiment_id} ({meta.date.strftime('%Y-%m-%d')}):")
        print(f"    Researcher: {meta.researcher}")
        print(
            f"    Conditions: {meta.temperature_c}°C, {meta.pressure_kpa} kPa"
        )
        print(f"    Peak value: {meta.peak_value:.2f}")
        if meta.notes:
            print(f"    Notes: {meta.notes}")

    # Find experiments with peaks above a threshold
    high_peak_experiments = [
        (meta.experiment_id, meta.peak_value, meta.date)
        for i, meta in experiments.get_existing_metadata()
        if meta.peak_value > 7.0
    ]

    print("\nExperiments with peak values > 7.0:")
    if high_peak_experiments:
        for exp_id, peak, date in high_peak_experiments:
            print(f"  {exp_id}: {peak:.2f} on {date.strftime('%Y-%m-%d')}")
    else:
        print("  None found")

    # Apply a temperature correction to one experiment
    def temperature_correction(data, scale_factor=0.05):
        """Apply temperature correction to data"""
        return data * (1 + scale_factor)

    corrected_experiments = experiments.with_updated_slice(
        0, temperature_correction(experiments.get_slice(0))
    )

    print("\nTemperature Correction Results:")
    original_peak = experiments.get_metadata(0).peak_value
    corrected_peak = corrected_experiments.get_metadata(0).peak_value
    print(f"  Original peak: {original_peak:.2f}")
    print(f"  Corrected peak: {corrected_peak:.2f}")
    print(f"  Difference: {corrected_peak - original_peak:.2f}")

    #########################################################################
    # Example 4: Multi-dimensional Tensor with Feature Metadata
    #########################################################################
    print("\n\n" + "-" * 80)
    print(
        "EXAMPLE 4: MULTI-DIMENSIONAL TENSOR WITH FEATURE METADATA".center(80)
    )
    print("-" * 80)

    class FeatureMetadata(BaseModel):
        name: str = ""
        description: str = ""
        data_type: str = "numeric"
        importance: Optional[float] = None
        feature_range: ComputedField = data_based_field(
            lambda x: (float(np.min(x)), float(np.max(x)))
        )
        is_categorical: bool = False

    # Create a 100x20 dataset (100 samples, 20 features)
    np.random.seed(42)
    feature_data = np.random.randn(100, 20)

    print(
        f"Feature dataset shape: {feature_data.shape} (100 samples, 20 features)"
    )

    # Create a MetadataArray with column metadata (each column is a feature)
    dataset = MetadataArray(
        data=feature_data, metadata_class=FeatureMetadata, major_axis=1
    )

    # Set metadata for features
    dataset.update_metadata(
        0, name="age", description="Age in years", importance=0.8
    )
    dataset.update_metadata(
        1, name="income", description="Annual income", importance=0.9
    )
    dataset.update_metadata(
        2,
        name="category",
        description="Product category",
        is_categorical=True,
        data_type="categorical",
    )
    dataset.update_metadata(
        3, name="height", description="Height in cm", importance=0.4
    )
    dataset.update_metadata(
        4, name="weight", description="Weight in kg", importance=0.5
    )

    # Display feature metadata
    print("\nFeature Metadata:")
    for i in range(5):  # Show first 5 features
        meta = dataset.get_metadata(i)
        print(f"  Feature {i}: {meta.name}")
        print(f"    Description: {meta.description}")
        print(f"    Type: {meta.data_type}")
        print(f"    Range: {meta.feature_range}")
        if meta.importance is not None:
            print(f"    Importance: {meta.importance:.2f}")
        print(f"    Categorical: {meta.is_categorical}")

    # Update data and automatically refresh computed fields
    print("\nUpdating Feature Data:")
    new_data = feature_data.copy()
    new_data[:, 0] = np.random.uniform(18, 65, size=100)  # Update age column
    updated_dataset = dataset.with_updated_data(new_data)

    # The feature_range for age will be automatically updated
    original_range = dataset.meta(0).feature_range
    new_range = updated_dataset.meta(0).feature_range
    print(f"  Feature: {dataset.meta(0).name}")
    print(f"  Original range: {original_range}")
    print(f"  New range after update: {new_range}")

    # Find important features
    important_features = [
        (i, meta.name, meta.importance)
        for i, meta in dataset.get_existing_metadata()
        if meta.importance is not None and meta.importance > 0.7
    ]

    print("\nImportant Features (importance > 0.7):")
    for idx, name, importance in important_features:
        print(f"  {name}: {importance:.2f}")

    print("\n" + "=" * 80)
    print("End of Examples".center(80))
    print("=" * 80)


if __name__ == "__main__":
    usage_example()
