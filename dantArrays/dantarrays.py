from typing import Optional, Type, TypeVar, List, Callable
import numpy as np
from pydantic import BaseModel


# Define a context generator for default values
class MetaDefault:
    """Smart default generator with access to array context"""

    def __init__(self, factory_func: Optional[Callable] = None):
        self.factory_func = factory_func or (
            lambda idx, shape, axis: f"Item_{idx}"
        )

    def __call__(self, idx, shape, axis):
        return self.factory_func(idx, shape, axis)


# Convenience function to generate index-based names
def index_field(prefix: str = "Item_", suffix: str = ""):
    return MetaDefault(lambda idx, shape, axis: f"{prefix}{idx}{suffix}")


# Convenience function for dimension-based defaults
def dimension_field():
    return MetaDefault(
        lambda idx, shape, axis: f"Dimension {axis} of size {shape[axis]}"
    )


T = TypeVar("T", bound=BaseModel)


class MetadataArray:
    def __init__(
        self,
        data: np.ndarray,
        metadata_class: Type[T],
        major_axis: int = 0,
        metadata: Optional[List[T]] = None,
    ):
        """
        Wrapper for numpy arrays with rich metadata for rows or columns.

        Args:
            data: NumPy array to wrap
            metadata_class: Pydantic model class to use for metadata
            major_axis: 0 for row metadata, 1 for column metadata
            metadata: Optional pre-existing metadata list
        """
        self.data = np.asarray(data)
        self.major_axis = major_axis
        self.metadata_class = metadata_class

        # Size of the dimension we're attaching metadata to
        major_dim_size = self.data.shape[major_axis]

        # Initialize metadata container
        if metadata is None:
            self._metadata = [None] * major_dim_size
        else:
            if len(metadata) != major_dim_size:
                raise ValueError(
                    f"Metadata length ({len(metadata)}) must match the major dimension size ({major_dim_size})"
                )
            self._metadata = metadata

    def _resolve_defaults(self, instance: T, idx: int) -> None:
        """Resolve any MetaDefault objects in the instance"""
        for field_name in instance.model_fields.keys():
            value = getattr(instance, field_name)
            if isinstance(value, MetaDefault):
                setattr(
                    instance,
                    field_name,
                    value(idx, self.data.shape, self.major_axis),
                )

    def __getitem__(self, idx):
        """Support for standard NumPy indexing"""
        return self.data[idx]

    def get_metadata(self, idx: int) -> T:
        """Get metadata for a specific index, creating default if needed"""
        if self._metadata[idx] is None:
            # Create default metadata
            self._metadata[idx] = self.metadata_class()
            # Resolve any smart defaults
            self._resolve_defaults(self._metadata[idx], idx)

        return self._metadata[idx]

    def set_metadata(self, idx: int, metadata: T) -> None:
        """Set complete metadata for a specific index"""
        if not isinstance(metadata, self.metadata_class):
            raise TypeError(
                f"Metadata must be a {self.metadata_class.__name__} object"
            )
        self._metadata[idx] = metadata
        # Resolve any smart defaults in the new metadata
        self._resolve_defaults(self._metadata[idx], idx)

    def update_metadata(self, idx: int, **kwargs) -> None:
        """Update specific metadata fields for the given index"""
        # Get existing metadata or create default
        metadata = self.get_metadata(idx)

        # Process any MetaDefault values in the updates
        for field, value in list(kwargs.items()):
            if isinstance(value, MetaDefault):
                kwargs[field] = value(idx, self.data.shape, self.major_axis)

        # Use Pydantic's model_copy with update
        self._metadata[idx] = metadata.model_copy(update=kwargs)

    def batch_update(self, indices, **kwargs) -> None:
        """Update metadata for multiple indices at once"""
        for idx in indices:
            self.update_metadata(idx, **kwargs)

    def apply_with_metadata(self, func):
        """Apply a function to each slice along major axis with its metadata"""
        results = []
        for i in range(self.data.shape[self.major_axis]):
            # Extract the slice
            if self.major_axis == 0:
                slice_data = self.data[i]
            else:
                slice_data = self.data[:, i]

            results.append(func(slice_data, self.get_metadata(i)))
        return results

    @property
    def metadata(self) -> List[T]:
        """Return all metadata (creating defaults where needed)"""
        return [
            self.get_metadata(i)
            for i in range(self.data.shape[self.major_axis])
        ]

    @property
    def shape(self):
        return self.data.shape

    def meta(self, idx: int):
        """Return a metadata accessor for easier field-by-field updates"""
        return MetadataAccessor(self, idx)


class MetadataAccessor:
    """Helper class for ergonomic access to metadata fields"""

    def __init__(self, parent: MetadataArray, idx: int):
        self._parent = parent
        self._idx = idx

    def __getattr__(self, name: str):
        """Access metadata fields as attributes"""
        if name.startswith("_"):
            return super().__getattr__(name)

        metadata = self._parent.get_metadata(self._idx)
        if name in metadata.model_fields:
            return getattr(metadata, name)
        raise AttributeError(
            f"{metadata.__class__.__name__} has no attribute '{name}'"
        )

    def __setattr__(self, name: str, value):
        """Set metadata fields as attributes"""
        if name in ["_parent", "_idx"]:
            super().__setattr__(name, value)
            return

        # Update the specific field
        self._parent.update_metadata(self._idx, **{name: value})

    def update(self, **kwargs):
        """Update multiple fields at once"""
        self._parent.update_metadata(self._idx, **kwargs)
