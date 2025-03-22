import copy
from typing import Optional, Type, TypeVar, Callable, Any, cast
import numpy as np
from pydantic import BaseModel
from warnings import warn
from contextlib import contextmanager

from .utils import (
    internal_method,
    get_array_slice,
    UnsafeDataAccessWarning,
    _internal_access,
)
from .computed import (
    ComputedFieldContext,
    COMPUTED_FACTORY_KEY,
    COMPUTED_WRITABLE_KEY,
)

T = TypeVar("T", bound=BaseModel)


class DantArray:
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
            major_axis: Axis for which metadata is tracked (can be negative to index from the end)
            metadata: Optional pre-existing metadata list

        Raises:
            ValueError: If major_axis is invalid or metadata length is inconsistent
        """
        self._data_unsafe: np.ndarray = np.array([])
        self.metadata_class = metadata_class

        # Convert to numpy array early to access ndim
        data_array = np.asarray(data)

        # Handle negative major_axis (normalize to positive index)
        normalized_axis = (
            major_axis if major_axis >= 0 else data_array.ndim + major_axis
        )

        # Validate normalized_axis is within bounds
        if (
            not isinstance(normalized_axis, int)
            or normalized_axis < 0
            or normalized_axis >= data_array.ndim
        ):
            raise ValueError(
                f"major_axis must be an integer in range [-{data_array.ndim}, {data_array.ndim - 1}], got {major_axis}"
            )

        self.major_axis = normalized_axis

        # Set data (this will initialize _metadata through the setter)
        self.data = data

        # Initialize metadata if provided
        if metadata is not None:
            if len(metadata) != self.data.shape[self.major_axis]:
                raise ValueError(
                    f"Metadata length ({len(metadata)}) must match the major dimension size ({self.data.shape[self.major_axis]})"
                )
            self._metadata = list(metadata)

    def __getattribute__(self, name):
        """Catch direct access to _data_unsafe"""
        if (
            name == "_data_unsafe"
            and not getattr(_internal_access, "active", False)
            and object.__getattribute__(self, "_data_unsafe") is not None
        ):
            warn(
                "Direct access to _data_unsafe detected. This bypasses computed field updates. Consider using:\n"
                "1. array.data for read-only access\n"
                "2. with array.edit_data() as data: ... for mutations\n"
                "3. array.with_updated_data(new_data) for immutable updates\n"
                "4. array.update_slice(idx, new_slice) for single slice updates",
                UnsafeDataAccessWarning,
                stacklevel=2,
            )
        return object.__getattribute__(self, name)

    @property
    @internal_method
    def data(self) -> np.ndarray:
        """Get a read-only copy of the underlying numpy array."""
        return self._data_unsafe.copy()

    @data.setter
    @internal_method
    def data(self, new_data: np.ndarray) -> None:
        """
        Set the underlying numpy array, validating size consistency with metadata and refreshing computed fields automatically
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
                self._data_unsafe.shape[self.major_axis]
                if self._data_unsafe.size > 0
                else 0
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
            # Initialize metadata container for first time
            major_dim_size = new_data.shape[self.major_axis]
            if major_dim_size < 0:  # Implicit validation
                raise ValueError(
                    f"Invalid major dimension size: {major_dim_size}"
                )
            self._metadata = [None] * major_dim_size

        self._data_unsafe = new_data
        self.refresh_computed_fields()

    @contextmanager
    def edit_data(self, refresh: bool = True):
        """
        Context manager for safely editing the underlying data with automatic refresh.

        Args:
            refresh: Whether to automatically refresh computed fields after edits

        Example:
            with array.edit_data() as data:
                data[0, 0] = 5  # Modify data safely
        """
        try:
            yield self._data_unsafe
        finally:
            if refresh:
                self.refresh_computed_fields()

    def update_slice(self, idx: int, new_slice: np.ndarray) -> None:
        """
        Update a slice along the major axis with automatic refresh of computed fields.

        Args:
            idx: Index of the slice to update
            new_slice: New data for the slice
        """
        self._validate_index(idx)

        with self.edit_data(refresh=True) as data:
            if self.major_axis == 0:
                data[idx] = new_slice
            elif self.major_axis == 1:
                data[:, idx] = new_slice
            else:
                slices = tuple(
                    idx if i == self.major_axis else slice(None)
                    for i in range(len(data.shape))
                )
                data[slices] = new_slice

    @internal_method
    def _validate_index(self, idx: int) -> None:
        """
        Ensure index is within bounds of major dimension

        Raises:
            IndexError: If index is out of bounds
        """
        major_dim_size = self._data_unsafe.shape[self.major_axis]
        if not 0 <= idx < major_dim_size:
            raise IndexError(
                f"Index {idx} out of bounds for axis {self.major_axis} with size {major_dim_size}"
            )

    def _is_computed_field(self, field_or_info: Any) -> bool:
        """
        Check if a field is computed based on field metadata

        Args:
            field_or_info: Either a field name, field value, or field_info object

        Returns:
            True if this is a computed field
        """
        if isinstance(field_or_info, str):
            if field_or_info not in self.metadata_class.model_fields:
                return False
            field_info = self.metadata_class.model_fields[field_or_info]
        else:
            field_info = field_or_info

        if hasattr(field_info, "json_schema_extra"):
            extra = field_info.json_schema_extra or {}
        else:
            return False

        return COMPUTED_FACTORY_KEY in extra

    def _resolve_computed_fields(self, instance: T, idx: int) -> None:
        """
        Resolve computed fields for a metadata instance, respecting user overrides
        """
        context = ComputedFieldContext(idx, self.data, self.major_axis)

        # Initialize storage for computed field factories if needed
        if not hasattr(instance, "_computed_field_factories"):
            instance._computed_field_factories = {}

        # Process each field in the model
        for field_name, field_info in instance.model_fields.items():
            extra = getattr(field_info, "json_schema_extra", {}) or {}

            # Check if this is a computed field
            if self._is_computed_field(field_info):
                factory = extra[COMPUTED_FACTORY_KEY]
                writable = extra.get(COMPUTED_WRITABLE_KEY, False)

                # Ensure we track override status
                if field_name not in instance._computed_field_factories:
                    instance._computed_field_factories[field_name] = {
                        "factory": factory,
                        "writable": writable,
                        "user_overridden": False,
                        "last_computed_value": None,
                    }
                meta_info = instance._computed_field_factories[field_name]

                # If field not writable, always recompute; if writable but not overridden, still recompute
                if (not writable) or (not meta_info["user_overridden"]):
                    computed_value = factory(context)
                    setattr(instance, field_name, computed_value)
                    meta_info["last_computed_value"] = computed_value

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
        self, idx: int, create_default: bool = False
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
        Update specific metadata fields for the given index.

        For writable computed fields, if the user-provided value differs from
        the last automatically computed value, we consider it a user override.
        """
        self._validate_index(idx)

        # Get existing metadata or create default
        metadata = self.get_metadata(idx, create_default)
        if metadata is None:
            return  # No metadata and not creating default

        # Validate field names
        for field in kwargs:
            if field not in metadata.model_fields:
                valid_fields = list(metadata.model_fields.keys())
                raise ValueError(
                    f"Invalid field '{field}'. Valid fields: {valid_fields}"
                )

        # Prepare a new metadata object by copying/updating
        old_metadata = metadata
        new_metadata = metadata.model_copy(update=kwargs)

        # Replicate computed field factories so we keep override info
        if not hasattr(old_metadata, "_computed_field_factories"):
            old_metadata._computed_field_factories = {}
        new_metadata._computed_field_factories = copy.deepcopy(
            old_metadata._computed_field_factories
        )

        # Check for computed fields that shouldn't be updated or mark them overridden
        for field_name, new_value in kwargs.items():
            if field_name in new_metadata._computed_field_factories:
                info = new_metadata._computed_field_factories[field_name]
                if not info["writable"]:
                    # Non-writable computed field => error
                    raise ValueError(
                        f"Field '{field_name}' is computed and not writable. "
                        f"You cannot directly update this field."
                    )
                else:
                    # If user sets a value that differs from the last auto-computed,
                    # we consider it an override
                    last_val = info.get("last_computed_value", None)
                    if last_val is None or new_value != last_val:
                        info["user_overridden"] = True

        # Store the updated metadata
        self._metadata[idx] = new_metadata

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

    @internal_method
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

    @internal_method
    def get_slice(self, idx: int) -> np.ndarray:
        """
        Get a slice of data along the major axis

        Args:
            idx: Index to get slice for

        Returns:
            Array slice at the specified index
        """
        self._validate_index(idx)
        return get_array_slice(self._data_unsafe, idx, self.major_axis)

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

    @internal_method  # not strictly needed, but future-proofing
    def refresh_computed_fields(self) -> None:
        """Refresh all computed fields in all metadata instances."""
        for idx in range(self.data.shape[self.major_axis]):
            metadata = self.get_metadata(idx, create_default=False)
            if metadata is None:
                continue

            # Skip if no computed fields are stored
            if not hasattr(metadata, "_computed_field_factories"):
                continue

            self._resolve_computed_fields(metadata, idx)

    @property
    @internal_method
    def shape(self) -> tuple:
        """Get the shape of the underlying array"""
        return self._data_unsafe.shape

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

    def copy(self, deep_metadata: bool = False) -> "DantArray":
        """
        Create a copy of this DantArray

        vbnet
        Copy code
        Args:
            deep_metadata: Whether to perform a deep copy of metadata (True)
                        or reuse existing references (False).
                        - True => new metadata objects with no shared state
                        - False => new list object, but references to original
                                    metadata objects

        Returns:
            New DantArray with copied data and shallow or deep-copied metadata
        """
        # Always copy the NumPy array data
        new_data = self.data.copy()

        if deep_metadata:
            # Deep copy: create entirely new metadata model objects
            copied_metadata = []
            for m in self._metadata:
                if m is not None:
                    copied_metadata.append(m.model_copy(deep=True))
                else:
                    copied_metadata.append(None)
        else:
            # Shallow copy: new list, but referencing the same metadata objects
            copied_metadata = list(self._metadata)

        return DantArray(
            data=new_data,
            metadata_class=self.metadata_class,
            major_axis=self.major_axis,
            metadata=copied_metadata,
        )

    # Immutable modification methods
    def with_updated_data(self, new_data: np.ndarray) -> "DantArray":
        """
        Create a new DantArray with updated data and recomputed fields

        Args:
            new_data: New array data

        Returns:
            New DantArray instance with updated data and refreshed computed fields
        """
        result = self.copy()
        result.data = new_data
        result.refresh_computed_fields()
        return result

    def with_updated_slice(
        self, idx: int, new_slice: np.ndarray
    ) -> "DantArray":
        """
        Create a new DantArray with an updated slice and recomputed fields

        Args:
            idx: Index of slice to update
            new_slice: New data for the slice

        Returns:
            New DantArray instance with updated slice and refreshed computed fields
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
            context = ComputedFieldContext(idx, result.data, result.major_axis)
            for field_name, info in metadata._computed_field_factories.items():
                if (not info["writable"]) or (not info["user_overridden"]):
                    computed_value = info["factory"](context)
                    setattr(metadata, field_name, computed_value)
                    info["last_computed_value"] = computed_value

        return result

    def re_enable_computed_field(
        self, idx: int, field_name: str, recompute: bool = True
    ) -> None:
        """
        Re-enable automatic recomputation for a previously overridden writable computed field.

        Optionally triggers an immediate recomputation of the field.

        Args:
            idx: The index of the metadata item
            field_name: The name of the computed field
            recompute: Whether to immediately recompute the field after re-enabling

        Raises:
            ValueError: If the field is not found or is not writable
        """
        self._validate_index(idx)
        metadata = self.get_metadata(idx, create_default=False)
        if metadata is None:
            raise ValueError(f"No metadata exists at index {idx}.")

        if not hasattr(metadata, "_computed_field_factories"):
            raise ValueError("No computed fields exist in this metadata.")

        field_info = metadata._computed_field_factories.get(field_name)
        if field_info is None:
            raise ValueError(
                f"Field '{field_name}' is not recognized as computed."
            )
        if not field_info["writable"]:
            raise ValueError(
                f"Cannot re-enable a non-writable field '{field_name}'."
            )

        # Reset override status
        field_info["user_overridden"] = False

        # Optionally recompute immediately
        if recompute:
            context = ComputedFieldContext(idx, self.data, self.major_axis)
            new_value = field_info["factory"](context)
            setattr(metadata, field_name, new_value)
            field_info["last_computed_value"] = new_value


class MetadataAccessor:
    """Helper class for access to metadata fields"""

    def __init__(self, parent: DantArray, idx: int):
        """
        Initialize a metadata accessor

        Args:
            parent: Parent DantArray
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
