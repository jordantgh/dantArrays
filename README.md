# dantArrays: Enhanced NumPy Arrays with Rich, Computed Metadata

---

## Overview

`dantArrays` extends standard NumPy arrays by embedding rich, computed metadata directly into array slices, rows, or columns. This integration allows you to keep data tightly coupled with its descriptive context, computed metrics, or dynamic annotations—without losing NumPy's efficiency and versatility.

---

## Key Features

- **Rich Metadata Integration:** Attach structured metadata (using Pydantic models) to slices of NumPy arrays.
- **Automatic Computations:** Define metadata fields computed directly from array data.
- **Data Safety:** Safe mechanisms to access and modify underlying array data without corrupting computed metadata.
- **Flexibility:** Easily update array slices or metadata, automatically recalculating dependent computed fields.
- **Usability:** Clear, Pythonic API for both simple and complex data manipulation tasks.

---

## Installation

```bash
pip install dantarrays
```

---

## Quickstart Example

Here's how easily you can add metadata to your arrays:

```python
import numpy as np
from pydantic import BaseModel
from dantarrays import DantArray, mean_value_field

class StockMetadata(BaseModel):
    symbol: str
    mean_price: float = mean_value_field()

price_data = np.array([
    [100, 101, 102],
    [50, 49, 51],
])

stocks = DantArray(data=price_data, metadata_class=StockMetadata, major_axis=0)

stocks.update_metadata(0, symbol="AAPL")
stocks.update_metadata(1, symbol="MSFT")

print(stocks.meta(0).mean_price)  # Automatically computed mean price
# Output: 101.0
```

---

## Usage

### Defining Metadata Models

Use Pydantic models to define structured metadata:

```python
from pydantic import BaseModel
from dantarrays import max_value_field, mean_value_field

class ExperimentMetadata(BaseModel):
    experiment_id: str
    max_measurement: float = max_value_field()
    avg_measurement: float = mean_value_field()
```

### Creating a DantArray

```python
import numpy as np
from dantarrays import DantArray

data = np.random.rand(5, 10)
experiments = DantArray(data=data, metadata_class=ExperimentMetadata, major_axis=0)
```

### Updating Metadata

```python
experiments.update_metadata(0, experiment_id="EXP001")
```

### Accessing Metadata

```python
metadata = experiments.get_metadata(0)
print(metadata.max_measurement)
```

### Using the `.meta()` API for Convenient Field Access

The `.meta()` method provides ergonomic, attribute-style access to metadata fields:

```python
experiments.meta(0).experiment_id = "EXP002"
print(experiments.meta(0).experiment_id)
# Output: EXP002
```

You can also update multiple fields conveniently:

```python
experiments.meta(0).update(experiment_id="EXP003", avg_measurement=0.75)
```

### Editing Data Safely

```python
with experiments.edit_data() as data:
    data[0, :] = np.random.rand(10)
```

### Immutable Updates

```python
updated_experiments = experiments.with_updated_slice(0, np.zeros(10))
```

---

## Advanced Examples

### Batch Metadata Creation

```python
indices = [0, 1, 2]
experiments.batch_create(indices, experiment_id="default")
```

### Computed Fields Based on Data

Create a metadata field dynamically computed from the data slice:

```python
from dantarrays import data_based_field

class ImageMetadata(BaseModel):
    brightness: float = data_based_field(lambda slice: float(np.mean(slice)))
```

### Accessing Full Metadata

```python
for idx, meta in experiments.get_existing_metadata():
    print(f"Experiment {meta.experiment_id} avg: {meta.avg_measurement}")
```

### Data Validation and Safety

Direct access to the internal data array triggers warnings, ensuring safe data manipulation:

```python
experiments._data_unsafe  # Warns about unsafe access
```

### Contextual Computations

Computed fields have access to context about their array slice:

```python
from dantarrays import ComputedFieldContext

class CustomComputation:
    def __call__(self, ctx: ComputedFieldContext):
        return ctx.size * np.mean(ctx.get_slice())
```

---

## API Reference

### `DantArray`

- `.update_metadata(idx, **kwargs)`: Update metadata fields.
- `.get_metadata(idx)`: Retrieve metadata.
- `.meta(idx)`: Convenient, attribute-style metadata accessor.
- `.batch_create(indices, **kwargs)`: Batch create metadata entries.
- `.with_updated_data(new_data)`: Immutable data update.
- `.apply_with_metadata(func)`: Apply function to array slices and metadata.
- `.refresh_computed_fields()`: Refresh computed fields after data changes.

---

## Use Cases

- Financial time-series analysis
- Image datasets with annotations
- Scientific experiment tracking
- Feature metadata management in machine learning
