"""
Example usage of dantArrays package showing various use cases.
"""

import numpy as np
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

# Import from the refactored package
from dantArrays import (
    DantArray,
    mean_value_field,
    max_value_field,
    index_field,
    data_based_field,
)


def usage_example():
    """Demonstrate various use cases for the DantArray class"""
    print("\n" + "=" * 80)
    print("DantArray Usage Examples".center(80))
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
        max_price: float = max_value_field()
        mean_price: float = mean_value_field()

    # Create sample price data for 3 stocks over 100 days
    np.random.seed(42)  # For reproducibility
    price_data = np.random.normal(
        loc=[[100], [50], [200]], scale=[[10], [5], [20]], size=(3, 100)
    )

    print(f"Stock price data shape: {price_data.shape} (3 stocks, 100 days)")

    # Create a DantArray with row metadata (each row is a stock)
    stocks = DantArray(
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
        filename: str = index_field(prefix="img_", suffix=".jpg")
        label: Optional[str] = None
        confidence: Optional[float] = None
        brightness: float = data_based_field(lambda x: float(np.mean(x)))
        is_augmented: bool = False
        tags: List[str] = Field(default_factory=list)

    # Create a 20-image dataset with 64x64 images (grayscale)
    np.random.seed(42)
    images = np.random.rand(20, 64, 64)

    print(f"Image dataset shape: {images.shape} (20 images, 64x64 pixels)")

    # Create a DantArray for the image dataset
    image_dataset = DantArray(
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
        peak_value: float = max_value_field()
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

    # Create a DantArray for experiment data
    experiments = DantArray(
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
        feature_range: tuple = data_based_field(
            lambda x: (float(np.min(x)), float(np.max(x)))
        )
        is_categorical: bool = False

    # Create a 100x20 dataset (100 samples, 20 features)
    np.random.seed(42)
    feature_data = np.random.randn(100, 20)

    print(
        f"Feature dataset shape: {feature_data.shape} (100 samples, 20 features)"
    )

    # Create a DantArray with column metadata (each column is a feature)
    dataset = DantArray(
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
