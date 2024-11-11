# SpectralCNNAutoencoder

`SpectralCNNAutoencoder` is a Python project designed for analyzing spectral data using a Convolutional Neural Network (CNN) autoencoder. This tool identifies and visualizes anomalies in spectral data by reconstructing spectra, highlighting anomalous regions, and overlaying associated galaxy images from the Sloan Digital Sky Survey (SDSS).

## Features

- **Autoencoder Model for Spectra**: Train a CNN autoencoder to compress and reconstruct spectral data.
- **Anomaly Detection**: Detects anomalies based on reconstruction residuals and custom thresholds, allowing fine control over anomaly sensitivity.
- **Visualization**: 
  - Generates side-by-side plots of original and reconstructed spectra, with highlighted residuals.
  - Provides options to plot only anomalous spectra.
  - Fetches and displays SDSS galaxy images next to spectra.
  - Saves plots with unique file names to prevent overwrites.
- **Customizable Parameters**: Control thresholds, plotting options, and sample size.

## Getting Started

### Prerequisites

- **Python 3.x**
- **Dependencies**: Install the required packages:
  ```bash
  pip install -r requirements.txt
  ```

### Installation

Clone the repository:
```bash
git clone https://github.com/your-username/SpectralCNNAutoencoder.git
cd SpectralCNNAutoencoder
```

Install dependencies if you haven't already:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Model Initialization and Training

The autoencoder is set up and trained on the spectral data. After training, it is used to reconstruct the spectra.

```python
# Initialize and reset the model
autoencoder = CNNAutoencoderWithSkip()
reset_training_state(autoencoder, optim.Adam(autoencoder.parameters()))

# Train the model
train_autoencoder(autoencoder, all_fluxes_tensor, mask_tensor)
```

### 2. Anomaly Detection

After model evaluation, detect anomalies by comparing the original and reconstructed spectra.

```python
# Evaluate model and perform anomaly detection
autoencoder.eval()
reconstructed_fluxes = autoencoder(all_fluxes_tensor).detach().numpy().squeeze()

# Detect anomalies
(anomalous_regions, spectrum_anomalies, anomaly_metadata,
 abs_residual_threshold, rel_residual_threshold, range_mismatch_factor) = detect_anomalous_regions(
    original_fluxes=all_fluxes_tensor.numpy().squeeze(),
    reconstructed_fluxes=reconstructed_fluxes,
    window_size=50,
    percentile_threshold=95,
    range_mismatch_factor=1.5,
    overall_anomaly_threshold=0.3
)
```

### 3. Visualization and Plotting

The `plot_combined_spectra` function visualizes original and reconstructed spectra, with residuals and SDSS images side-by-side. Set `only_anomalous=True` to display only anomalous spectra.

```python
plot_combined_spectra(
    original_fluxes=all_fluxes_tensor.numpy().squeeze(),
    reconstructed_fluxes=reconstructed_fluxes,
    anomalous_regions=anomalous_regions,
    spectrum_anomalies=spectrum_anomalies,
    zpix_cat=zpix_cat,
    wavelengths=None,       # Optional: provide wavelength data if available
    save_directory=IMG_DIR, # Specify the directory for saved images
    only_anomalous=True     # Set to True to plot only anomalous spectra
)
```

#### File Saving

The `create_save_path` function ensures saved files have unique names by appending a sequential numeric suffix.

```python
# Example usage within plotting
base_filename = "combined_spectra_with_images"
save_path = create_save_path(IMG_DIR, base_filename)
```

### 4. Metadata Saving

Save anomaly detection thresholds, sampling info, and metadata for reproducibility.

```python
save_sampling_info(
    anomalous_indices, plot_indices, seed, anomaly_metadata, 
    abs_residual_threshold, rel_residual_threshold, range_mismatch_factor, 
    json_directory=JSON_DIR, zpix_cat=zpix_cat
)
```

## Directory Structure

- `models/`: Contains the model architecture files.
- `utils/`: Helper functions for data processing, anomaly detection, and plotting.
- `output_images/`: Default directory for saved plots.
- `metadata/`: Stores metadata files with anomaly information.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for bug fixes or feature requests.
