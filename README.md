# ObjectDetect3D

A repository for 3D object detection in volumetric images using supervised learning approaches.

## Project Overview

This project focuses on detecting objects in 3D images by predicting bounding boxes. The implementation generates synthetic data with Gaussian blobs and uses a 3D CNN to predict a 6-dimensional vector representing:
- 3D center position (x, y, z)
- Box side lengths (width, height, depth)

The model is trained and evaluated across different Signal-to-Noise Ratio (SNR) levels to assess robustness to noise.

## Features

- Synthetic 3D data generation with controllable SNR levels
- 3D CNN model for bounding box prediction
- Performance evaluation across different noise conditions
- Visualization tools for 3D predictions

## Dataset

The implementation uses synthetic data consisting of Gaussian blobs embedded in 3D volumes:

```python
# Generate synthetic blob dataset
num_samples = 1000
data_cutoff = []
labels_cutoff = []
for _ in range(num_samples):
   blob, label = generate_gaussian_blob_with_cutoff_pixel_labels(shape=IMG_SHAPE, snr_db=SNR_DB)
   data_cutoff.append(blob)
   labels_cutoff.append(label)
```

Key characteristics:
- Gaussian blobs with randomized positions and sizes
- Adjustable SNR levels (-10dB, -20dB, etc.) to simulate different imaging conditions
- Dataset split into training (80%), validation (10%), and test (10%) sets

## Model Architecture

The model uses a 3D CNN architecture to predict bounding boxes:

```python
class GaussianBlobNet(nn.Module):
    def __init__(self, input_size):
        super(GaussianBlobNet, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )

        # Calculate output size after conv layers
        conv_output_size = input_size // (2 ** 3)  # 3 max pooling layers
        flattened_size = 64 * conv_output_size * conv_output_size * conv_output_size

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Linear(128, 6)  # Output 6 bounding box parameters
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
```

## Installation

```bash
# Clone this repository
git clone https://github.com/parks9/ObjectDetect3D.git
cd ObjectDetect3D

# Set up Python environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Data Generation

```python
from blob_maker import generate_gaussian_blob_with_cutoff_pixel_labels

# Generate synthetic data with specific SNR
blob, label = generate_gaussian_blob_with_cutoff_pixel_labels(
    shape=(64, 64, 64), 
    snr_db=-10
)

# The label includes center coordinates and box dimensions
x_center, y_center, z_center, x_width, y_width, z_width = label
```

### Training

```python
# Initialize model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GaussianBlobNet(input_size=IMG_SHAPE[0]).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop (simplified)
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
    # Save best model
    if avg_val_loss < best_val_loss:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, model_save_path)
```

### Evaluation

```python
from test_model import evaluate_model, print_metrics

# Load the best model
model = GaussianBlobNet(input_size=IMG_SHAPE[0]).to(device)
checkpoint = torch.load(model_save_path)
model.load_state_dict(checkpoint['model_state_dict'])

# Evaluate model
metrics = evaluate_model(model, test_loader, device)
print_metrics(metrics)

# Example output:
# === 3D Bounding Box Metrics ===
# Center Error: 3.124 ± 1.852
# Size Error: 5.678 ± 2.341
# Volume Error: 18.5%
# Mean IoU: 0.721 ± 0.158
# 
# Precision at IoU thresholds:
# IoU > 0.25: 97.8%
# IoU > 0.50: 85.6%
# IoU > 0.75: 42.3%
```

### Visualization

```python
# Visualize ground truth and predicted boxes
def plot_comparison(blob_data, ground_truth, prediction_label):
    # Create figure with 3 planes (XY, XZ, YZ)
    figure, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot each plane with ground truth (green) and prediction (red)
    ax[0].imshow(blob_data[:, :, int(z_center)].T, cmap="jet")
    ax[0].set_title("XY Plane")
    # Add bounding boxes...
    
    plt.tight_layout()
    plt.show()
```

## Results

The model's performance is evaluated across different SNR levels using several metrics:
- Center Error: L2 distance between predicted and ground truth centers
- Size Error: L2 distance between predicted and ground truth dimensions
- Volume Error: Relative error in predicted box volume
- IoU (Intersection over Union): Measure of 3D bounding box overlap
- Precision at various IoU thresholds (0.25, 0.50, 0.75)

## Future Work

- Implement more sophisticated network architectures
- Extend to real-world 3D imaging data
- Incorporate instance segmentation alongside bounding box prediction
- Explore semi-supervised and self-supervised approaches

## License

[MIT](LICENSE)

## Contact

For questions or collaborations, please open an issue on this repository.