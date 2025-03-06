import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn

def generate_gaussian_blob_with_cutoff_pixel_labels(shape=(32, 32, 32), center=None, sigma=None, snr_db=None):
    """
    Generate a 3D Gaussian blob with background noise specified in dB SNR.

    Parameters:
    -----------
    shape : tuple
        Shape of the 3D volume (default: (32, 32, 32))
    center : ndarray or None
        Center coordinates of the blob. If None, randomly generated.
    sigma : ndarray or None
        Standard deviations of the blob. If None, randomly generated.
    snr_db : float or None
        Signal-to-Noise Ratio in decibels. If None, no noise is added.
        Typical values might be 20 dB (good signal), 10 dB (moderate noise),
        0 dB (signal power equals noise power), -10 dB (very noisy)

    Returns:
    --------
    tuple : (blob_values, label_target)
        blob_values : ndarray
            3D array containing the Gaussian blob with noise
        label_target : tuple
            (center_x, center_y, center_z, width_x, width_y, width_z)
    """
    x = np.linspace(0, shape[0] - 1, shape[0])
    y = np.linspace(0, shape[1] - 1, shape[1])
    z = np.linspace(0, shape[2] - 1, shape[2])
    x_mesh, y_mesh, z_mesh = np.meshgrid(x, y, z, indexing='ij')

    # Randomize center and sigma if not provided
    if center is None:
        center = np.random.uniform(shape[0] * 0.10, shape[0] * 0.90, size=3)
    if sigma is None:
        sigma = np.random.uniform(0.01 * shape[0], 0.1 * shape[0], size=3)

    # Compute Gaussian
    blob_values = np.exp(-((x_mesh - center[0]) ** 2 / (2 * sigma[0] ** 2) +
                           (y_mesh - center[1]) ** 2 / (2 * sigma[1] ** 2) +
                           (z_mesh - center[2]) ** 2 / (2 * sigma[2] ** 2)))

    # Normalize
    blob_values /= np.max(blob_values)

    # Apply cutoff: set values beyond 2 sigma to zero
    mask = ((x_mesh - center[0]) ** 2 / (sigma[0] ** 2) +
            (y_mesh - center[1]) ** 2 / (sigma[1] ** 2) +
            (z_mesh - center[2]) ** 2 / (sigma[2] ** 2)) <= 4  # 2 sigma squared is 4
    blob_values *= mask

    # Add noise if SNR is specified
    if snr_db is not None:
        # Calculate signal power
        signal_power = np.mean(blob_values ** 2)

        # Calculate required noise power based on SNR
        snr_linear = 10 ** (snr_db / 10)  # Convert dB to linear scale
        noise_power = signal_power / snr_linear

        # Generate Gaussian noise with calculated power
        noise = np.random.normal(0, np.sqrt(noise_power), shape)

        # Add noise to signal
        blob_values = blob_values + noise

        # Clip negative values to zero (optional, depending on your needs)
        blob_values = np.clip(blob_values, 0, None)

        # Re-normalize after adding noise
        if np.max(blob_values) > 0:  # Avoid division by zero
            blob_values /= np.max(blob_values)

    # Labels in pixel units: (center_x, center_y, center_z, width_x, width_y, width_z)
    label_target = (int(center[0]), int(center[1]), int(center[2]),
                    int(4 * sigma[0]), int(4 * sigma[1]), int(4 * sigma[2]))

    return blob_values, label_target


# Define a PyTorch Dataset class
class GaussianBlobDataset(Dataset):
   def __init__(self, data, labels):
       self.data = torch.tensor(data, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
       self.labels = torch.tensor(labels, dtype=torch.float32)
   def __len__(self):
       return len(self.data)
   def __getitem__(self, idx):
       return self.data[idx], self.labels[idx]


class GaussianBlobNet(nn.Module):
    def __init__(self, input_size):
        super(GaussianBlobNet, self).__init__()

        # Validate input size
        if not isinstance(input_size, int):
            raise ValueError("input_size must be an integer")

        self.input_size = input_size

        # Conv layers remain the same
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
        # After each MaxPool3d(2), size is halved
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