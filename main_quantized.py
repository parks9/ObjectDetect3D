import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import brevitas.nn as qnn
from brevitas.export import export_qonnx

# Import the data generation function and dataset class
from blob_maker import generate_gaussian_blob_with_cutoff_pixel_labels, GaussianBlobDataset


# Define the quantized model directly here to avoid import issues
class QuantGaussianBlobNet(nn.Module):
    def __init__(self, input_size, weight_bit_width=1, act_bit_width=1):
        super(QuantGaussianBlobNet, self).__init__()

        # Validate input size
        if not isinstance(input_size, int):
            raise ValueError("input_size must be an integer")

        self.input_size = input_size

        # First quantization layer to convert input to 8-bit
        self.quant_inp = qnn.QuantIdentity(
            bit_width=8,
            return_quant_tensor=True
        )

        # Quantized Conv layers
        self.conv_layers = nn.Sequential(
            qnn.QuantConv3d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
                weight_bit_width=weight_bit_width,
                bias=False,
                return_quant_tensor=True
            ),
            qnn.QuantReLU(
                bit_width=act_bit_width,
                return_quant_tensor=True
            ),
            nn.MaxPool3d(2),

            qnn.QuantConv3d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
                weight_bit_width=weight_bit_width,
                bias=False,
                return_quant_tensor=True
            ),
            qnn.QuantReLU(
                bit_width=act_bit_width,
                return_quant_tensor=True
            ),
            nn.MaxPool3d(2),

            qnn.QuantConv3d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                weight_bit_width=weight_bit_width,
                bias=False,
                return_quant_tensor=True
            ),
            qnn.QuantReLU(
                bit_width=act_bit_width,
                return_quant_tensor=True
            ),
            nn.MaxPool3d(2)
        )

        # Calculate output size after conv layers
        # After each MaxPool3d(2), size is halved
        conv_output_size = input_size // (2 ** 3)  # 3 max pooling layers
        flattened_size = 64 * conv_output_size * conv_output_size * conv_output_size

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            qnn.QuantLinear(
                in_features=flattened_size,
                out_features=128,
                bias=False,
                weight_bit_width=weight_bit_width,
                return_quant_tensor=True
            ),
            qnn.QuantReLU(
                bit_width=act_bit_width,
                return_quant_tensor=True
            ),
            qnn.QuantLinear(
                in_features=128,
                out_features=6,  # Output 6 bounding box parameters
                bias=False,
                weight_bit_width=weight_bit_width
            )
        )

    def forward(self, x):
        x = self.quant_inp(x)
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# Configuration
SNR_DB = -10
IMG_SHAPE = (64, 64, 64)
WEIGHT_BIT_WIDTH = 2  # Binary weights
ACT_BIT_WIDTH = 2  # Binary activations
BATCH_SIZE = 8
NUM_EPOCHS = 100
LEARNING_RATE = 0.0001

# Create directories for outputs
if not os.path.exists("saved_models"):
    os.makedirs("saved_models")
if not os.path.exists("figures"):
    os.makedirs("figures")
if not os.path.exists("finn_export"):
    os.makedirs("finn_export")

# Generate a sample dataset with pixel-level labels
print("Generating dataset...")
num_samples = 1000
data_cutoff = []
labels_cutoff = []
for i in range(num_samples):
    if i % 100 == 0:
        print(f"  Generated {i}/{num_samples} samples")
    blob, label = generate_gaussian_blob_with_cutoff_pixel_labels(shape=IMG_SHAPE, snr_db=SNR_DB)
    data_cutoff.append(blob)
    labels_cutoff.append(label)

# Convert to numpy arrays
data_cutoff = np.array(data_cutoff)
labels_cutoff = np.array(labels_cutoff)

# Visualize a sample
sample_blob = data_cutoff[0]
x_c, y_c, z_c, x_w, y_w, z_w = labels_cutoff[0]

# Plot the central slice in each dimension
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
# XY Plane
axes[0].imshow(sample_blob[:, :, z_c].T, cmap="jet")
axes[0].set_title("XY Plane")
rect_xy = Rectangle((x_c - x_w / 2, y_c - y_w / 2), x_w, y_w,
                    facecolor='none', edgecolor='g', linewidth=2)
axes[0].add_patch(rect_xy)
# XZ Plane
axes[1].imshow(sample_blob[:, y_c, :].T, cmap="jet")
axes[1].set_title("XZ Plane")
rect_xz = Rectangle((x_c - x_w / 2, z_c - z_w / 2), x_w, z_w,
                    facecolor='none', edgecolor='g', linewidth=2)
axes[1].add_patch(rect_xz)
# YZ Plane
axes[2].imshow(sample_blob[x_c, :, :].T, cmap="jet")
axes[2].set_title("YZ Plane")
rect_yz = Rectangle((y_c - y_w / 2, z_c - z_w / 2), y_w, z_w,
                    facecolor='none', edgecolor='g', linewidth=2)
axes[2].add_patch(rect_yz)
plt.savefig("figures/example_data.png")
plt.close()

# Split dataset into training (80%), validation (10%), and test (10%)
print("Preparing data loaders...")
dataset = GaussianBlobDataset(data_cutoff, labels_cutoff)
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize the quantized model
model = QuantGaussianBlobNet(
    input_size=IMG_SHAPE[0],
    weight_bit_width=WEIGHT_BIT_WIDTH,
    act_bit_width=ACT_BIT_WIDTH
).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Add variables to track best model
best_val_loss = float('inf')
best_epoch = 0
model_save_path = f"saved_models/best_quant_blob_model_{SNR_DB}_snr_w{WEIGHT_BIT_WIDTH}a{ACT_BIT_WIDTH}.pth"

# Training loop
print("Starting training...")
train_losses = []
val_losses = []

for epoch in range(NUM_EPOCHS):
    # Training phase
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    # Print progress
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

    # Save the best model
    if avg_val_loss < best_val_loss and epoch > 10:
        best_val_loss = avg_val_loss
        best_epoch = epoch + 1

        # Save the model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
        }, model_save_path)

        print(f"Saved new best model at epoch {epoch + 1} with validation loss: {avg_val_loss:.4f}")

print(f"\nTraining completed. Best model was saved at epoch {best_epoch} with validation loss: {best_val_loss:.4f}")

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig(f"figures/training_loss_w{WEIGHT_BIT_WIDTH}a{ACT_BIT_WIDTH}.png")
plt.close()

# Load the best model for evaluation
checkpoint = torch.load(model_save_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)  # Ensure model is on the correct device
model.eval()

# Final evaluation on test set using the best model
test_loss = 0.0
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item()

avg_test_loss = test_loss / len(test_loader)
print(f"Test Loss with best model: {avg_test_loss:.4f}")


# Function to visualize ground truth and predicted boxes
def plot_comparison(blob_data, ground_truth, prediction_label):
    # Extract ground truth values
    x_center, y_center, z_center, x_width, y_width, z_width = ground_truth

    # Extract predicted values
    pred_x_c, pred_y_c, pred_z_c, pred_x_w, pred_y_w, pred_z_w = prediction_label

    # Create figure
    figure, ax = plt.subplots(1, 3, figsize=(15, 5))

    # XY Plane
    ax[0].imshow(blob_data[:, :, int(z_center)].T, cmap="jet")
    ax[0].set_title("XY Plane")

    # Ground truth box (green)
    rect_xy_gt = Rectangle((x_center - x_width / 2, y_center - y_width / 2), x_width, y_width,
                           facecolor='none', edgecolor='g', linewidth=2, label='Ground Truth')
    ax[0].add_patch(rect_xy_gt)

    # Predicted box (red)
    rect_xy_pred = Rectangle((pred_x_c - pred_x_w / 2, pred_y_c - pred_y_w / 2),
                             pred_x_w, pred_y_w,
                             facecolor='none', edgecolor='r', linewidth=2,
                             linestyle='--', label='Prediction')
    ax[0].add_patch(rect_xy_pred)

    # XZ Plane
    ax[1].imshow(blob_data[:, int(y_center), :].T, cmap="jet")
    ax[1].set_title("XZ Plane")

    # Ground truth box
    rect_xz_gt = Rectangle((x_center - x_width / 2, z_center - z_width / 2), x_width, z_width,
                           facecolor='none', edgecolor='g', linewidth=2)
    ax[1].add_patch(rect_xz_gt)

    # Predicted box
    rect_xz_pred = Rectangle((pred_x_c - pred_x_w / 2, pred_z_c - pred_z_w / 2),
                             pred_x_w, pred_z_w,
                             facecolor='none', edgecolor='r', linewidth=2,
                             linestyle='--')
    ax[1].add_patch(rect_xz_pred)

    # YZ Plane
    ax[2].imshow(blob_data[int(x_center), :, :].T, cmap="jet")
    ax[2].set_title("YZ Plane")

    # Ground truth box
    rect_yz_gt = Rectangle((y_center - y_width / 2, z_center - z_width / 2), y_width, z_width,
                           facecolor='none', edgecolor='g', linewidth=2)
    ax[2].add_patch(rect_yz_gt)

    # Predicted box
    rect_yz_pred = Rectangle((pred_y_c - pred_y_w / 2, pred_z_c - pred_z_w / 2),
                             pred_y_w, pred_z_w,
                             facecolor='none', edgecolor='r', linewidth=2,
                             linestyle='--')
    ax[2].add_patch(rect_yz_pred)

    # Add legend to the first subplot
    ax[0].legend()

    # Adjust layout and display
    plt.tight_layout()
    plt.savefig(f"figures/predict_box_w{WEIGHT_BIT_WIDTH}a{ACT_BIT_WIDTH}.png")
    plt.close()


# Example visualization on a test sample
model.eval()
with torch.no_grad():
    # Get a sample from the test dataset
    sample_input, sample_target = next(iter(test_loader))

    # Move to device and get prediction
    sample_input = sample_input[0].unsqueeze(0).to(device)  # Add batch dimension
    sample_target = sample_target[0].cpu().numpy()

    # Get prediction
    prediction = model(sample_input).cpu().numpy()[0]

    # Get the corresponding blob data
    sample_blob = sample_input[0, 0].cpu().numpy()  # Remove batch and channel dimensions

    # Plot the comparison
    plot_comparison(sample_blob, sample_target, prediction)

# Export to ONNX for FINN
print("Exporting model to ONNX...")
dummy_input = torch.randn(1, 1, IMG_SHAPE[0], IMG_SHAPE[1], IMG_SHAPE[2], device=device)
export_path = f"finn_export/quant_blob_net_w{WEIGHT_BIT_WIDTH}a{ACT_BIT_WIDTH}.onnx"

export_qonnx(model, dummy_input, export_path)
print(f"Model exported to {export_path}")

print("All done!")