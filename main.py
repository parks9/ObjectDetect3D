import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from blob_maker import generate_gaussian_blob_with_cutoff_pixel_labels, GaussianBlobNet, GaussianBlobDataset

SNR_DB = -10
IMG_SHAPE = (64,64,64)


# Generate a sample dataset with pixel-level labels
num_samples = 1000
data_cutoff = []
labels_cutoff = []
for _ in range(num_samples):
   blob, label = generate_gaussian_blob_with_cutoff_pixel_labels(shape=IMG_SHAPE, snr_db=SNR_DB)
   data_cutoff.append(blob)
   labels_cutoff.append(label)
# Convert to numpy arrays
data_cutoff = np.array(data_cutoff)
labels_cutoff = np.array(labels_cutoff)
# Select the sample for visualization
sample_blob = data_cutoff[0]
# Extract label values
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


# Split dataset into training (80%), validation (10%), and test (10%)
dataset = GaussianBlobDataset(data_cutoff, labels_cutoff)
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
# Create DataLoaders
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Initialize model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GaussianBlobNet(input_size=IMG_SHAPE[0]).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Add variables to track best model
best_val_loss = float('inf')
best_epoch = 0

if not os.path.exists("saved_models"):
    os.makedirs("saved_models")
model_save_path = f"saved_models/best_blob_model_{SNR_DB}_snr.pth"

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

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

    # Print progress
    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

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

# To load the best model later, you can use:
# checkpoint = torch.load(model_save_path)
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Final evaluation on test set using the best model
checkpoint = torch.load(model_save_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

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
def plot_comparison(blob_data, ground_truth, prediction_label=None):
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
    plt.savefig("figures/predict_box.png")


# Example usage (add this after training the model):
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