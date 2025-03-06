import numpy as np
import torch
from torch.utils.data import DataLoader
from blob_maker import generate_gaussian_blob_with_cutoff_pixel_labels, GaussianBlobDataset, GaussianBlobNet

SNR_DB = -20
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


# Split dataset into training (80%), validation (10%), and test (10%)
dataset = GaussianBlobDataset(data_cutoff, labels_cutoff)

test_loader = DataLoader(dataset, batch_size=8, shuffle=False)


def calculate_3d_metrics(pred_boxes, gt_boxes):
    """
    Calculate various metrics for 3D bounding box prediction.

    Parameters:
    -----------
    pred_boxes : numpy array
        Predicted boxes in format (center_x, center_y, center_z, width_x, width_y, width_z)
    gt_boxes : numpy array
        Ground truth boxes in same format

    Returns:
    --------
    dict : Dictionary containing various metrics
    """
    metrics = {}

    # 1. Center Error (L2 distance between centers)
    center_errors = np.sqrt(np.sum((pred_boxes[:, :3] - gt_boxes[:, :3]) ** 2, axis=1))
    metrics['center_error_mean'] = np.mean(center_errors)
    metrics['center_error_std'] = np.std(center_errors)

    # 2. Size Error (L2 distance between dimensions)
    size_errors = np.sqrt(np.sum((pred_boxes[:, 3:] - gt_boxes[:, 3:]) ** 2, axis=1))
    metrics['size_error_mean'] = np.mean(size_errors)
    metrics['size_error_std'] = np.std(size_errors)

    # 3. Volume Error (absolute and relative)
    pred_volumes = np.prod(pred_boxes[:, 3:], axis=1)
    gt_volumes = np.prod(gt_boxes[:, 3:], axis=1)
    volume_errors_abs = np.abs(pred_volumes - gt_volumes)
    volume_errors_rel = volume_errors_abs / gt_volumes
    metrics['volume_error_abs_mean'] = np.mean(volume_errors_abs)
    metrics['volume_error_rel_mean'] = np.mean(volume_errors_rel) * 100  # as percentage

    # 4. IoU (Intersection over Union)
    ious = calculate_3d_iou_batch(pred_boxes, gt_boxes)
    metrics['iou_mean'] = np.mean(ious)
    metrics['iou_std'] = np.std(ious)

    # 5. Precision at different IoU thresholds
    iou_thresholds = [0.25, 0.5, 0.75]
    for thresh in iou_thresholds:
        metrics[f'precision_iou_{thresh}'] = np.mean(ious >= thresh) * 100  # as percentage

    return metrics


def calculate_3d_iou_batch(boxes1, boxes2):
    """
    Calculate IoU between two batches of 3D bounding boxes.

    Parameters:
    -----------
    boxes1, boxes2 : numpy arrays
        Arrays of boxes in format (center_x, center_y, center_z, width_x, width_y, width_z)

    Returns:
    --------
    numpy array : IoU values for each pair of boxes
    """
    ious = []
    for box1, box2 in zip(boxes1, boxes2):
        # Convert to min-max coordinates
        box1_min = box1[:3] - box1[3:] / 2
        box1_max = box1[:3] + box1[3:] / 2
        box2_min = box2[:3] - box2[3:] / 2
        box2_max = box2[:3] + box2[3:] / 2

        # Calculate intersection
        intersection_min = np.maximum(box1_min, box2_min)
        intersection_max = np.minimum(box1_max, box2_max)

        # Check if boxes intersect
        if np.any(intersection_max < intersection_min):
            ious.append(0.0)
            continue

        # Calculate volumes
        intersection_volume = np.prod(intersection_max - intersection_min)
        box1_volume = np.prod(box1[3:])
        box2_volume = np.prod(box2[3:])
        union_volume = box1_volume + box2_volume - intersection_volume

        # Calculate IoU
        iou = intersection_volume / union_volume
        ious.append(iou)

    return np.array(ious)


# Example usage in training loop:
def evaluate_model(model, data_loader, device):
    """
    Evaluate model performance using various 3D bounding box metrics.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            # Convert to numpy for metric calculation
            pred_boxes = outputs.cpu().numpy()
            gt_boxes = targets.numpy()

            all_preds.append(pred_boxes)
            all_targets.append(gt_boxes)

    # Concatenate all predictions and targets
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate metrics
    metrics = calculate_3d_metrics(all_preds, all_targets)

    return metrics


# Example of printing metrics:
def print_metrics(metrics):
    """
    Print metrics in a formatted way.
    """
    print("\n=== 3D Bounding Box Metrics ===")
    print(f"Center Error: {metrics['center_error_mean']:.3f} ± {metrics['center_error_std']:.3f}")
    print(f"Size Error: {metrics['size_error_mean']:.3f} ± {metrics['size_error_std']:.3f}")
    print(f"Volume Error: {metrics['volume_error_rel_mean']:.1f}%")
    print(f"Mean IoU: {metrics['iou_mean']:.3f} ± {metrics['iou_std']:.3f}")
    print("\nPrecision at IoU thresholds:")
    print(f"IoU > 0.25: {metrics['precision_iou_0.25']:.1f}%")
    print(f"IoU > 0.50: {metrics['precision_iou_0.5']:.1f}%")
    print(f"IoU > 0.75: {metrics['precision_iou_0.75']:.1f}%")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GaussianBlobNet(input_size=IMG_SHAPE[0]).to(device)
model_save_path = f"saved_models\\best_blob_model_{SNR_DB}_snr.pth"
checkpoint = torch.load(model_save_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

metrics = evaluate_model(model, test_loader, device)

print_metrics(metrics)
