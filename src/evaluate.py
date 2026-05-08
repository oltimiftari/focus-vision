import torch

from sklearn.metrics import precision_score, recall_score, f1_score


def calculate_iou(prediction, target, threshold=0.5):

    prediction = (prediction > threshold).float()
    target = (target > threshold).float()

    prediction = prediction.view(-1)
    target = target.view(-1)

    intersection = (prediction * target).sum()
    union = prediction.sum() + target.sum() - intersection

    smooth = 1e-6

    iou = (intersection + smooth) / (union + smooth)

    return iou.item()


def evaluate_model(model, data_loader, device):

    model.eval()

    iou_scores = []
    all_predictions = []
    all_targets = []

    with torch.no_grad():

        for images, masks in data_loader:

            images = images.to(device)
            masks = masks.to(device)

            predictions = model(images)

            batch_iou = calculate_iou(predictions, masks)
            iou_scores.append(batch_iou)

            predictions = (predictions > 0.5).float()
            masks = (masks > 0.5).float()

            all_predictions.extend(
                predictions.cpu().numpy().flatten()
            )

            all_targets.extend(
                masks.cpu().numpy().flatten()
            )

    average_iou = sum(iou_scores) / len(iou_scores)

    precision = precision_score(
        all_targets,
        all_predictions,
        zero_division=0
    )

    recall = recall_score(
        all_targets,
        all_predictions,
        zero_division=0
    )

    f1 = f1_score(
        all_targets,
        all_predictions,
        zero_division=0
    )

    return {
        "iou": average_iou,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }