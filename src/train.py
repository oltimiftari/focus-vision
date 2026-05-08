import torch
import torch.nn as nn
from tqdm import tqdm


class BCEDiceIoULoss(nn.Module):

    def __init__(self):
        super(BCEDiceIoULoss, self).__init__()

        self.bce = nn.BCELoss()

    def forward(self, predictions, targets):

        bce_loss = self.bce(predictions, targets)

        smooth = 1e-6

        predictions_flat = predictions.view(-1)
        targets_flat = targets.view(-1)

        intersection = (predictions_flat * targets_flat).sum()

        union = (
            predictions_flat.sum()
            + targets_flat.sum()
            - intersection
        )

        iou = (intersection + smooth) / (union + smooth)

        iou_loss = 1 - iou

        total_loss = bce_loss + iou_loss

        return total_loss


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    epochs
):

    train_losses = []
    val_losses = []

    for epoch in range(epochs):

        model.train()

        running_train_loss = 0.0

        for images, masks in tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{epochs} - Training"
        ):

            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, masks)

            loss.backward()

            optimizer.step()

            running_train_loss += loss.item()

        avg_train_loss = (
            running_train_loss / len(train_loader)
        )

        train_losses.append(avg_train_loss)

        model.eval()

        running_val_loss = 0.0

        with torch.no_grad():

            for images, masks in tqdm(
                val_loader,
                desc=f"Epoch {epoch+1}/{epochs} - Validation"
            ):

                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)

                loss = criterion(outputs, masks)

                running_val_loss += loss.item()

        avg_val_loss = (
            running_val_loss / len(val_loader)
        )

        val_losses.append(avg_val_loss)

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"- Train Loss: {avg_train_loss:.4f} "
            f"- Val Loss: {avg_val_loss:.4f}"
        )

    return train_losses, val_losses