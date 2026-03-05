import os
import logging
import argparse
import copy
import time
import torch
import torch.nn as nn
from config.config import vit_testing
from training.data_loader import load_stl10
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.utils.helper_functions import get_lr
from src.model.vit import VisionTransformer



def train_model(model, loss_func, dataloader, optimizer, device, sanity_check=False):
    """
    Train the model for one epoch.

    Args:
        model (torch.nn.Module): The model to train.
        loss_func (function): The loss function to compute the loss.
        dataloader (torch.utils.data.DataLoader): The DataLoader for training data.
        optimizer (torch.optim.Optimizer): The optimizer to update the model's parameters.
        sanity_check (bool): If True, runs only one batch for debugging purposes.

    Returns:
        tuple: Training loss and training accuracy for the epoch.
    """

    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0



    for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        # reset the optimizer.
        optimizer.zero_grad()

        # forward pass.
        outputs = model(x_batch)
        loss = loss_func(outputs, y_batch)

        # backward pass
        # compute the gradients
        loss.backward()

        # Update model parameters based on gradients.
        optimizer.step()

        # calculate the accuracy  metrics for this batch.
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item()
        running_corrects += torch.sum(preds == y_batch.data).item()
        total_samples += x_batch.size(0)

        if sanity_check:
            break

    epoch_loss = running_loss / float(len(dataloader.dataset))
    epoch_acc = running_corrects / total_samples
    print(epoch_acc)

    return epoch_loss, epoch_acc


def validate_model(model, loss_func, dataloader, device, sanity_check=False):
    """
    Validate the model for one epoch.

    Args:
        model (torch.nn.Module): The model to validate.
        loss_func (function): The loss function to compute the loss.
        dataloader (torch.utils.data.DataLoader): The DataLoader for validation data.
        sanity_check (bool): If True, runs only one batch for debugging purposes.

    Returns:
        tuple: Validation loss and validation accuracy for the epoch.
    """

    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    with torch.no_grad():
        batch_count = 0
        for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            batch_count+= 1

            # forward pass.
            outputs = model(x_batch)
            loss = loss_func(outputs, y_batch)

            # calculate the accuracy  metrics for this batch.
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item()
            running_corrects += torch.sum(preds == y_batch.data).item()

            # get the number of samples.
            total_samples += x_batch.size(0)

            if sanity_check:
                break

    epoch_loss = running_loss / float(len(dataloader.dataset))
    epoch_acc = running_corrects / total_samples

    return epoch_loss, epoch_acc


def train_val(model, params):
    """
    Train and validate the model for a specified number of epochs.

    Args:
        model (torch.nn.Module): The model to be trained and evaluated.
        params (dict): A dictionary containing training parameters:
            - 'num_epochs' (int): The number of epochs to train the model.
            - 'loss_func' (function): The loss function to compute the loss.
            - 'optimizer' (torch.optim.Optimizer): The optimizer to update the model's parameters.
            - 'train_dataloader' (torch.utils.data.DataLoader): The DataLoader for the training data.
            - 'val_dataloader' (torch.utils.data.DataLoader): The DataLoader for the validation data.
            - 'sanity_check' (bool): If True, runs only one batch for debugging purposes.
            - 'lr_scheduler' (torch.optim.lr_scheduler): The learning rate scheduler.
            - 'path2weights' (str): The path to save the best model's weights.

    Returns:
        tuple: A tuple containing:
            - The model with the best weights after training.
            - A dictionary with loss history for training and validation.
            - A dictionary with performance metrics history for training and validation.
    """

    # extract parameters from the provided dictionary.
    num_epochs = params["num_epochs"]
    loss_func = params["loss_func"]
    optimizer = params["optimizer"]
    train_dataloader = params["train_dataloader"]
    val_dataloader = params["val_dataloader"]
    sanity_check = params["sanity_check"]
    lr_scheduler = params["lr_scheduler"]
    path2weights = params["path2weights"]
    device = params["device"]

    # initialize dictionaries to store loss and metric history for both train and validation.
    loss_history = {"train": [], "val": []}
    metric_history = {"train": [], "val": []}

    # deep copy of the model's initial state to store the best weights.
    best_model_weights = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    # loop through each epoch.
    for epoch in range(1, num_epochs + 1):

        # get the current learning rate for the optimizer.
        current_lr = get_lr(optimizer)
        print(f"Epoch: {epoch}/{num_epochs}, current lr: {current_lr}")

        # train phase: set model to training mode and compute training loss and metric.
        train_loss, train_metric = train_model(model, loss_func, train_dataloader, optimizer, device, sanity_check)
        loss_history["train"].append(train_loss)
        metric_history["train"].append(train_metric)

        # validation phase: set model to evaluation mode and compute validation loss and metric.
        val_loss, val_metric = validate_model(model, loss_func, val_dataloader, device, sanity_check)
        loss_history["val"].append(val_loss)
        metric_history["val"].append(val_metric)


        # save the model weights if the validation loss is the best so far.
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            # save model weights.
            torch.save(model.state_dict(), path2weights)
            print("copied best model weights!")

        # step the learning rate scheduler at the end of each epoch.
        lr_scheduler.step()

        # print training and validation loss, along with the validation metric (accuracy).
        print(f"train loss: {train_loss:.6f}, val loss: {val_loss:.6f}, accuracy: {100 * val_metric:.2f}")
        print("-" * 10)

    # load the best model weights after training is complete.
    model.load_state_dict(best_model_weights)

    # return the model with the best weights, and the loss and metric history for both training and validation.
    return model, loss_history, metric_history




# configure logging.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """main training loop for vision transformer."""

    # parse command line arguments.
    parser = argparse.ArgumentParser(description="train vision transformer model")

    parser.add_argument("--data_path", type=str, default="./data", help="path to dataset")
    parser.add_argument("--checkpoint_dir", type=str, default="./models", help="directory to save checkpoints")
    parser.add_argument("--train_batch_size", type=int, default=32, help="training batch size")
    parser.add_argument("--val_batch_size", type=int, default=64, help="validation batch size")
    parser.add_argument("--num_epochs", type=int, default=20, help="number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--t_max", type=int, default=5, help="cosine annealing t_max")
    parser.add_argument("--eta_min", type=float, default=1e-6, help="cosine annealing minimum lr")
    parser.add_argument("--sanity_check", action="store_true", help="run one batch for debugging")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="device")

    # parse arguments.
    args = parser.parse_args()

    # create checkpoint directory.
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    logger.info(f"checkpoint directory: {args.checkpoint_dir}")

    # set device.
    device = torch.device(args.device)
    logger.info(f"using device: {device}")

    # load vit configuration.
    config = vit_testing
    logger.info("loaded vit_testing configuration")

    # initialize model and move to device.
    model = VisionTransformer(config).to(device)

    # log trainable parameters.
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"model trainable parameters: {trainable_params}")

    # load train and validation dataloaders.
    logger.info(f"loading stl10 dataset from: {args.data_path}")
    train_loader, val_loader = load_stl10(
        data_dir=args.data_path,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
    )
    logger.info(f"train batches: {len(train_loader)}, val batches: {len(val_loader)}")

    # define loss function.
    loss_func = nn.CrossEntropyLoss(reduction="sum")
    logger.info("loss function: cross entropy with sum reduction")

    # initialize sgd optimizer.
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    logger.info(f"optimizer: sgd with lr={args.lr}")

    # configure cosine annealing scheduler.
    lr_scheduler = CosineAnnealingLR(
        optimizer=optimizer,
        T_max=args.t_max,
        eta_min=args.eta_min
    )
    logger.info(f"scheduler: cosine annealing with t_max={args.t_max}, eta_min={args.eta_min}")

    # define path to save best model weights.
    path2weights = os.path.join(args.checkpoint_dir, "vit_best_weights.pt")

    # define training parameters.
    params_train = {
        "num_epochs": args.num_epochs,
        "optimizer": optimizer,
        "loss_func": loss_func,
        "train_dataloader": train_loader,
        "val_dataloader": val_loader,
        "sanity_check": args.sanity_check,
        "lr_scheduler": lr_scheduler,
        "path2weights": path2weights,
        "device": device
    }


    # start training timer.
    start_time = time.time()
    logger.info("starting training...")

    # train and validate the model.
    model, loss_hist, metric_hist = train_val(model, params_train)

    # log training duration.
    duration = (time.time() - start_time) / 60
    logger.info(f"training completed in {duration:.2f} minutes")
    logger.info(f"best model weights saved to: {path2weights}")


if __name__ == "__main__":

    # run the main training loop.
    main()