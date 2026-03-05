import os
import logging
import copy
import time
import yaml
import torch
import torch.nn as nn
from config.config import resnet18_config
from training.train_loader import load_stl10
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.utils import get_lr
from src.model import ResNet18



# configure logging.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train_model(model, loss_func, dataloader, optimizer, device, sanity_check=False):
    """trains the model for one epoch."""

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

        # backward pass and compute gradients.
        loss.backward()

        # update model parameters based on gradients.
        optimizer.step()

        # calculate accuracy metrics for this batch.
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item()
        running_corrects += torch.sum(preds == y_batch.data).item()
        total_samples += x_batch.size(0)

        if sanity_check:
            break

    epoch_loss = running_loss / float(len(dataloader.dataset))
    epoch_acc = running_corrects / total_samples

    return epoch_loss, epoch_acc


def validate_model(model, loss_func, dataloader, device, sanity_check=False):
    """validates the model for one epoch."""

    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            # forward pass.
            outputs = model(x_batch)
            loss = loss_func(outputs, y_batch)

            # calculate accuracy metrics for this batch.
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item()
            running_corrects += torch.sum(preds == y_batch.data).item()
            total_samples += x_batch.size(0)

            if sanity_check:
                break

    epoch_loss = running_loss / float(len(dataloader.dataset))
    epoch_acc = running_corrects / total_samples

    return epoch_loss, epoch_acc


def train_val(model, params):
    """trains and validates the model for a specified number of epochs."""

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

    # initialize dictionaries to store loss and metric history.
    loss_history = {"train": [], "val": []}
    metric_history = {"train": [], "val": []}

    # deep copy of the model's initial state to store the best weights.
    best_model_weights = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    for epoch in range(1, num_epochs + 1):

        # get the current learning rate.
        current_lr = get_lr(optimizer)
        logger.info(f"epoch: {epoch}/{num_epochs}, current lr: {current_lr}")

        # train phase.
        train_loss, train_metric = train_model(
            model, loss_func, train_dataloader, optimizer, device, sanity_check
        )
        loss_history["train"].append(train_loss)
        metric_history["train"].append(train_metric)

        # validation phase.
        val_loss, val_metric = validate_model(
            model, loss_func, val_dataloader, device, sanity_check
        )
        loss_history["val"].append(val_loss)
        metric_history["val"].append(val_metric)

        # save the model weights if validation loss is the best so far.
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), path2weights)
            logger.info("copied best model weights!")

        # step the learning rate scheduler.
        lr_scheduler.step()

        # log training and validation metrics.
        logger.info(f"train loss: {train_loss:.6f}, val loss: {val_loss:.6f}, accuracy: {100 * val_metric:.2f}")
        logger.info("-" * 10)

    # load the best model weights after training.
    model.load_state_dict(best_model_weights)

    return model, loss_history, metric_history



def load_config(config_path: str) -> dict:
    """loads yaml config file and returns a dictionary."""

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config



def main():
    """main training loop for resnet18."""

    # load yaml config.
    config_path = "config/default.yaml"
    cfg = load_config(config_path)
    logger.info(f"loaded config from: {config_path}")

    # extract training and data settings.
    num_epochs = cfg["training"]["num_epochs"]
    lr = cfg["training"]["lr"]
    t_max = cfg["training"]["t_max"]
    eta_min = cfg["training"]["eta_min"]
    sanity_check = cfg["training"]["sanity_check"]
    path2weights = cfg["training"]["path2weights"]
    train_batch_size = cfg["data"]["train_batch_size"]
    val_batch_size = cfg["data"]["val_batch_size"]

    # create checkpoint directory.
    checkpoint_dir = os.path.dirname(path2weights)
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger.info(f"checkpoint directory: {checkpoint_dir}")

    # set device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"using device: {device}")

    # load resnet18 configuration.
    model_config = resnet18_config
    logger.info("loaded resnet18 configuration")

    # initialize model and move to device.
    model = ResNet18(model_config).to(device)

    # log trainable parameters.
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"model trainable parameters: {trainable_params}")

    # load train and validation dataloaders.
    logger.info("loading stl10 dataset from: ./data")
    train_loader, val_loader = load_stl10(
        data_dir="./data",
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
    )
    logger.info(f"train batches: {len(train_loader)}, val batches: {len(val_loader)}")

    # define loss function.
    loss_func = nn.CrossEntropyLoss(reduction="sum")
    logger.info("loss function: cross entropy with sum reduction")

    # initialize sgd optimizer.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    logger.info(f"optimizer: sgd with lr={lr}")

    # configure cosine annealing scheduler.
    lr_scheduler = CosineAnnealingLR(
        optimizer=optimizer,
        T_max=t_max,
        eta_min=eta_min
    )
    logger.info(f"scheduler: cosine annealing with t_max={t_max}, eta_min={eta_min}")

    # define training parameters.
    params_train = {
        "num_epochs": num_epochs,
        "optimizer": optimizer,
        "loss_func": loss_func,
        "train_dataloader": train_loader,
        "val_dataloader": val_loader,
        "sanity_check": sanity_check,
        "lr_scheduler": lr_scheduler,
        "path2weights": path2weights,
        "device": device,
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