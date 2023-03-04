import functools
import datetime
import importlib
import logging
import torch
from tqdm import tqdm

def get_logger(filename):
    logger = logging.getLogger(__name__)
    if logger.hasHandlers():
        return logger
    fh = logging.FileHandler(filename=filename, mode="w+")
    ch = logging.StreamHandler()
    logger.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info("")
    logger.info(
        "######################################################################")
    logger.info(f"Starting new run {datetime.datetime.now()}")
    logger.info(
        "######################################################################")
    logger.info("")
    return logger

def import_class(model_class: str):
    return getattr(importlib.import_module(".".join(model_class.split(".")[:-1])), 
                                            model_class.split(".")[-1])

def train(dataloader, model, loss_fn, optimizer, writer, step, device):
    r"""Train the model.

    Args:
        dataloader: the dataloader to extract the data (required).
        model: the model to train (required).
        loss_fn: the loss to minimize (required).
        optimizer: to perform the minimization (required).
        writer: writes variables to tensorboard (required).
        step: training step (required).
        device: cpu or cuda (required).
    """
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in tqdm(enumerate(dataloader), total=len(dataloader)):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        writer.add_scalar("Loss/train", loss, step)
        step += 1

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            logging.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return step


def validation(dataloader: torch.utils.data.DataLoader, model: torch.nn.Module,
               loss_fn, writer, epoch: int, device: str) -> float:
    r"""validation the model performance.

    Args:
        dataloader: the dataloader to extract the data (required).
        model: the model to validation (required).
        loss_fn: the loss to apply (required).
        device: cpu or cuda (required).
    """
    num_batches = len(dataloader)
    model.eval()
    validation_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in tqdm(dataloader, total=num_batches):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            validation_loss += loss.item()
    validation_loss /= num_batches
    writer.add_scalar("Loss/validation", loss, epoch)
    logging.info(
        f"validation Error: \n Avg loss: {validation_loss:>8f} \n")
    return validation_loss

def train_epochs(epochs, train_dataloader, model, optimizer, loss_fn,
                device, validation_dataloader, scheduler, checkpoint_epoch_period,
                ):
    writer = SummaryWriter()
    step = 0
    for epoch in tqdm(range(epochs)):
        writer.add_scalar("lr/train", optimizer.param_groups[0]["lr"], epoch)
        step = train(train_dataloader, model, loss_fn,
                     optimizer, writer, step, device)
        validation_loss = validation(
            validation_dataloader, model, loss_fn, writer, epoch, device)
        scheduler.step()
        writer.flush()
        if epoch % checkpoint_epoch_period == 0:
            ckpt_name = f"{datetime.now()}-checkpoint-epoch-{epoch}-step-{step}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'step': step,
                'validation_loss': validation_loss
            }, f"Checkpoints/{ckpt_name}")
            logging.info(f"Checkpoint saved to Checkpoints/{ckpt_name}")
    logging.info("Done!")
    writer.close()
    ckpt_name = f"{datetime.now()}-checkpoint-epoch-{epochs - 1}-step-{step}.pth"
    if not os.path.exists(f"Checkpoints/{ckpt_name}"):
        torch.save({
            'epoch': epochs - 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step': step,
            'validation_loss': validation_loss
        }, f"Checkpoints/{ckpt_name}")
        logging.info(f"Model saved to Models/{ckpt_name}")