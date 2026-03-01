import os
import logging
import argparse

import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import random_split
import argparse
from pathlib import Path
from helpers_functions import calculate_loss_batch, evaluate_model, generate_and_print_sample
import time

# import project modules.
from src.tokenizers import BPETokenizer
from src.utils.config import GPT2_SMALL_124M
from src.models.gpt import GPTModel
from src.data.dataset import GPTTextDataset, create_dataloader
from src.utils.helpers import save_checkpoint



def train_model(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, starting_prompt, tokenizer):
    """trains model for multiple epochs and tracks losses."""

    # initialize lists to track losses and tokens seen.
    train_losses, val_losses, track_tokens_seen = [], [], []

    tokens_seen, global_step = 0, -1

    # main training loop.
    for epoch in range(num_epochs):

        # set model to training mode.
        model.train()

        for input_batch, target_batch in train_loader:

            # reset loss gradients from previous batch iteration.
            optimizer.zero_grad()

            # calculate loss for current batch.
            loss = calculate_loss_batch(input_batch, target_batch, model, device)

            # calculate loss gradients.
            loss.backward()

            # update model weights using loss gradients.
            optimizer.step()

            # track total tokens processed.
            tokens_seen += input_batch.numel()
            global_step += 1

            # evaluate model at specified frequency.
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"epoch {epoch+1} (step {global_step:06d}): "
                      f"train loss {train_loss:.3f}, val loss {val_loss:.3f}")

        # generate sample text after each epoch.
        generate_and_print_sample(model, tokenizer, device, starting_prompt)

    return train_losses, val_losses, track_tokens_seen



def main():
    """main training loop."""

    # parse command line arguments.
    parser = argparse.ArgumentParser(description="train gpt-2 model")

    parser.add_argument("--data_path", type=str, default="data/processed", help="path to processed data")
    parser.add_argument("--checkpoint_dir", type=str, default="results/checkpoints", help="directory to save checkpoints")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--epochs", type=int, default=15, help="number of epochs")  # you can set your desired number of epochs here. 
    parser.add_argument("--lr", type=float, default=0.0004, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="weight decay")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="device")


    # parse command line arguments.
    args = parser.parse_args()

    # create checkpoint directory.
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # set device.
    device = torch.device(args.device)
    print(f"using device: {device}\n")

    # load text data from processed directory.
    data_path = Path(args.data_path)
    text_files = list(data_path.glob("*.txt"))

    if not text_files:
        raise ValueError(f"no text files found in {args.data_path}")

    # read and combine all text files.
    raw_text_content = ""
    for text_file in text_files:
        with open(text_file, 'r', encoding='utf-8') as f:
            raw_text_content += f.read()

    print(f"loaded {len(text_files)} files with {len(raw_text_content)} characters\n")

    # initialize tokenizer.
    tokenizer = BPETokenizer()

    # load gpt-2 small configuration.
    config = GPT2_SMALL_124M


    # initialize model and move to device.
    model = GPTModel(config).to(device)

    # count trainable parameters.
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model trainable parameters: {trainable_params}\n")

    # create dataset from text.
    text_data = GPTTextDataset(raw_text_content, tokenizer, config.context_length, stride=128)

    # set train/validation split ratio.
    train_ratio = 0.90
    dataset_len = len(text_data)

    # calculate split sizes.
    train_len = int(train_ratio * dataset_len)
    val_len = dataset_len - train_len

    # set seed for reproducible split.
    torch.manual_seed(66)

    # split dataset into train and validation sets.
    train_data, val_data = random_split(text_data, [train_len, val_len])

    # set seed for reproducible data loading.
    torch.manual_seed(66)

    # create training data loader.
    train_loader = create_dataloader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0
    )

    # create validation data loader.
    val_loader = create_dataloader(
        val_data,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0
    )

    # get total tokens for sanity check.
    total_tokens = len(tokenizer.encode(raw_text_content))

    # sanity check for sufficient training tokens.
    if total_tokens * train_ratio < config.context_length:
        print("not enough tokens for the training loader. "
              "try to lower the context_length or increase the training_ratio")

    # sanity check for sufficient validation tokens.
    if total_tokens * (1 - train_ratio) < config.context_length:
        print("not enough tokens for the validation loader. "
              "try to lower the context_length or decrease the training_ratio")

    # start timer for training duration.
    start_time = time.time()

    # initialize adamw optimizer.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # train model and track losses.
    train_losses, val_losses, tokens_seen = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs=args.epochs,
        eval_freq=5,
        eval_iter=5,
        starting_prompt="explain who an outlier is?",
        tokenizer=tokenizer
    )

    # end timer for training duration.
    end_time = time.time()

    # calculate and display training duration.
    execution_time_minutes = (end_time - start_time) / 60
    print(f"training completed in {execution_time_minutes:.2f} minutes.")

    # save final checkpoint.
    checkpoint_path = f"{args.checkpoint_dir}/checkpoint_epoch_{args.epochs}.pt"

    # save model checkpoint to disk.
    save_checkpoint(model, optimizer, args.epochs, train_losses[-1], checkpoint_path)


if __name__ == "__main__":

    # run the main training loop.
    main()