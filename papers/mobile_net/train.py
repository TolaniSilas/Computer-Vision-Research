# Import the neccessary libraries.
import os
import torch
from torch import nn
import torch.optim as optim
from model import MobileNet
from train_loader import load_MNIST_dataset


# Load the training and validation dataset.
train_loader, valid_loader = load_MNIST_dataset()
print("Data are loaded and are ready to use!")



def train_model(device, model, train_loader, epoch, optimizer):

    # Set the model to training mode.
    model.train()
    train_loss = 0
    n= 0
    print(f"\nEpoch: {epoch}")

    # process the images in batches.
    for batch_idx, (data, labels) in enumerate(train_loader):
        # Use the device for hardware acceleration.
        data = data.to(device)
        label = labels.to(device)

        # Reset the optimizer.
        optimizer.zero_grad()

        # Push the data forward through the model layers.
        output = model(data)

        # Get the loss.
        loss = loss_criteria(output, label)

        # Keep a running total.
        train_loss += loss.item()

        # Backpropagate.
        loss.backward()
        optimizer.step()

        if n==2:
            n+=1
            break
        
     
        # # Print metrics for every 5 batches.
        # if batch_idx % 5 == 0:
        #     print(f"Training set [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100 * batch_idx / len(train_loader):.0f}%)] Loss: {loss.item():.5f} ")


    # Return the average loss for each epoch.
    avg_loss = train_loss / (batch_idx + 1)
    print(f"Average Training Loss: {avg_loss:.5f}")

    print("Training Completed!")

    return avg_loss



def validate_model(device, model, val_loader):
    # Switch the model to evaluation mode to see how the model is performing.
    model.eval()
    print("Model Validation Starts!")

    # Initialize the validation loss.
    val_loss = 0
    correct = 0

    with torch.no_grad():
        batch_count = 0
        for data, target in val_loader:
            batch_count += 1
            data, target = data.to(device), target.to(device)

            # Get the predicted class for this  batch.
            output = model(data)

            # Calculate the loss for this batch.
            val_loss += loss_criteria(output, target).item()

            # Calculate the accuracy  metrics for this batch.
            _, predicted = torch.max(output.data, 1)
            correct += torch.sum(target == predicted).item()
            

        # Calculate the total accuracy and average loss for each epoch.
        avg_loss = val_loss / batch_count
        print(f'Average Validation Loss: {avg_loss:.5f}, Accuracy: {correct}/{len(val_loader.dataset)} ({100 * correct / len(val_loader.dataset):.0f}%)')

        # Return average loss for each epoch.
        return avg_loss




# It is time to use the train function to train and the validation function to evaluate how the model is performing.

device = "cpu"
# Check if GPU is present or avaliable as the hardware accelerator.
if (torch.cuda.is_available()):
    device = "cuda"
print(f"Training on {device}")


# Create an instance of the model class and allocate it to the device available.
model = MobileNet(num_classes=10).to(device)

# Define the learning rate.
learning_rate = 1e-4
# # Use Adaptive Moment Estimation Optimization to adjust and update the model weights.
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Using RMSProp optimizer.
optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

# Specify the loss criteria.
loss_criteria = nn.CrossEntropyLoss()

# Initialize empty arrays to track metrics.
epoch_nums, training_loss, validation_loss,  = [], [], []

# Train over 30 epochs.
epochs = 3
for epoch in range(1, epochs +1):
    train_loss = train_model(device, model, train_loader, epoch, optimizer)
    val_loss = validate_model(device, model, valid_loader)

    # Append the metrics to the predefined empty arrays.
    epoch_nums.append(epoch)
    training_loss.append(train_loss)
    validation_loss.append(val_loss)