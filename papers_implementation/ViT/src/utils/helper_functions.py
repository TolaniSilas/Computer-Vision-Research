"""
ViT-specific helpers (e.g. GELU, LayerNorm, Calculate Mean and STD, show images).
"""
import numpy as np
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt


"""
GELU Activation Function Implementation from paper 'Gaussian Error Linear Units (GELUs)' by Hendrycks and Gimpel (2016).
https://arxiv.org/abs/1606.08415
"""

class GELU(nn.Module):
    """gaussian error linear unit (gelu) activation function. mathematical implementation of gelu."""

    def __init__(self):
        """initializes gelu activation."""

        super().__init__()

    def forward(self, x):
        """applies gelu activation to input tensor."""

        # compute gelu activation using tanh approximation.
        return 0.5 * x * (
            1 + torch.tanh(
                torch.sqrt(torch.tensor(2.0 / torch.pi)) *
                (x + 0.044715 * torch.pow(x, 3))
            )
        )


"""
Layer Normalization Implementation from paper 'Layer Normalization' by Ba, Kiros, and Hinton (2016).
https://arxiv.org/abs/1607.06450
"""

class LayerNorm(nn.Module):
    """layer normalization for stabilizing neural network training."""

    def __init__(self, embed_dim):
        """initializes layer normalization with learnable parameters."""

        super().__init__()

        # small constant to prevent division by zero.
        self.eps = 1e-5

        # learnable scale parameter.
        self.scale = nn.Parameter(torch.ones(embed_dim))

        # learnable shift parameter.
        self.shift = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x):
        """applies layer normalization to input tensor."""

        # calculate mean across last dimension.
        mean = x.mean(dim=-1, keepdim=True)

        # calculate variance across last dimension.
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # normalize input.
        norm_x = (x - mean) / torch.sqrt(var + self.eps)

        # apply scale and shift.
        return self.scale * norm_x + self.shift
    


def calculate_image_mean_std(train_dataset, gray_scale=False):
    """
    Calculate the mean and standard deviation of a dataset of images.

    Parameters:
    -----------
    train_dataset : Dataset
        A dataset containing image-label pairs, where each image is assumed
        to be a PyTorch tensor with shape (C, H, W).
    gray_scale : bool, optional
        If True, the function assumes grayscale images and calculates statistics accordingly.
        If False, it calculates statistics for RGB channels separately.

    Returns:
    --------
    tuple
        A tuple containing two elements:
        - A tuple of means for each channel (e.g., (mean_R, mean_G, mean_B) or (mean,)).
        - A tuple of standard deviations for each channel (e.g., (std_R, std_G, std_B) or (std,)).
    """

    # calculate mean values for each channel.
    mean_RGB = [np.mean(a=x.numpy(), axis=(1, 2)) for x, y in train_dataset]

    mean_R = np.mean([m[0] for m in mean_RGB])
    mean_G = np.mean([m[1] for m in mean_RGB])
    mean_B = np.mean([m[2] for m in mean_RGB])

    # calculate standard deviation for each channel.
    std_RGB = [np.std(a=x.numpy(), axis=(1, 2)) for x, y in train_dataset]
    std_R = np.mean([s[0] for s in std_RGB])
    std_G = np.mean([s[1] for s in std_RGB])
    std_B = np.mean([s[2] for s in std_RGB])

    # return grayscale statistics if specified.
    if gray_scale:
        mean_gray = np.mean([np.mean(m) for m in mean_RGB])
        std_gray = np.mean([np.mean(s) for s in std_RGB])
        return (mean_gray,), (std_gray,)

    # return RGB statistics.
    return (mean_R, mean_G, mean_B), (std_R, std_G, std_B)


def show_image(img, y=None, color=True):
    """
    Display an image using matplotlib.

    Parameters:
    img (Tensor): The image tensor to be displayed. Expected shape (C, H, W).
    y (int or str, optional): The label associated with the image. Default is None.
    color (bool, optional): Determines if the image should be displayed in color. Default is True.

    Behavior:
    - Converts the PyTorch tensor to a NumPy array and transposes it to match the shape expected by matplotlib (H, W, C).
    - Displays the image using `plt.imshow`.
    - If a label (`y`) is provided, it is displayed as the title of the image.
    """

    # convert the PyTorch tensor to a NumPy array.
    np_img = img.numpy()

    # transpose dimensions to match matplotlib's expected input shape (H, W, C).
    np_img_transpose = np.transpose(np_img, (1, 2, 0))

    # display the image.
    plt.imshow(np_img_transpose)

    # display the label as a title, if provided.
    if y is not None:
        plt.title("Label: " + str(y))



def get_lr(opt):
    """
    Retrieve the current learning rate from the optimizer.

    Args:
        opt (torch.optim.Optimizer): The optimizer used for training.

    Returns:
        float: The current learning rate of the optimizer.
    """

    # loop through the parameter groups in the optimizer.
    for param_group in opt.param_groups:

        # return the learning rate from the first parameter group.
        return param_group['lr']