import numpy as np


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

    # Calculate mean values for each channel.
    mean_RGB = [np.mean(a=x.numpy(), axis=(1, 2)) for x, y in train_dataset]

    mean_R = np.mean([m[0] for m in mean_RGB])
    mean_G = np.mean([m[1] for m in mean_RGB])
    mean_B = np.mean([m[2] for m in mean_RGB])

    # Calculate standard deviation for each channel.
    std_RGB = [np.std(a=x.numpy(), axis=(1, 2)) for x, y in train_dataset]
    std_R = np.mean([s[0] for s in std_RGB])
    std_G = np.mean([s[1] for s in std_RGB])
    std_B = np.mean([s[2] for s in std_RGB])

    # Return grayscale statistics if specified.
    if gray_scale:
        mean_gray = np.mean([np.mean(m) for m in mean_RGB])
        std_gray = np.mean([np.mean(s) for s in std_RGB])
        return (mean_gray,), (std_gray,)

    # Return RGB statistics.
    return (mean_R, mean_G, mean_B), (std_R, std_G, std_B)

# # Get the mean and standard deviation of the training dataset.
# mean_rgb, std_rgb = calculate_image_mean_std(train_dataset)