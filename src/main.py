import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from scipy.stats import gamma



############## ---- 1 ---- ##############
# Write a Python function to choose a patch on an image
# Input: Image, Patch center, Half-side of the patch
# Output: Image patch

def mirror_image(image: np.ndarray) -> np.ndarray:
    """
    Mirror the image at the borders to extrapolate the patch.

    Parameters:
    image (np.ndarray): The input image.

    Returns:
    np.ndarray: Original image with mirrored copies around the borders.
    """
    # Flipping the image
    flip_ud = cv2.flip(image, 0)    # vertical flip
    flip_lr = cv2.flip(image, 1)    # horizontal flip
    flip_both = cv2.flip(image, -1)

    # Mirroring the image at the borders
    top_row = np.hstack([flip_both, flip_ud, flip_both])
    middle_row = np.hstack([flip_lr, image, flip_lr])
    bottom_row = np.hstack([flip_both, flip_ud, flip_both])

    # Putting all together
    mirrored_image = np.vstack([top_row, middle_row, bottom_row])
    
    return mirrored_image

def choose_patch(image: np.ndarray, center: tuple, half_side: int) -> np.ndarray:
    """
    Choose a patch from the image.

    Parameters:
    image (np.ndarray): The input image.
    center (tuple): The center of the patch (y, x).
    half_side (int): Half the side length of the patch.

    Returns:
    np.ndarray: The selected patch.
    """
    img_bounds = image.shape[:2]
    
    # get mirrored image
    mirrored_image = mirror_image(image)
    # get new center for staying in the original image
    x = center[0] + img_bounds[0]
    y = center[1] + img_bounds[1]
    
    # get the patch from the mirrored image
    patch = mirrored_image[y-half_side:y+half_side+1, x-half_side:x+half_side+1]
    return patch


############### ---- 2 ---- ##############
# Write a Python function that adds noise on the image patch
# Input: Image patch, type of noise (Gaussian for optique, speckle for SAR), standard deviation
# Output: Noisy patch

def add_noise_optique(patch: np.ndarray, std_dev: float, mean: float) -> np.ndarray:
    """
    Add Gaussian noise to the optique image patch.

    Parameters:
    patch (np.ndarray): The input image patch.
    std_dev (float): Standard deviation of the noise.
    mean (float): Mean of the noise.

    Returns:
    np.ndarray: Noisy image patch.
    """
    # Generate Gaussian noise
    noise = np.random.normal(mean, std_dev, patch.shape).astype(np.uint8)
    # Add noise to the patch
    noisy_patch = patch + noise
    # Clip the values to be in the valid range [0, 255]
    noisy_patch = np.clip(noisy_patch, 0, 255).astype(np.uint8)
    return noisy_patch

def add_noise_sar(patch: np.ndarray, std_dev: float) -> np.ndarray:
    """
    Add speckle noise to the SAR image patch.

    Parameters:
    patch (np.ndarray): The input image patch.
    std_dev (float): Standard deviation of the noise.

    Returns:
    np.ndarray: Noisy image patch.
    """
    # Generate speckle noise
    noise = np.random.gamma(2, 1, patch.shape) * std_dev
    # Add speckle noise to the patch
    noisy_patch = patch * noise
    # Clip the values to be in the valid range [0, 255]
    noisy_patch = np.clip(noisy_patch, 0, 255).astype(np.uint8)
    return noisy_patch


############### ---- 3 ---- ##############
# Write a Python function that filter the image patch
# Input: Image patch to filter, size of the filter, standard deviation of the filter
# Output: Filtered image
# Please take a special care with the border of the image patch (propose an extrapolation method)
# Compare your filter with Python implemented function


############### ---- 4 ---- ##############
# Write a Python function to compute the histogram of the image
# Input: Image patch
# Output: Histogram (vector providing the gray value occurences from 0 to 255).
# image patch as vector

def histogram(patch: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Compute the histogram of the image patch.

    Parameters:
    patch (np.ndarray): The input image patch.

    Returns:
    np.ndarray: Histogram of the image patch.
    """
    # Flatten the patch to a 1D array
    pixelList = patch.flatten(order='F')
    
    # Count gray value occurrences
    bins = np.zeros(n_bins, dtype=int)
    
    for i in range(n_bins):
        bins[i] = len(pixelList[pixelList == i])
    
    return bins


############### ---- 5 ---- ##############
# Write a Python function that compute the central moments of an histogram
# Input: Histogram (e.g. 256 vector)
# Output: Mean, variance, standard deviation, skewness, curtosis, excess


############### ---- 6 ---- ##############
# Write a Python function that calculate the normalized and cumulative histogram
# Input: Histogram
# Output: Two vectors for respectively the normalized and cumulative histogram


############### ---- 7 ---- ##############
# Write a Python function that performs the 2-sample Kolmogorov-Smirnov test
# Input: Cumulative histogram 1, Cumulative histogram 2, significance level
# Output: Vector of differences D, Decision (0 or 1)




if __name__ == "__main__":
    # Loading filepaths
    datapath = './data/'
    fileExtension = '.tif'
    filenames = [f for f in os.listdir(datapath) if f.endswith(fileExtension)]
    filepaths = [datapath + filename for filename in filenames]    
    
    # Loading specific image
    spec_img = 10
    
    # Load an example image (grayscale)
    image = cv2.imread(filepaths[spec_img], cv2.IMREAD_GRAYSCALE)
    
    
    ## TASK 1 ##
    # Define the center and half-side of the patch
    center = (789, 512)  # Example center
    half_side = 30       # Example half-side length

    # Get the patch
    patch = choose_patch(image, center, half_side)

    # # Display the original image and the patch
    # plt.subplot(1, 2, 1)
    # plt.imshow(image, cmap='gray')
    # plt.title('Original Image')
    
    # plt.subplot(1, 2, 2)
    # plt.imshow(patch, cmap='gray')
    # plt.title('Image Patch')
    
    # plt.show()
    
    
    ## TASK 2 ##
    if "optik" in filenames[spec_img]:
        # Add Gaussian noise to the patch
        noisy_patch = add_noise_optique(patch, std_dev=10, mean=0)
        print(filenames[spec_img])
    elif "SAR" in filenames[spec_img]:
        # Add speckle noise to the patch
        noisy_patch = add_noise_sar(patch, std_dev=0.1)
        print(filenames[spec_img])
        
    # plot the noisy patch
    plt.subplot(1, 2, 1)
    plt.imshow(patch, cmap='gray')
    plt.title('Original Patch')
    
    plt.subplot(1, 2, 2)
    plt.imshow(noisy_patch, cmap='gray')
    plt.title('Noisy Patch')
    
    plt.show()
        
    
    
    ## TASK 3 ##
    
    ## TASK 4 ##
    # Compute the histogram of the patch
    n_bins = 256
    hist = histogram(patch, n_bins)
    
    # # Plot the histogram
    # plt.figure()
    # plt.bar(np.arange(n_bins), hist, width=1, color='black', edgecolor='black')
    # plt.show()
    
    ## TASK 5 ##
    
    ## TASK 6 ##
    
    ## TASK 7 ##