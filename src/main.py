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

##### sehr WIP und noch nicht sinnvoll
# seperat, da die Funktion erst auf alle Zeilen, dann auf alle Spalten angewendet wird
def gauss_filter(x, filter_std):
     gauss = np.exp(-(x**2)/(2*filter_std**2)) /(filter_std*np.sqrt(2*np.pi))
     return gauss

def img_filter(patch: np.ndarray, filter_size: int, filter_std: float) -> np.ndarray:
    """
    Filter the image (patch).

    Parameters: 
    patch (np.ndarray): The input image patch.
    filter_size (int): size of the row filter in pixels.
    filter_std (float): standard deviation of the filter.

    Returns:
    np.ndarray: Filtered image patch. 
    """
# da std übergeben wird, kommen Boxfilter wie der Mittelwertfilter nicht in Frage -> Binomialfilter oder Gaußfilter
# Patch nur um die halbe Filterbreite erweitern -> entweder aus Originalbild abgreifen oder den Patch spiegeln
# Bsp: Filtergröße 5 -> startet am linken Rand
#      Es fehlen links außerhalb des Patches 2 Pixel [=floor(filter_size/2) Pixel]
#      2D Gaußfilter in Zeilen/Spalten seperierbar -> erst in Zeilen, dann das Zwischenergebnis nach Spalten filtern 
#                                                  => gesamtgefilteres Bild = Ergebnis
    patch_bounds = patch.shape[:2]
    extended_patch = mirror_image(patch)
    half_filter_size = filter_size//2
    x = np.arange(-half_filter_size+1,half_filter_size+1,1)
    gauss_kernel = gauss_filter(x, filter_std)
    # in einer Schleife muss an jeder Stelle des Patches der Filter zwei mal angesetzt werden
    
    filtered_patch = np.zeros(patch_bounds)
    for row in range(patch_bounds[0]):
        for column in range(patch_bounds[1]):
            new_value = gauss_kernel @ extended_patch[patch_bounds[0]+row, patch_bounds[1]+column-half_filter_size:patch_bounds[1]+column+half_filter_size]
            print(f"new_value: {new_value}")
            filtered_patch[row, column] = 0

        # pxl are rows
        # run over all positions in the patch, but use the values of the extended patch
        print('.')

    filtered_patch = row_filtered_patch
    return filtered_patch

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
    n_bins (int): Number of bins.

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


def central_moments(hist):
    # Wahrscheinlichkeitsverteilung
    p = hist / np.sum(hist)

    # mean
    mue = np.sum(np.arange(len(hist)) * p)

    # variance
    var = np.sum((np.arange(len(hist)) - mue)**2 * p)

    # standard deviation
    sigma = np.sqrt(var)

    # skewness
    gamma_1 = (1 / (sigma**3)) * np.sum((np.arange(len(hist)) - mue)**3 * p)

    # curtosis
    cur = (1 / (sigma**4)) * np.sum((np.arange(len(hist)) - mue)**4 * p)

    # excess
    gamma_2 = cur - 3

    return mue, var, sigma, gamma_1, cur, gamma_2

def plot_histogram_with_moments(hist, moments, n_bins=256):
    """
    Plottet das Histogramm, markiert Mittelwert und Standardabweichung und zeigt die zentralen Momente als Text an.
    """
    x = np.arange(n_bins)
    mean, var, std, skew, kurt, excess = moments

    plt.figure(figsize=(10,5))
    plt.bar(x, hist, width=1, color='gray', alpha=0.7, label='Histogramm')

    # Vertikale Linien für Mittelwert und Standardabweichung
    plt.axvline(mean, color='red', linestyle='--', linewidth=2, label='Mittelwert')
    plt.axvline(mean + std, color='blue', linestyle=':', linewidth=2, label='Mittelwert ± Std')
    plt.axvline(mean - std, color='blue', linestyle=':', linewidth=2)

    # Textbox mit Momenten
    textstr = '\n'.join((
        r'$\mathrm{Mittelwert}=%.2f$' % (mean, ),
        r'$\mathrm{Varianz}=%.2f$' % (var, ),
        r'$\mathrm{Std}=%.2f$' % (std, ),
        r'$\mathrm{Schiefe}=%.2f$' % (skew, ),
        r'$\mathrm{Kurtosis}=%.2f$' % (kurt, ),
        r'$\mathrm{Exzess}=%.2f$' % (excess, )))
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    plt.text(0.98, 0.98, textstr, transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.title('Histogramm mit zentralen Momenten und Mittelwert/Std')
    plt.xlabel('Grauwert')
    plt.ylabel('Anzahl')
    plt.legend()
    plt.tight_layout()
    plt.show()


############### ---- 6 ---- ##############
# Write a Python function that calculate the normalized and cumulative histogram
# Input: Histogram
# Output: Two vectors for respectively the normalized and cumulative histogram
def normalized_and_cumulative_histogram(hist: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Compute the normalized and cumulative histogram.

    Parameters:
    hist (np.ndarray): The input histogram (z.B. 256er Vektor).

    Returns:
    tuple: (normalized_hist, cumulative_hist), beide als np.ndarray
    """
    # Summe berechnen 
    total = 0
    for h in hist:                                      # Schleife summiert alle Werte im Histogramm, um die Gesamtanzahl zu bestimmen
        total += h

    normalized_hist = np.zeros_like(hist, dtype=float)
    cumulative_hist = np.zeros_like(hist, dtype=float)
    cumulative_sum = 0.0

    for i in range(len(hist)):
        # Normalisieren
        if total > 0:
            normalized_hist[i] = hist[i] / total        # Wert wird durch die Gesamtsumme geteilt, um die relative Häufigkeit zu berechnen
        else:
            normalized_hist[i] = 0.0
        # Kumulieren
        cumulative_sum += normalized_hist[i]            # normalisierten Werte werden aufsummiert, um das kumulierte Histogramm zu erstellen
        cumulative_hist[i] = cumulative_sum

    return normalized_hist, cumulative_hist

############### ---- 7 ---- ##############
# Write a Python function that performs the 2-sample Kolmogorov-Smirnov test
# Input: Cumulative histogram 1, Cumulative histogram 2, significance level
# Output: Vector of differences D, Decision (0 or 1)


def ks_test(cumulative_hist1, cumulative_hist2, alpha):
    # vector of differences
    D = np.abs(cumulative_hist1 - cumulative_hist2)

    # maximum difference
    D_max = np.max(D)

    # value KS-test
    n = len(cumulative_hist1)
    m = len(cumulative_hist2)
    c = 1.628 # alpha = 0.01
    temp = c * np.sqrt((n + m) / (n * m))

    # decision
    decision = 1 if D_max > temp else 0

    return D, decision




# Test

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
    # plt.subplot(1, 2, 1)
    # plt.imshow(patch, cmap='gray')
    # plt.title('Original Patch')
    
    # plt.subplot(1, 2, 2)
    # plt.imshow(noisy_patch, cmap='gray')
    # plt.title('Noisy Patch')
    
    # plt.show()
        
    
    
    ## TASK 3 ##
    filtered_patch = img_filter(noisy_patch,5,0.5)
    
    ## TASK 4 ##
    # Compute the histogram of the patch
    n_bins = 256
    hist = histogram(patch, n_bins)
    
    # # Plot the histogram
    plt.figure()
    plt.bar(np.arange(n_bins), hist, width=1, color='black', edgecolor='black')
    plt.show()
    
    ## TASK 5 ##
    central_moments_hist = central_moments(hist)
    plot_histogram_with_moments(hist, central_moments_hist, n_bins)
    
    ## TASK 6 ##
    hist_norm_kum = normalized_and_cumulative_histogram(hist)
    norm_hist = hist_norm_kum[0]
    cum_hist = hist_norm_kum[1]

    # Histogramme plotten
    x = np.arange(n_bins)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    # Absolutes Histogramm
    axs[0].bar(x, hist, width=1, color='gray')
    axs[0].set_title('Absolutes Histogramm')
    axs[0].set_xlabel('Grauwert')
    axs[0].set_ylabel('Anzahl')
    # Normalisiertes Histogramm
    axs[1].bar(x, norm_hist, width=1, color='blue')
    axs[1].set_title('Normalisiertes Histogramm')
    axs[1].set_xlabel('Grauwert')
    axs[1].set_ylabel('Relative Häufigkeit')
    # Kumuliertes Histogramm (als Linie)
    axs[2].plot(x, cum_hist, color='red')
    axs[2].set_title('Kumuliertes Histogramm')
    axs[2].set_xlabel('Grauwert')
    axs[2].set_ylabel('Kumulierte Summe')
    plt.tight_layout()
    plt.show()

    ## alle drei Histogramme überlagert ##
    fig, ax1 = plt.subplots(figsize=(10,5))
    # Absolutes Histogramm (graue Balken)
    ax1.bar(x, hist, width=1, color='gray', alpha=0.4, label='Absolut')
    # Normalisiertes Histogramm (blaue Balken, kleinere Höhe)
    ax1.bar(x, norm_hist * hist.max(), width=1, color='blue', alpha=0.4, label='Normalisiert')
    ax1.set_xlabel('Grauwert')
    ax1.set_ylabel('Anzahl (links)')
    #ax1.legend(loc='upper left')
    # Zweite y-Achse für das kumulierte Histogramm
    ax2 = ax1.twinx()
    ax2.plot(x, cum_hist, color='red', linewidth=2, label='Kumuliert')
    ax2.set_ylabel('Kumulierte Summe (rechts, normiert)')
    ax2.set_ylim(0, 1.05)
    # Legenden zusammenführen
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower right')
    plt.title('Absolutes, normalisiertes und kumuliertes Histogramm')
    plt.tight_layout()
    plt.show()
    
    ## TASK 7 ##
    # 2 Varianten die geteste werden sollen: 
    #    1)Zwei Bildpatches vergleichen
    #    2) Patch vs. Gaußverteilung(Normalverteilung) vergleichen

    # Variante 1
    # zweites Bildpatch erstellen
    #center2 = (789, 562)  # z.B. 50 Pixel weiter rechts: Decision = 0
    center2 = (150, 176) # weit genug verschoben für Decision = 1
    half_side = 30
    patch2 = choose_patch(image, center2, half_side)

    #Histogramm für Bildpatch 2
    hist2 = histogram(patch2, n_bins)
    _, cum_hist2 = normalized_and_cumulative_histogram(hist2)

    # Testaufruf für beide Bildpatches
    D, decision = ks_test(cum_hist, cum_hist2, alpha=0.05)
    print("KS-Test Patch vs. Patch: D_max =", np.max(D), "Entscheidung:", decision)

    #Visualisierung
    plt.plot(cum_hist, label='Patch 1')
    plt.plot(cum_hist2, label='Patch 2')
    plt.legend()
    plt.title('Kumulierte Histogramme der Patches')
    plt.show()

    # Variante 2
    # Gaußverteilung erzeugen
    mean = np.mean(patch)
    std = np.std(patch)
    x = np.arange(n_bins)
    gauss_pdf = (1/(std * np.sqrt(2 * np.pi))) * np.exp(-((x - mean)**2) / (2 * std**2))

    # Normierung und Kumulierung durch eigene Funktion
    _, gauss_cum = normalized_and_cumulative_histogram(gauss_pdf)

    # Funktionsaufruf
    D_gauss, decision_gauss = ks_test(cum_hist, gauss_cum, alpha=0.05)
    print("KS-Test Patch vs. Gauß: D_max =", np.max(D_gauss), "Entscheidung:", decision_gauss)

    # Visualisierung
    plt.plot(cum_hist, label='Patch')
    plt.plot(gauss_cum, label='Gauß')
    plt.legend()
    plt.title('Kumuliertes Histogramm vs. Gauß')
    plt.show()
