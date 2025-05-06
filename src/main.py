############## ---- 1 ---- ##############
# Write a Matlab/Python function to choose a patch on an image
# Input: Image, Patch center, Half-side of the patch
# Output: Image patch


############### ---- 2 ---- ##############
# Write a Matlab/Python function that adds noise on the image patch
# Input: Image patch, type of noise (Gaussian for optique, speckle for SAR), standard deviation
# Output: Noisy patch


############### ---- 3 ---- ##############
# Write a Matlab/Python function that filter the image patch
# Input: Image patch to filter, size of the filter, standard deviation of the filter
# Output: Filtered image
# Please take a special care with the border of the image patch (propose an extrapolation method)
# Compare your filter with Python implemented function


############### ---- 4 ---- ##############
# Write a Matlab/Python function to compute the histogram of the image
# Input: Image patch
# Output: Histogram (vector providing the gray value occurences from 0 to 255).
# image patch as vector
pixelList = patch.flatten(order='F')

# count gray value occurences (all possible 256)
# schleife über alle Pixel suche nach einem Wert
# gesuchten wert um 1 erhöhen
# repeat

bins = np.zeros(256)
targetValue = 0
for i in range(256):
    bins[i] = len(pixelList[pixelList == targetValue])
    targetValue += 1

plt.figure()
plt.plot(bins)
plt.show()

############### ---- 5 ---- ##############
# Write a Matlab/Python function that compute the central moments of an histogram
# Input: Histogram (e.g. 256 vector)
# Output: Mean, variance, standard deviation, skewness, curtosis, excess


############### ---- 6 ---- ##############
# Write a Matlab/Python function that calculate the normalized and cumulative histogram
# Input: Histogram
# Output: Two vectors for respectively the normalized and cumulative histogram


############### ---- 7 ---- ##############
# Write a Matlab/Python function that performs the 2-sample Kolmogorov-Smirnov test
# Input: Cumulative histogram 1, Cumulative histogram 2, significance level
# Output: Vector of differences D, Decision (0 or 1)