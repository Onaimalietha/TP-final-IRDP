from scipy.stats import skew, kurtosis
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2 
from sklearn.manifold import TSNE
import keras
from keras import layers
from keras import ops

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
assert x_train.shape == (50000, 32, 32, 3)
assert x_test.shape == (10000, 32, 32, 3)
assert y_train.shape == (50000, 1)
assert y_test.shape == (10000, 1)

classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

def plot_sample(x, y, index):
    plt.figure(figsize = (15,2))
    plt.imshow(x_train[index])
    plt.title(classes[int(y[index])])
    plt.show()

def show_histograms(img : np.ndarray):
    plt.axis("off")
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    blue_color = cv2.calcHist([img], [0], None, [256], [0, 256])
    green_color = cv2.calcHist([img], [1], None, [256], [0, 256])
    red_color = cv2.calcHist([img], [2], None, [256], [0, 256])

    # Blue histogram.
    plt.subplot(3, 1, 1)
    plt.title("Blue Histogram")
    plt.plot(blue_color, color = "blue")
    
    # Green histogram.
    plt.subplot(3, 1, 2)
    plt.title("Green Histogram")
    plt.plot(green_color, color = "green")

    # Blue histogram.
    plt.subplot(3, 1, 3)
    plt.title("Red Histogram")
    plt.plot(red_color, color = "red")
    
    # for clear view 
    plt.tight_layout() 
    plt.show() 

#show_histograms(x_train[0, :])

def compute_color_histograms(images, bins=16):
    """
    Compute color histograms for a set of images.
    
    Args:
        images: NumPy array of shape (n_samples, height, width, 3)
        bins: Number of bins for each color channel histogram.
    
    Returns:
        histograms: NumPy array of shape (n_samples, bins * 3)
    """
    histograms = []
    for img in images:
        # Compute histograms for each channel (R, G, B)
        r_hist = np.histogram(img[:, :, 0], bins=bins, range=(0, 1), density=True)[0]
        g_hist = np.histogram(img[:, :, 1], bins=bins, range=(0, 1), density=True)[0]
        b_hist = np.histogram(img[:, :, 2], bins=bins, range=(0, 1), density=True)[0]
        # Concatenate histograms
        hist = np.concatenate([r_hist, g_hist, b_hist])
        histograms.append(hist)
    return np.array(histograms)
#plot_sample(x_train, y_train, 17357)

def compute_color_features(images):
    """
    Compute 4 statistical features (mean, std, skewness, kurtosis) for each color channel (R, G, B).
    
    Args:
        images: NumPy array of shape (n_samples, height, width, 3)
    
    Returns:
        features: NumPy array of shape (n_samples, 12), where 12 = 4 features * 3 channels
    """
    features = []
    for img in images:
        img_features = []
        for channel in range(3):  # Loop over R, G, B channels
            pixels = img[:, :, channel].flatten()  # Flatten channel values
            img_features.extend([
                np.mean(pixels),
                np.std(pixels),
                skew(pixels),
                kurtosis(pixels)
            ])
        features.append(img_features)
    return np.array(features)

# Normalize images and compute features
x_train_norm = x_train / 255.0  # Ensure normalization
features = compute_color_features(x_train)

# Reduce dataset size for efficiency
sample_size = 500
indices = np.random.choice(features.shape[0], sample_size, replace=False)
x_sample = features[indices]
y_sample = y_train[indices]

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=50, n_iter=250, learning_rate=300)
x_embedded = tsne.fit_transform(x_sample)

# Plot the t-SNE embedding
plt.figure(figsize=(12, 8))
scatter = plt.scatter(x_embedded[:, 0], x_embedded[:, 1], c=y_sample.flatten(), cmap="tab10", alpha=0.7)
plt.colorbar(scatter, ticks=range(len(classes)), label='Classes')
plt.title("t-SNE Embedding of CIFAR-10 Subset (Statistical Features)")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.show()