import numpy as np
from tensorflow.keras.datasets import cifar10
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Step 1: Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Flatten the labels
y_train = y_train.flatten()
y_test = y_test.flatten()

# Step 2: Preprocess the data
# Reshape images (32x32x3 to 3072) and normalize pixel values
x_train = x_train.reshape((x_train.shape[0], -1)) / 255.0
x_test = x_test.reshape((x_test.shape[0], -1)) / 255.0

# Standardize features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Step 3: Reduce dimensionality using PCA (optional)
pca = PCA(n_components=50)  # Retain top 50 components
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)

# Step 4: Train SVM (binary classification for classes 0 and 1)
binary_indices_train = np.where((y_train == 0) | (y_train == 1))[0]
binary_indices_test = np.where((y_test == 0) | (y_test == 1))[0]

x_train_binary = x_train_pca[binary_indices_train]
y_train_binary = y_train[binary_indices_train]
x_test_binary = x_test_pca[binary_indices_test]
y_test_binary = y_test[binary_indices_test]

svm = SVC(kernel='rbf', C=1, gamma='scale')
svm.fit(x_train_binary, y_train_binary)

# Step 5: Evaluate the model
y_pred = svm.predict(x_test_binary)
accuracy = accuracy_score(y_test_binary, y_pred)

print(f"Binary Classification Accuracy: {accuracy:.2f}")

# Step 6: Multi-class classification (one-vs-all)
svm_multiclass = SVC(kernel='rbf', C=1, gamma='scale', decision_function_shape='ovr')
svm_multiclass.fit(x_train_pca, y_train)

y_pred_multiclass = svm_multiclass.predict(x_test_pca)
accuracy_multiclass = accuracy_score(y_test, y_pred_multiclass)

print(f"Multi-class Classification Accuracy: {accuracy_multiclass:.2f}")
