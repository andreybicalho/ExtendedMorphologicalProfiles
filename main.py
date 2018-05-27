import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io

# Importing the dataset from Matlab format
dataset = io.loadmat('indianpines_dataset.mat')
number_of_bands = int(dataset['number_of_bands'])
number_of_rows = int(dataset['number_of_rows'])
number_of_columns = int(dataset['number_of_columns'])
pixels = np.transpose(dataset['pixels'])

# Applying Principal Components Analysis (PCA)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pc = pca.fit_transform(pixels)
pc_1 = pc[:, 0]
pc_2 = pc[:, 1]

# Visualizing PCs
pc_1_img = np.reshape(pc_1, (number_of_rows, number_of_columns))
pc_2_img = np.reshape(pc_2, (number_of_rows, number_of_columns))

plt.subplot(223)
plt.title('PC 1')
plt.imshow(pc_1_img, cmap='gray', interpolation='bicubic')

plt.subplot(224)
plt.title('PC 2')
plt.imshow(pc_2_img, cmap='gray', interpolation='bicubic')

plt.show()
