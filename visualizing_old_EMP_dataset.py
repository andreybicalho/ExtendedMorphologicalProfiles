#
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io

# Importing the dataset from Matlab format
dataset = io.loadmat('indianpines_empPCAbased18_2pc_4se_2sei_4oc__toCompareWith.mat')

number_of_bands = int(dataset['number_of_bands'])
number_of_rows = int(dataset['number_of_rows'])
number_of_columns = int(dataset['number_of_columns'])
pixels = np.transpose(dataset['pixels'])

images = np.zeros(shape=(number_of_rows, number_of_columns, number_of_bands))
for i in range(number_of_bands):
    img = pixels[:, i]
    images[:, :, i] = np.reshape(img, (number_of_rows, number_of_columns))

fig = plt.figure(figsize=(15, 15))
columns = 9
rows = 2

for i in range(1, number_of_bands+1):
    fig.add_subplot(rows, columns, i)
    plt.imshow(images[:, :, i-1], cmap='gray', interpolation='bicubic')

plt.show()
