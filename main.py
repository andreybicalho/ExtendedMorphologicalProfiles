import numpy as np
from skimage.morphology import reconstruction
from skimage.morphology import erosion
from skimage.morphology import disk
from skimage import util
from skimage import io
import itertools


class ExtendedMorphologicalProfiles:

    def build_emp(self, base_image, se_size=4, se_size_increment=2, num_openings_closings=4):
        self.base_image = base_image
        base_image_rows, base_image_columns, base_image_channels = self.base_image.shape
        se_size = se_size
        se_size_increment = se_size_increment
        num_openings_closings = num_openings_closings
        self.morphological_profile_size = (num_openings_closings * 2) + 1
        self.emp_size = self.morphological_profile_size * base_image_channels
        self.emp = np.zeros(
            shape=(base_image_rows, base_image_columns, self.emp_size))

        cont = 0
        for i in range(base_image_channels):
            # build MPs
            mp_temp = self.build_morphological_profiles(
                base_image[:, :, i], se_size, se_size_increment, num_openings_closings)

            aux = self.morphological_profile_size * (i+1)

            # build the EMP
            cont_aux = 0
            for k in range(cont, aux):
                self.emp[:, :, k] = mp_temp[:, :, cont_aux]
                cont_aux += 1

            cont = self.morphological_profile_size * (i+1)

        return self.emp

    def build_morphological_profiles(self, image, se_size=4, se_size_increment=2, num_openings_closings=4):
        x, y = image.shape

        cbr = np.zeros(shape=(x, y, num_openings_closings))
        obr = np.zeros(shape=(x, y, num_openings_closings))

        it = 0
        tam = se_size
        while it < num_openings_closings:
            se = disk(tam)
            temp = self.closing_by_reconstruction(image, se)
            cbr[:, :, it] = temp[:, :]
            temp = self.opening_by_reconstruction(image, se)
            obr[:, :, it] = temp[:, :]
            tam += se_size_increment
            it += 1

        mp = np.zeros(shape=(x, y, (num_openings_closings*2)+1))
        cont = num_openings_closings - 1
        for i in range(num_openings_closings):
            mp[:, :, i] = cbr[:, :, cont]
            cont = cont - 1

        mp[:, :, num_openings_closings] = image[:, :]

        cont = 0
        for i in range(num_openings_closings+1, num_openings_closings*2+1):
            mp[:, :, i] = obr[:, :, cont]
            cont += 1

        return mp

    def opening_by_reconstruction(self, image, se):
        eroded = erosion(image, se)
        reconstructed = reconstruction(eroded, image)
        return reconstructed

    def closing_by_reconstruction(self, image, se):
        obr = self.opening_by_reconstruction(image, se)

        obr_inverted = util.invert(obr)
        obr_inverted_eroded = erosion(obr_inverted, se)
        obr_inverted_eroded_rec = reconstruction(
            obr_inverted_eroded, obr_inverted)
        obr_inverted_eroded_rec_inverted = util.invert(obr_inverted_eroded_rec)
        return obr_inverted_eroded_rec_inverted

    def get_emp(self):
        return self.emp

    def get_base_image(self):
        return self.base_image

    def get_emp_size(self):
        return self.emp_size

    def get_morphological_profile_size(self):
        return self.morphological_profile_size
    
    def get_number_of_base_images(self):
        base_image_rows, base_image_columns, base_image_channels = self.base_image.shape
        return base_image_channels



# 
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io

# Importing the dataset from Matlab format
dataset = io.loadmat('indianpines_dataset.mat')
number_of_bands = int(dataset['number_of_bands'])
number_of_rows = int(dataset['number_of_rows'])
number_of_columns = int(dataset['number_of_columns'])
pixels = np.transpose(dataset['pixels'])
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
pixels = sc.fit_transform(pixels)

groundtruth = io.loadmat('indianpines_gt.mat')
gt = np.transpose(groundtruth['pixels'])

# Preprocessing
# Applying Principal Components Analysis (PCA)
from sklearn.decomposition import PCA
number_of_pc = 4
pca = PCA(n_components=number_of_pc)
pc = pca.fit_transform(pixels)

# Visualizing PCs
fig = plt.figure(figsize=(15, 15))
columns = number_of_pc
rows = 1
pc_images = np.zeros(shape=(number_of_rows, number_of_columns, number_of_pc))
for i in range(number_of_pc):
    pc_images[:, :, i] = np.reshape(pc[:, i], (number_of_rows, number_of_columns))
    fig.add_subplot(rows, columns, i+1)
    plt.imshow(pc_images[:, :, i], cmap='gray', interpolation='bicubic')

plt.show()

# Building the Extended Morphological Profiles (EMP)
pc_images.shape
emp = ExtendedMorphologicalProfiles()
emp_image = emp.build_emp(base_image=pc_images)

# Visualizing the EMP
fig = plt.figure(figsize=(15, 15))
columns = emp.get_morphological_profile_size()
rows = emp.get_number_of_base_images()
print("EMP = "+str(emp_image.shape))
print("Number of Base Images: "+str(rows))
print("Morphological Profiles size: "+str(columns))

for i in range(1, emp.get_emp_size()+1):
    fig.add_subplot(rows, columns, i)
    plt.imshow(emp_image[:, :, i-1], cmap='gray', interpolation='bicubic')

plt.show()


# building dataset for classification
dim_x, dim_y, dim_z = emp_image.shape
dim = dim_x * dim_y

x = np.zeros(shape=(dim, dim_z))
y = gt

cont = 0
for i in range(dim_x):
    for j in range(dim_y):
        x[cont, :] = emp_image[i, j,:]
        cont += 1

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.75, random_state=0)

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualizing the results
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Plot non-normalized confusion matrix
class_names = ['background', 'alfalfa', 'corn-notill', 'corn-min', 'corn',
               'grass/pasture', 'grass/trees', 'grass/pasture-mowed', 'hay-windrowed', 'oats', 'soybeans-notill', 
               'soybeans-min', 'soybean-clean', 'wheat', 'woods', 'bldg-grass-tree-drives', 'stone-steel towers']

plt.figure()
plot_confusion_matrix(cm, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cm, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
