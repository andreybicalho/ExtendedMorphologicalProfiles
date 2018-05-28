# Remote Sensed Hyperspectral Image Classification With Extended Morphological Profiles and Support Vector Machines

This is an example of how to use the Extended Morphological Profiles to classify remote sensed hyperspectral images.

# Extended Morphological Profiles

Extended Morphological Profiles (EMP) is a simple and effective technique to encode both spectral and spatial information in the classification process. This method connects similar structures through morphological operations and keeps the essential spectral information by using some feature-extraction method such as Principal Component Analysis (PCA).

# Indian Pines Dataset

This scene was gathered by [AVIRIS sensor](https://aviris.jpl.nasa.gov/) recorded over Northwestern Indiana, USA, and consists of 145x145 pixels and 224 spectral reflectance bands in the wavelength range 0.4–2.5 10^(-6) meters. The Indian Pines scene contains two-thirds agriculture, and one-third forest or other natural perennial vegetation. There are two major dual lane highways, a rail line, as well as some low density housing, other built structures, and smaller roads. The ground truth available is designated into sixteen classes (seventeen if you consider the background) and is not all mutually exclusive. It is also a very common practice reducing the number of bands to 200 by removing bands covering the region of water absorption: [104-108], [150-163], 220. Indian Pines data are available through [Pursue's univeristy MultiSpec site](https://engineering.purdue.edu/~biehl/MultiSpec/hyperspectral.html).

# Classification

In this example the Support Vector Machine (SVM) machine learning algorithm, with the Radial Basis Function (RBF) Kernel, was used for the classification.

Consider [this paper](http://ieeexplore.ieee.org/abstract/document/6490435/) for further reading:
A. B. Santos, A. de Albuquerque Araujo, and D. Menotti, “Combining multiple classification methods for hyperspectral data interpretation,” IEEE J. Sel. Topics Appl. Earth Observ. Remote Sens., vol. 6, no. 3, pp. 1450–1459, Jun. 2013.