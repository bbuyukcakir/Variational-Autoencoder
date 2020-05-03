This repository houses my code for the variational autoencoder models and my results.
The present model is built with *PyTorch* and uses the "UTK Face Cropped" dataset from Kaggle 

(https://www.kaggle.com/abhikjha/utk-face-cropped)

The initial results are demonstrated below. The left-most column contains pictures from the dataset, and columns to the right of it contain
the corresponding "varied" outputs based on the original image.

![](results/from_faces.png)

#### Future work
The next step in this search is to change the model better compress the underlying distribution (currently 500-dimensional).
The model should also be adjusted in order to provide clearer images.
More traning time or an engineered loss function is likely necessary.
