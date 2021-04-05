# Reg_GAP_for_BMI

Keras Version = 2.2
Tensorflow version=1.14.0
Python 3.0

For inference use BMI_REG_GAP.py
Set path to the directory which contains the files.
cd path

load the features and ground truth

test_embeddings=np.load("path")

test_bmi=np.load("path")

Load the provided weights of Bollywood dataset for REG_GAP.


model99.load_weights("path.h5")


BMI_Reg_GAP.ipynb contains all the step by step information regarding how to execute and it also shows results already computed in it.

Currently it includes model, weights and features for inference purpose, the complete code will be uploaded here subject to the acceptance of our paper.
