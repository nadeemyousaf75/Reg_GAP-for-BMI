
#!pip install tensorflow==1.14.0
#import tensorflow as tf
#!pip install keras==2.2

import keras

from keras.models import Sequential
from keras.optimizers import adam
from keras.constraints import max_norm
import numpy as np
import keras.backend as K
import sys
import numpy
import tensorflow as tf
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, GlobalAveragePooling1D, GlobalAveragePooling2D, Flatten,Add , Multiply ,Concatenate ,Dot
from keras import models
from keras.optimizers import SGD,Adam, RMSprop
#from keras.models import Sequential, Model
from keras.engine import  Model
from keras import regularizers
from matplotlib import pyplot as plt
from keras.layers import BatchNormalization
from google.colab.patches import cv2_imshow
import imutils
import cv2
from PIL import Image
from keras.preprocessing import image
from scipy import stats
from sklearn.metrics import mean_absolute_error

#"""#Connection with Google Collab"""

#from google.colab import drive
#drive.mount('/content/drive')

"""# Set Path

"""

#cd drive/My Drive/bmi/bolloywood

"""# Load Features"""

test_embeddings=np.load("/content/drive/My Drive/bmi/bolloywood/paper/Bollywood_Reg_GAP_test_features.npy")
test_bmi=np.load("/content/drive/My Drive/bmi/bolloywood/paper/Bollywood_Reg_GAP_test_bmi.npy")

"""# Model"""

model99=Sequential([
    Dense(512, activation='sigmoid', input_shape=(512,),kernel_initializer='glorot_uniform', bias_initializer='zeros'),
    Dropout(0.4),
    Dense(256, activation='sigmoid',kernel_initializer='glorot_uniform', bias_initializer='zeros' ),
    Dense(1, activation='linear',kernel_initializer='glorot_uniform', bias_initializer='zeros')])

adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=0.48, decay=0.0, amsgrad=True)
model99.compile(optimizer='adam',
             loss="mse",
             metrics=['accuracy'])

"""# Loading Weights"""

model99.load_weights("paper/Bollywood_Reg_GAP_weights.h5")

"""# Predict"""

test_bmii=model99.predict(test_embeddings)
    test_bmii_t = np.array([x[0] for x in test_bmii])
    print(mean_absolute_error(test_bmi, test_bmii_t))

