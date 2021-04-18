import glob
import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

import get_data as gd
from tensorflow.keras import preprocessing
from tensorflow import keras

# Load images
image_set_gs = []
for file in glob.glob("examples_post/*.jpg"):
    img = cv2.imread(file)
    image_set_gs.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    print("Reading: " + file)
    
# Store in np arrays
np_filter = np.array(image_set_gs, dtype=np.object)
np_original = np.copy(np_filter)

# Preprocessing
gd.filter_method_canny(np_filter, 3)

# umber of sampling blocks
res_xy = (24, 32)

# btains candidates for speech bubbles
bubble_set = []
gd.get_speech_bubble_candidates(np_original, np_filter, bubble_set, res_xy, 127)
bubble_set = np.array(bubble_set)
bubble_set = bubble_set

# Loads CNN and displays its arch.
model = keras.models.load_model("models/trained_model_50")
model.summary()

# For illustrative purposes, plots the convolutional kernels
layer = model.get_layer("conv2d")
w_t = layer.get_weights()[0]
norm_val = 1/np.amax(w_t)
for k in range(0, w_t.shape[3]):
    plt.subplot(4, 8, k+1)
    plt.imshow(w_t[:,:,:,k]*norm_val)
plt.show()

# Classifies and stores images
# No preprocessing is done on these
# i.e: It just outputs text
i = 0
for img in bubble_set:
    t_img = cv2.resize(img, (50, 50))
    t_img = np.reshape(t_img, (-1, *t_img.shape))
    pred_vec = model.predict([t_img])
    pred_class = np.argmax(pred_vec)
    if(pred_class != 0):
        out_img = img[:,:,1]
        cv2.imwrite("examples_post/output/a" + str(i) + ".png", out_img)
        i += 1

print("From " + str(bubble_set.shape[2]) + " candidates, extracted " + str(i) + " speech bubbles.")
print("EOP")
