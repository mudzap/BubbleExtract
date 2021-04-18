import glob
import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

import get_data as gd
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
#from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import preprocessing
from tensorflow import keras

# Cargar imagenes
image_set_gs = []
for file in glob.glob("examples_post/*.jpg"):
    img = cv2.imread(file)
    image_set_gs.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    print("Reading: " + file)
    
# Hacerlas en una matriz de numpy
np_filter = np.array(image_set_gs, dtype=np.object)
np_original = np.copy(np_filter)

# Filtrarlas por cualquier metodo
gd.filter_method_canny(np_filter, 3)

bubble_set = []
res_xy = (24, 32)

gd.get_speech_bubble_candidates(np_original, np_filter, bubble_set, res_xy, 127)
bubble_set = np.array(bubble_set)
bubble_set = bubble_set

#Carga la red
model = keras.models.load_model("models/trained_model_50")
model.summary()

layer = model.get_layer("conv2d")
w_t = layer.get_weights()[0]

norm_val = 1/np.amax(w_t)

for k in range(0, w_t.shape[3]):
    plt.subplot(4, 8, k+1)
    plt.imshow(w_t[:,:,:,k]*norm_val)
plt.show()


i = 0
for img in bubble_set:
    t_img = cv2.resize(img, (50, 50))
    t_img = np.reshape(t_img, (-1, *t_img.shape))
    pred_vec = model.predict([t_img])
    pred_class = np.argmax(pred_vec)
    if(pred_class != 0):
        out_img = img[:,:,1] #cv2.threshold(img[:,:,1], 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        cv2.imwrite("examples_post/output/a" + str(i) + ".png", out_img)
        i += 1

#cv2.waitKey(0)
#cv2.destroyAllWindows()
print("EOP")
