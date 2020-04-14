import tensorflow as tf 
from keras.models import load_model
import cv2
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt
print('hi')
model = tf.keras.models.load_model('model1.h5')
print('doone !')

model.summary()
def labels(i):
    if i == 0: return 'Anime'
    else : return 'Human'

image_path='./52.jpg'
img = image.load_img(image_path, target_size=(150, 150))
#plt.imshow(img)
img = np.expand_dims(img, axis=0)
result=model.predict_classes(img)
print(labels(result[0][0]))
#plt.title(labels(result[0][0]))
plt.show()
