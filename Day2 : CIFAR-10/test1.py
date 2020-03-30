import tensorflow as tf 
from keras.models import load_model
import cv2
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt
print('hi')
model = tf.keras.models.load_model('model19march.h5')
print('doone !')

model.summary()


image_path='static/6.png'
img = image.load_img(image_path, target_size=(32, 32))
plt.imshow(img)
img = np.expand_dims(img, axis=0)

labels =  ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']


result=model.predict(img)
print(result.dtype)
print((result))
tag = (result.argmax(axis=-1))[0]
ans = labels[tag]
print(ans)
# ans = labels(result)
plt.title(ans)

# print(ans)
plt.show()
