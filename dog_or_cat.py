from tensorflow import keras
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.utils import array_to_img
from tensorflow.python.ops.numpy_ops import np_config
from PIL import ImageDraw, ImageFont

np_config.enable_numpy_behavior()
import numpy as np

IMG_SIZE = 160 # All images will be resized to 160x160

model = keras.models.load_model('dogs_vs_cats.h5')

# load the image
picture='cat_vs_dog/dog3.jpg'
img = load_img(picture)
img_conv = load_img(picture, target_size=(IMG_SIZE, IMG_SIZE))
print("NumPy array info:")
img_array=img_to_array(img_conv)
img_array = (img_array / 127.5) - 1
img_conv = array_to_img(img_array)
img_array = img_array.reshape(1, IMG_SIZE, IMG_SIZE, 3)  # reshape image
prediction = model(img_array)
class_names = ['Cat', 'Dog']
#print(model.summary())
print(prediction[0][0])
print(np.sign((int)(prediction[0][0])))
print ("It is a " + class_names[np.sign((int)(prediction[0][0]))] + "!")

answer = (class_names[np.sign((int)(prediction[0][0]))])

draw = ImageDraw.Draw(img)

font = ImageFont.truetype("arial.ttf", 30, encoding="unic")
draw.text((0, 0),answer,(0,0,0),font)

img.show()
