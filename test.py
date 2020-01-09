import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

model = tf.keras.models.load_model("number_guesser_2.model")
IMG_SIZE = 28

test_dir = "./test_images"

def get_img_array(img):
    img_array = cv2.imread(os.path.join(test_dir, img), cv2.IMREAD_GRAYSCALE)
    img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

    plt.imshow(img_array)
    plt.show()

    img_array = img_array.reshape(-1, IMG_SIZE, IMG_SIZE)
    img_array = np.abs((img_array / 255.0) - 1)

    return img_array


for img in os.listdir(test_dir):
    img_array = get_img_array(img)
    prediction = model.predict([img_array])
    print(np.argmax(prediction[0]))
