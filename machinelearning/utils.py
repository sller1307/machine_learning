import json
from os import listdir
import random
import tensorflow as tf


def load_json(filename):
    with open(filename, "r") as fp:
        data = json.load(fp)
    return data


def get_a_random_image(file_dir):
    imagefiles = [f for f in listdir(file_dir)]
    imagefile = file_dir + "/" + random.choice(imagefiles)
    return tf.keras.preprocessing.image.load_img(imagefile, target_size=(224, 224))
