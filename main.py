import tensorflow as tf
from tensorflow import keras
from keras import layers, utils
import numpy as nm
import matplotlib.pyplot as plt
from PIL import Image
 
import FrameGenerator

print (tf.__version__)

# model = keras.application.DenseNet121(weight = "imagenet", include_top = False)

NUM_CLASSES = 10
FILES_PER_CLASS = 50
