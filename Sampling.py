import tqdm
import random
import pathlib
import itertools
import collections

import os
import cv2
import numpy as np
import remotezip as rz

import tensorflow as tf

# Some modules to display an animation using imageio.
import imageio
from IPython import display
from urllib import request
from tensorflow_docs.vis import embed

import misc
from preprocessing import *

NUM_CLASSES = 10
FILES_PER_CLASS = 50

# Path to your video
video_path = "./SAMPLE_WALK.mp4"

arr = frames_from_video_file(video_path, 20,(224,224))
rand_arr = random_frames_from_video(video_path, 20,(224,224))
to_gif(arr, "uniform_gamma_corrected.gif")
to_gif(rand_arr, "random.gif")