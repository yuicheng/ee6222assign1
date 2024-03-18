import cv2
import tensorflow as tf
import numpy as np
import random
import imageio
import pathlib
from tensorflow_docs.vis import embed
from PIL import Image

def format_frames(frame, output_size):
  """
    Pad and resize an image from a video.
    
    Args:
      frame: Image that needs to resized and padded. 
      output_size: Pixel size of the output frame image.

    Return:
      Formatted frame with padding of specified output size.
  """
  
  frame = tf.image.convert_image_dtype(frame, tf.float32)
  # frame = tf.image.adjust_gamma(frame, 0.5)
  # layer = tf.keras.layers.Normalization(mean=[0.07,0.07,0.07], variance=[(lambda x : x ** 2) (x) for x in [0.1, 0.09, 0.08]])
  # frame = layer(frame)
  frame = tf.image.per_image_standardization(frame)
  frame = tf.image.resize_with_pad(frame, *output_size)
  return frame

def frames_from_video_file(video_path: str, frame_cnt: int, output_size = (224, 224), step = 10):
  """
    Creates frames from each video file present for each category.

    Args:
      video_path: File path to the video.
      frame_cnt: frames to be extracted from the video
      output_size: Output resolution, default to 320x240 as the resolution provided in the dataset
      step: interval of uniform sampling, default is 10.

    Return:
      An NumPy array of frames in the shape of (frame_cnt, height, width, channels).
  """
  # Read each video frame by frame
  result = []
  captured = cv2.VideoCapture(str(video_path))  

  video_length = captured.get(cv2.CAP_PROP_FRAME_COUNT)

  need_length = 1 + (frame_cnt - 1) * step

  if need_length > video_length:
    start = 0
  else:
    max_start = video_length - need_length
    start = random.randint(0, max_start + 1)

  captured.set(cv2.CAP_PROP_POS_FRAMES, start)

  # ret is a boolean indicating whether read was successful, frame is the image itself
  ret, frame = captured.read()
  result.append(format_frames(frame, output_size))

  for _ in range(frame_cnt - 1):
    for _ in range(step):
      ret, frame = captured.read()
    if ret:
      frame = format_frames(frame, output_size)
      result.append(frame)
    else:
      # add padding
      result.append(np.zeros_like(result[0]))

  captured.release()

  # BGR2RGB
  result = np.array(result)[..., [2, 1, 0]]

  return result

def random_frames_from_video(video_path: str, frame_cnt: int, output_size = (224, 224)):
  
  result = []
  captured = cv2.VideoCapture(video_path)
  if not captured.isOpened():
    return result
  
  frames = int(captured.get(cv2.CAP_PROP_FRAME_COUNT))
  indices = random.sample(range(frames), frame_cnt)
  
  
  for idx in indices:
    captured.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = captured.read()
    frame = format_frames(frame, output_size)
    result.append(frame)
    
  captured.release()
  
  # BGR2RGB
  result = np.array(result)[..., [2, 1, 0]]
  
  return result
    

def to_gif(images, gif_fname = 'animation.gif'):
  '''
    Generate a gif from an NumPy array.

    Args:
        images: A NumPy Array.
        gif_fname: Filename of the GIF file to save, default is animation.gif
  '''
  converted_images = np.clip(images * 255, 0, 255).astype(np.uint8)
  imageio.mimsave(gif_fname, converted_images, fps=10)
  return embed.embed_file(gif_fname)

