import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

def grey_save(prediction, filename):
  formatted = (prediction * 255 / np.max(prediction)).astype("uint8")
  depth = Image.fromarray(formatted)
  depth.save(filename)

def grey(prediction):
  formatted = (prediction * 255 / np.max(prediction)).astype("uint8")
  depth = Image.fromarray(formatted)
  return depth

def write_depth(prediction, bits, filename):
  depth_min = prediction.min()
  depth_max = prediction.max()

  max_val = (2 ** (8 * bits)) - 1

  if depth_max - depth_min > np.finfo("float").eps:
      out = max_val * (prediction - depth_min) / (depth_max - depth_min)
  else:
      out = np.zeros(prediction.shape, dtype=prediction.dtype)

  cv2.imwrite(filename, out.astype("uint16"), [cv2.IMWRITE_PNG_COMPRESSION, 0])
  
  return

def inferno(prediction, filename):
  plt.imsave(filename, prediction, cmap="inferno")
