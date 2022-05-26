#!/usr/bin/python

import os
import requests
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import torch
from image_preprocessing import *
from GLPN import GLPN
from NeWCFs import NeWCFs
import pykitti
import numpy as np
import pickle
from visualiser import *
from read_depth import depth_read, depth_read_nyu
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
import glob
from multiprocessing.dummy import Pool as ThreadPool
import pathlib

def compute_errors(gt, pred, drive, preprocessing_type, dataset, kitti: bool = False):
  mask = np.logical_and(gt < 80.0, gt > 1e-3)

  pred[np.isinf(pred)] = 80.0
  pred[np.isnan(pred)] = 1e-3
  pred = pred[mask].astype(np.float32) * (8 if kitti and model_name == 'glpn' else 1)
  gt = gt[mask].astype(np.float32)

  thresh = np.maximum((gt / pred), (pred / gt))
  d1 = (thresh < 1.25 ).mean()
  d2 = (thresh < 1.25 ** 2).mean()
  d3 = (thresh < 1.25 ** 3).mean()
  
  rmse = (gt - pred) ** 2
  rmse = np.sqrt(rmse.mean())

  rmse_log = (np.log(gt) - np.log(pred)) ** 2
  rmse_log = np.sqrt(rmse_log.mean())

  abs_rel = np.mean(np.abs(gt - pred) / gt)
  sq_rel = np.mean(((gt - pred)**2) / gt)

  err = np.log(pred) - np.log(gt)
  silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

  err = np.abs(np.log10(pred) - np.log10(gt))
  log10 = np.mean(err)

  row = {"Preprocessing Type":preprocessing_type, 
          "Dataset":dataset, 
          "silog":silog, 
          "log10":log10, 
          "abs_rel":abs_rel, 
          "sq_rel":sq_rel, 
          "rmse":rmse, 
          "rmse_log":rmse_log,    
          "d1":d1, 
          "d2":d2, 
          "d3":d3}
          
  return row


def execute_kitti(drive:str, pbar):
  basedir = 'data/KITTI'
  date = '2011_09_26'
  df = pd.DataFrame(columns=["Preprocessing Type", "Dataset", "Drive", "silog", "log10", "rmse", "rmse_log", "abs_rel", "sq_rel", "d1", "d2", "d3"])

  data = pykitti.raw(basedir, date, drive)
  
  path = lambda x: os.path.join(os.curdir, "Image", model_name, "kitti", x, f"{drive}_{i}.png")

  i = 0
  for camera in ["02", "03"]:
    for gt_filename in os.listdir(f"data/KITTI/train/{date}_drive_{drive}_sync/proj_depth/groundtruth/image_{camera}"):
      pbar.update(0)
      gt = depth_read(f"data/KITTI/train/{date}_drive_{drive}_sync/proj_depth/groundtruth/image_{camera}/{gt_filename}")
      image = Image.open(f"data/KITTI/{date}/{date}_drive_{drive}_sync/image_{camera}/data/{gt_filename}")
      
      height = image.height
      width = image.width
      top_margin = int(height - 352)
      left_margin = int((width - 1216) / 2)
      gt = gt[top_margin:top_margin + 352, left_margin:left_margin + 1216]
      image = image.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))

      inferno(gt, path("GT"))

      
      # Original
      prediction = model.predict_cpu(image)
      inferno(prediction, path("Original"))
      df.loc[len(df) + 1] = compute_errors(gt, prediction, drive, "Original", "KITTI", True)

      # Grey
      prediction_grey = model.predict_cpu(greyscale(image))
      inferno(prediction_grey, path("Grey"))
      df.loc[len(df) + 1] = compute_errors(gt, prediction_grey, drive, "Greyscale", "KITTI", True)

      # Edge
      prediction_edge = model.predict_cpu(edge(image))
      inferno(prediction_edge, path("Edge"))
      df.loc[len(df) + 1] = compute_errors(gt, prediction_edge, drive, f"Edge Detection", "KITTI", True)

      # Edge Color 5
      prediction_gdege_grey_5 = model.predict_cpu(edge_col(image, 5))
      inferno(prediction_gdege_grey_5, path("EdgeCol5"))
      df.loc[len(df) + 1] = compute_errors(gt, prediction_gdege_grey_5, drive, f"Edge Detection and Colour Threshold 5", "KITTI", True)

      # Edge Color 50
      prediction_gdege_grey_50 = model.predict_cpu(edge_col(image, 50))
      inferno(prediction_gdege_grey_50, path("EdgeCol50"))
      df.loc[len(df) + 1] = compute_errors(gt, prediction_gdege_grey_50, drive, f"Edge Detection and Colour Threshold 50", "KITTI", True)

      # Edge Grey 5
      prediction_gdege_grey_5 = model.predict_cpu(edge_grey(image, 5))
      inferno(prediction_gdege_grey_5, path("EdgeGrey5"))
      df.loc[len(df) + 1] = compute_errors(gt, prediction_gdege_grey_5, drive, f"Edge Detection and Greyscale Threshold 5", "KITTI", True)

      # Edge Grey 50
      prediction_gdege_grey_50 = model.predict_cpu(edge_grey(image, 50))
      inferno(prediction_gdege_grey_50, path("EdgeGrey50"))
      df.loc[len(df) + 1] = compute_errors(gt, prediction_gdege_grey_50, drive, f"Edge Detection and Greyscale Threshold 50", "KITTI", True)

      # Desaturated
      prediction_desat = model.predict_cpu(desaturated(image, 0.5))
      inferno(prediction_desat, path("Desat"))
      df.loc[len(df) + 1] = compute_errors(gt, prediction_desat, drive, "Desaturated", "KITTI", True)

      # Over Saturated
      prediction_sat = model.predict_cpu(desaturated(image, 2))
      inferno(prediction_sat, path("Sat"))
      df.loc[len(df) + 1] = compute_errors(gt, prediction_sat, drive, "Oversaturated", "KITTI", True)
      
      i += 1
    pbar.update(1)
    df.to_csv(f"eval/{model_name}/eval_kitti_{drive}.csv")


def execute_nyu(directory:str, pbar):
  i = 0
  df = pd.DataFrame(columns=["Preprocessing Type", "Dataset", "silog", "log10", "rmse", "rmse_log", "abs_rel", "sq_rel", "d1", "d2", "d3"])
  drive = 0
  path = lambda x: os.path.join(os.curdir, "Image", model_name, "nyu", x, f"{directory}_{i}.png")

  for filename in glob.glob(f'data/nyu/{directory}/rgb_[0-9]*.jpg'):
    pbar.update(0)
    image = Image.open(f"{filename}")
    gt = depth_read_nyu(f"{filename[:-13]}/sync_depth_{filename[-9:-4]}.png")

    inferno(gt, path("GT"))

    # Original
    prediction = model.predict_cpu(image)
    inferno(prediction, path("Original"))
    df.loc[len(df) + 1] = compute_errors(gt, prediction, drive, "Original", "nyu")

    # Grey
    prediction_grey = model.predict_cpu(greyscale(image))
    inferno(prediction_grey, path("Grey"))
    df.loc[len(df) + 1] = compute_errors(gt, prediction_grey, drive, "Greyscale", "nyu")

    # Edge
    prediction_edge = model.predict_cpu(edge(image))
    inferno(prediction_edge, path("Edge"))
    df.loc[len(df) + 1] = compute_errors(gt, prediction_edge, drive, f"Edge Detection", "nyu")

    # Edge Color 5
    prediction_gdege_grey_5 = model.predict_cpu(edge_col(image, 5))
    inferno(prediction_gdege_grey_5, path("EdgeCol5"))
    df.loc[len(df) + 1] = compute_errors(gt, prediction_gdege_grey_5, drive, f"Edge Detection and Colour Threshold 5", "nyu")

    # Edge Color 50
    prediction_gdege_grey_50 = model.predict_cpu(edge_col(image, 50))
    inferno(prediction_gdege_grey_50, path("EdgeCol50"))
    df.loc[len(df) + 1] = compute_errors(gt, prediction_gdege_grey_50, drive, f"Edge Detection and Colour Threshold 50", "nyu")

    # Edge Grey 5
    prediction_gdege_grey_5 = model.predict_cpu(edge_grey(image, 5))
    inferno(prediction_gdege_grey_5, path("EdgeGrey5"))
    df.loc[len(df) + 1] = compute_errors(gt, prediction_gdege_grey_5, drive, f"Edge Detection and Greyscale Threshold 5", "nyu")

    # Edge Grey 50
    prediction_gdege_grey_50 = model.predict_cpu(edge_grey(image, 50))
    inferno(prediction_gdege_grey_50, path("EdgeGrey50"))
    df.loc[len(df) + 1] = compute_errors(gt, prediction_gdege_grey_50, drive, f"Edge Detection and Greyscale Threshold 50", "nyu")

    # Desaturated
    prediction_desat = model.predict_cpu(desaturated(image, 0.5))
    inferno(prediction_desat, path("Desat"))
    df.loc[len(df) + 1] = compute_errors(gt, prediction_desat, drive, "Desaturated", "nyu")

    # Over Saturated
    prediction_sat = model.predict_cpu(desaturated(image, 2))
    inferno(prediction_sat, path("Sat"))
    df.loc[len(df) + 1] = compute_errors(gt, prediction_sat, drive, "Oversaturated", "nyu")
    
    i += 1

  pbar.update(1)

  df.to_csv(f"eval/{model_name}/eval_nyu_{directory}.csv")

  return i

def kitti():
  drives = ['0001', '0002', '0005', '0009', '0011', '0013', '0014', '0015', '0017', '0018', '0019', '0020', '0022', '0023', '0027', '0028', '0029', '0032', '0035']
  
  num_samples = len(drives)
  with tqdm(total=2*num_samples) as pbar:
    with ThreadPool(2) as p:
      res = p.starmap(execute_kitti,[(drive, pbar) for drive in drives])
      

def nyu():
  num_samples = len(os.listdir('data/nyu'))
  
  with tqdm(total=num_samples) as pbar:
    with ThreadPool(2) as p:
      res = p.starmap(execute_nyu,[(dir, pbar) for dir in os.listdir('data/nyu')])

if __name__=="__main__":
  global model, model_name

  t = str(input("(G)LPN or (N)eWCF predictor:"))
  glpn = False
  if t.lower() in ['g', 'glpn']:
    glpn = True
    model_name = 'glpn'
  elif t.lower() in ['n', 'newcf']:
    glpn = False
    model_name = 'newcf'
  else:
    print('Invalid input!')
    exit(-1)

  t = str(input("(K)itti or (N)yu dataset:"))

  if t.lower() in ['k', 'kitti']:
    if glpn:
      model = GLPN(True)
    else:
      model = NeWCFs(True)

    kitti()
    exit(0)
  elif t.lower() in ['n', 'nyu']:
    if glpn:
      model = GLPN(False)
    else:
      model = NeWCFs(False)
    nyu()
    exit(0)
  else:
    print('Invalid input!')
    exit(-1)