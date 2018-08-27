import cv2
import os
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from skimage.exposure import equalize_adapthist
from skimage.color import gray2rgb
cv2.setNumThreads(0)

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def restricted_float(x):
    x = float(x)
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x

def to_np(x):
    x = x.cpu().numpy()
    if len(x.shape)>3:
        return x[:,0:3,:,:]
    else:
        return x

