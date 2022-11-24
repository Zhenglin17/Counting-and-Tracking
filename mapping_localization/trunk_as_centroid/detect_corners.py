'''
    File name:
    Author: https://github.com/nishagandhi/detect_those_corners
    Date created:
    Date last modified:
    Python Version: 3.6.5
'''

import numpy as np
import cv2
import sys
import copy as cp
import matplotlib.pyplot as plt
import argparse
import os
import random
from mapping_localization.trunk_as_centroid.corners import harris_corners,shi_tomasi
import glob

#Checks if the path exists
# def check_path(path):
#
#     if not os.path.exists(path):
#         print('ERROR! The given path does not exist.')
#         sys.exit(0)

#Finds corners : harris and shitomasi
