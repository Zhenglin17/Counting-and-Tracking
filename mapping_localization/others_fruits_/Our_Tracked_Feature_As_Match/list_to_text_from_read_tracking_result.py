from __future__ import (print_function, division, absolute_import)
import pickle
import cv2
import numpy as np
import sys
sys.path.append('C:\\Users\\UPenn Robotics LX\\Documents\\GitHub\\Tracking_my_implementation')
from scpye.improc.image_processing import enhance_contrast
from scpye.track.assignment import hungarian_assignment
from scpye.track.bounding_box import (bboxes_assignment_cost, extract_bbox)
from scpye.track.fruit_track import FruitTrack
from scpye.track.optical_flow import (calc_optical_flow, calc_average_flow)
from scpye.utils.drawing import (Colors, draw_bboxes, draw_optical_flows,
                                 draw_text, draw_bboxes_matches, draw_line)

# TODO: load the npy files in, after loading they are 2D array containing feature positions
cur_pos = np.load('cur_pos.npy')
pre_pos = np.load('pre_pos.npy')
pre_pre_pos = np.load('pre_pre_pos.npy')
pre_pre_pre_pos = np.load('pre_pre_pre_pos.npy')
num_features = len(cur_pos)
scale = 1.1
orientation = 0.3
descriptor = (np.ones((1,128))).astype(int)
scale_ori = np.array([[scale, orientation]])
sift_arr = np.concatenate((scale_ori, descriptor), axis=1)
sift_arr_rep = np.repeat(sift_arr, num_features, axis=0)
cur_pos = np.append(cur_pos, sift_arr_rep, axis=1)
pre_pos = np.append(pre_pos, sift_arr_rep, axis=1)
pre_pre_pos = np.append(pre_pre_pos, sift_arr_rep, axis=1)
pre_pre_pre_pos = np.append(pre_pre_pre_pos, sift_arr_rep, axis=1)
num_features_str = str(num_features)
np.savetxt('cur_pos_str.txt', cur_pos, fmt='%.1f', delimiter=' ', newline='\n', header=num_features_str + ' 128', footer='', comments='')
np.savetxt('pre_pos_str.txt', pre_pos, fmt='%.1f', delimiter=' ', newline='\n', header=num_features_str + ' 128', footer='', comments='')
np.savetxt('pre_pre_pos_str.txt', pre_pre_pos, fmt='%.1f', delimiter=' ', newline='\n', header=num_features_str + ' 128', footer='', comments='')
np.savetxt('pre_pre_pre_pos_str.txt', pre_pre_pre_pos, fmt='%.1f', delimiter=' ', newline='\n', header=num_features_str + ' 128', footer='', comments='')

match_ind = np.arange(num_features).reshape(num_features,1)
match_ind = np.repeat(match_ind, 2, axis = 1)
np.savetxt('match_ind.txt', match_ind, fmt='%d', delimiter=' ', newline='\n', header='inp_frame0000.png inp_frame0001.png', footer='', comments='')

print ('features in every frame are:', num_features)
