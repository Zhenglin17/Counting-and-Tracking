'''
    File name:
    Author: Xu Liu and https://github.com/strawlab/pymvg
    Date created:
    Date last modified:
    Python Version: 2.7
'''

from __future__ import (print_function, division, absolute_import)

import logging
import cv2
import numpy as np
import math

from scpye.improc.image_processing import enhance_contrast
from scpye.track.assignment import hungarian_assignment
from scpye.track.bounding_box import (bboxes_assignment_cost, extract_bbox, bbox_area)
from scpye.track.fruit_track import FruitTrack
from scpye.track.optical_flow import (calc_optical_flow, calc_average_flow)
from scpye.utils.drawing import (Colors, draw_bboxes, draw_optical_flows,
                                 draw_text, draw_text_mask, draw_bboxes_matches, draw_line, draw_bbox_centers)
from scpye.track.bounding_box import (bbox_center, shift_bbox)

import glob
import cv2
import numpy as np
from mapping_localization.trunk_as_centroid.corners import harris_corners,shi_tomasi
import copy
import shutil
import pickle
import os

class TrunkTrack(object):
    def __init__(self, init_pos, init_fr_idx):
        self.hist_fr_idx_dict = {}
        self.hist_fr_idx_dict[init_fr_idx] = np.array(init_pos)
        self.hist = [np.array(init_pos)]

def extract_track_trunk_corners(data_dir_pre):

    img_fname_list = glob.glob(data_dir_pre+'/trunk_img_seg/*.png')
    img_fname_list_sorted = sorted(img_fname_list)


    img_fname = img_fname_list_sorted[0]
    img_frame_format = (img_fname.split('/'))[-1]
    img_frame = (img_frame_format.split('.'))[0]

    img_label_fname = glob.glob(data_dir_pre+'/trunk_img_seg/trunk_segmentation/*.png')[0]#+img_frame_format


    img = cv2.imread(img_fname)
    img_label = cv2.imread(img_label_fname)

    shitomasi, corners_img = shi_tomasi(img, increase_points = 0)

    tracked = True
    valid_corners = []
    for corners in corners_img:
        x,y = corners.ravel()
        if img_label[int(y), int(x), 0] > 0:
            valid_corners.append((x,y))

    if len(valid_corners) <= 4:
        valid_corners = []
        print('too few corners, lowering the threshold and increasing corners to track')
        shitomasi, corners_img = shi_tomasi(img, increase_points = 1)
        for corners in corners_img:
            x,y = corners.ravel()
            if img_label[int(y), int(x), 0] > 0:
                valid_corners.append((x,y))

    if len(valid_corners) <= 4:
        valid_corners = []
        print('too few corners, lowering the threshold and increasing corners to track')
        shitomasi, corners_img = shi_tomasi(img, increase_points = 2)
        for corners in corners_img:
            x,y = corners.ravel()
            if img_label[int(y), int(x), 0] > 0:
                valid_corners.append((x,y))

    if len(valid_corners) <= 3:
        valid_corners = []
        print('way too few corners, bad case')
        tracked = False
        return tracked

    trunk_tracks = []
    prev_pts = []
    for valid_corner in valid_corners:
        x = valid_corner[0]
        y = valid_corner[1]
        cv2.circle(shitomasi,(x,y),3,[255,255,0],-1)
        prev_pts.append([x,y])
        trunk_tracks.append(TrunkTrack([x,y], img_frame))

    # cv2.imwrite(data_dir_pre+'/corner_extracted/'+'shitomasi_'+img_frame_format,shitomasi)
    multi_view_tracks = 3

    if os.path.exists(data_dir_pre+'/trunk_tracking_results'):
        shutil.rmtree((data_dir_pre+'/trunk_tracking_results'))
    os.mkdir((data_dir_pre+'/trunk_tracking_results'))

    for img_idx in np.arange(multi_view_tracks):
        filename_prev = img_fname_list_sorted[img_idx]
        img_prev = cv2.imread(filename_prev)
        prev_gray = cv2.cvtColor(img_prev,cv2.COLOR_BGR2GRAY)

        filename = img_fname_list_sorted[img_idx+1]
        img_frame_format = (filename.split('/'))[-1]
        img_frame = (img_frame_format.split('.'))[0]

        img = cv2.imread(filename)
        img_full_hist = img.copy()
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        init_pts = copy.deepcopy(prev_pts)
        curr_pts, status = calc_optical_flow(prev_gray, gray, prev_pts, init_pts, win_size = [20, 5], max_level = 15, use_init_flow= False)


        draw_optical_flows(img, prev_pts, curr_pts, status,
                            radius=4, color=Colors.magenta)
        tracked_tracks = []
        for cur_pt, prev_pt, stat, trunk_track in zip(curr_pts, prev_pts, status, trunk_tracks):
            if stat:
                trunk_track.hist.append(cur_pt)
                trunk_track.hist_fr_idx_dict[img_frame] = cur_pt
                tracked_tracks.append(trunk_track)
                draw_line(img_full_hist, trunk_track.hist, color=[255,255,255], thickness = 2, dash_gap=0)
        trunk_tracks = tracked_tracks
        prev_pts = copy.deepcopy(curr_pts)
        cv2.imwrite(data_dir_pre+'/trunk_tracking_results/two_frame_'+img_frame_format,img)
        cv2.imwrite(data_dir_pre+'/trunk_tracking_results/full_hist_'+img_frame_format,img_full_hist)

    trunk_tracks_fname = data_dir_pre+'/generated_docs_this_run/trunk_tracks.pkl'
    with open(trunk_tracks_fname, 'wb') as output:
        pickle.dump(trunk_tracks, output)
    return tracked

if __name__ == "__main__":
    data_dir_pre = '/home/sam/Desktop/ACFR_Trunk_Centroid/20171206T002056.045986/png'
    extract_track_trunk_corners(data_dir_pre)




