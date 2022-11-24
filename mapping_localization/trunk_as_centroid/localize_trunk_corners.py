'''
    File name:
    Author: Xu Liu
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
from mapping_localization.trunk_as_centroid.corners import harris_corners,shi_tomasi
import copy
import pickle
from mapping_localization.trunk_as_centroid.extract_track_trunk_corners import extract_track_trunk_corners
from mapping_localization.trunk_as_centroid.trunk_corners_tri import trunk_corners_multi_view_triangulation

def localize_trunk_corners(data_dir_pre):
    tracked = extract_track_trunk_corners(data_dir_pre)
    if tracked == False:
        np.save(data_dir_pre + '/generated_docs_this_run/trunk_corners_3D_pos.npy',[])
        return
    with open(data_dir_pre+'/generated_docs_this_run/cameras.txt') as f:
        lines=f.readlines()
        for line in lines:
            # skip the comments
            if line[0] == '#':
                continue
            else:
                if '\n' in line:
                    line = line [:-1]
                cam_info = line.split(' ')#np.fromstring(line, dtype=float, sep='\n')
                if len(cam_info) != 8:
                    print(len(cam_info))
                    raise Exception('camera info not correct, check cameras.txt!')
    f_x =float(cam_info[4])
    f_y =float(cam_info[5])
    c_x =float(cam_info[6])
    c_y =float(cam_info[7])
    # camera intrinsic matrix
    K_mat = np.array([[f_x, 0, c_x],
                      [0, f_y, c_y],
                      [0,   0,  1]])
    trunk_tracks_fname = data_dir_pre+'/generated_docs_this_run/trunk_tracks.pkl'
    with open(trunk_tracks_fname, 'rb') as input:
        trunk_tracks = pickle.load(input)

    rot_trans_dict_fname = data_dir_pre + '/generated_docs_this_run/every_frame_rot_trans_dict.pkl'
    with open(rot_trans_dict_fname, 'rb') as input:
        rot_trans_dict = pickle.load(input)
    # save the 3D position of every corner point on the trunk
    print('Trunk corners positions are:')
    trunk_corners_3D_pos = trunk_corners_multi_view_triangulation(K_mat, rot_trans_dict,trunk_tracks)
    np.save(data_dir_pre + '/generated_docs_this_run/trunk_corners_3D_pos.npy',trunk_corners_3D_pos)
    print('Trunk corner positions have been saved as .npy file')

    # rot_unseen = rot_trans_dict[str(unseen_fr_idx)][0]
    # trans_unseen = rot_trans_dict[str(unseen_fr_idx)][1]
    # # points locations in the camera coordinate
    # pos_cam_frame_unseen = np.dot(rot_unseen, pos_world_frame) + trans_unseen
    # # points locations in the image coordinate
    # pos_img_unseen = np.dot(K_mat, pos_cam_frame_unseen)

if __name__ == "__main__":
    data_dir_pre = glob.glob('/home/sam/Desktop/Fruit-count-dataset/ACFR_RCNN_GT_HA1.2_age3/doing/*')[0]
    localize_trunk_corners(data_dir_pre)





