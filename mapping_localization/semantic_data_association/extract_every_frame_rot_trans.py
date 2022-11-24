from __future__ import (print_function, division, absolute_import)
import pickle
import cv2
import numpy as np
import sys
import transforms3d as trans3d
import os
from scpye.improc.image_processing import enhance_contrast
from scpye.track.assignment import hungarian_assignment
from scpye.track.bounding_box import (bboxes_assignment_cost, extract_bbox)
from scpye.track.fruit_track import FruitTrack
from scpye.track.optical_flow import (calc_optical_flow, calc_average_flow)
from scpye.utils.drawing import (Colors, draw_bboxes, draw_optical_flows,
                                 draw_text, draw_bboxes_matches, draw_line, draw_bbox_centers)
from scpye.track.bounding_box import (bbox_center, shift_bbox)
import time

def extract_every_frame_rot_trans(data_dir_pre = '..'):
    every_frame_rot_trans_dict = {}
    with open(data_dir_pre+'/generated_docs_this_run/image_and_camera_id.txt') as f:
        # Image list with two lines of data per image:
        #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID
        #   POINTS2D[] as (X, Y, POINT3D_ID)
        lines=f.readlines()
        ind = -1
        length = len(lines) - 1
        for line in lines:
            # skip comments in the first four lines
            if ind >= 0:
                # image pose information
                img_info_arr = (np.fromstring(line, dtype=float, sep=' '))
                quat_img = img_info_arr[1:5]
                rot_mat_world_to_img = trans3d.quaternions.quat2mat(quat_img)
                trans_world_to_img = img_info_arr[5:8]
                frame_id_temp = int(img_info_arr[0])
                every_frame_rot_trans_dict[str(frame_id_temp)] = [rot_mat_world_to_img,trans_world_to_img]
            ind += 1

    with open(data_dir_pre+'/generated_docs_this_run/every_frame_rot_trans_dict.pkl', 'wb') as output_loc:
        pickle.dump(every_frame_rot_trans_dict, output_loc)

if __name__ == "__main__":
    extract_every_frame_rot_trans()
