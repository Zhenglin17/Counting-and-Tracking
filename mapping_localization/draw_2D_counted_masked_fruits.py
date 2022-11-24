import cv2
import os
import time
import pickle
import numpy as np
import sys
sys.path.append('/home/sam/git/Tracking_my_implementation')

from mapping_localization.save_valid_points import save_valid_points
from mapping_localization.find_camera_pos import find_camera_pos

start = time.time()
print("timer started \n")


from scpye.improc.binary_cleaner import BinaryCleaner
from scpye.improc.blob_analyzer import BlobAnalyzer
from mapping_localization.semantic_data_association.landmark_based_fruit_tracker import LandmarkFruitTracker
from scpye.utils.data_manager import DataManager
from scpye.utils.bag_manager import BagManager
from scpye.utils.landmark_bag_manager import LandmarkBasedBagManager
from scpye.utils.fruit_visualizer import FruitVisualizer
import copy
from scpye.track.bounding_box import (bbox_center, shift_bbox)


from mapping_localization.semantic_data_association.extract_landmark_3D_pos_bboxes_hist import  extract_landmark_3D_pos_bboxes_hist
from mapping_localization.semantic_data_association.extract_every_frame_rot_trans import  extract_every_frame_rot_trans
from scpye.utils.drawing import (Colors, draw_bboxes, draw_optical_flows,
                                 draw_text, draw_text_mask, draw_bboxes_matches, draw_line, draw_bbox_centers)



def check_bbox_inside_mask(mask,bbox):
    x,y = bbox_center(bbox)
    h,w = mask.shape
    x1 = min(x, w-1)
    y1 = min(y, h-1)
    x2 = max(x1, 0)
    y2 = max(y1, 0)
    return mask[int(y2),int(x2)]


def draw_2D_counted_masked_fruits(inp_format,data_dir_pre, min_age):
    with open(os.path.join(data_dir_pre,'tracks_frame_idx_whole_list.pkl'), 'rb') as input:
        tracks_frame_idx_whole_list = pickle.load(input)
#TODO: 2D starts here *****************************************************************************************************
    total_counts = 0
    # total_counts_all = 0
    total_counts_prev = 0
    # total_counts_all_prev = 0
    for lost_tracks_frame_idx in tracks_frame_idx_whole_list:
        lost_tracks = lost_tracks_frame_idx[0]
        actual_fr_idx = int(lost_tracks_frame_idx[1]) - 1
        if actual_fr_idx < 0:
            continue
        bgr = cv2.imread(os.path.join(data_dir_pre, 'tracking_results', 'bgr%04d' %actual_fr_idx + inp_format))
        bgr_vis = bgr.copy()
        h,w,_ = bgr.shape
        if lost_tracks != []:
            binary_mask_name =(os.path.join(data_dir_pre, 'target_tree_binary_label', 'frame%04d' %actual_fr_idx + inp_format))
            if os.path.exists(binary_mask_name):
                mask = cv2.imread(binary_mask_name, 0)
                for track in lost_tracks:
                    if track.age >= min_age:
                        # total_counts_all += 1
                        bbox = track.detected_bbox_hist[-1]
                        if check_bbox_inside_mask(mask, bbox):
                            total_counts+=1
                            color_value = 255
                            # draw_line(bgr, track.hist, color=[color_value,color_value,color_value], thickness = 0, dash_gap=10)
                            # draw_line(bgr_vis, track.hist, color=[color_value,color_value,color_value], thickness = 0, dash_gap=10)
                            draw_bboxes(bgr_vis, track.bbox, thickness=3,color = Colors.blue)

        # draw_text_mask(bgr, total_counts_all_prev, 'Total count:', (10, h - 50), scale=1,
        #       color=Colors.cyan)
        draw_text_mask(bgr, total_counts_prev, 'Target tree count:', (10, h - 90), scale=1,
              color=Colors.cyan)
        draw_text_mask(bgr, '', 'Green: valid tracks. Blue: new detections ', (10, h - 10), scale=1,
              color=Colors.cyan)



        # draw_text_mask(bgr_vis, total_counts_all_prev, 'Total count:', (10, h - 50), scale=1,
        #       color=Colors.cyan)
        draw_text_mask(bgr_vis, total_counts_prev, 'Target tree count:', (10, h - 90), scale=1,
              color=Colors.cyan)
        draw_text_mask(bgr_vis, '', 'Red: target fruits being counted. Green: valid tracks. Blue: new detections ', (10, h - 10), scale=1,
              color=Colors.cyan)


        total_counts_prev = total_counts
        # total_counts_all_prev = total_counts_all
        fname_bgr_output = os.path.join(data_dir_pre,'tracking_results_masked','bgr'+ '%04d' %actual_fr_idx +inp_format)
        cv2.imwrite(fname_bgr_output,bgr)

        fname_bgr_vis = os.path.join(data_dir_pre,'tracking_results_masked_visulization','bgr'+ '%04d' %actual_fr_idx +inp_format)
        cv2.imwrite(fname_bgr_vis,bgr_vis)

        print ('writing image:',fname_bgr_output)
#TODO: 2D ends here *****************************************************************************************************


if __name__ == '__main__':
    # original images format
    inp_format = '.png'
    # data folders directory
    data_dir_pre = '/home/sam/Desktop/Fruit-count-dataset/ACFR_RCNN_GT/doing/20171205T235708.045986'
    min_age = 4

    # distance_threshold = np.load('distance_threshold.npy')
    draw_2D_counted_masked_fruits(inp_format,data_dir_pre, min_age)
