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
                                 draw_text_mask,draw_text, draw_bboxes_matches, draw_line, draw_bbox_centers)



def check_bbox_inside_mask(mask,bbox):
    x,y = bbox_center(bbox)
    h,w = mask.shape
    x1 = min(x, w-1)
    y1 = min(y, h-1)
    x2 = max(x1, 0)
    y2 = max(y1, 0)
    return mask[int(y2),int(x2)]

def draw_landmark_counted_fruits(start_idx_2D_tracking,start_idx_diff, end_idx,inp_format,pred_format,data_dir_pre, min_age = 3, rot_trans_dict = None):
        # TODO: 3D landmark based tracking dictionary key: tuple (3D landmark position), value is another dictionary as: {'track':the last track of this landmark, 'frame number': 1 + the last frame that observes this landmark}
    with open(os.path.join(data_dir_pre,'landmark_pos_tracks_dict.pkl'), 'rb') as input:
        landmark_pos_tracks_dict = pickle.load(input)

    trunk_corners_pos = np.load(os.path.join(data_dir_pre,'generated_docs_this_run','trunk_corners_3D_pos.npy'))
    # trunk_pos_arr = np.array(trunk_pos)
    # trunk_z = np.average(trunk_pos_arr[:,2])


#**************************************************************conduct the duplicate landmark rejection***********************************************************************************************************
    # #TODO: in the form of: {the duplicate (to remove) landmark : corresponding original (to keep) landmark}
    # with open(data_dir_pre+'/duplicate_landmarks_dict.pkl', 'rb') as input:
    #     duplicate_landmarks_dict = pickle.load(input)
    # # TODO: in the form of: {laste observe frame number : [tracks, is_dup_landmark]}
    # track_last_observe_fr_num_dict = {}
    # for landmark_pos, track_fr_num in landmark_pos_tracks_dict.items():
    #     last_observe_fr_num = track_fr_num['frame number'] - 1
    #     this_is_dup_lanmark = False
    #     if landmark_pos in duplicate_landmarks_dict:
    #         this_is_dup_lanmark = True
    #     if last_observe_fr_num not in track_last_observe_fr_num_dict:
    #         track_last_observe_fr_num_dict[last_observe_fr_num] = [[copy.deepcopy(track_fr_num['track'][0]), this_is_dup_lanmark]]
    #     else:
    #         track_last_observe_fr_num_dict[last_observe_fr_num].append([copy.deepcopy(track_fr_num['track'][0]), this_is_dup_lanmark])
# ****************************************************************if do not conduct the duplicate landmark rejection, replaced with the following:*****************************************************************************
    track_last_observe_fr_num_dict = {}
    for landmark_pos, track_fr_num in landmark_pos_tracks_dict.items():
        last_observe_fr_num = track_fr_num['frame number'] - 1
        this_is_dup_lanmark = False
        if last_observe_fr_num not in track_last_observe_fr_num_dict:
            track_last_observe_fr_num_dict[last_observe_fr_num] = [[copy.deepcopy(track_fr_num['track'][0]), this_is_dup_lanmark]]
        else:
            track_last_observe_fr_num_dict[last_observe_fr_num].append([copy.deepcopy(track_fr_num['track'][0]), this_is_dup_lanmark])
# ****************************************************************replacement ends here**********************************************************************************************************************************

    total_counts = 0
    total_counts_all = 0
    total_counts_prev = 0
    total_counts_all_prev = 0
    total_counts_before_centroid = 0
    total_counts_before_centroid_prev = 0
    # records the trunk depth in corresponding fr_idx camera coordinate
    final_trunk_depth_fr_idx_dict = {}
    # TODO: note: the bgr****.png name of landmark tracking and 2D tracking are already synchronized!
    for i in range(start_idx_diff, end_idx):
        fname_bgr = os.path.join(data_dir_pre,'landmark_tracking_results_idx_sync','bgr'+ '%04d' %i +inp_format)

        if os.path.isfile(fname_bgr)==False:
            print(fname_bgr + ' is not found (if the last frame has not been tracked, check whether the data dir and the first frame number is correct) \n Tracking process is ceased!','\n')
            break
        bgr = cv2.imread(fname_bgr)
        bgr_vis = bgr.copy()
        h,w,_ = bgr.shape

        # TODO: be careful of the index, since there is a difference between landmark tracking and the 2D tracking, the actual output of landmark tracking should + start_idx_diff
        binary_mask_name =(os.path.join(data_dir_pre, 'target_tree_binary_label', 'frame%04d' %(start_idx_2D_tracking+i) + inp_format))
        if os.path.exists(binary_mask_name):
            mask = cv2.imread(binary_mask_name, 0)
        else:
            mask = None

        bboxes = []

        # if start idx difference is larger than 2, this means 2D tracking begins earilier than 3D, lowering 3D tracking age  threshold temporaily
        if start_idx_diff >= 2 and i - start_idx_diff <= 2:
            print('start_idx_diff is larger than 2',start_idx_diff)
            # raise Exception('change code to read 2D tracking to initialize the tracking process!')
            age_threshold = 2
        else:
            print('start_idx_diff is no larger than 2',start_idx_diff)
            age_threshold = min_age
        if i in track_last_observe_fr_num_dict:

            for track_dupflag in track_last_observe_fr_num_dict[i]:
                track = track_dupflag[0]
                dupflag = track_dupflag[1]
                # sum_track_observed = len(track.hist_unobserved) - sum(track.hist_unobserved)
                if track.age >= age_threshold:
                    if i == end_idx-1:
                        bbox = shift_bbox(track.bbox, track.pos)
                        bboxes.append(bbox)
                    else:
                        bbox = shift_bbox(track.bbox, track.prev_pos)
                        bboxes.append(bbox)
                        # draw_bboxes(bgr, bbox, color=Colors.blue, thickness=2, margin=0)
                    if dupflag:
                        continue
                        # draw_bboxes(bgr, bbox, color=Colors.magenta, thickness=3, margin=0)
                    else:
                        total_counts_all += 1
                        if os.path.exists(binary_mask_name):
                            if check_bbox_inside_mask(mask, bbox):
                                draw_bboxes(bgr_vis, bbox, color=Colors.blue, thickness=3, margin=0)
                                total_counts+=1
                                total_counts_before_centroid+=1
                                # look back for multiple frames, and determine whether or not this landmark is before tree centroid

                                if trunk_corners_pos != []:
                                    before_centroid_count = 0
                                    after_centroid_count = 0

                                    for idx_sub in np.arange(15):
                                        back_prop_idx = i - idx_sub
                                        if  back_prop_idx>= 0:
                                            if str(back_prop_idx) not in rot_trans_dict:
                                                continue
                                            rot = rot_trans_dict[str(back_prop_idx)][0]
                                            trans = rot_trans_dict[str(back_prop_idx)][1]
                                            if str(back_prop_idx) not in final_trunk_depth_fr_idx_dict:
                                                # Tree centroid: find the depth of trunk corner points
                                                # points locations in the camera coordinate
                                                depth = []
                                                for trunk_corner_pos in trunk_corners_pos:
                                                    trunk_corner_pos_cam_frame = np.dot(rot, trunk_corner_pos) + trans
                                                    depth.append(trunk_corner_pos_cam_frame[2])
                                                # sort depth (from large to small)
                                                sorted_depth = sorted(depth, reverse = True)
                                                quater_idx = int(len(depth) / 4.0)
                                                final_trunk_depth = sorted_depth[max(quater_idx, 1)]
                                                final_trunk_depth_fr_idx_dict[str(back_prop_idx)] = final_trunk_depth
                                                print('final trunk depth is the', max(quater_idx, 0), 'th largest depth value, which is',final_trunk_depth)

                                            landmark_pos = copy.deepcopy(track.landmark_pos)
                                            landmark_pos_cam_frame = np.dot(rot, landmark_pos) + trans
                                            if landmark_pos_cam_frame[2] <= 1.15 * final_trunk_depth_fr_idx_dict[str(back_prop_idx)]:
                                                before_centroid_count += 1
                                            else:
                                                after_centroid_count += 1
                                    print('current landmark before tree centroid count:', before_centroid_count, 'after centroid count:',after_centroid_count)
                                    if before_centroid_count <= 3 and before_centroid_count*2 <= after_centroid_count:
                                        # draw_bboxes(bgr_vis, bbox, color=Colors.cyan, thickness=3, margin=0)
                                        total_counts_before_centroid-=1

        if i == end_idx -1:
            print ('this is the last frame, counting all valid tracked fruits now! num of counted fruits here is:',total_counts - total_counts_prev)



        if i == end_idx - 1:

            draw_text_mask(bgr, total_counts_before_centroid, 'Target tree count (before centroid):', (10, h - 170), scale = 1.5, color = Colors.cyan)
            draw_text_mask(bgr, total_counts, 'Target tree count:', (10, h - 120), scale = 1.5, color = Colors.cyan)
            draw_text_mask(bgr, total_counts_all, 'Total count:', (10, h - 70), scale = 1.5, color = Colors.cyan)

            draw_text_mask(bgr_vis, total_counts_before_centroid, 'Target tree count (before centroid):', (10, h - 220), scale = 1.5, color = Colors.cyan)
            draw_text_mask(bgr_vis, total_counts, 'Target tree count:', (10, h - 170), scale = 1.5, color = Colors.cyan)
            draw_text_mask(bgr_vis, total_counts_all, 'Total count:', (10, h - 120), scale = 1.5, color = Colors.cyan)

        else:
            draw_text_mask(bgr, total_counts_before_centroid_prev, 'Target tree count (before centroid):', (10, h - 170), scale = 1.5, color = Colors.cyan)
            draw_text_mask(bgr, total_counts_prev, 'Target tree count:', (10, h - 120), scale=1.5,color=Colors.cyan)
            draw_text_mask(bgr, total_counts_all_prev, 'Total count:', (10, h - 70), scale = 1.5, color = Colors.cyan)

            draw_text_mask(bgr_vis, total_counts_before_centroid_prev, 'Target tree count (before centroid):', (10, h - 220), scale = 1.5, color = Colors.cyan)
            draw_text_mask(bgr_vis, total_counts_prev, 'Target tree count:', (10, h - 170), scale=1.5,color=Colors.cyan)
            draw_text_mask(bgr_vis, total_counts_all_prev, 'Total count:', (10, h - 120), scale = 1.5, color = Colors.cyan)


        draw_text_mask(bgr, '', 'Green: valid tracks. Blue: new detections ', (10, h - 20), scale=1.5,color=Colors.cyan)

        draw_text_mask(bgr_vis, '', 'Red: target-tree fruits being counted.', (10, h - 70), scale=1.5,
            color=Colors.cyan)
        draw_text_mask(bgr_vis, '', 'Green: valid tracks. Blue: new detections.', (10, h - 20), scale=1.5,
            color=Colors.cyan)

        total_counts_all_prev = total_counts_all
        total_counts_prev = total_counts
        total_counts_before_centroid_prev = total_counts_before_centroid

        fname_bgr_output = os.path.join(data_dir_pre,'landmark_counting_results','bgr'+ '%04d' %i +inp_format)
        cv2.imwrite(fname_bgr_output,bgr)

        fname_bgr_vis_output = os.path.join(data_dir_pre,'landmark_counting_visulization','bgr'+ '%04d' %i +inp_format)
        cv2.imwrite(fname_bgr_vis_output,bgr_vis)


        bgr_vis_masked = bgr_vis.copy()
        if os.path.exists(binary_mask_name):
            mask = cv2.imread(binary_mask_name, 0)
            bgr_vis_masked[mask == 0,:] -= 50


        fname_bgr_masked_vis_output = os.path.join(data_dir_pre,'check_mask_correctness_for_landmark_tracking','bgr'+ '%04d' %i +inp_format)
        cv2.imwrite(fname_bgr_masked_vis_output,bgr_vis_masked)


        print ('writing image:',fname_bgr_output)

if __name__ == '__main__':
    # first frame number, e.g., 1 means your end frame is named frame0001.XXX
    start_idx = 0
    # end frame number, e.g., 5000 means your end frame is named frame5000.XXX, you can just set a large number if you want, the tracking process will automatically end once the final image has been read
    end_idx = 2000
    # original images format
    inp_format = '.png'
    # prediction images format
    pred_format = '.npy'
    # data folders directory
    # data_dir_pre = '/home/sam/Desktop/Fruit-count-dataset/ACFR_RCNN_GT/20171206T042534.518455'
    # data_dir_pre = '/home/sam/Desktop/Fruit-count-dataset/ACFR_RCNN_GT/20171206T000729.045986'
    # data_dir_pre = '/home/sam/Desktop/Fruit-count-dataset/ACFR_RCNN_GT/20171206T000629.445986'
    data_dir_pre = '/home/sam/Desktop/Fruit-count-dataset/ACFR_RCNN_GT/20171205T235708.045986'

    # distance_threshold = np.load('distance_threshold.npy')
    draw_landmark_counted_fruits(start_idx_2D_tracking = start_idx,start_idx_diff = 0,end_idx = end_idx,
                                 inp_format= inp_format,pred_format = pred_format,data_dir_pre = data_dir_pre, min_age = 3)
