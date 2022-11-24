import logging
import cv2
import os
import time
import pickle
import numpy as np
import sys
import glob
sys.path.append('/home/sam/git/Tracking_my_implementation')

from mapping_localization.save_valid_points import save_valid_points
from mapping_localization.find_camera_pos import find_camera_pos

start = time.time()
print ("timer started \n")

from mapping_localization.semantic_data_association.extract_landmark_3D_pos_bboxes_hist import  extract_landmark_3D_pos_bboxes_hist
from mapping_localization.semantic_data_association.extract_every_frame_rot_trans import  extract_every_frame_rot_trans
from mapping_localization.semantic_data_association.landmark_based_tracking_save_docs_for_SfM import run_landmark_based_tracking_save_docs_for_SfM
from mapping_localization.semantic_data_association.draw_full_landmark_tracking_history import draw_landmark_counted_fruits
from mapping_localization.semantic_data_association.draw_full_landmark_tracking_history_without_trunk import draw_landmark_counted_fruits_without_trunk
from mapping_localization.semantic_data_association.duplicate_landmarks_identification import duplicate_landmarks_identification

from mapping_localization.trunk_as_centroid.localize_trunk_corners import localize_trunk_corners

import shutil


def run_landmark_tracking(parent_data_dir, folder_dir_working_on):
    min_age = 3

    # original images format
    inp_format = '.png'
    # prediction images format
    pred_format = '.npy'
    # data folders directory
    folder_dir_working_on = glob.glob(r"C:\Users\tom\Ag_Lab\tracking_stack\test_data")

    change_current_results = False
    # if only visualize counting process, set this to be False
    run_tracking = True
    # if conduct the depth based outlier rejection, set this to be true
    depth_outlier_rej_trunk_as_centroid = False

    if change_current_results:
        print('Be careful: change current results == True, if Enter is pressed, will start reading all folders iteratively!!!')
        raw_input("Press Enter to continue...")
        folder_dir_working_on = glob.glob('/home/sam/Desktop/Fruit-count-dataset/ACFR_RCNN_GT_HA1.2_age3/done/*')
        print('Be careful: change current results == True, if Enter is pressed, will start reading all folders iteratively!!!')
        # raw_input("Press Enter to continue...")

    if len(folder_dir_working_on) != 1 and change_current_results == False:
        raise Exception('folder_dir_working_on != 1, check data folder!')


    folder_dir_working_on = sorted(folder_dir_working_on)
    for cur_folder_dir_working_on in folder_dir_working_on:
        # first frame number, e.g., 1 means your end frame is named frame0001.XXX
        start_idx = 2 # TODO: minimun start index should be set 2, becasue need 2 frames to initialize the landmark. this will be changed in line 70 if reconstruction starts index > start_idx
        # end frame number, e.g., 5000 means your end frame is named frame5000.XXX, you can just set a large number if you want, the tracking process will automatically end once the final image has been read
        end_idx = 1000 # TODO: this will be changed in line 70 if reconstruction ends index < end_idx

        data_dir_pre = cur_folder_dir_working_on

        print('current data directory:', data_dir_pre, '\n')

        if change_current_results == True:
            if 'hard' in data_dir_pre or 'trunk-invisible' in data_dir_pre:
                print ('This is one of the hard cases, continue to the next one!')
                continue

            # if '20171206T042534.518455' not in data_dir_pre:
            #     continue

            shutil.rmtree(os.path.join(data_dir_pre,'landmark_counting_results'))
            shutil.rmtree(os.path.join(data_dir_pre,'landmark_counting_visulization'))
            if run_tracking:
                shutil.rmtree(os.path.join(data_dir_pre,'landmark_tracking_results_idx_sync'))
                shutil.rmtree(os.path.join(data_dir_pre,'projection_results'))

            os.mkdir(os.path.join(data_dir_pre,'landmark_counting_results'))
            os.mkdir(os.path.join(data_dir_pre,'landmark_counting_visulization'))
            if run_tracking:
                os.mkdir(os.path.join(data_dir_pre,'landmark_tracking_results_idx_sync'))
                os.mkdir(os.path.join(data_dir_pre,'projection_results'))

        rerun_fruit_3D_pos_bboxes_hist_dict_fname_and_rot_trans_dict_fname = True

        save_valid_points(data_dir=data_dir_pre)
        find_camera_pos(data_dir=data_dir_pre)
        # fruit_3D_pos_bboxes_hist_dict: key is 3D point index, value is a dictionary: {'3Dpos':fruit_3D_loc, 'frame_idx': [frame_ind] (order: small to large), '2Dpos_bbox':fruit}, which contains fruits' 3D position, frame index where it appears (a list of integers) and corresponding frame's 2D_position_and_bbox (a list where each element is another list: [array(x_center, y_center),array(bbox)])
        fruit_3D_pos_bboxes_hist_dict_fname = data_dir_pre+'/generated_docs_this_run/fruit_3D_pos_bboxes_hist_dict.pkl'
        # every_frame_rot_trans_dict is recorded as follows: every_frame_rot_trans_dict[str(frame_id_temp)] = [rot_mat_world_to_img,trans_world_to_img]
        rot_trans_dict_fname = data_dir_pre+'/generated_docs_this_run/every_frame_rot_trans_dict.pkl'
        if os.path.isfile(fruit_3D_pos_bboxes_hist_dict_fname) == False or rerun_fruit_3D_pos_bboxes_hist_dict_fname_and_rot_trans_dict_fname:
            print(fruit_3D_pos_bboxes_hist_dict_fname + 'does not exist, running extract_landmark_3D_pos_bboxes_hist.py to generate the file\n')
            #computing weighted average (sometimes the same point repeated for multiple times) is inside this function
            extract_landmark_3D_pos_bboxes_hist(data_dir_pre)
        else:
            print(fruit_3D_pos_bboxes_hist_dict_fname + 'already exists, NOT re-running extract_landmark_3D_pos_bboxes_hist.py\n')

        if os.path.isfile(rot_trans_dict_fname) == False or rerun_fruit_3D_pos_bboxes_hist_dict_fname_and_rot_trans_dict_fname:
            print(rot_trans_dict_fname + 'does not exist, running extract_every_frame_rot_trans.py to generate the file\n')
            extract_every_frame_rot_trans(data_dir_pre)
        else:
            print(rot_trans_dict_fname + 'already exists, NOT re-running extract_every_frame_rot_trans.py\n')


        with open(fruit_3D_pos_bboxes_hist_dict_fname, 'rb') as input:
            fruit_3D_pos_bboxes_hist_dict = pickle.load(input)
        with open(rot_trans_dict_fname, 'rb') as input:
            rot_trans_dict = pickle.load(input)


        recon_start_fr = 5000
        recon_ends_fr = -1
        for fr_idx in rot_trans_dict:
            if int(fr_idx) < recon_start_fr:
                recon_start_fr = int(fr_idx)
            if int(fr_idx) >= recon_ends_fr:
                recon_ends_fr = int(fr_idx)

        print('3D reconstruction begins at:', recon_start_fr, ' and ends at:',recon_ends_fr)
        if start_idx < recon_start_fr:
            print ('3D reconstruction starts idx > set start_idx, here let start_idx = recon_start_fr')
            start_idx = recon_start_fr
        if end_idx > recon_ends_fr:
            end_idx = recon_ends_fr
            print ('3D reconstruction ends idx > set end_idx, here let end_idx = recon_ends_fr')
        np.save(os.path.join(data_dir_pre,'lmft_start_end_idx.npy'),np.array([start_idx,end_idx]))


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



        tracking_2D_start_end_idx = np.load(data_dir_pre+'/2Dft_start_end_idx.npy')
        start_idx_diff = start_idx - tracking_2D_start_end_idx[0]
        # end_idx_diff = end_idx - tracking_2D_start_end_idx[1]

        if os.path.isdir(os.path.join(data_dir_pre,'landmark_tracking_results'))== False:
            os.mkdir(os.path.join(data_dir_pre,'landmark_tracking_results'))
        if run_tracking:
            run_landmark_based_tracking_save_docs_for_SfM(start_idx,end_idx,inp_format,pred_format, data_dir_pre, fruit_3D_pos_bboxes_hist_dict, rot_trans_dict, K_mat, min_age= min_age)
            for i in range(start_idx_diff, end_idx):
                fname_bgr = os.path.join(data_dir_pre,'landmark_tracking_results','bgr'+ '%04d' %(i-start_idx_diff) +inp_format)
                if os.path.isfile(fname_bgr)==False:
                    print(fname_bgr + ' is not found (if the last frame has not been tracked, check whether the data dir and the first frame number is correct) \n Tracking process is ceased!','\n')
                    break
                fname_bgr_sync_with_2D_tracking = os.path.join(data_dir_pre,'landmark_tracking_results_idx_sync','bgr'+ '%04d' %(i) +inp_format)
                bgr = cv2.imread(fname_bgr)
                cv2.imwrite(fname_bgr_sync_with_2D_tracking, bgr)
            shutil.rmtree(os.path.join(data_dir_pre,'landmark_tracking_results'))

        # duplicate_landmarks_identification(data_dir_pre, rerun_dis_stat = False, rerun_thresholding = True, visualize = False, average_div_by_threshold = 100)

        start_idx_2D_tracking = tracking_2D_start_end_idx[0]
        if depth_outlier_rej_trunk_as_centroid:
            localize_trunk_corners(data_dir_pre)
            draw_landmark_counted_fruits(start_idx_2D_tracking,start_idx_diff,end_idx,inp_format,pred_format,data_dir_pre, min_age = min_age, rot_trans_dict = rot_trans_dict)
        else:
            draw_landmark_counted_fruits_without_trunk(start_idx_2D_tracking,start_idx_diff,end_idx,inp_format,pred_format,data_dir_pre, min_age = min_age, rot_trans_dict = rot_trans_dict)

if __name__ == '__main__':
    run_landmark_tracking(parent_data_dir="", folder_dir_working_on="")

