import glob
import os

import sys
# sys.path.append('/home/sam/git/Tracking_my_implementation')

from mapping_localization.run_tracking_save_docs_for_SfM import run_tracking_save_docs_for_SfM
from mapping_localization.save_valid_points import save_valid_points
from mapping_localization.find_object_pos_and_bbox import find_object_pos_and_bbox
from mapping_localization.find_camera_pos import find_camera_pos
from mapping_localization.trans_glob2cam import trans_glob2cam
from mapping_localization.conduct_correction_and_visualize import conduct_correction_and_visualize
from mapping_localization.draw_2D_counted_masked_fruits import draw_2D_counted_masked_fruits

import time
import numpy as np


def two_dim_tracking(parent_data_dir, folder_dir_working_on):
    data_set = 'ACFR'
    min_age = 3
    # whether or not to only count trees that are masked:
    only_count_masked_trees = True

    # whether or not to re-run the tracking code
    rerun_tracking = True
    only_run_tracking = True
    # whether or not to re-run the mapping and localization code
    rerun = True
    # whether or not to only display results (do not run step 1-4)
    only_visualize = False

    # Part 1: tracking *************************************************************************************************
    pkl_files_found = False

    if os.path.isfile('tracks_frame_idx_whole_list.pkl') and os.path.isfile(
            'counted_fruits_and_last_frame_whole_list.pkl'):
        print ('tracking process output .pkl files is found!')
        pkl_files_found = True

    if pkl_files_found == False or rerun_tracking == True or only_run_tracking == True:
        print('.pkl files not found or .rerun_tracking is True, launch tracking process')
        time.sleep(2)

    # first frame number, e.g., 1 means your end frame is named frame0001.XXX
    start_idx = 0
    # end frame number, e.g., 5000 means your end frame is named frame5000.XXX, you can just set a large number if you want, the tracking process will automatically end once the final image has been read
    end_idx = 1000
    # original images format
    inp_format = '.png'
    # prediction images format
    pred_format = '.npy'
    # data folders directory
    folder_dir_working_on = glob.glob(folder_dir_working_on)

    # Change current results or run on new dataset
    change_current_results = False
    if change_current_results:
        folder_dir_working_on = glob.glob(folder_dir_working_on)
        print(
            'Be careful: change current results == True, if Enter is pressed, will start reading all folders iteratively!!!')
        raw_input("Press Enter to continue...")

    if len(folder_dir_working_on) != 1 and change_current_results == False:
        raise Exception('folder_dir_working_on != 1, check data folder!')

    folder_dir_working_on = sorted(folder_dir_working_on)
    for cur_folder_dir_working_on in folder_dir_working_on:

        data_dir_pre = cur_folder_dir_working_on

        print('current data directory:', data_dir_pre, '\n')

        np.save(os.path.join(data_dir_pre, '2Dft_start_end_idx.npy'), np.array([start_idx, end_idx]))
        # Optional: specify the major movement direction of the fruits in the image plane for better initialization! Left: -1; Not sure: 0; Right: 1.
        fruits_moving_direction = 0

        # Hungarian assignment cost, change in assignment.py
        # unassigned_cost = 1.2

        run_tracking = True
        # run tracking, results will be save in data folders directory + '/tracking_results' folder
        if run_tracking:
            run_tracking_save_docs_for_SfM(start_idx=start_idx, end_idx=end_idx, inp_format=inp_format,
                                           pred_format=pred_format, data_dir_pre=data_dir_pre,
                                           fruits_moving_direction=fruits_moving_direction, min_age=min_age)

        if only_count_masked_trees:
            draw_2D_counted_masked_fruits(inp_format, data_dir_pre, min_age)

    # TODO: ends here

    # TODO: If need to use the following code, change the data folder directory
    if only_run_tracking == False:
        # Part 2: mapping and localization *********************************************************************************
        num_files = len(glob.glob('./generated_docs_this_run/*'))

        # parameters for step 5:
        # whether or not to conduct distance outliers rejection
        distance_outliers_rejection = False
        # whether or not to conduct double tracks rejection
        double_tracks_rejection = False
        # what to display
        display_semantic_mapping = True
        display_distance_histogram = True

        if num_files == 3:
            print('Only 3 files, this is the initial run, start running!')

        else:
            print(
                'There is no file in generated_docs_this_run or number of files in generated_docs_this_run are more than three.'
                'If this is the first run of localization and mapping process, please check your generated_docs_this_run folder.\n '
                'If not, if you want to re-run the whole process, please set rerun == True and only_visualize == False; If you want to only'
                'display the results, please set rerun == False and only_visualize == True')

        if rerun == True or num_files == 3 and only_visualize == False:
            # step 1: find valid 3D points
            save_valid_points()
            print('1st step of 5 steps finished!')

            # step 2: find trunk position and its 3D bounding box vertices
            find_object_pos_and_bbox()
            print('2nd step of 5 steps finished!')

            # step 3: find camera position
            find_camera_pos()
            print('3rd step of 5 steps finished!')

            # step 4: transform from global frame to camera frame
            trans_glob2cam()
            print('4th step of 5 steps finished!')

            # step 5: conduct correction and visualize final results
            conduct_correction_and_visualize(distance_outliers_rejection, double_tracks_rejection,
                                             display_semantic_mapping,
                                             display_distance_histogram)
            print('5th step of 5 steps finished!')

        if only_visualize:
            conduct_correction_and_visualize(distance_outliers_rejection, double_tracks_rejection,
                                             display_semantic_mapping,
                                             display_distance_histogram)
