from __future__ import (print_function, division, absolute_import)
import pickle
import cv2
import numpy as np
import sys

from scpye.improc.image_processing import enhance_contrast
from scpye.track.assignment import hungarian_assignment
from scpye.track.bounding_box import (bboxes_assignment_cost, extract_bbox)
from scpye.track.fruit_track import FruitTrack
from scpye.track.optical_flow import (calc_optical_flow, calc_average_flow)
from scpye.utils.drawing import (Colors, draw_bboxes, draw_optical_flows,
                                 draw_text, draw_bboxes_matches, draw_line)
import os
import glob
import copy
def semantic_feature_matches(parent_data_dir, folder_dir_working_on):
    # TODO: if this is for extract semantic data association from 2D tracking, set this to 2D; If this is for refined data association from landmark based tracking, set this to landmark
    landmark_or_2Dtracking = '2D'

    folder_dir_working_on = glob.glob(folder_dir_working_on)
    if len(folder_dir_working_on) != 1:
        raise Exception('folder_dir_working_on != 1, check data folder!')
    else:
        data_dir_pre = folder_dir_working_on[0]
        print('current data directory:', data_dir_pre)

    if landmark_or_2Dtracking == 'landmark':
        start_fr_idx = 2

        if os.path.exists('landmark_refined_frame_match_idx/inp_match_idx.txt'):
            os.remove('landmark_refined_frame_match_idx/inp_match_idx.txt')
            print ('old txt files are removed!')
        for fname in glob.glob('landmark_refined_landmark_frame_features/*.txt'):
                os.remove(fname)



        # TODO: be careful: lost tracks only have information before (k-1)th frame if lost in kth frame
        with open(os.path.join(data_dir_pre, 'landmarked_disappeared_tracks_frame_idx_whole_list.pkl'), 'rb') as input_tracks:
            print('Extracting landmark refined data association...\n')
            print('Pickling...\n')
            lost_tracks_frame_idx_whole_list = pickle.load(input_tracks)

    else:
        start_fr_idx = 0

        if os.path.exists('frame_match_idx/inp_match_idx.txt'):
            os.remove('frame_match_idx/inp_match_idx.txt')
            print ('old txt files are removed!')
        for fname in glob.glob('frame_features/*.txt'):
                os.remove(fname)


        # TODO: be careful: lost tracks only have information before (k-1)th frame if lost in kth frame
        with open(os.path.join(data_dir_pre, 'tracks_frame_idx_whole_list.pkl'), 'rb') as input_tracks:
            print('Extracting 2D tracking data association...\n')
            print('Pickling...\n')
            lost_tracks_frame_idx_whole_list = pickle.load(input_tracks)


    #TODO: change this according to data!!!!!!!!!!!!!!!!!!!!!!
    image_name_prefix = 'frame'
    image_type = '.png'

    min_initialization_age = 3

    # lost_tracks_frame_idx_whole_list: first element: track objects; second element: frame index string, e.g., '0001'
    # because lost_tracks_frame_idx_whole_list sometimes does not start from 0000 frame, add this parameter to align up
    #TODO: be careful: lost tracks only have information before (k-1)th frame if lost in kth frame, therefore the start frame_idx = recorded frame index - 1
    lost_tracks_frame_idx_whole_list_start_idx = int(lost_tracks_frame_idx_whole_list[0][1]) - 1
    isfirst = 1
    list_all_match_idx = []
    # fr_idx is saved idx - 3!!!!!!!!!!!!!!!!!!(the pre_pre_pre frame's index)

    # # feature counter records number of features in every frame in order to match the index of tracked features in different frames!!!!!!!!!!!
    # feature_counter = {}
    # feature_pos = {}
    # # dictionary, key: frame1 and frame2, element: 2d array of corresopnding matching feature idx
    # match_idx = {}

    already_started = 0

    # reject outliers according to color portion (green channel value / sum of 3 colors); and illumination (avg of all 3 colors).
    outlier_rejection = 0
    img_features_dict = {}
    feature_matches_dict = {}
    feature_idx_dict = {}

    scale = 1.1
    orientation = 0.3
    descriptor = (np.ones((1, 128))).astype(int)
    scale_ori = np.array([[scale, orientation]])
    sift_arr = np.concatenate((scale_ori, descriptor), axis=1)

    if landmark_or_2Dtracking == 'landmark':
        print('landmark part was commented out, should revert this commenting!')
        proportion_of_KLT_optimized_small_ignored_track = 0
        total_track = 0
        for fr_idx in range(start_fr_idx,1500,1):
            #TODO: be careful: lost tracks only have information before (k-1)th frame if lost in kth frame, therefore the start frame_idx = recorded frame index - 1
            if fr_idx >= len(lost_tracks_frame_idx_whole_list) - 1:
                print('All tracks has been parsed, this script ends!')
                break
            tracks = lost_tracks_frame_idx_whole_list[fr_idx-lost_tracks_frame_idx_whole_list_start_idx][0]
            if len(tracks) == 0:
                print('no lost fruit tracks (counted fruis) in frame:', fr_idx, 'continue to the next frame')
                continue

            # TODO: be careful: lost tracks only have information before (k-1)th frame if lost in kth frame, therefore the start frame_idx = recorded frame index - 1
            frame_idx = int(lost_tracks_frame_idx_whole_list[fr_idx-lost_tracks_frame_idx_whole_list_start_idx][1])-1
            fr_idx_str = ('%04d' %fr_idx)
            if frame_idx != fr_idx:
                raise Exception('frame_idx in lost_tracks_frame_idx_whole_list are not matched with fr_idx in the script. Check implementation!')

            # TODO: change margin values!!!
            # define the visualization red box dimension (mar*mar)
            mar2 = 3
            # outlier_rejection part herae already removed, please see code before 8.13 if you want to add it back

            for track in tracks:

                if len(track.KLT_hist) != len(track.hist) or len(track.KLT_status) != len(track.hist):
                    raise Exception('len(track.KLT_hist) != len(track.hist) or len(track.KLT_status) != len(track.hist), check landmark fruit tracker implementation!')

                if track.age>=min_initialization_age:
                    total_track += 1
                    proportion_of_KLT_optimized = sum(track.KLT_status) / float(len(track.KLT_status))
                    if proportion_of_KLT_optimized < 0.5:
                        print('proportion_of_KLT_optimized= ',proportion_of_KLT_optimized,' too small, ignore this track')
                        proportion_of_KLT_optimized_small_ignored_track += 1
                        continue
                    # pix_cords (or track.hist) is a list where every element represents (x,y) or (col_num, row_num) of the track
                    # Here remove code because no need to check coords in margin, otherwise refer code before 8.15
                    # outlier_rejection part here already removed, please see code before 8.13 if you want to add it back
                    f = []
                    # TODO: be careful: lost tracks only have information before (k-1)th frame if lost in kth frame
                    for fr_idx_sub in np.arange(track.age):
                        fr_idx_actual = fr_idx-fr_idx_sub
                        fr_idx_str = ('%04d' %(fr_idx_actual))
                        if fr_idx_actual not in img_features_dict:
                            img_features_dict[fr_idx_actual] = []
                        img_features_dict[fr_idx_actual].append(track.KLT_hist[track.age-1-fr_idx_sub])#, track.detected_bbox_hist[track.age-1-fr_idx_sub]])

                        # get current feature index
                        feature_idx_dict[fr_idx_actual] = len(img_features_dict[fr_idx_actual]) - 1

                    for fr_idx_sub in np.arange(0,track.age):
                        # build feature correspondence between current frame and current+1 frame
                        for fr_idx_sub_sub in np.arange(fr_idx_sub+1, track.age):
                            print(fr_idx - fr_idx_sub,fr_idx - fr_idx_sub_sub)
                            print(track.age)
                            if (fr_idx - fr_idx_sub, fr_idx - fr_idx_sub_sub) not in feature_matches_dict:
                                feature_matches_dict[(fr_idx - fr_idx_sub,fr_idx- fr_idx_sub_sub)] = [[feature_idx_dict[fr_idx - fr_idx_sub], feature_idx_dict[fr_idx- fr_idx_sub_sub]]]
                            else:
                                feature_matches_dict[(fr_idx - fr_idx_sub,fr_idx- fr_idx_sub_sub)].append([feature_idx_dict[fr_idx - fr_idx_sub], feature_idx_dict[fr_idx- fr_idx_sub_sub]])
        with open (data_dir_pre+'/landmark_refined_frame_match_idx/inp_match_idx.txt', 'w') as match_file:
            for fr_idx_1_and_0, value in feature_matches_dict.items():
                # the key in feature_idx_dict is [frame(i+1), framei]
                idx_str_1 = '%04d' % fr_idx_1_and_0[0]
                idx_str_0 = '%04d' % fr_idx_1_and_0[1]
                string1_0 = image_name_prefix + idx_str_1 + image_type + ' ' + image_name_prefix + idx_str_0 + image_type
                # list append for the last 3 frames
                list_match_idx = ['\n'+string1_0, '\n'.join(' '.join(str(cell) for cell in row) for row in value)]
                for subitem in list_match_idx:
                    match_file.write("%s\n" % subitem)

        for fr_idx, features in img_features_dict.items():
            features_list = []
            # last feature's idx + 1 is the total number of features
            num_features = feature_idx_dict[fr_idx] + 1
            sift_arr_rep = np.repeat(sift_arr, num_features, axis=0)
            for feature in features:
                features_list.append(feature)
            features_arr = np.array(features_list)
            feature_pos = np.append(features_arr, sift_arr_rep, axis = 1)
            num_features_str = str(num_features)
            fr_idx_str = '%04d' %fr_idx
            np.savetxt(data_dir_pre+'/landmark_refined_frame_features/'+image_name_prefix+fr_idx_str+image_type+'.txt', feature_pos, fmt='%.2f', delimiter=' ', newline='\n', header=str(num_features) + ' 128',
                   footer='', comments='')

            res_vis = cv2.imread(data_dir_pre+'/input/'+image_name_prefix+fr_idx_str+image_type)
            for cord in features_arr:
                cord = cord.astype(int)
                res_vis[cord[1] - mar2 : cord[1] + mar2 + 1, cord[0] - mar2 : cord[0] + mar2 + 1, :] = 255
            cv2.imwrite(data_dir_pre+'/landmark_refined_semantic_features_results/result_'+fr_idx_str+image_type,res_vis)
            print('writing result images:'+fr_idx_str+image_type)

        with open(data_dir_pre+'/landmark_refined_image_features_dict.pkl', 'wb') as output:
            # img_features_dict is in the form of: key: image frame index, value: a list with features
            pickle.dump(img_features_dict, output)

        print('Build_semantic_feature_matches finished! \n number of proportion_of_KLT_optimized_small_ignored_track =', proportion_of_KLT_optimized_small_ignored_track, 'number of total track is',total_track,'\n')




    elif landmark_or_2Dtracking == '2D':
        # TODO: new feature added: associate the image_features_dict with corresponding track id!!!
        global_track_idx = -1
        for fr_idx in range(start_fr_idx,1500,1):
            #TODO: be careful: lost tracks only have information before (k-1)th frame if lost in kth frame, therefore the start frame_idx = recorded frame index - 1
            if fr_idx >= len(lost_tracks_frame_idx_whole_list) - 1:
                print('All tracks has been parsed, this script ends!')
                break

            tracks = lost_tracks_frame_idx_whole_list[fr_idx-lost_tracks_frame_idx_whole_list_start_idx][0]
            if len(tracks) == 0:
                print('no lost fruit tracks (counted fruis) in frame:', fr_idx, 'continue to the next frame')
                continue


            # TODO: be careful: lost tracks only have information before (k-1)th frame if lost in kth frame, therefore the start frame_idx = recorded frame index - 1
            frame_idx = int(lost_tracks_frame_idx_whole_list[fr_idx-lost_tracks_frame_idx_whole_list_start_idx][1])-1
            fr_idx_str = ('%04d' %fr_idx)
            if frame_idx != fr_idx:
                raise Exception('frame_idx in lost_tracks_frame_idx_whole_list are not matched with fr_idx in the script. Check implementation!')

            # TODO: change margin values!!!
            # define the visualization red box dimension (mar*mar)
            mar2 = 3
            # outlier_rejection part herae already removed, please see code before 8.13 if you want to add it back

            for track in tracks:
                if track.age>=min_initialization_age:
                    global_track_idx += 1
                    # pix_cords (or track.hist) is a list where every element represents (x,y) or (col_num, row_num) of the track
                    # Here remove code because no need to check coords in margin, otherwise refer code before 8.15
                    # outlier_rejection part here already removed, please see code before 8.13 if you want to add it back
                    f = []
                    # TODO: be careful: lost tracks only have information before (k-1)th frame if lost in kth frame
                    for fr_idx_sub in np.arange(track.age):
                        fr_idx_actual = fr_idx-fr_idx_sub
                        fr_idx_str = ('%04d' %(fr_idx_actual))
                        if fr_idx_actual not in img_features_dict:
                            img_features_dict[fr_idx_actual] = []
                        # TODO: new feature added: associate the image_features_dict with corresponding track id!!!
                        # e
                        img_features_dict[fr_idx_actual].append([copy.deepcopy(track.hist[track.age-1-fr_idx_sub]), copy.deepcopy(track.detected_bbox_hist[track.age-1-fr_idx_sub]), copy.deepcopy(global_track_idx)])

                        # get current feature index
                        feature_idx_dict[fr_idx_actual] = len(img_features_dict[fr_idx_actual]) - 1

                    for fr_idx_sub in np.arange(0,track.age):
                        # build feature correspondence between current frame and current+1 frame
                        for fr_idx_sub_sub in np.arange(fr_idx_sub+1, track.age):
                            # print(fr_idx - fr_idx_sub,fr_idx - fr_idx_sub_sub)
                            # print(track.age)
                            if (fr_idx - fr_idx_sub, fr_idx - fr_idx_sub_sub) not in feature_matches_dict:
                                feature_matches_dict[(fr_idx - fr_idx_sub,fr_idx- fr_idx_sub_sub)] = [[feature_idx_dict[fr_idx - fr_idx_sub], feature_idx_dict[fr_idx- fr_idx_sub_sub]]]
                            else:
                                feature_matches_dict[(fr_idx - fr_idx_sub,fr_idx- fr_idx_sub_sub)].append([feature_idx_dict[fr_idx - fr_idx_sub], feature_idx_dict[fr_idx- fr_idx_sub_sub]])




        with open (data_dir_pre+'/frame_match_idx/inp_match_idx.txt', 'w') as match_file:
            for fr_idx_1_and_0, value in feature_matches_dict.items():
                # the key in feature_idx_dict is [frame(i+1), framei]
                idx_str_1 = '%04d' % fr_idx_1_and_0[0]
                idx_str_0 = '%04d' % fr_idx_1_and_0[1]
                string1_0 = image_name_prefix + idx_str_1 + image_type + ' ' + image_name_prefix + idx_str_0 + image_type
                # list append for the last 3 frames
                list_match_idx = ['\n'+string1_0, '\n'.join(' '.join(str(cell) for cell in row) for row in value)]
                for subitem in list_match_idx:
                    match_file.write("%s\n" % subitem)

        for fr_idx, feature_bboxes in img_features_dict.items():
            features_list = []
            # last feature's idx + 1 is the total number of features
            num_features = feature_idx_dict[fr_idx] + 1
            sift_arr_rep = np.repeat(sift_arr, num_features, axis=0)
            for feature_bbox in feature_bboxes:
                features_list.append(feature_bbox[0])
            features_arr = np.array(features_list)
            feature_pos = np.append(features_arr, sift_arr_rep, axis = 1)
            num_features_str = str(num_features)
            fr_idx_str = '%04d' %fr_idx
            np.savetxt(data_dir_pre+'/frame_features/'+image_name_prefix+fr_idx_str+image_type+'.txt', feature_pos, fmt='%.2f', delimiter=' ', newline='\n', header=str(num_features) + ' 128',
                   footer='', comments='')

            res_vis = cv2.imread(data_dir_pre+'/input/'+image_name_prefix+fr_idx_str+image_type)
            for cord in features_arr:
                cord = cord.astype(int)
                res_vis[cord[1] - mar2 : cord[1] + mar2 + 1, cord[0] - mar2 : cord[0] + mar2 + 1, :] = 255
            cv2.imwrite(data_dir_pre+'/semantic_features_results/result_'+fr_idx_str+image_type,res_vis)
            print('writing result images:'+fr_idx_str+image_type)

        with open(data_dir_pre+'/image_features_bbox_dict.pkl', 'wb') as output:
            # img_features_dict is in the form of: key: image frame index, value: a list with features and their bboxes, inside every element of this list is a sub-list, where the first element is feature coordinate in the image plane, the second element is the bbox x,y,w,h
            pickle.dump(img_features_dict, output)

    else:
        raise Exception('set landmark_or_2Dtracking to be either landmark or 2D')
