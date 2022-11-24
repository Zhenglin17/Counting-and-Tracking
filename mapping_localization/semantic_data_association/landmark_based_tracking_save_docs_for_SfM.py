import logging
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
print ("timer started \n")


from scpye.improc.binary_cleaner import BinaryCleaner
from scpye.improc.blob_analyzer import BlobAnalyzer
from mapping_localization.semantic_data_association.landmark_based_fruit_tracker import LandmarkFruitTracker
from scpye.utils.data_manager import DataManager
from scpye.utils.bag_manager import BagManager
from scpye.utils.landmark_bag_manager import LandmarkBasedBagManager
from scpye.utils.fruit_visualizer import FruitVisualizer
import copy

from mapping_localization.semantic_data_association.extract_landmark_3D_pos_bboxes_hist import  extract_landmark_3D_pos_bboxes_hist
from mapping_localization.semantic_data_association.extract_every_frame_rot_trans import  extract_every_frame_rot_trans

import matplotlib.pyplot as plt




def run_landmark_based_tracking_save_docs_for_SfM(start_idx,end_idx,inp_format,pred_format, data_dir_pre, fruit_3D_pos_bboxes_hist_dict, rot_trans_dict, K_mat, min_age):

    # %%
    frame_interval = 1
    #base_dir = #'/home/chao/Workspace/dataset/agriculture'
    base_dir = data_dir_pre
    fruit = ''
    color = ''
    mode = ''
    side = ''
    bag_ind = 1

    # %%
    dm = DataManager(base_dir, fruit = fruit, color=color, mode=mode, side=side)
    bm = LandmarkBasedBagManager(dm.data_dir, bag_ind)

    bc = BinaryCleaner(ksize=5, iters=1)
    ba = BlobAnalyzer()
    lmft = LandmarkFruitTracker(min_age=min_age)

    # RCNN
    def find_previous_frame(fr_num):
        j = '%04d' % (fr_num - 1)
        prev_fname_bgr = os.path.join(data_dir_pre,'input','frame'+str(j)+inp_format)
        prev_fname_bgr_frame_number_only = 'frame' + str(j)
        print ('prev_fname_bgr_frame_number_only', prev_fname_bgr_frame_number_only)
        if os.path.isfile(prev_fname_bgr):
            return cv2.imread(prev_fname_bgr), prev_fname_bgr_frame_number_only
        else:
            raise Exception('previous frame not found!!!!!!!!!!!!!!!!!!!!!!!')
    # for bgr, bw in tqdm(bm.load_detect()):
    is_start_frame = 1
    tracks_frame_idx_whole_list = []
    recorded_disappeared_tracks = []
    prev_time = time.time()
    for i in range(start_idx, end_idx,frame_interval):
        if i %5 ==0:
            cur_time = time.time()
            print('time for every 5 iterations:',cur_time-prev_time,'\n')
            prev_time = cur_time
        i_str = '%04d' % i
        print('current frame:' + i_str)
        fname = os.path.join(data_dir_pre,'pred','pred_frame'+i_str+pred_format)
        fname_bgr = os.path.join(data_dir_pre,'input','frame'+i_str+inp_format)
        if os.path.isfile(fname_bgr)==False:
            print(fname_bgr + ' is not found (if the last frame has not been tracked, check whether the data dir and the first frame number is correct) \n Tracking process is ceased!','\n')
            break
            # raise Exception('image not found!!!!!!!!!!!!!!!!!!!!!!!')

        if os.path.isfile(fname):
            bgr = cv2.imread(fname_bgr)
            # TODO: check whether it is the first frame, if yes, no previous frame can be found, just let previous frame = first frame
            # print is_start_frame
            # RCNN
            if is_start_frame:
                prev_bgr = bgr
                prev_fname_bgr = fname_bgr
                prev_fname_bgr_frame_number_only = 'frame' + i_str
            else:
                prev_bgr, prev_fname_bgr_frame_number_only = find_previous_frame(i)

            # RCNN: following no use

            # RCNN:
            fruits = np.load(fname)
            # bw = np.zeros((bgr.shape[0],bgr.shape[1]))
            bw = bgr
            bw = cv2.cvtColor(bw, cv2.COLOR_BGR2GRAY)

            # TODO: add prev_bgr parameter to record previous frame
            if is_start_frame:
                print('first iteration ft.track is none!!!!!!')
                # RCNN
                lmft.track(bgr, fruits, bw, prev_bgr=prev_bgr, current_frame_idx= i, fruit_3D_pos_bboxes_hist_dict = fruit_3D_pos_bboxes_hist_dict, rot_trans_dict= rot_trans_dict, K_mat = K_mat, data_dir_pre=data_dir_pre)
                is_start_frame = 0
                counted_fruits_and_last_frame = None
            else:
                # TODO: for landmark based, record current disappeard tracks (be careful: disappear tracks only have information before (k-1)th frame if lost in kth frame) instead of all tracks for direct matching
                recorded_disappeared_tracks = lmft.track(bgr, fruits, bw, prev_bgr = prev_bgr, current_frame_idx= i, fruit_3D_pos_bboxes_hist_dict = fruit_3D_pos_bboxes_hist_dict, rot_trans_dict= rot_trans_dict, K_mat = K_mat, data_dir_pre=data_dir_pre)
            # counted_fruits_and_last_frame_whole_list.append([counted_fruits_and_last_frame,prev_fname_bgr_frame_number_only])
            # TODO: copy and save in tracks_frame_idx_whole_list.pkl(first element: all tracks in frame, second element: frame index))
            copied_recorded_disappeared_tracks = copy.deepcopy(recorded_disappeared_tracks)
            tracks_frame_idx_whole_list.append([copied_recorded_disappeared_tracks, i_str])
            # TODO: use this function to save tracking images, green boxes are counted, blue are new detections, cyan are tracked, refer to bag_manager.py
            bm.save_track(lmft.disp_bgr, lmft.disp_bw, lmft.disp_prev_bgr,lmft.save_prev_bgr, save_disp=True)
            # plt.pause(0)
            actual_end_frame_idx = i
        else:
            print (fname+' does not exist!\n')
            # actual_end_frame_idx = i-1


    # RCNN
    # print('finishing the track, current idx:',i,'actual finish idx', actual_end_frame_idx)
    # actual_end_frame_idx_str = '%04d' %actual_end_frame_idx

    # tracks lost in the last frame should be recorded to lost tracks for (the last frame + 1)
    last_fr_lost_tracks_idx = actual_end_frame_idx+1
    last_fr_lost_tracks_idx_str = '%04d' %(last_fr_lost_tracks_idx)
    recorded_disappeared_tracks = lmft.finish(prev_bgr, current_frame_idx= last_fr_lost_tracks_idx)
    copied_recorded_disappeared_tracks = copy.deepcopy(recorded_disappeared_tracks)
    tracks_frame_idx_whole_list.append([copied_recorded_disappeared_tracks, last_fr_lost_tracks_idx_str])



    # flag = 0
    # for idx,track_fr_list in enumerate(tracks_frame_idx_whole_list):
    #     if track_fr_list[1] == actual_end_frame_idx_str:
    #         tracks_frame_idx_whole_list[idx][0].extend(copied_recorded_disappeared_tracks)
    #         flag = 1
    # if flag == 0:
    #     raise Exception('lmft implementation has problem, check it!')
    # counted_fruits_and_last_frame = lmft.finish(prev_bgr)
    # counted_fruits_and_last_frame_whole_list.append([counted_fruits_and_last_frame,prev_fname_bgr_frame_number_only])
    # with open('counted_fruits_and_last_frame_whole_list.pkl', 'wb') as output_counted_fruits_and_last_frame_whole_list:
    #     pickle.dump(counted_fruits_and_last_frame_whole_list, output_counted_fruits_and_last_frame_whole_list)
    # First element, all 'tracks' objects, second element: frame index
    with open(data_dir_pre+'/landmarked_disappeared_tracks_frame_idx_whole_list.pkl', 'wb') as output_tracks_frame_idx_whole_list:
        pickle.dump(tracks_frame_idx_whole_list, output_tracks_frame_idx_whole_list)

    # TODO: 3D landmark based tracking dictionary key: tuple (3D landmark position), value is another dictionary as: {'track':the last track of this landmark, 'frame number': 1 + the last frame that observes this landmark}
    landmark_pos_tracks_dict = lmft.landmark_pos_tracks_dict
    with open(data_dir_pre+'/landmark_pos_tracks_dict.pkl', 'wb') as output:
        pickle.dump(landmark_pos_tracks_dict, output)

    end = time.time()
    print ('timer ended, total time is:', end - start)


if __name__ == '__main__':

    # first frame number, e.g., 1 means your end frame is named frame0001.XXX
    start_idx = 9
    # end frame number, e.g., 5000 means your end frame is named frame5000.XXX, you can just set a large number if you want, the tracking process will automatically end once the final image has been read
    end_idx = 190
    # original images format
    inp_format = '.png'
    # prediction images format
    pred_format = '.npy'
    # data folders directory
    data_dir_pre = '/home/sam/Desktop/Fruit-count-dataset/ACFR_RCNN_GT/20171206T042534.518455'



    rerun_fruit_3D_pos_bboxes_hist_dict_fname_and_rot_trans_dict_fname = False

    save_valid_points(data_dir='../')
    find_camera_pos(data_dir='../')
    # fruit_3D_pos_bboxes_hist_dict: key is 3D point index, value is a dictionary: {'3Dpos':fruit_3D_loc, 'frame_idx': [frame_ind] (order: small to large), '2Dpos_bbox':fruit}, which contains fruits' 3D position, frame index where it appears (a list of integers) and corresponding frame's 2D_position_and_bbox (a list where each element is another list: [array(x_center, y_center),array(bbox)])
    fruit_3D_pos_bboxes_hist_dict_fname = data_dir_pre+'/generated_docs_this_run/fruit_3D_pos_bboxes_hist_dict.pkl'
    # every_frame_rot_trans_dict is recorded as follows: every_frame_rot_trans_dict[str(frame_id_temp)] = [rot_mat_world_to_img,trans_world_to_img]
    rot_trans_dict_fname = data_dir_pre+'/generated_docs_this_run/every_frame_rot_trans_dict.pkl'
    if os.path.isfile(fruit_3D_pos_bboxes_hist_dict_fname) == False or rerun_fruit_3D_pos_bboxes_hist_dict_fname_and_rot_trans_dict_fname:
        print(fruit_3D_pos_bboxes_hist_dict_fname + 'does not exist, running extract_landmark_3D_pos_bboxes_hist.py to generate the file\n')
        extract_landmark_3D_pos_bboxes_hist()
    else:
        print(fruit_3D_pos_bboxes_hist_dict_fname + 'already exists, NOT re-running extract_landmark_3D_pos_bboxes_hist.py\n')

    if os.path.isfile(rot_trans_dict_fname) == False or rerun_fruit_3D_pos_bboxes_hist_dict_fname_and_rot_trans_dict_fname:
        print(rot_trans_dict_fname + 'does not exist, running extract_every_frame_rot_trans.py to generate the file\n')
        extract_every_frame_rot_trans()
    else:
        print(rot_trans_dict_fname + 'already exists, NOT re-running extract_every_frame_rot_trans.py\n')

    with open(fruit_3D_pos_bboxes_hist_dict_fname, 'rb') as input:
        fruit_3D_pos_bboxes_hist_dict = pickle.load(input)
    with open(rot_trans_dict_fname, 'rb') as input:
        rot_trans_dict = pickle.load(input)

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



    run_landmark_based_tracking_save_docs_for_SfM(start_idx,end_idx,inp_format,pred_format,data_dir_pre)
