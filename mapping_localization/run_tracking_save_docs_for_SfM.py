import logging
import cv2
import os
import time
import pickle
import numpy as np
import sys

# sys.path.append('/home/sam/git/Tracking_my_implementation')

start = time.time()
print("timer started \n")

from scpye.improc.binary_cleaner import BinaryCleaner
from scpye.improc.blob_analyzer import BlobAnalyzer
from scpye.track.fruit_tracker import FruitTracker
from scpye.utils.data_manager import DataManager
from scpye.utils.bag_manager import BagManager
from scpye.utils.fruit_visualizer import FruitVisualizer
import copy

import matplotlib.pyplot as plt


def run_tracking_save_docs_for_SfM(start_idx=1, end_idx=5000, inp_format='.jpg', pred_format='.png',
                                   data_dir_pre = r'C:\Users\tomdo\Ag_Lab\tracking_stack\test_data',
                                   fruits_moving_direction=0, min_age=3):
    # %%
    frame_interval = 1
    # base_dir = #'/home/chao/Workspace/dataset/agriculture'
    base_dir = data_dir_pre
    fruit = ''
    color = ''
    mode = ''
    side = ''
    bag_ind = 1

    # %%
    dm = DataManager(base_dir, fruit=fruit, color=color, mode=mode, side=side)
    bm = BagManager(dm.data_dir, bag_ind)

    bc = BinaryCleaner(ksize=5, iters=1)
    ba = BlobAnalyzer()
    ft = FruitTracker(fruits_moving_direction=fruits_moving_direction, min_age=min_age)

    # RCNN
    def find_previous_frame(fr_num):
        j = '%04d' % (fr_num - 1)
        prev_fname_bgr = os.path.join(data_dir_pre, 'input', 'frame' + str(j) + inp_format)
        prev_fname_bgr_frame_number_only = 'frame' + str(j)
        print ('prev_fname_bgr_frame_number_only', prev_fname_bgr_frame_number_only)
        if os.path.isfile(prev_fname_bgr):
            return cv2.imread(prev_fname_bgr), prev_fname_bgr_frame_number_only
        else:
            raise Exception('previous frame not found!!!!!!!!!!!!!!!!!!!!!!!')

    # for bgr, bw in tqdm(bm.load_detect()):
    is_start_frame = 1
    # TODO: record counted_fruits_and_last_frame_whole_list for localization (details see fruit_tracker comment(search counted_fruits_and_last_frame))
    counted_fruits_and_last_frame_whole_list = []
    tracks_frame_idx_whole_list = []
    recorded_lost_tracks = []
    prev_time = time.time()
    for i in range(start_idx, end_idx, frame_interval):
        if i % 5 == 0:
            cur_time = time.time()
            print('time for every 5 iterations:', cur_time - prev_time, '\n')
            prev_time = cur_time
        i = '%04d' % i
        print('current frame:' + i)
        fname = os.path.join(data_dir_pre, 'pred', 'pred_frame' + str(i) + pred_format)
        fname_bgr = os.path.join(data_dir_pre, 'input', 'frame' + str(i) + inp_format)
        if os.path.isfile(fname_bgr) == False:
            print(
            fname_bgr + ' is not found (if the last frame has not been tracked, check whether the data dir and the first frame number is correct) \n Tracking process is ceased!',
            '\n')
            break
            # raise Exception('image not found!!!!!!!!!!!!!!!!!!!!!!!')

        if os.path.isfile(fname):
            bgr = cv2.imread(fname_bgr)
            # TODO: check whether it is the first frame, if yes, no previous frame can be found, just let previous frame = first frame
            # print is_start_frame
            # RCNN
            if is_start_frame:
                print ('i', i)
                prev_bgr = bgr
                prev_fname_bgr = fname_bgr
                prev_fname_bgr_frame_number_only = 'frame' + str(i)
            else:
                prev_bgr, prev_fname_bgr_frame_number_only = find_previous_frame(int(i))

            # RCNN: following no use
            # bw = cv2.imread(fname)
            # bw[bw==1] = 250
            # bw = cv2.cvtColor(bw, cv2.COLOR_BGR2GRAY)
            # # The following process is to change the greyscale bw (segmented, binary) image, into another binary image which only contains contours (fruits) which has area > 4
            # bw = bc.clean(bw)
            # # fruits here are a 2D array of [x y w h] of bboxes extracted in bolb_analyzer.py
            # fruits, _ = ba.analyze(bgr, bw)
            # bw[bw==1] = 255

            # RCNN:
            fruits = np.load(fname)
            # bw = np.zeros((bgr.shape[0],bgr.shape[1]))
            bw = bgr
            bw = cv2.cvtColor(bw, cv2.COLOR_BGR2GRAY)

            # TODO: add prev_bgr parameter to record previous frame
            # TODO: record counted_fruits_and_last_frame for localization
            if is_start_frame:
                print('first iteration ft.track is none!!!!!!')
                # RCNN
                counted_fruits_and_last_frame = ft.track(bgr, fruits, bw, prev_bgr=prev_bgr)
                is_start_frame = 0

            else:
                # if len(fruits) == 0:
                #     print ('no detections in this frame!!!')
                #     ft.disp_bgr = bgr
                #     ft.disp_bw = bw_copy
                # else:
                # RCNN
                counted_fruits_and_last_frame, recorded_lost_tracks = ft.track(bgr, fruits, bw, prev_bgr=prev_bgr)
            # TODO: record counted_fruits_and_last_frame_whole list for localization (details see fruit_tracker comment(search counted_fruits_and_last_frame))
            counted_fruits_and_last_frame_whole_list.append(
                [counted_fruits_and_last_frame, prev_fname_bgr_frame_number_only])
            # TODO: copy and save in tracks_frame_idx_whole_list.pkl(first element: all tracks in frame, second element: frame index))
            copied_recorded_lost_tracks = copy.deepcopy(recorded_lost_tracks)
            tracks_frame_idx_whole_list.append([copied_recorded_lost_tracks, i])
            # TODO: use this function to save tracking images, green boxes are counted, blue are new detections, cyan are tracked, refer to bag_manager.py
            bm.save_track(ft.disp_bgr, ft.disp_bw, save_disp=True)
            # plt.pause(0)
        else:
            print (fname + ' does not exist!\n')

    # TODO: record counted_fruits_and_last_frame for localization
    # RCNN
    counted_fruits_and_last_frame = ft.finish(prev_bgr)
    # TODO: record counted_fruits_and_last_frame_whole list for localization (details see fruit_tracker comment(search counted_fruits_and_last_frame))
    # TODO: counted_fruits_and_last_frame_whole_list is a list, element in it is also a list, in which the first element is counted_fruits_and_last_frame and the second element is frame number, e.g., frame0000
    counted_fruits_and_last_frame_whole_list.append([counted_fruits_and_last_frame, prev_fname_bgr_frame_number_only])
    with open(data_dir_pre + '/counted_fruits_and_last_frame_whole_list.pkl',
              'wb') as output_counted_fruits_and_last_frame_whole_list:
        pickle.dump(counted_fruits_and_last_frame_whole_list, output_counted_fruits_and_last_frame_whole_list)
    # First element, all 'tracks' objects, second element: frame index
    with open(data_dir_pre + '/tracks_frame_idx_whole_list.pkl', 'wb') as output_tracks_frame_idx_whole_list:
        pickle.dump(tracks_frame_idx_whole_list, output_tracks_frame_idx_whole_list)

    end = time.time()
    print ('timer ended, total time is:', end - start)


if __name__ == '__main__':
    run_tracking_save_docs_for_SfM()
