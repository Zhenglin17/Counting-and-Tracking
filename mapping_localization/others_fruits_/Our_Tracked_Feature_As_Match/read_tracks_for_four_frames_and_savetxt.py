from __future__ import (print_function, division, absolute_import)
import pickle
import cv2
import numpy as np
import sys
sys.path.append('C:\\Users\\UPenn Robotics LX\\Documents\\GitHub\\Tracking_my_implementation')
from scpye.improc.image_processing import enhance_contrast
from scpye.track.assignment import hungarian_assignment
from scpye.track.bounding_box import (bboxes_assignment_cost, extract_bbox)
from scpye.track.fruit_track import FruitTrack
from scpye.track.optical_flow import (calc_optical_flow, calc_average_flow)
from scpye.utils.drawing import (Colors, draw_bboxes, draw_optical_flows,
                                 draw_text, draw_bboxes_matches, draw_line)

with open('tracks_frame_idx_whole_list.pkl', 'rb') as input_tracks:
    tracks_frame_idx_whole_list = pickle.load(input_tracks)
start_idx = int(tracks_frame_idx_whole_list[0][1])
for fr_idx in range(10,100,4):
    cur_pos = []
    pre_pos = []
    pre_pre_pos = []
    pre_pre_pre_pos = []
    # pre_pre_pos = []
    idx_str_0 = ('%04d' %fr_idx)
    idx_str_1 = ('%04d' % (fr_idx + 1))
    idx_str_2 = ('%04d' % (fr_idx + 2))
    idx_str_3 = ('%04d' % (fr_idx + 3))
    f0 = cv2.imread('input\\inp_frame'+idx_str_0+'.png')
    f1 = cv2.imread('input\\inp_frame'+idx_str_1+'.png')
    f2 = cv2.imread('input\\inp_frame'+idx_str_2+'.png')
    f3 = cv2.imread('input\\inp_frame'+idx_str_3+'.png')
    tracks = []
    tracks = tracks_frame_idx_whole_list[fr_idx+3-start_idx][0]
    frame_idx = tracks_frame_idx_whole_list[fr_idx+3-start_idx][1]
    for track in tracks:
        if frame_idx != idx_str_3:
                print('Check tracks_frame_idx_whole_list!!!!!!!!!!!!!')
        if len(track.hist)>=4:

            #TODO: be very careful: track.hist is [col_num:x, row_num:y]!!!!!!!!!!!!!!!!!!!!!!!
            cur_pos.append(track.hist[-1])
            pre_pos.append(track.hist[-2])
            pre_pre_pos.append(track.hist[-3])
            pre_pre_pre_pos.append(track.hist[-4])

            r3 = int(track.hist[-1][1])
            c3 = int(track.hist[-1][0])
            r2 = int(track.hist[-2][1])
            c2 = int(track.hist[-2][0])
            r1 = int(track.hist[-3][1])
            c1 = int(track.hist[-3][0])
            r0 = int(track.hist[-4][1])
            c0 = int(track.hist[-4][0])

            f3[r3-3:r3+3,c3-3:c3+3, :] = 255
            f2[r2-3:r2+3,c2-3:c2+3, :] = 255
            f1[r1-3:r1+3,c1-3:c1+3,:] = 255
            f0[r0-3:r0+3,c0-3:c0+3, :] = 255
            # pre_pre_pos.append(track.hist[-3])

        cv2.imshow('f0',f0)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()

        cv2.imshow('f1',f1)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()

        cv2.imshow('f2',f2)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()

        cv2.imshow('f3',f3)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()

    np.save('inp_results\\inp_frame'+idx_str_3+'.png',cur_pos)
    np.save('inp_results\\inp_frame'+idx_str_2+'.png',pre_pos)
    np.save('inp_results\\inp_frame'+idx_str_1+'.png',pre_pre_pos)
    np.save('inp_results\\inp_frame'+idx_str_0+'.png',pre_pre_pre_pos)
    print ('features in every frame are:', len(cur_pos))

    cv2.imwrite('inp_results\\result_'+idx_str_0+'.png',f0)
    cv2.imwrite('inp_results\\result_'+idx_str_1+'.png',f1)
    cv2.imwrite('inp_results\\result_'+idx_str_2+'.png',f2)
    cv2.imwrite('inp_results\\result_'+idx_str_3+'.png',f3)
    print(fr_idx)