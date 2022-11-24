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

if os.path.exists('frame_match_idx/inp_match_idx.txt'):
    raise Exception ('frame_match_idx/inp_match_idx.txt already exists! Please delete frame_features and frame_match_idx folders before running this script!!')

os.mkdir('./frame_match_idx')
os.mkdir('./frame_features')

dir = '/home/sam/Desktop/ACFR_PKL/8.15_200frs'
with open(os.path.join(dir, 'tracks_frame_idx_whole_list.pkl'), 'rb') as input_tracks:
    print('Pickling...\n')
    tracks_frame_idx_whole_list = pickle.load(input_tracks)

#TODO: change this according to data!!!!!!!!!!!!!!!!!!!!!!
image_name_prefix = 'frame'
image_type = '.png'

# tracks_frame_idx_whole_list: first element: track objects; second element: frame index string, e.g., '0001'
start_idx = int(tracks_frame_idx_whole_list[0][1])
isfirst = 1
list_all_match_idx = []
# fr_idx is saved idx - 3!!!!!!!!!!!!!!!!!!(the pre_pre_pre frame's index)

# feature counter records number of features in every frame in order to match the index of tracked features in different frames!!!!!!!!!!!
feature_counter = {}
feature_pos = {}
# dictionary, key: frame1 and frame2, element: 2d array of corresopnding matching feature idx
match_idx = {}

start_fr_idx = 0
already_started = 0

# reject outliers according to color portion (green channel value / sum of 3 colors); and illumination (avg of all 3 colors).
outlier_rejection = 0
for fr_idx in range(start_fr_idx,1050,1):
    cur_pos = []
    pre_pos = []
    pre_pre_pos = []
    pre_pre_pre_pos = []
    # pre_pre_pos = []
    idx_str_0 = ('%04d' %fr_idx)
    idx_str_1 = ('%04d' % (fr_idx + 1))
    idx_str_2 = ('%04d' % (fr_idx + 2))
    idx_str_3 = ('%04d' % (fr_idx + 3))
    if os.path.isfile('input/'+image_name_prefix+idx_str_0+image_type) == False:
        raise Exception ('input/'+image_name_prefix+idx_str_0+image_type+' does not exist!!!!!!!!')
    f0 = cv2.imread('input/'+image_name_prefix+idx_str_0+image_type)
    image_rows, image_cols, _ = f0.shape
    f1 = cv2.imread('input/'+image_name_prefix+idx_str_1+image_type)
    f2 = cv2.imread('input/'+image_name_prefix+idx_str_2+image_type)
    f3 = cv2.imread('input/'+image_name_prefix+idx_str_3+image_type)
    tracks = tracks_frame_idx_whole_list[fr_idx+3-start_idx][0]
    frame_idx = tracks_frame_idx_whole_list[fr_idx+3-start_idx][1]
    if frame_idx != idx_str_3:
        raise Exception('Check tracks_frame_idx_whole_list!!!!!!!!!!!!!')


    # TODO: change margin values!!!
    mar1 = 2
    mar2 = 3
    # outlier_rejection part here already removed, please see code before 8.13 if you want to add it back
    for track in tracks:
        if len(track.hist)>=4:
            #TODO: be very careful: track.hist is [col_num:x, row_num:y]!!!!!!!!!!!!!!!!!!!!!!!
            r3 = int(track.hist[-1][1])
            c3 = int(track.hist[-1][0])
            r2 = int(track.hist[-2][1])
            c2 = int(track.hist[-2][0])
            r1 = int(track.hist[-3][1])
            c1 = int(track.hist[-3][0])
            r0 = int(track.hist[-4][1])
            c0 = int(track.hist[-4][0])
            if r3 >= image_rows-mar1 or r2 >= image_rows-mar1 or r1 >= image_rows-mar1 or r0 >= image_rows-mar1:
                print ('row index over maximun! row maximun is:',image_rows - mar1,'idx is:',r0,r1,r2,r3)
                continue
            if c3 >= image_cols - mar1 or c2 >= image_cols - mar1 or c1 >= image_cols - mar1 or c0 >= image_cols - mar1:
                print ('col index over maximun! col maximun is:',image_cols - mar1,'idx is:',c0,c1,c2,c3)
                continue
            if r3 <= 0+ mar1 or r2 <= 0+ mar1 or r1 <= 0+ mar1 or r0 <= 0+ mar1:
                print ('row index <margin, idx is:',r0,r1,r2,r3)
                continue
            if c3 <= 0+ mar1 or c2 <= 0+ mar1 or c1 <= 0+ mar1 or c0 <= 0+ mar1:
                print ('col index <margin, idx is:',c0,c1,c2,c3)
                continue

            # outlier_rejection part here already removed, please see code before 8.13 if you want to add it back

            if isfirst == 1:
                print('This ( frame num:', fr_idx, ') is the first frame with fruits which have hist >= 4!')
                # apple_pixels_green_portion = cur_green_portion
                isfirst =0
                f3[r3 - mar2 : r3 + mar2 + 1, c3 - mar2 : c3 + mar2 + 1, 2] = 255
                f2[r2 - mar2: r2 + mar2 + 1, c2 - mar2: c2 + mar2 + 1, 2] = 255
                f1[r1 - mar2: r1 + mar2 + 1, c1 - mar2: c1 + mar2 + 1, 2] = 255
                f0[r0 - mar2: r0 + mar2 + 1, c0 - mar2: c0 + mar2 + 1, 2] = 255
            else:
                if outlier_rejection:
                    raise Exception('this outlier_rejection part already removed, please see code before 8.13 if you want to add it back')

                else:
                    f3[r3 - mar2 : r3 + mar2 + 1, c3 - mar2 : c3 + mar2 + 1, 2] = 255
                    f2[r2 - mar2 : r2 + mar2 + 1, c2 - mar2 : c2 + mar2 + 1, 2] = 255
                    f1[r1 - mar2 : r1 + mar2 + 1, c1 - mar2 : c1 + mar2 + 1, 2] = 255
                    f0[r0 - mar2 : r0 + mar2 + 1, c0 - mar2 : c0 + mar2 + 1, 2] = 255
                    cur_pos.append(track.hist[-1])
                    pre_pos.append(track.hist[-2])
                    pre_pre_pos.append(track.hist[-3])
                    pre_pre_pre_pos.append(track.hist[-4])


    # f0_show = cv2.resize(f0,(int(f0.shape[1]/2),int(f0.shape[0]/2)))
    # cv2.imshow(idx_str_0,f0_show)
    # cv2.waitKey(5000)
    # cv2.destroyAllWindows()
    #
    # f1_show = cv2.resize(f1, (int(f0.shape[1]/2),int(f0.shape[0]/2)))
    # cv2.imshow(idx_str_1,f1_show)
    # cv2.waitKey(5000)
    # cv2.destroyAllWindows()
    #
    # f2_show = cv2.resize(f2, (int(f0.shape[1]/2),int(f0.shape[0]/2)))
    # cv2.imshow(idx_str_2,f2_show)
    # cv2.waitKey(5000)
    # cv2.destroyAllWindows()
    #
    # f3_show = cv2.resize(f3, (int(f0.shape[1]/2),int(f0.shape[0]/2)))
    # cv2.imshow(idx_str_3,f3_show)
    # cv2.waitKey(5000)
    # cv2.destroyAllWindows()

    # np.save('inp_results\\inp_frame'+idx_str_3+'.png',cur_pos)
    # np.save('inp_results\\inp_frame'+idx_str_2+'.png',pre_pos)
    # np.save('inp_results\\inp_frame'+idx_str_1+'.png',pre_pre_pos)
    # np.save('inp_results\\inp_frame'+idx_str_0+'.png',pre_pre_pre_pos)

    if isfirst:
        print('no fruits with age > 4 in frame:',fr_idx,', continue to the next frame\n')
        continue
    elif already_started == 0:
        actual_start_fr_idx = fr_idx
        print('actual start frame (the first frame with fruits that have hist >= 4) is:',fr_idx, '\n')
        already_started = 1
    num_features = len(cur_pos)
    print('features in every frame are:', num_features)
    if num_features == 0:
        print('\n\nno fruits in frame:',fr_idx,', this process break here!\n\n')
        break

    cv2.imwrite('results/result_'+idx_str_0+image_type,f0)
    cv2.imwrite('results/result_'+idx_str_1+image_type,f1)
    cv2.imwrite('results/result_'+idx_str_2+image_type,f2)
    cv2.imwrite('results/result_'+idx_str_3+image_type,f3)
    scale = 1.1
    orientation = 0.3
    descriptor = (np.ones((1, 128))).astype(int)
    scale_ori = np.array([[scale, orientation]])
    sift_arr = np.concatenate((scale_ori, descriptor), axis=1)
    # make descriptor look different every iteration to avoid SfM making them the same feature -- no longer needed
    # sift_arr += (fr_idx%4)*1
    sift_arr_rep = np.repeat(sift_arr, num_features, axis=0)
    cur_pos = np.append(cur_pos, sift_arr_rep, axis=1)
    pre_pos = np.append(pre_pos, sift_arr_rep, axis=1)
    pre_pre_pos = np.append(pre_pre_pos, sift_arr_rep, axis=1)
    pre_pre_pre_pos = np.append(pre_pre_pre_pos, sift_arr_rep, axis=1)

    feature_counter[idx_str_3] = num_features
    feature_pos[idx_str_3] = cur_pos
    if idx_str_2 in feature_counter:
        feature_counter[idx_str_2] += num_features
        feature_pos[idx_str_2] = np.append(feature_pos[idx_str_2],pre_pos,0)
        feature_counter[idx_str_1] += num_features
        feature_pos[idx_str_1] = np.append(feature_pos[idx_str_1], pre_pre_pos, 0)
        feature_counter[idx_str_0] += num_features
        feature_pos[idx_str_0] = np.append(feature_pos[idx_str_0], pre_pre_pre_pos, 0)
    # elif idx_str_1 in feature_counter:
    #     feature_counter[idx_str_2] = num_features
    #     feature_pos[idx_str_2] = pre_pos
    #     feature_counter[idx_str_1] += num_features
    #     feature_pos[idx_str_1] = np.append(feature_pos[idx_str_1], pre_pre_pos, 0)
    #     feature_counter[idx_str_0] += num_features
    #     feature_pos[idx_str_0] = np.append(feature_pos[idx_str_0], pre_pre_pre_pos, 0)
    # elif idx_str_0 in feature_counter:
    #     feature_counter[idx_str_2] = num_features
    #     feature_pos[idx_str_2] = pre_pos
    #     feature_counter[idx_str_1] = num_features
    #     feature_pos[idx_str_1] = pre_pre_pos
    #     feature_counter[idx_str_0] += num_features
    #     feature_pos[idx_str_0] = feature_pos[idx_str_0].append(pre_pre_pre_pos)
    else:
        feature_counter[idx_str_2] = num_features
        feature_pos[idx_str_2] = pre_pos
        feature_counter[idx_str_1] = num_features
        feature_pos[idx_str_1] = pre_pre_pos
        feature_counter[idx_str_0] = num_features
        feature_pos[idx_str_0] = pre_pre_pre_pos

    num_features_str = str(num_features)
    np.savetxt('frame_features/'+image_name_prefix+idx_str_3+image_type+'.txt', feature_pos[idx_str_3], fmt='%.2f', delimiter=' ', newline='\n', header=str(feature_counter[idx_str_3]) + ' 128',
               footer='', comments='')
    np.savetxt('frame_features/'+image_name_prefix+idx_str_2+image_type+'.txt',feature_pos[idx_str_2], fmt='%.2f', delimiter=' ', newline='\n', header=str(feature_counter[idx_str_2]) + ' 128',
               footer='', comments='')
    np.savetxt('frame_features/'+image_name_prefix+idx_str_1+image_type+'.txt',feature_pos[idx_str_1], fmt='%.2f', delimiter=' ', newline='\n',
               header=str(feature_counter[idx_str_1]) + ' 128', footer='', comments='')
    np.savetxt('frame_features/'+image_name_prefix+idx_str_0+image_type+'.txt',feature_pos[idx_str_0], fmt='%.2f', delimiter=' ', newline='\n',
               header=str(feature_counter[idx_str_0]) + ' 128', footer='', comments='')

    match_idx0 = np.arange(feature_counter[idx_str_0]-num_features, feature_counter[idx_str_0]).reshape(num_features, 1)
    match_idx1 = np.arange(feature_counter[idx_str_1]-num_features, feature_counter[idx_str_1]).reshape(num_features, 1)
    match_idx2 = np.arange(feature_counter[idx_str_2]-num_features, feature_counter[idx_str_2]).reshape(num_features, 1)
    match_idx3 = np.arange(num_features).reshape(num_features, 1)

    string3_2 = image_name_prefix + idx_str_3 + image_type + ' ' + image_name_prefix + idx_str_2 + image_type
    string2_1 = image_name_prefix + idx_str_2 + image_type + ' ' + image_name_prefix + idx_str_1 + image_type
    string1_0 = image_name_prefix + idx_str_1 + image_type + ' ' + image_name_prefix + idx_str_0 + image_type
    string3_0 = image_name_prefix + idx_str_3 + image_type + ' ' + image_name_prefix + idx_str_0 + image_type
    string3_1 = image_name_prefix + idx_str_3 + image_type + ' ' + image_name_prefix + idx_str_1 + image_type
    string2_0 = image_name_prefix + idx_str_2 + image_type + ' ' + image_name_prefix + idx_str_0 + image_type

    match_idx[idx_str_3 + idx_str_2] = np.append(match_idx3, match_idx2, 1)
    match_idx[idx_str_3 + idx_str_1] = np.append(match_idx3, match_idx1, 1)
    match_idx[idx_str_3 + idx_str_0] = np.append(match_idx3, match_idx0, 1)


    if fr_idx != actual_start_fr_idx:
        match_idx[idx_str_1 + idx_str_0]= np.append(match_idx[idx_str_1 + idx_str_0],np.append(match_idx1, match_idx0, 1),0)
        match_idx[idx_str_2 + idx_str_1] = np.append(match_idx[idx_str_2 + idx_str_1],np.append(match_idx2, match_idx1, 1),0)
        match_idx[idx_str_2 + idx_str_0] = np.append(match_idx[idx_str_2 + idx_str_0],np.append(match_idx2, match_idx0, 1),0)
        list_match_idx = ['\n'+string1_0,'\n'.join(' '.join(str(cell) for cell in row) for row in match_idx[idx_str_1 + idx_str_0]),
                          '\n' + string3_0, '\n'.join(' '.join(str(cell) for cell in row) for row in match_idx[idx_str_3 + idx_str_0]),
                          '\n' + string2_0, '\n'.join(' '.join(str(cell) for cell in row) for row in match_idx[idx_str_2 + idx_str_0]),
                          ]
    else:
        match_idx[idx_str_1 + idx_str_0] = np.append(match_idx1, match_idx0, 1)
        match_idx[idx_str_2 + idx_str_1] = np.append(match_idx2, match_idx1, 1)
        match_idx[idx_str_2 + idx_str_0] = np.append(match_idx2, match_idx0, 1)
        list_match_idx = ['\n'+string3_2, '\n'.join(' '.join(str(cell) for cell in row) for row in match_idx[idx_str_3 + idx_str_2]),
                          '\n'+string2_1, '\n'.join(' '.join(str(cell) for cell in row) for row in match_idx[idx_str_2 + idx_str_1]),
                          '\n'+string1_0,'\n'.join(' '.join(str(cell) for cell in row) for row in match_idx[idx_str_1 + idx_str_0]),
                          '\n' + string3_0, '\n'.join(' '.join(str(cell) for cell in row) for row in match_idx[idx_str_3 + idx_str_0]),
                          '\n' + string3_1, '\n'.join(' '.join(str(cell) for cell in row) for row in match_idx[idx_str_3 + idx_str_1]),
                          '\n' + string2_0, '\n'.join(' '.join(str(cell) for cell in row) for row in match_idx[idx_str_2 + idx_str_0]),
                          ]

    list_all_match_idx.append(list_match_idx)
    if fr_idx % 10== 0:
        with open ('frame_match_idx/inp_match_idx.txt', 'w') as match_file:
            for item in list_all_match_idx:
                for subitem in item:
                    match_file.write("%s\n" % subitem)

    # np.savetxt('frame_match_idx/inp_match_idx'+image_name_prefix+idx_str_3+'.txt', list_match_idx, fmt='%d', delimiter=' ', newline='\n',
    #            header='', footer='', comments='')
    print('Finished for frame:'+idx_str_3)


# list append for the last 3 frames
list_match_idx = ['\n'+string3_2, '\n'.join(' '.join(str(cell) for cell in row) for row in match_idx[idx_str_3 + idx_str_2]),
                  '\n'+string2_1, '\n'.join(' '.join(str(cell) for cell in row) for row in match_idx[idx_str_2 + idx_str_1]),
                  '\n' + string3_1, '\n'.join(' '.join(str(cell) for cell in row) for row in match_idx[idx_str_3 + idx_str_1])
                  ]
list_all_match_idx.append(list_match_idx)
with open ('frame_match_idx/inp_match_idx.txt', 'w') as match_file:
    for item in list_all_match_idx:
        for subitem in item:
            match_file.write("%s\n" % subitem)
