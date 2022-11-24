from __future__ import (print_function, division, absolute_import)
import pickle
import cv2
import numpy as np
import sys

def find_object_pos_and_bbox():

    with open('semantic_data_association/counted_fruits_and_last_frame_whole_list.pkl', 'rb') as input:
        # counted_fruits_and_last_frame is a list, every element is: [[x,y,w,h],[array(xc,yc)]]
        # counted_fruits_and_last_frame_whole_list is a list, element in it is also a list, in which the first element is counted_fruits_and_last_frame and the second element is frame number, e.g., frame0000
        fruits_frame_list = pickle.load(input)

    # valid_points is a list of points containing valid_point 's index and corresponding frame number!!!
    # i.e. valid_points[i][0] is an array of all valid points (x,y) in the frame, and valid_points[i][1] is the number of the frame (number of the frame = camer_id - 1 since its index starts from 0)
    valid_points = np.load('generated_docs_this_run/valid_points.npy')



    # 3D point list with one line of data per point:
    #   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
    # use three_dim_pts_ind_pos_dict to record the 3D points index, X, Y, Z as a dictionary, key is: POINT3D_ID, element is: [X, Y, Z]
    three_dim_pts_ind_pos_dict = {}
    with open('generated_docs_this_run/points3D.txt') as f:
        # Image list with two lines of data per image:
        #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID(use this image_id[ind][-1] to extract information we need!!!ID-1 = frame number!!!), NAME
        #   POINTS2D[] as (X, Y, POINT3D_ID)
        lines=f.readlines()
        ind = 0
        for line in lines:
            # skip the comments

            if line[0] == '#':
                continue
            else:
                #only need first four: POINT3D_ID, X, Y, Z
                pt_info_temp = np.fromstring(line, dtype=float, sep=' ')
                three_dim_pts_ind_pos_dict[pt_info_temp[0]] = pt_info_temp[1:4]

    print('total number of reconstructed 3D points are:', len(three_dim_pts_ind_pos_dict))



    frame_points = []
    frame_num_strs = []
    for valid_point in valid_points:
        frame_points.append(valid_point[0])
        frame_num_str_temp = 'frame%04d' % (valid_point[1])
        frame_num_strs.append(frame_num_str_temp)

    num_frames = len(frame_num_strs)

    pos_area_fridx_of_fruits = []
    # frame_num_corr_pos_of_fruits = []
    for element in fruits_frame_list:
        if element[0]== None or element[0] ==[] or element[1] == None or element[1] == []:
            print('fruits_frame_list element is empty!')
            continue
        # element in list fruits is: [array(bbox), array(x_center, y_center)]
        fruits = element[0]
        frame = element[1]

        corresponding_frame_found = 0
        for ind in range(num_frames):
            # find the match frame, record corresponding valid points
            if frame_num_strs[ind] == frame:
                corresponding_frame_found = 1
                a = frame_num_strs[ind]
                frame_ind = int(frame_num_strs[ind][-4:])
                print('found match frame', frame_num_strs[ind])
                cur_valid_points = frame_points[ind]
                num_pts = cur_valid_points.shape[0]
                # find the location of fruit one-by-one
                for fruit in fruits:
                    # bbox is [x,y,w,h]
                    ##TODO: change this value!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    another_margin_ratio = 1
                    bbox = fruit[0]
                    bbox_area = bbox[2]*bbox[3]
                    x_low = bbox[0] - bbox[2]*(another_margin_ratio/2.0)
                    x_up = bbox[0]+bbox[2] + bbox[2]*(another_margin_ratio/2.0)

                    y_low = bbox[1] - bbox[3]*(another_margin_ratio/2.0)
                    y_up = bbox[1]+bbox[3] + bbox[3]*(another_margin_ratio/2.0)


                    # average over all neighbor points' location (that are inside the fruit's bounding box)
                    # TODO: new edit: I vectorize here to speed up
                    neighbor_points_three_dim_loc = []
                    good_idx_x1 = (x_low <= cur_valid_points[:, 0])
                    good_idx_x2 = (cur_valid_points[:, 0] <= x_up)
                    good_idx_y1 = (y_low <= cur_valid_points[:, 1])
                    good_idx_y2 = (cur_valid_points[:, 1] <= y_up)
                    good_idx =np.logical_and(np.logical_and(np.logical_and(good_idx_x1, good_idx_x2), good_idx_y1), good_idx_y2)
                    three_dim_pt_idx_all = cur_valid_points[good_idx,2]
                    fruit_center_position = fruit[1]
                    print('feature position and fruit center position are (should be the same in semantic matching process):',cur_valid_points[good_idx,:2], fruit_center_position)
                    for three_dim_pt_idx in three_dim_pt_idx_all:
                        neighbor_points_three_dim_loc.append(three_dim_pts_ind_pos_dict[three_dim_pt_idx])

                    # **********before vectorizing
                    # average over all neighbor points' location (that are inside the fruit's bounding box)
                    # neighbor_points_three_dim_loc = []
                    # for pts_ind in range(num_pts):
                    #     # check if the point is in the fruit bounding box, if yes, regard that as one of the fruit's neighbour feature point
                    #     x_pt = cur_valid_points[pts_ind,0]
                    #     y_pt = cur_valid_points[pts_ind,1]
                    #     if x_low<=x_pt<=x_up and y_low<=y_pt<=y_up:
                    #         # find corresponding 3D point index of the feature point, which is just 3rd column of valid_point array
                    #         three_dim_pt_ind = cur_valid_points[pts_ind,2]
                    #         # find the corresponding 3D point location
                    #         for key in three_dim_pts_ind_pos_dict:
                    #             if key == three_dim_pt_ind:
                    #                 # print('found! 3D point index is:', three_dim_pt_ind)
                    #                 neighbor_points_three_dim_loc.append(three_dim_pts_ind_pos_dict[key])
                    # *********before vectorizing ends

                    # calculate the average as our fruit's location
                    num_neighbor_pts = len(neighbor_points_three_dim_loc)
                    if num_neighbor_pts == 0:
                        print('NO fruit position found!')
                        continue
                    print('fruit position found!')
                    x_avg = 0
                    y_avg = 0
                    z_avg = 0
                    for loc in neighbor_points_three_dim_loc:
                        x_avg += loc[0] / float(num_neighbor_pts)
                        y_avg += loc[1] / float(num_neighbor_pts)
                        z_avg += loc[2] / float(num_neighbor_pts)
                    pos_temp = np.array([x_avg,y_avg,z_avg,bbox_area,frame_ind])
                    pos_area_fridx_of_fruits.append(pos_temp)
                    # frame_num_corr_pos_of_fruits.append(frame)
        if corresponding_frame_found == 0:
             print('No corresponding reconstructed frame found!!!!!')






    print('over')
    pos_area_fridx_of_fruits_arr = np.array(pos_area_fridx_of_fruits)
    np.savetxt('generated_docs_this_run/pos_area_fridx_of_fruits_text.txt',pos_area_fridx_of_fruits_arr,'%.2f',header='3D positions and areas of fruits(X, Y, Z, bbox area, fruit frame ID)')
    with open('generated_docs_this_run/pos_area_fridx_of_fruits.pkl', 'wb') as output_loc:
        pickle.dump(pos_area_fridx_of_fruits, output_loc)
    # with open('generated_docs_this_run\\frame_num_corr_pos_of_fruits.pkl', 'wb') as output_frame_num:
    #     pickle.dump(frame_num_corr_pos_of_fruits, output_frame_num)

if __name__ == "__main__":
    find_object_pos_and_bbox()
