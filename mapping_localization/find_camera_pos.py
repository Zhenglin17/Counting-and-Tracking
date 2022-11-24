from __future__ import (print_function, division, absolute_import)
import transforms3d as trans3d
import pickle
import cv2
import numpy as np
import sys
import os

def find_camera_pos(data_dir):
    # x = np.genfromtxt('images.txt',dtype='str')
    # x = np.loadtxt('images.txt', dtype='str', comments='#', delimiter=' ', converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0)
    image_id = []
    points = []
    frame_name = []
    # read image and points in
    with open(os.path.join(data_dir,'generated_docs_this_run/image_and_camera_id.txt')) as f:
        # Image list with two lines of data per image:
        #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID(use this image_id[ind][-1] to extract information we need!!!ID-1 = frame number!!!), NAME
        #   POINTS2D[] as (X, Y, POINT3D_ID)
        lines=f.readlines()
        ind = -1
        length = len(lines) - 1
        img_pos_id = np.zeros((length,4))
        img_pos_id_list = []
        for line in lines:
            # skip comments in the first four lines
            if ind >= 0:
                img_id_temp = line[-9:-5]
                img_info_arr = (np.fromstring(line, dtype=float, sep=' '))
                quat_img = img_info_arr[1:5]
                rot_mat_img = trans3d.quaternions.quat2mat(quat_img)
                trans_img = img_info_arr[5:8]
                img_pos_temp = - np.dot(rot_mat_img.T, trans_img)
                img_pos_id[ind,:] = np.append(int(img_info_arr[0]),img_pos_temp)
                img_pos_id_list.append(img_pos_id[ind,:])
            ind += 1
    np.save(os.path.join(data_dir,'generated_docs_this_run/img_id_position.npy'),img_pos_id)
    np.savetxt(os.path.join(data_dir,'generated_docs_this_run/img_id_position.txt'),img_pos_id_list,fmt='%.3f',delimiter=' ', newline='\n', header='#FRAME_ID, X, Y, Z', footer='', comments='')
    # np.savetxt('valid_points.txt',valid_points,fmt='%.3f',delimiter=' ', newline='\n', header='', footer='')
    print('over')

if __name__ == "__main__":
    find_camera_pos()
