from __future__ import (print_function, division, absolute_import)
import pickle
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import transforms3d as trans3d

def trans_glob2cam():
    with open('generated_docs_this_run/pos_area_fridx_of_fruits.pkl', 'rb') as input:
        fruits_pos_area_list = pickle.load(input)
    fruits_pos_area_arr = np.array(fruits_pos_area_list)
    xs = fruits_pos_area_arr[:,0]
    ys = fruits_pos_area_arr[:,1]
    zs = fruits_pos_area_arr[:,2]
    area = fruits_pos_area_arr[:,3]
    frame_ids = fruits_pos_area_arr[:,4]
    rot_trans_dict = {}
    with open('generated_docs_this_run/image_and_camera_id.txt') as f:
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
                img_info_arr = (np.fromstring(line, dtype=float, sep=' '))
                quat_img = img_info_arr[1:5]
                rot_mat_world_to_img = trans3d.quaternions.quat2mat(quat_img)
                trans_world_to_img = img_info_arr[5:8]
                frame_id_temp = int(img_info_arr[0])
                rot_trans_dict[str(frame_id_temp)] = [rot_mat_world_to_img,trans_world_to_img]
            ind += 1
    num_1 = frame_ids.shape[0]
    pos_cam_frame = np.zeros((num_1,3))
    xs_cam_frame = np.zeros_like(xs)
    ys_cam_frame = np.zeros_like(ys)
    zs_cam_frame = np.zeros_like(zs)
    pos_cam_frame_area_fridx_of_fruits = fruits_pos_area_arr
    for pt_idx in range(num_1):
        frid_int = int(frame_ids[pt_idx])
        # TODO: here I directly let camera poses which are not obtained to be its neighbor (last frame) camera poses!!!!!!!!!!!!!!!!!! Maybe change this !!!!!!
        if str(frid_int) not in rot_trans_dict:
            frid_int = frid_int-1
            print(frid_int, 'not good!!!\n')
            if str(frid_int) not in rot_trans_dict:
                frid_int = frid_int + 2
                print(frid_int, 'not good!!!\n')
                if str(frid_int) not in rot_trans_dict:
                    frid_int = frid_int - 3
                    print(frid_int, 'not good!!!\n')
                    if str(frid_int) not in rot_trans_dict:
                        frid_int = frid_int - 1
                        print(frid_int, 'not good!!!\n')
                        if str(frid_int) not in rot_trans_dict:
                            frid_int = frid_int - 1
                            print(frid_int, 'not good!!!\n')
                            if str(frid_int) not in rot_trans_dict:
                                frid_int = frid_int - 1
                                print(frid_int, 'not good!!!\n')
                                if str(frid_int) not in rot_trans_dict:
                                    frid_int = frid_int - 1
                                    print(frid_int, 'not good!!!\n')
                                    if str(frid_int) not in rot_trans_dict:
                                        raise Exception('frame:', frid_int, 'and its neighbor camera poses are all not obtained!!!' )
        rot = rot_trans_dict[str(frid_int)][0]
        trans = rot_trans_dict[str(frid_int)][1]
        pos_world_frame = np.array([xs[pt_idx],ys[pt_idx],zs[pt_idx]])
        pos_cam_frame[pt_idx,:] = np.dot(rot, pos_world_frame) + trans
        xs_cam_frame[pt_idx] = pos_cam_frame[pt_idx, 0]
        ys_cam_frame[pt_idx] = pos_cam_frame[pt_idx, 1]
        zs_cam_frame[pt_idx] = pos_cam_frame[pt_idx, 2]
        pos_cam_frame_area_fridx_of_fruits[pt_idx,0:3] = pos_cam_frame[pt_idx,:].T
        # area3D = area2D * depth^2
        pos_cam_frame_area_fridx_of_fruits[pt_idx, 3] = pos_cam_frame_area_fridx_of_fruits[pt_idx, 3] * (pos_cam_frame_area_fridx_of_fruits[pt_idx, 2])**2

    np.save('generated_docs_this_run/pos_cam_frame_area_fridx_of_fruits.npy',pos_cam_frame_area_fridx_of_fruits)
    np.savetxt('generated_docs_this_run/pos_cam_frame_area_fridx_of_fruits.txt',pos_cam_frame_area_fridx_of_fruits,fmt = '%.2f',header='3D positions in camera frame and areas*depth^2(z^2) of fruits, format: X, Y, Z, bbox area*Z^2, fruit frame ID')
    print(1)


if __name__=="__main__":
   trans_glob2cam()

