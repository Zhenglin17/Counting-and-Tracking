from __future__ import (print_function, division, absolute_import)
import pickle
import cv2
import numpy as np
import sys
import os

def save_valid_points(data_dir = '.'):
    # x = np.genfromtxt('images.txt',dtype='str')
    # x = np.loadtxt('images.txt', dtype='str', comments='#', delimiter=' ', converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0)
    image_id = []
    points = []
    image_id_full = []
    # read image and points in
    with open(os.path.join(data_dir,'generated_docs_this_run/images.txt')) as f:
        # Image list with two lines of data per image:
        #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID(use this image_id[ind][-1] to extract information we need!!!ID-1 = frame number!!!), NAME
        #   POINTS2D[] as (X, Y, POINT3D_ID)
        lines=f.readlines()
        ind = 0
        for line in lines:
            # skip comments in the first four lines
            if ind >= 4:
                if ind % 2 == 0:
                    # actual frame number
                    image_id_temp = int(line[-9:-5])
                    arr = np.fromstring(line, dtype=float, sep=' ')
                    arr[0] = image_id_temp
                    image_id.append(arr)
                    # extract the image name out, e.g. 'bgr0000.png'
                    image_id_full.append(line[-12:-1])
                    # frame_ind = image_id[int(ind/2)-2][-1]
                    # frame_ind = '%04d' % (frame_ind - 1)
                    # image_id_full.append('bgr'+frame_ind+'.png')
                else:
                    points.append(np.fromstring(line, dtype=float, sep=' '))
            ind += 1
    # np.savetxt('match_ind.txt', match_ind, fmt='%d', delimiter=' ', newline='\n', header='inp_frame0000.png inp_frame0001.png', footer='', comments='')

    # store points which are valid in reconstruction in a new list named valid_points (remove points which have no correspondence(==-1))
    # valid points is a list, every element in it is an array, every row of the array is: X, Y, POINT3D_ID
    valid_points = []
    ind_of_point = 0
    for point in points:
        total_num = len(point)
        is_first_point = True
        #initialize valid_point to zeros in case that there is no valid point in this iteration
        valid_point = np.zeros((1,3))
        for i in range(total_num):
            if (i+1) % 3 == 0:
                if point[i] != -1:
                    # print ('yes')
                    if is_first_point == True:
                        valid_point = (point[i-2:i+1]).reshape(1,3)
                        is_first_point = False
                    else:
                        valid_point = np.append(valid_point,(point[i-2:i+1]).reshape(1,3),axis = 0)
                        # print (point[i-2:i+1])
                # else:
                #     print('no')
                #     print (point[i-2:i+1])

        # valid_points is a list of points containing valid_point 's index and corresponding frame number!!!
        # i.e. valid_points[i][0] is an array of all valid points in the frame, and valid_points[i][1] is the number of the frame
        # TODO: HERE be CAREFUL!!! store the frame number instead of its name!!!!!
        valid_points.append([valid_point, float(image_id_full[ind_of_point][-8:-4])])
        ind_of_point += 1

    np.savetxt(os.path.join(data_dir,'generated_docs_this_run/image_and_camera_id.txt'),image_id,fmt='%.3f',delimiter=' ', newline='\n', header='#IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID', footer='', comments='')
    np.save(os.path.join(data_dir,'generated_docs_this_run/valid_points.npy'), valid_points)
    # np.savetxt('valid_points.txt',valid_points,fmt='%.3f',delimiter=' ', newline='\n', header='', footer='')
    print('over')

if __name__ == "__main__":
    save_valid_points()
