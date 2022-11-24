from __future__ import (print_function, division, absolute_import)
import pickle
import cv2
import numpy as np
import sys
sys.path.append('C:\\Users\\UPenn Robotics LX\\Documents\\GitHub\\Tracking_my_implementation')

with open('three_dim_pos_of_fruits.pkl', 'rb') as input:
    fruits_three_dim_pos_list = pickle.load(input)
fruits_three_dim_pos_arr = np.array(fruits_three_dim_pos_list)

np.savetxt('three_dim_pos_of_fruits.txt',fruits_three_dim_pos_arr,'%.2f',header='3D positions of fruits(X, Y, Z)')


print('over')

# with open('three_dim_pos_of_fruits.pkl', 'wb') as output_loc:
#     pickle.dump(three_dim_pos_of_fruits, output_loc)
# with open('frame_num_corr_pos_of_fruits.pkl', 'wb') as output_frame_num:
#     pickle.dump(frame_num_corr_pos_of_fruits, output_frame_num)
