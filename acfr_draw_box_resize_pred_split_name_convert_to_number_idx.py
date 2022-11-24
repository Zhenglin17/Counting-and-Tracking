# Modifed for use on windows or linux/mac

import glob
import cv2
import os
import csv
import numpy as np
from shutil import copyfile
import sys
from pathlib import PurePath, PurePosixPath

# ---Replace--- with location needed for utilites, not needed for clones
# sys.path.append('/home/sam/git/Tracking_my_implementation')
from scpye.utils.drawing import (Colors, draw_bboxes, draw_optical_flows,
                                 draw_text, draw_bboxes_matches, draw_line)
from scpye.improc.image_processing import enhance_contrast


def acfr_draw_boxes(parent_data_dir, folder_dir_working_on):
    # ---Replace---
    parent_data_dir = parent_data_dir
    print(folder_dir_working_on)
    folder_dir_working_on = glob.glob(folder_dir_working_on)

    if len(folder_dir_working_on) != 1:
        raise Exception('folder_dir_working_on != 1, check data fold into that folder!')
    else:
        data_dir_pre = folder_dir_working_on[0]
        print('current data directory:', data_dir_pre)

    sorted_name_list = sorted(glob.glob(os.path.join(data_dir_pre, 'png', '*.png')))
    i = -1
    # is_first = 1
    resize_factor = 1
    for img_name in sorted_name_list:
        i += 1
        idx = '%04d' % i
        c = os.path.split(img_name)[-1]
        actual_name = c[:-4] #split on .png
        if os.path.isfile(os.path.join(data_dir_pre, 'png', actual_name + '.png')):

            img_ori = cv2.imread(os.path.join(data_dir_pre, 'png', actual_name + '.png'))
            w, h, _ = img_ori.shape
            # if is_first:
            # img_bw = np.zeros((w,h))

            # copyfile(csv_name, './pred/pred_frame'+idx+'.csv')
            cv2.imwrite(os.path.join(data_dir_pre, 'input', 'frame' + idx + '.png'), img_ori)

            if os.path.isfile(os.path.join(data_dir_pre, 'target_tree_binary_label', actual_name + '.png')):
                img_mask_bi = cv2.imread(os.path.join(data_dir_pre, 'target_tree_binary_label', actual_name + '.png'),
                                         0)
                cv2.imwrite(os.path.join(data_dir_pre, 'target_tree_binary_label', 'frame' + idx + '.png'), img_mask_bi)
                img_mask = cv2.imread(os.path.join(data_dir_pre, 'target_tree_mask', actual_name + '.png'))
                cv2.imwrite(os.path.join(data_dir_pre, 'target_tree_binary_label', 'frame' + idx + '.png'), img_mask_bi)

            img_ori = enhance_contrast(img_ori)
            csv_name = (os.path.join(parent_data_dir, 'det_all', actual_name + '.csv'))
            if os.path.isfile(csv_name):
                with open(csv_name, 'r') as csvfile:
                    bboxes = []
                    box_reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
                    for row in box_reader:
                        x_min = row[1] * float(resize_factor)
                        y_min = row[2] * float(resize_factor)
                        w_box = (row[3] - row[1]) * float(resize_factor)
                        h_box = (row[4] - row[2]) * float(resize_factor)
                        bbox = [x_min, y_min, w_box, h_box]
                        bboxes.append(bbox)
                    if bboxes != []:
                        draw_bboxes(img_ori, bboxes, color=[0, 255, 0], thickness=3)
                    # img_bw[int(round(y_min)):int(round(y_min+h_box)), int(round(x_min)):int(round(x_min+w_box))] = 255
            else:
                print('not detections in current image:', actual_name)
                bboxes = []

            np.save(os.path.join(data_dir_pre, 'pred', 'pred_frame' + idx + '.npy'), bboxes)
            cv2.imwrite(os.path.join(data_dir_pre, 'pred_bbox', 'pred_frame' + idx + '.png'), img_ori)
            # cv2.imwrite('./pred/pred_frame'+idx+'.png',img_bw)
            print('current image:', actual_name)
        else:
            print((os.path.join(data_dir_pre, 'png', actual_name + '.png') + '.png does not exist!'))
