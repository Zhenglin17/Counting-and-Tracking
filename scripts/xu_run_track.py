import logging
import cv2
import os
import time
start = time.time()
print "timer started \n"

# from tqdm import tqdm
import sys
sys.path.append('C:\\Users\\UPenn Robotics LX\\Documents\\GitHub\\Tracking_my_implementation')

from scpye.improc.binary_cleaner import BinaryCleaner
from scpye.improc.blob_analyzer import BlobAnalyzer
from scpye.track.fruit_tracker import FruitTracker
from scpye.utils.data_manager import DataManager
from scpye.utils.bag_manager import BagManager
from scpye.utils.fruit_visualizer import FruitVisualizer

import matplotlib.pyplot as plt

# %%
#base_dir = #'/home/chao/Workspace/dataset/agriculture'
base_dir = 'C:\\Users\\UPenn Robotics LX\\Desktop\\Fruit-count-dataset'
color = 'red'
mode = 'slow_flash'
side = 'north'
bag_ind = 1

# %%
dm = DataManager(base_dir, color=color, mode=mode, side=side)
bm = BagManager(dm.data_dir, bag_ind)

bc = BinaryCleaner(ksize=5, iters=1)
ba = BlobAnalyzer()
ft = FruitTracker()
# fv = FruitVisualizer(pause_time=0.1)


# %%
#TODO: refer to image_processing.py and blob_analyzer: the read-in bw is greyscale image, bgr here is color image, fruits are just a list of bboxes and bw is binary image with bounding boxes in it(bw: binary image)
#TODO: CHANGE This To Read Our Own Data!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# TODO: this function is for reading the previous frame
def find_previous_frame(fr_num):
    j = '%04d' % (fr_num - 1)

    # prev_fname_bgr = 'C:\\Users\\UPenn Robotics LX\\Desktop\\Fruit-count-dataset\\screenshots\\Screenshot-000' + str(
    #     j) + '.jpg'
    prev_fname_bgr = 'C:\\Users\\UPenn Robotics LX\\Desktop\\Fruit-count-dataset\\orange-large\\input\\inp_frame' + str(j) + '.png'
    if os.path.isfile(prev_fname_bgr):
        return cv2.imread(prev_fname_bgr)
    else:
        raise Exception('previous frame not found!!!!!!!!!!!!!!!!!!!!!!!')
        # return find_previous_frame(int(i)-1)

# ckp4 = time.time()
# ckp3 = ckp4
# print ('time for initializing:', ckp4 - start)

#for bgr, bw in tqdm(bm.load_detect()):
is_start_frame = 1
for i in range(0, 602):
    # ckp2 = ckp3
    # ckp3 = time.time()
    # print ('time for whole loop', ckp3 - ckp2)
    i = '%04d' % i
    print 'current frame:' + i
    # fname = 'C:\\Users\\UPenn Robotics LX\\Desktop\\Fruit-count-dataset\\ral-2017-apples-oranges\\Labels\\orange\\orange_label_frame00' + str(i) + '.png'
    # fname_bgr = 'C:\\Users\\UPenn Robotics LX\\Desktop\\Fruit-count-dataset\\ral-2017-apples-oranges\\Images\\orange\\frame00' + str(i) + '.jpg'
    fname = 'C:\\Users\\UPenn Robotics LX\\Desktop\\Fruit-count-dataset\\orange-large\\pred\\pred_frame' + str(i) + '.png'
    # fname = 'C:\\Users\\UPenn Robotics LX\\Desktop\\Fruit-count-dataset\\screenshots\\Screenshot-000' + str(i) + '.jpg'
    #TODO:
    #TODO:
    fname_bgr = 'C:\\Users\\UPenn Robotics LX\\Desktop\\Fruit-count-dataset\\orange-large\\input\\inp_frame' + str(i) + '.png'
    if os.path.isfile(fname):
        bgr = cv2.imread(fname_bgr)
        # TODO: check whether it is the first frame, if yes, no previous frame can be found, just let previous frame = first frame
        # print is_start_frame
        if is_start_frame:
            # print ('i', i)
            prev_bgr = bgr
        else:
            prev_bgr = find_previous_frame(int(i))
        is_start_frame = 0

        bw = cv2.imread(fname)
        bgr[bgr==1] = 250
        prev_bgr[prev_bgr==1] = 250
        bw[bw==1] = 250
        # bgr = cv2.resize(bgr, (500,400))
        # prev_bgr = cv2.resize(prev_bgr, (500, 400))
        # bw = cv2.resize(bw,(500,400))
        # bgr = cv2.resize(bw,(700,800))
        # bw = cv2.resize(bw, (700, 800))
        # cv2.imshow('bw', bw)
        # cv2.waitKey(200)
        # cv2.imshow('bgr', bgr)
        # cv2.destroyAllWindows()
        bw = cv2.cvtColor(bw, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('bw', bw)
        # cv2.waitKey(2000)

        # cv2.imwrite('C:\\Users\\UPenn Robotics LX\\Desktop\\bw.png', bw)

        # TODO: The following process is to change the greyscale bw (segmented, binary) image, into another binary image which only contains contours (fruits) which has area > 4, in other words, many noise points are ruled out
        bw = bc.clean(bw)
        # TODO: fruits here are a 2D array of [x y w h] of bboxes extracted in bolb_analyzer.py
        fruits, bw = ba.analyze(bgr, bw)
        # ckp0 = time.time()
        # TODO: add prev_bgr parameter to record previous frame
        ft.track(bgr, fruits, bw, prev_bgr = prev_bgr)
        # fv.show(ft.disp_bgr, ft.disp_bw)
        # ckp1 = time.time()
        # print ('time for tracking', ckp1 - ckp0)
        # fv.show(prev_bgr, ft.disp_bw)

        # TODO: use this function to save tracking images, green boxes are counted, blue are new detections, cyan are tracked, refer to bag_manager.py
        bm.save_track(ft.disp_bgr, ft.disp_bw, save_disp=True)
        # plt.pause(0)
    else:
        print (fname+' does not exist!\n')

ft.finish(prev_bgr)

end = time.time()
print ('timer ended, total time is:', end - start)