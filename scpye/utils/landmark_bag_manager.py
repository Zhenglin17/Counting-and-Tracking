from __future__ import (print_function, division, absolute_import)

import os
import logging

import cv2
import pyrosbag as rosbag
import numpy as np
# from cv_bridge import CvBridge, CvBridgeError

class LandmarkBasedBagManager(object):
    def __init__(self, data_dir, index, bag='bag', detect='detect',
                 track='track'):
        self.data_dir = data_dir
        self.index = index

        self.bag_fmt = "frame{0}.bag"
        self.bgr_fmt = "bgr{0:04d}.png"
        self.bw_fmt = "bw{0:04d}.png"
        self.disp_fmt = "disp{0:04d}.png"

        self.i_detect = 0
        self.i_track = 0

        self.bag_dir = os.path.join(self.data_dir, bag)
        self.detect_dir = os.path.join(self.bag_dir, detect,
                                       "frame{0}".format(index))
        self.track_dir = os.path.join(self.bag_dir, track,
                                      "frame{0}".format(index))

        self.logger = logging.getLogger(__name__)
        self.logger.info("BagManger: {}".format(self.bag_dir))

    def load_bag(self, topic='/color/image_rect_color'):
        """
        A generator for image
        :param topic: image message topic
        :return:
        """
        bagname = os.path.join(self.bag_dir,
                               self.bag_fmt.format(self.index))
        self.logger.info('loading bag: {0}'.format(bagname))
        bridge = CvBridge()
        with rosbag.Bag(bagname) as bag:
            for topic, msg, t in bag.read_messages(topic):
                try:
                    image = bridge.imgmsg_to_cv2(msg)
                except CvBridgeError as e:
                    self.logger.error(e.message)
                    continue
                yield image

    def save_detect(self, bgr, bw):
        self.logger.info('saving image {}'.format(self.i_detect))

        bgr_name = os.path.join(self.detect_dir,
                                self.bgr_fmt.format(self.i_detect))
        bw_name = os.path.join(self.detect_dir,
                               self.bw_fmt.format(self.i_detect))

        cv2.imwrite(bgr_name, bgr)
        cv2.imwrite(bw_name, bw)

        self.i_detect += 1

        self.logger.debug("save bgr: {}".format(bgr_name))
        self.logger.debug("save bw: {}".format(bw_name))

    def load_detect(self):
        i = 0
        while True:
            self.logger.info('loading image {}'.format(i))
            bgr_name = os.path.join(self.detect_dir,
                                    self.bgr_fmt.format(i))
            bw_name = os.path.join(self.detect_dir,
                                   self.bw_fmt.format(i))
            bgr = cv2.imread(bgr_name, cv2.IMREAD_COLOR)
            bw = cv2.imread(bw_name, cv2.IMREAD_GRAYSCALE)

            self.logger.debug("load bgr: {}".format(bgr_name))
            self.logger.debug("load bw: {}".format(bw_name))

            if bgr is None or bw is None:
                self.logger.debug("No image left at {}".format(i))
                break
            else:
                i += 1
                yield bgr, bw

    # TODO: use this function to save tracking images, green boxes are counted, blue are new detections, cyan are tracked
    # TODO: bgr_name = 'C:\\Users\\UPenn Robotics LX\\Desktop\\Fruit-count-dataset\\apple\\red\\slow_flash\\north\\bag\\track\\frame1\\bgr0002.png'
    # TODO: bw_name = 'C:\\Users\\UPenn Robotics LX\\Desktop\\Fruit-count-dataset\\apple\\red\\slow_flash\\north\\bag\\track\\frame1\\bw0002.png'
    # TODO: disp_name = 'C:\\Users\\UPenn Robotics LX\\Desktop\\Fruit-count-dataset\\apple\\red\\slow_flash\\north\\bag\\track\\frame1\\disp0002.png'
    # TODO: self.i_track = number of current frames
    def save_track(self, disp_bgr, disp_bw, prev_bgr, save_prev_bgr, save_disp=False):

        # TODO: I commented out old file saving code and applied new one which has simpler folder directory:
        self.logger.info('saving image {}'.format(self.i_track))

        # name_extension = '_trunk_walnut'
        bgr_name = os.path.join(self.data_dir, 'landmark_tracking_results',
                                self.bgr_fmt.format(self.i_track))
        cv2.imwrite(bgr_name, disp_bgr)
        if save_prev_bgr:
            prev_bgr_name = os.path.join(self.data_dir, 'landmark_tracking_results',
                                    self.bgr_fmt.format(self.i_track-1))
            cv2.imwrite(prev_bgr_name, prev_bgr)

        # RCNN:
        # cv2.imwrite(bw_name, disp_bw)
        # if save_disp:
        #     disp_name = os.path.join(self.data_dir,'tracking_results',
        #                              self.disp_fmt.format(self.i_track))
        #     h, w, _ = np.shape(disp_bgr)
        #     disp = np.ones((h, w * 2 + 50, 3), np.uint8) * 50
        #     disp[:, :w, :] = disp_bgr
        #     disp[:, w + 50:, :] = disp_bw
        #     # cv2.imwrite(disp_name, disp)

        self.i_track += 1
        if os.path.isfile(bgr_name) == 0:
            print(bgr_name+'is not saved! Create the folder according to its directory first!\n')




        # self.logger.info('saving image {}'.format(self.i_track))
        #
        # name_extension = '_trunk_walnut'
        # bgr_name = os.path.join(self.track_dir+name_extension,
        #                         self.bgr_fmt.format(self.i_track))
        # bw_name = os.path.join(self.track_dir+name_extension,
        #                        self.bw_fmt.format(self.i_track))
        # cv2.imwrite(bgr_name, disp_bgr)
        # cv2.imwrite(bw_name, disp_bw)
        #
        # if save_disp:
        #     disp_name = os.path.join(self.track_dir+name_extension,
        #                              self.disp_fmt.format(self.i_track))
        #     h, w, _ = np.shape(disp_bgr)
        #     disp = np.ones((h, w * 2 + 50, 3), np.uint8) * 50
        #     disp[:, :w, :] = disp_bgr
        #     disp[:, w + 50:, :] = disp_bw
        #     cv2.imwrite(disp_name, disp)
        #
# self.i_track += 1
