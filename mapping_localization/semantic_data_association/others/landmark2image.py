
from __future__ import (print_function, division, absolute_import)

import logging
from itertools import izip
import pickle
import cv2
import numpy as np
import math

from scpye.improc.image_processing import enhance_contrast
from scpye.track.assignment import hungarian_assignment
from scpye.track.bounding_box import (bboxes_assignment_cost, extract_bbox)
from scpye.track.fruit_track import FruitTrack
from scpye.track.optical_flow import (calc_optical_flow, calc_average_flow)
from scpye.utils.drawing import (Colors, draw_bboxes, draw_optical_flows,
                                 draw_text, draw_bboxes_matches, draw_line, draw_bbox_centers)
from scpye.track.bounding_box import (bbox_center, shift_bbox)


# TODO: all inputs for the tracking part: images, bbox(contains a list of [x, y, w, h]),
# TODO: adjustable parameters: state_cov(P0, state covariance matrix) and proc_cov(Q, transition cov matrix representing how confident we are in our state trans matrix F, refer to kalman_filter.py)
# TODO: adjustable parameters: min_age=3(how many tracked times are regarded as valid tracks), win_size(window_size of fruits), max_level (levels of pyramid), init_flow (40,0) (initial flow velocity(automatically updated using avarage_flow));
# TODO: adjustable parameters: unassigned_cost=1.2 in assignment.py file, this controls threshold of hungarian assignment (if cost > un.._cost, this assignment will be classified to unassgnment)
# TODO: adjustable parameters: : margin pixels = COUNT_CUTOUT_PADDING, should be the same as Steven's code, control the number of margin pixels when extracting bounding boxes
# TODO: adjustable parameters: flow_cov(2,2) and bbox_cov (1,1): model noise of optical flow and bbox detection (if both are 0, no updates by KF, directly use the result of optical flow and detection)
# TODO: update:understanding of the visulization: the bboxes are final (optical flow and kalman filter) bbox positions, the lines of tracking are only optical flow traking process (not including the final update)
# TODO: understanding of kalman filter's usage: both in optical flow updates and in detection updates, and the init_flow is the same, which is mean_optical_flow, flow_cov and bbox_cov: model noise of optical flow and bbox detection
# TODO: intuition of adding KF: the general moving direction and distance should be similar for all fruits
# TODO: further understand flow_cov and bbox_cov: the row vector is diagnoalized in kalman_filter.py file, the two elements represent x and y noises (which are independent here) of the optical flow or detection model.
# TODO: important!!! the fruit is counted only if it is tracked more than 2 times (appeared in 3 frames), because the age is initialized to 1 instead of 0
# TODO: Update: Optical_flow tracking is based on grayscale images!
 # TODO: win_size is: optical flow window size = win_size*win_size (our apples diameter: 20-40 pixels). max_level is: (optical flow maximum level of pyramid - 1). can change the win_size, max_level to change the performace. refer link: https://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html
 #TODO:In countour analysis, I changed min_size from 4 to 100, for trunk tracking

 # TODO:  min_movement_threshold: only fruits with flow >  min_movement_threshold will be thresholded by flow direction, set a large value means disable it

class FruitTracker(object):
    COUNT_CUTOUT_PADDING = 1
    # COUNT_STD_SIZE = (128,128)

    # discarded********************************************************************************
    # this is for tracking
    # def __init__(self, min_age=12, win_size=30, max_level=3, init_flow=(0, 0),
    #              state_cov=(5, 5, 5, 5), proc_cov=(8, 4, 4, 2),
    #              flow_cov=(2, 2), bbox_cov=(0, 0), margin_pixels = COUNT_CUTOUT_PADDING):
    # this is only for direct-matching
    # def __init__(self, min_age=5, win_size=31, max_level=3, init_flow=(-20, 0),
    #              state_cov=(5, 5, 5, 5), proc_cov=(8, 4, 4, 2),
    #              flow_cov=(1e-10, 1e-10), bbox_cov=(100, 100), margin_pixels = COUNT_CUTOUT_PADDING):
    # discarded ends********************************************************************************

    # TODO: ORANGE
    # def __init__(self, min_age=4, win_size=31, max_level=3, init_flow=(-10, 0), init_flow_dir = (-1, 0), min_movement_threshold = 10,
    #              state_cov=(6, 2, 6, 2), proc_cov=(6, 2, 6, 2),
    #              flow_cov=(3, 1), bbox_cov=(1, 0.5), vel_cov=(2, 0.7), margin_pixels = COUNT_CUTOUT_PADDING):
    # TODO:
    def __init__(self, min_age=4, win_size= np.array([30, 45]), max_level=7, init_flow=(0, 0), init_flow_dir = (1, 0), min_movement_threshold = 10000,
                 state_cov=(3, 1, 3, 1), proc_cov=(9, 3, 9, 3),
                 flow_cov=(0.3, 0.1), bbox_cov=(3, 1), vel_cov=(1, 0.3), margin_pixels = COUNT_CUTOUT_PADDING):
    # def __init__(self, min_age=5, win_size=31, max_level=3, init_flow=(-10, 0),
    #      state_cov=(5, 5, 5, 5), proc_cov=(8, 4, 4, 2),
    #      flow_cov=(2, 1), bbox_cov=(0.2, 0.1), vel_cov=(0.2, 0.1), margin_pixels = COUNT_CUTOUT_PADDING):
        """
        :param min_age: minimum age of a tracking to be considered for counting
        """
        self.margin_pixels = margin_pixels
        # Tracking
        self.tracks = []
        # TODO: age measures how many times the bbox is tracked
        self.min_age = min_age
        self.total_counts = 0
        self.frame_counts = 0

        self.prev_gray = None
        # Optical flow parameters
        self.win_size = win_size
        self.max_level = max_level

        # Kalman filter parameters
        self.state_cov = np.array(state_cov)

        self.init_flow = np.array(init_flow)
        self.init_flow_avg = np.array(init_flow)
        self.init_flow_dir = np.array(init_flow_dir)
        self.min_movement_threshold = min_movement_threshold


        self.proc_cov = np.array(proc_cov)
        self.flow_cov = np.array(flow_cov)
        self.bbox_cov = np.array(bbox_cov)
        self.vel_cov = np.array(vel_cov)

        self.logger = logging.getLogger(__name__)
        # Visualization
        self.vis = True
        self.disp_bgr = None
        self.disp_bw = None

        # TODO: new from comp
        self.prev_detection_exist = 1

    @property
    def initialized(self):
        return self.prev_gray is not None


    def add_new_tracks(self, tracks, fruits):
        """
        Add new fruits to tracks
        :param tracks: a list of tracks
        :param fruits: a list of fruits
        """
        for fruit in fruits:
            track = FruitTrack(fruit, self.init_flow, self.state_cov,
                               self.proc_cov)
            tracks.append(track)

    # TODO: track here contain all info about the fruit, age, bbox, pos, vel, etc. (instance of FruitTrack class which corresponds to fruit), tracks is a list contains all track
    # TODO: fruits here are a 2D array of 4 vertices of bboxes extracted in bolb_analyzer.py
    # TODO: add prev_bgr parameter to input previous frame (for counting)


    ############here is the main !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # RCNN
    def track(self, image, fruits, bw):#, prev_bgr):
        """
        Main tracking step
        :param image: greyscale image --> no, it is color image
        :param fruits: new fruits #TODO: fruits here are a 2D array of [x y w h] of bboxes extracted in bolb_analyzer.py
        :param bw: binary image
        """
        # Convert to greyscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('before',image)
        # cv2.waitKey(1000000)
        cv2.destroyAllWindows()
        self.disp_bgr = enhance_contrast(image)
        # cv2.imshow('after',self.disp_bgr)
        # cv2.waitKey(1000000)
        # cv2.destroyAllWindows()


        self.disp_bw = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)

        # VISUALIZATION: new detection
        # if self.vis:
        #     draw_bboxes(self.disp_bgr, fruits, color=Colors.blue, margin = 0)
        #     draw_bboxes(self.disp_bw, fruits, color=Colors.blue, margin = 0)


        # Initialization
        if not self.initialized:
            self.prev_gray = gray
            self.add_new_tracks(self.tracks, fruits)
            self.logger.info('Tracker initialized.')
            return

        np.set_printoptions(suppress=True)

        # for t_0 in self.tracks:
        #     print('cov before pred:',t_0.kf.P,'\nage:',t_0.age,'\n')
        # TODO: here is just the first step of kalman filter: state prediction based on prior
        # TODO: new from comp
        # self.predict_tracks()
        if self.tracks != []:
            self.predict_tracks()
        else:
            print('self.tracks == [] in fruit_tracker.py')
        self.logger.debug("Predicted tracks: {}".format(len(self.tracks)))

        # TODO: removed on 8.13.2018
        # VISUALIZATION: after prediction


        # TODO: NEWNEW: moved hungarian assign to here, which is before update
        # TODO: KF update pos&vel using FCN detection
        # TODO:match_tracks outputs the match detection results and predictions of optical flow + KF
        #TODO: here the matched_tracks are correted bboxes (detection and optical-flow-KF-prediction combined)
        if self.tracks != []:
            prev_pts = [t.prev_pos for t in self.tracks]
            #init_pts = [t.pos + self.init_flow for t in self.tracks]
            init_pts = [t.pos for t in self.tracks]

            # TODO: here the status is output status vector (of unsigned chars): each element of the vector is set to 1 if the flow for the corresponding features has been found, otherwise, 0.
            # TODO: can change the window_size, max_level (levels of pyramid) to change the performace
            curr_pts, status = calc_optical_flow(self.prev_gray, gray,
                                                 prev_pts, init_pts,
                                                 self.win_size,
                                                 self.max_level)
        else:
            curr_pts = np.array([])
            prev_pts = []
            status = []
        # # VISUALIZATION: optical flow
        # # TODO: draw the optical flow tracking lines
        draw_two_frame_only = 0
        if draw_two_frame_only and self.tracks != []:
            #****************************************************************************************************************draw_raw_optical_flow
            draw_optical_flows(self.disp_bgr, prev_pts, curr_pts, status,
                                radius=4, color=Colors.magenta)
            draw_optical_flows(self.disp_bw, prev_pts, curr_pts, status,
                                radius=4, color=Colors.magenta)

        if self.tracks != []:
            # Update init flow
            self.init_flow = calc_average_flow(prev_pts, curr_pts, status)
            self.init_flow_avg = 0.7*self.init_flow + 0.3*self.init_flow_avg
            norm_avg_flow = np.linalg.norm(self.init_flow)
            dir_avg_flow = self.init_flow / norm_avg_flow
            if norm_avg_flow > self.min_movement_threshold:
                self.init_flow_dir = 0.7*self.init_flow_dir + 0.3*dir_avg_flow

            self.init_flow_dir = self.init_flow_dir / np.linalg.norm(self.init_flow_dir)
            self.logger.debug("init flow: {}".format(self.init_flow))

            KLT_bboxes = []
            KLT_tracks = []
            unmatched_tracks_prior = []
            estimated_KLT_points = []
            if self.prev_detection_exist == 1:
                if fruits.shape[0] != 0:
                    if len(curr_pts.shape) == 1:
                        track = self.tracks[0]
                        point = curr_pts
                        stat = status
                        if stat:
                            new_bboxes = shift_bbox(track.bbox, point)
                            KLT_bboxes.append(new_bboxes)
                            KLT_tracks.append(track)
                            estimated_KLT_points.append(point)
                        else:
                            unmatched_tracks_prior.append(track)

                    else:
                        for track, point, stat in izip(self.tracks, curr_pts, status):
                            if stat:
                                new_bboxes = shift_bbox(track.bbox, point)
                                KLT_bboxes.append(new_bboxes)
                                KLT_tracks.append(track)
                                estimated_KLT_points.append(point)
                            else:
                                unmatched_tracks_prior.append(track)
        else:
            matched_tracks = []
            fruits_list = []
            new_fruits =  fruits
            KLT_bboxes = []
            KLT_tracks = []
            estimated_KLT_points = []
            unmatched_tracks_prior = []
            unmatched_tracks = []

        if fruits.shape[0] == 0:
            matched_tracks = []
            new_fruits = []
            unmatched_tracks = self.tracks
            fruits_list = []
            print ('fruits.shape[0] in fruit_tracker.py')

        elif self.tracks != []:
            matched_tracks, fruits_list, matched_KLT_points, new_fruits, unmatched_tracks = self.match_tracks(
                KLT_bboxes, KLT_tracks,
                fruits, estimated_KLT_points)

        unmatched_tracks.extend(unmatched_tracks_prior)
        self.logger.debug("matched/new/unmatched: {0}/{1}/{2}".format(
            len(matched_tracks), len(new_fruits), len(unmatched_tracks)
        ))

        # TODO: removed on 8.13.2018

        self.tracks = matched_tracks


        # TODO: KF update pos using optical flow
        # TODO: in this function, both optical flow and kalman filter (for position correction) are used to generate two returns: updated track and lost track

        # TODO: new from comp: if previous frame has no track, then here should initialize again (add new tracks instead of update tracks)
        if self.tracks != [] and self.prev_detection_exist == 1:
            updated_tracks, lost_tracks = self.update_tracks(gray,  fruits_list, matched_KLT_points)
            self.prev_detection_exist = 1
        elif self.prev_detection_exist == 0:
            self.prev_gray = gray
            self.add_new_tracks(self.tracks, fruits)
            updated_tracks = []
            lost_tracks = []
            self.prev_detection_exist = 1
            print ('prev_detection_exist == 0 in fruit_tracker.py')
        elif self.tracks == []:
            updated_tracks = []
            lost_tracks = []
            self.prev_detection_exist = 0
            print ('self.tracks == [] in fruit_tracker.py')

        # updated_tracks, lost_tracks = self.update_tracks(gray,  fruits_list, matched_KLT_points)
        self.logger.debug("update/lost: {0}/{1}".format(len(updated_tracks),
                                                        len(lost_tracks)))

        # TODO: removed on 8.13.2018
        # VISUALIZATION: after optical flow update
        # VISUALIZATION: after hungarian assignment update


        # Assemble all lost tracks and update tracks
        # TODO: NEW: because moving the match tracks upwards, the line below is also moved updwards.
        # self.tracks = matched_tracks
        self.add_new_tracks(self.tracks, new_fruits)
        lost_tracks.extend(unmatched_tracks)
        self.logger.debug(
            "tracks/lost: {0}/{1}".format(len(self.tracks), len(lost_tracks)))

        # VISUALIZATION:

        if self.vis:
            # diff between counted and tracked boxes: >min_age(3) or not
            #RCNN edited:
            # counted_bboxes = [t.bbox for t in self.tracks if
            #                   t.age >= self.min_age]
            # tracked_bboxes = [t.bbox for t in self.tracks if
            #                   1 < t.age < self.min_age]
            # new_bboxes = [t.bbox for t in self.tracks if t.age == 1]


            counted_bboxes = []
            tracked_bboxes = []
            new_bboxes = []
            tracked_and_new_fruits = fruits_list
            for new_fruit in new_fruits:
                tracked_and_new_fruits.append(new_fruit)
            for cur_fruit, cur_track in izip(tracked_and_new_fruits,self.tracks):
                if cur_track.age >= self.min_age:
                    counted_bboxes.append(cur_fruit)
                elif cur_track.age == 1:
                    new_bboxes.append(cur_fruit)
                else:
                    tracked_bboxes.append(cur_fruit)

            fruits_list = []

            #*****************************************************************************************************************draw_bboxes
            # bbox_center_radius = 2
            # if len(counted_bboxes):
            #     draw_bbox_centers(self.disp_bgr, counted_bboxes, color=Colors.green,
            #                 thickness=2, r = bbox_center_radius)
            #     draw_bbox_centers(self.disp_bw, counted_bboxes, color=Colors.green,
            #                 thickness=2, r = bbox_center_radius)
            # if len(tracked_bboxes):
            #     draw_bbox_centers(self.disp_bgr, tracked_bboxes, color=Colors.yellow,
            #                 thickness=2, r = bbox_center_radius)
            #     draw_bbox_centers(self.disp_bw, tracked_bboxes, color=Colors.yellow,
            #                 thickness=2,r = bbox_center_radius)
            # if len(new_bboxes):
            #     draw_bbox_centers(self.disp_bgr, new_bboxes, color=Colors.red,
            #                 thickness=2, r = bbox_center_radius)
            #     draw_bbox_centers(self.disp_bw, new_bboxes, color=Colors.red,
            #                 thickness=2, r = bbox_center_radius)
            bbox_margin = 0
            if len(counted_bboxes):
                draw_bboxes(self.disp_bgr, counted_bboxes, color=Colors.green,
                            thickness=2, margin = bbox_margin)
                draw_bboxes(self.disp_bw, counted_bboxes, color=Colors.green,
                            thickness=2, margin = bbox_margin)
            if len(tracked_bboxes):
                draw_bboxes(self.disp_bgr, tracked_bboxes, color=Colors.yellow,
                            thickness=2, margin = bbox_margin)
                draw_bboxes(self.disp_bw, tracked_bboxes, color=Colors.yellow,
                            thickness=2, margin = bbox_margin)
            if len(new_bboxes):
                draw_bboxes(self.disp_bgr, new_bboxes, color=Colors.red,
                            thickness=1, margin = bbox_margin)
                draw_bboxes(self.disp_bw, new_bboxes, color=Colors.red,
                            thickness=1, margin = bbox_margin)

        # TODO: maybe time consuming, plesae edit the dash_gap to 1000 if want optimal speed
        # TODO: here is visualizing the optical flow tracking in multiple concurrent frames
        if self.vis:
            if not draw_two_frame_only:
                for track in self.tracks:
                    color_value = 255
                    draw_line(self.disp_bgr, track.hist, color=[color_value,color_value,color_value], thickness = 0, dash_gap=15)
                    draw_line(self.disp_bw, track.hist, color=[color_value,color_value,color_value], thickness = 0, dash_gap=15)
                    draw_line(self.disp_bw, track.hist, color=[color_value,color_value,color_value], thickness = 0, dash_gap=15)

                # TODO: count the lost tracks (is valid track in previous frame but not shown in the following frames) which used to be counted tracks (min_age>=3) instead of current counted tracks!!!!!!!
        # Count fruits in lost tracks
        # TODO: add prev_bgr parameter to input previous frame(for counting)
        # TODO: record counted_fruits_and_last_frame for localization
        # RCNN
        counted_fruits_and_last_frame = self.count_in_tracks(lost_tracks)#, prev_bgr)
        self.logger.info(
            "Frame/Total counts: {0}/{1}".format(self.frame_counts,
                                                 self.total_counts))


        if self.vis:
            h, w = np.shape(bw)
            draw_text(self.disp_bgr, self.total_counts, (10, h - 10), scale=1.5,
                      color=Colors.cyan)
            draw_text(self.disp_bw, self.total_counts, (10, h - 10), scale=1.5,
                      color=Colors.cyan)

        # TODO: after RCNN, update to recording lost_tracks (be careful: lost tracks only have information before (k-1)th frame if lost in kth frame) instead of all tracks for direct matching
        # TODO: record counted_fruits_and_last_frame for localization
        return counted_fruits_and_last_frame, lost_tracks







    # TODO: here is just the first step of kalman filter: state prediction based on prior
    def predict_tracks(self):
        """
        Predict tracks in Kalman filter
        """
        for track in self.tracks:
            track.predict()


    #TODO: in this function, both optical flow and kalman filter (for position correction) are used to generate two returns: updated track and lost track
    def update_tracks(self, gray, fruits_list, matched_KLT_points):
        """
        Update tracks' position in Kalman filter via KLT
        :param gray: greyscale image
        :return: updated tracks, lost_tracks
        """
        # Get points in previous image and points in current image
        # TODO: tracks are lists containing all info such as bbox, age, pos, etc.
        # TODO: prev_pts and init_pts here are center pixels in the fruit contour
        prev_pts = [t.prev_pos for t in self.tracks]

        # TODO: FRCNN: edit init_pts to use the previous average flow
        init_pts = [t.pos for t in self.tracks]

        # TODO: here the status is output status vector (of unsigned chars): each element of the vector is set to 1 if the flow for the corresponding features has been found, otherwise, 0.
        # TODO: can change the window_size, max_level (levels of pyramid) to change the performace
        # TODO: NEWNEW: can consider integrating this optical flow calculation process with previous one !!!
        # curr_pts, status = calc_optical_flow(self.prev_gray, gray,
        #                                      prev_pts, init_pts,
        #                                      self.win_size,
        #                                      self.max_level)
        curr_pts = matched_KLT_points

        # # Update init flow
        # self.init_flow = calc_average_flow(prev_pts, curr_pts, status)
        # self.logger.debug("init flow: {}".format(self.init_flow))

        if len(curr_pts) == 1:
            print('len(curr_pts.shape) == 1 in fruit_tracker.py \n')
            print('NOT SURE THE CHANGE IS CORRECT, CHECK WHETHER ERROR OCCURS! \n')
            track=self.tracks[0]
            KLT_point=curr_pts[0]
            prev_point = prev_pts[0]
            FCN_bbox = fruits_list[0]
            vel_dir = self.init_flow / np.linalg.norm(self.init_flow)
            vel_mag = np.linalg.norm(KLT_point - prev_point)
            vel = vel_mag * vel_dir
            track.para_update_bbox(KLT_point, FCN_bbox, vel, self.flow_cov, self.bbox_cov, self.vel_cov)

        else:
            for track, KLT_point, prev_point, FCN_bbox in izip(self.tracks, curr_pts, prev_pts, fruits_list):
                vel_dir = self.init_flow / np.linalg.norm(self.init_flow)
                vel_mag = np.linalg.norm(KLT_point - prev_point)
                np.linalg.norm(self.init_flow)
                vel = vel_mag * vel_dir
                track.para_update_bbox(KLT_point, FCN_bbox, vel, self.flow_cov, self.bbox_cov, self.vel_cov)


        #TODO: NEWNEW: all following lines compreseed to this parallel update!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # for track, KLT_point, prev_point, FCN_bbox in izip(self.tracks, curr_pts, prev_pts, fruits_list):
        #     vel_dir = self.init_flow / np.linalg.norm(self.init_flow)
        #     vel_mag = np.linalg.norm(KLT_point - prev_point)
        #     vel = vel_mag * vel_dir
        #     track.para_update_bbox(KLT_point, FCN_bbox, vel, self.flow_cov, self.bbox_cov, self.vel_cov)




        # # TODO: moving correct_bbox from match_tracks to here
        # for track, fruit in izip(self.tracks, fruits_list):
        #     track.correct_bbox(fruit, self.bbox_cov)
        # # TODO: NEWNEW update velocity
        # for track, point, prev_point, stat in izip(self.tracks, curr_pts, prev_pts, status):
        #     if stat:
        #         vel_dir = self.init_flow / np.linalg.norm(self.init_flow)
        #         vel_mag = np.linalg.norm(point - prev_point)
        #         vel = vel_mag * vel_dir
        #         track.kf.update_vel(vel, self.vel_cov)
        #


        # # TODO: add tracked fruits to tracks (after updating using kalman filter), remove lost ones.
        # # Remove lost tracks
        # updated_tracks, lost_tracks = [], []
        # for track, point, stat in izip(self.tracks, curr_pts, status):
        #     if stat:
        #         #TODO: here use kalman filter to update the position!!!
        #         print('flow_update')
        #         track.correct_flow(point, self.flow_cov)
        #         updated_tracks.append(track)
        #     else:
        #         lost_tracks.append(track)

        # TODO: NEWNEW
        updated_tracks = self.tracks
        lost_tracks = []

        self.prev_gray = gray

        return updated_tracks, lost_tracks


    # TODO: use the hungarian_assignment to match_tracks outputs the match detection results and predictions of optical flow + KF
    # TODO: fruits here are a 2D array of 4 vertices of bboxes extracted in bolb_analyzer.py
    # TODO: track here is the instance of FruitTrack class which corresponds to fruit, tracks is a list contains all track




    def match_tracks(self, bboxes, tracks, fruits, estimated_KLT_points):
        """
        Match tracks to new detection
        :param tracks:
        :param fruits:#TODO: fruits here are a 2D array of [x y w h] of bboxes extracted in bolb_analyzer.py
        :return: matched_tracks, new_fruits, unmatched_tracks
        """
        # TODO: t.box should be 2d array; fruits should be 3d array
        # bboxes_update = np.array([t.bbox for t in tracks])
        # TODO: NEW Becasue I change inputing  tracks to inputing bboxes, here I changed correspondingly
        bboxes_update = np.array([bbox for bbox in bboxes])
        bboxes_detect = np.array(fruits)


        # TODO: new from comp
        if tracks == []:
            # self.add_new_tracks(self.tracks, fruits)
            cost = np.array([[1000]])
            match_inds = np.array([])
            lost_inds = np.array([])
            new_inds = np.arange((fruits).shape[0])
            print('tracks ==[] in fruit_tracker.py')
        else:
            cost = bboxes_assignment_cost(bboxes_update, bboxes_detect)
            match_inds, lost_inds, new_inds = hungarian_assignment(cost)

        # cost = bboxes_assignment_cost(bboxes_update, bboxes_detect)
       # match_inds, lost_inds, new_inds = hungarian_assignment(cost)



        # VISUALIZATION: hungarian assignment
        # draw_bboxes_matches(self.disp_bgr, match_inds, bboxes_update,
        #                     bboxes_detect, color=Colors.cyan)
        # draw_bboxes(self.disp_bgr, bboxes_detect, color=Colors.cyan,
        #             thickness=1)
        # draw_bboxes(self.disp_bw, bboxes_detect, color=Colors.cyan,
        #             thickness=1)

        # get matched tracks
        matched_tracks = []
        fruits_list = []
        matched_KLT_points = []
        #TODO: here the bboxes are correted from the original detections to detection-prediction combined result

        # extract new tracks
        new_fruits = fruits[new_inds]
        # get unmatched tracks
        unmatched_tracks = [tracks[ind] for ind in lost_inds]

        for match in match_inds:
            i_track, i_fruit = match
            track = tracks[i_track]
            fruit = fruits[i_fruit]
            # TODO: NEWNEW: move this into update tracks
            # track.correct_bbox(fruits[i_fruit], self.bbox_cov)
            movement = bbox_center(fruit) - track.prev_pos
            norm_movement = np.linalg.norm(movement)
            if norm_movement < self.min_movement_threshold:
                angle = 0
            else:
                dir_movement = (movement / norm_movement)
                angle = math.acos(np.dot(self.init_flow_dir, dir_movement))
            if angle <= np.pi/2:
                fruits_list.append(fruit)
                matched_tracks.append(track)
                matched_KLT_points.append(estimated_KLT_points[i_track])
            else:
                new_fruits = np.append(new_fruits,fruit[np.newaxis,:],0)
                unmatched_tracks.append(track)





        return matched_tracks, fruits_list, matched_KLT_points, new_fruits, unmatched_tracks


    # TODO: count the lost tracks which used to be counted tracks (min_age>=3) instead of current counted tracks!!!!!!!
    # TODO: add prev_bgr parameter to input previous frame(for counting)

    # RCNN
    def count_in_tracks(self, tracks):#, prev_bgr):
        """
        Count how many fruits there are in tracks
        :param tracks: list of tracks
        """
        # RCNN deleted some code here

        self.frame_counts = sum([1 for t in tracks if t.age >= self.min_age])
        print (self.frame_counts)
        # cv2.imshow('bbox_img_temp',(225,225,225))
        # cv2.waitKey(100000)
        # cv2.destroyAllWindows()

        # print('frame_counts:', self.frame_counts, '\n\n')
        self.total_counts += self.frame_counts

        # RCNN
        # #TODO: return this for Localiztion!
        # return counted_fruits_and_last_frame

    # RCNN
    def finish(self):#, prev_bgr):
        # TODO: because this is the last frame, all tracks are counted instead of only lost tracks as we did for previous frames!!!!!!!!
        # TODO: record counted_fruits_and_last_frame for localization
        counted_fruits_and_last_frame = self.count_in_tracks(self.tracks)#, prev_bgr)
        self.logger.info("Total counts: {}".format(self.total_counts))
        print('total counts are:', self.total_counts, '\n')
        #TODO: ############################################################################### remember to delete!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # with open('tracks.pkl', 'wb') as output_us:
        #     pickle.dump(self.tracks, output_us)

        print(1)
        # TODO: record counted_fruits_and_last_frame for localization
        return counted_fruits_and_last_frame



