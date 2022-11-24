from __future__ import (print_function, division, absolute_import)

import logging
import pickle
import cv2
import numpy as np
import math


from mapping_localization.semantic_data_association.project_landmarks import project_landmarks_add_unseen_observation

from scpye.improc.image_processing import enhance_contrast
from scpye.track.assignment import hungarian_assignment
from scpye.track.bounding_box import (bboxes_assignment_cost,landmark_based_bboxes_assignment_cost, extract_bbox)
from mapping_localization.semantic_data_association.landmark_fruit_track import LandmarkFruitTrack
from mapping_localization.semantic_data_association.extract_landmark_3D_pos_bboxes_hist import  extract_landmark_3D_pos_bboxes_hist
from scpye.track.optical_flow import (calc_optical_flow, calc_average_flow, calc_optical_flow_for_refinement)
from scpye.utils.drawing import (Colors, draw_bboxes, draw_optical_flows,
                                 draw_text, draw_bboxes_matches, draw_line, draw_bbox_centers)
from scpye.track.bounding_box import (bbox_center, shift_bbox)
import copy


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

class LandmarkFruitTracker(object):
    COUNT_CUTOUT_PADDING = 1
    # TODO:
    def __init__(self, min_age=3, win_size= np.array([15, 20]), max_level=2, init_flow=(0, 0), init_flow_dir = (1, 0), min_movement_threshold = 10000,
                 state_cov=(3, 1, 3, 1), proc_cov=(9, 3, 9, 3),
                 flow_cov=(0.3, 0.1), bbox_cov=(3, 1), vel_cov=(1, 0.3), margin_pixels = COUNT_CUTOUT_PADDING):

        """
        :param min_age: minimum age of a tracking to be considered for counting
        """
        self.margin_pixels = margin_pixels
        # Tracking
        self.tracks = []
        self.lost_tracks_prev_fr = []
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
        self.disp_prev_bgr = None
        self.save_prev_bgr = 0
        self.disp_bw = None


        # TODO: new from comp
        self.prev_detection_exist = 1

        # TODO: 3D landmark based tracking dictionary key: tuple (3D landmark position), value is another dictionary as: {'track':the last track of this landmark, 'frame number': 1 + the last frame that observes this landmark}
        self.landmark_pos_tracks_dict = {}

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
            track = LandmarkFruitTrack(fruit, self.init_flow, self.state_cov,
                               self.proc_cov)
            tracks.append(track)

    def add_unobserved_track(self, tracks,fruit, hist_pos_bboxes_list, hist_observed_all):
        track = LandmarkFruitTrack(fruit, self.init_flow, self.state_cov,self.proc_cov)
        track.hist = [hist_pos_bbox[0] for hist_pos_bbox in hist_pos_bboxes_list]
        track.hist_unobserved = [hist_observed for hist_observed in hist_observed_all]
        track.KLT_hist = [hist_pos_bbox[0] for hist_pos_bbox in hist_pos_bboxes_list]
        track.KLT_status = [False for _ in hist_pos_bboxes_list]
        track.age = len(track.hist)
        tracks.append(track)
        return track

    # TODO: track here contain all info about the fruit, age, bbox, pos, vel, etc. (instance of FruitTrack class which corresponds to fruit), tracks is a list contains all track
    # TODO: fruits here are a 2D array of 4 vertices of bboxes extracted in bolb_analyzer.py
    # TODO: add prev_bgr parameter to input previous frame (for counting)


    #TODO: *********************************************************************************************************************main function starts from here*********************************************************************************************************************
    # RCNN
    def track(self, image, fruits, bw, prev_bgr, current_frame_idx,fruit_3D_pos_bboxes_hist_dict, rot_trans_dict, K_mat, data_dir_pre):

        """
        Main tracking step
        :param image: greyscale image --> no, it is color image
        :param fruits: new fruits #TODO: fruits here are a 2D array of [x y w h] of bboxes extracted in bolb_analyzer.py
        :param bw: binary image
        """
        # Convert to greyscale
        global new_fruits, lost_tracks
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.destroyAllWindows()
        if self.disp_bgr is not None:
            self.disp_prev_bgr = self.disp_bgr
        self.save_prev_bgr = 0

        self.disp_bgr = enhance_contrast(image)
        self.disp_bw = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)

        # Initialization
        if not self.initialized:
            self.prev_gray = gray
            self.add_new_tracks(self.tracks, fruits)
            print('Initializing tracker......')
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

        if self.tracks != []:
            prev_pts = [t.prev_pos for t in self.tracks]
            #init_pts = [t.pos + self.init_flow for t in self.tracks]
            init_pts = [t.pos for t in self.tracks]


        #TODO:****************************************************Projecting landmarks to current frame starts****************************************************
        # only landmarks whose oldest observation's frame index is in the range of (current frame + landmark_horizon[0]) : (current frame + landmark_horizon[1]) will be considered --> if fruit_3D_pos_bboxes_hist_dict['frame_idx'][0] <= current_frame_idx + landmark_horizon[1] and fruit_3D_pos_bboxes_hist_dict['frame_idx'][0]  >= current_frame_idx + landmark_horizon[0]:
        # img_w_h is in the form of [img_width(cols), img_height(rows)]
        # projected_fruits_bboxes_with_hist_3D_pos_dict: projected_fruits_bboxes_with_hist_3D_pos_dict[three_dim_pt_idx] = {'3Dpos':value['3Dpos'], 'proj_bbox':bbox_shifted_to_proj_center, 'frame_idx':value['frame_idx'], '2Dpos_bbox':value['2Dpos_bbox'], 'unobserved':[True means the fruit bbox is not observed but predicted by projecting the landmark back]}, where value is from fruit_3D_pos_bboxes_hist_dict,
        # specifically  '3Dpos':fruit_3D_loc, 'frame_idx': [frame_ind] (order: small to large), '2Dpos_bbox':fruit}, which contains fruits' 3D position, frame index where it appears (a list of integers) and corresponding frame's 2D_position_and_bbox (a list where each element is another list: [array(x_center, y_center),array(bbox)])
        # Note: fruit 2D position is based on the Kalman Filter output, not the FRCNN bbox center!!!
        projected_fruits_bboxes_with_hist_3D_pos_dict = project_landmarks_add_unseen_observation(current_frame_idx= current_frame_idx, landmark_horizon=[-30,-1], img_w_h = [1236, 1648], fruit_3D_pos_bboxes_hist_dict = fruit_3D_pos_bboxes_hist_dict, rot_trans_dict= rot_trans_dict, K_mat = K_mat, data_dir_pre = data_dir_pre)
        cur_landmarks_3D_pos = []
        cur_landmarks_bboxes_shifted_to_proj_center = []
        # every element in the following two list contains the full history of the point
        cur_landmarks_fr_idxes = []
        cur_landmarks_every_fr_pos_bboxes = []
        cur_landmarks_every_fr_unobserved = []
        for three_dim_pt_idx, value in projected_fruits_bboxes_with_hist_3D_pos_dict.items():
            # cur_landmarks_fr_idxes is small --> large (cur_landmarks_fr_idxes[-1] is the most recent frame index!)
            fr_idxes_list = []
            every_fr_pos_bboxes_list = []
            every_fr_unobserved_list = []
            for frame_idx,pos_bbox,unobserved in zip(value['frame_idx'], value['2Dpos_bbox'], value['unobserved']):
                    if frame_idx < current_frame_idx:
                            fr_idxes_list.append(copy.deepcopy(frame_idx))
                            every_fr_pos_bboxes_list.append(copy.deepcopy(pos_bbox))
                            every_fr_unobserved_list.append(copy.deepcopy(unobserved))

            if fr_idxes_list[-1] != current_frame_idx - 1:
                raise Exception('The history of some landmarks are not extracted in the correct way, the most recent frame index (shoud = cur_fr_idx - 1) is ',fr_idxes_list[-1],' current frame index is',current_frame_idx,' check the cur_landmarks_fr_indxes!')
            cur_landmarks_bboxes_shifted_to_proj_center.append(copy.deepcopy(value['proj_bbox']))
            cur_landmarks_fr_idxes.append(copy.deepcopy(fr_idxes_list))
            cur_landmarks_every_fr_pos_bboxes.append(copy.deepcopy(every_fr_pos_bboxes_list))
            cur_landmarks_every_fr_unobserved.append(copy.deepcopy(every_fr_unobserved_list))
            cur_landmarks_3D_pos.append(copy.deepcopy(value['3Dpos']))


    #TODO:****************************************************Projecting landmarks to current frame ends****************************************************

        # TODO: here the optical flow process calc_optical_flow is removed!!!

        if fruits.shape[0] == 0:
            matched_tracks = []
            new_fruits = []
            unmatched_tracks = self.tracks
            matched_fruits_list = []
            disappeared_tracks = self.tracks
            print ('fruits.shape[0] in fruit_tracker.py')

        elif self.tracks != []:
            # TODO: remember to deal with counting problem: if the fruit has already lost once, we should never re-count it!
            matched_tracks, matched_fruits_list, unobserved_list, new_fruits, unmatched_tracks, disappeared_tracks = self.landmark_match_tracks( tracks = self.tracks, fruits = fruits, cur_landmarks_bboxes_shifted_to_proj_center = cur_landmarks_bboxes_shifted_to_proj_center, cur_landmarks_every_fr_pos_bboxes = cur_landmarks_every_fr_pos_bboxes, cur_landmarks_every_fr_unobserved = cur_landmarks_every_fr_unobserved, cur_landmarks_every_fr_idxes= cur_landmarks_fr_idxes, current_frame_index = current_frame_idx, cur_landmarks_3D_pos = cur_landmarks_3D_pos)

        #unmatched_tracks.extend(unmatched_tracks_prior)

        # TODO: removed on 8.13.2018

        self.tracks = matched_tracks


        # TODO: KF update pos using optical flow
        # TODO: in this function, both optical flow and kalman filter (for position correction) are used to generate two returns: updated track and lost track

        # TODO: new from comp: if previous frame has no track, then here should initialize again (add new tracks instead of update tracks)


        draw_two_frame_only = 0
        if self.tracks != [] and self.prev_detection_exist == 1:
            # TODO: TO IMPLEMENT FOR LANDMARK BASED UPDATE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # updated_tracks, lost_tracks = self.update_tracks(gray,  fruits_list, matched_KLT_points)
            updated_tracks, KLT_status = self.landmark_update_tracks(gray, matched_fruits_list, unobserved_list)

            #****************************************************************************************************************draw_raw_optical_flow
            # current_pts = [t.KLT_pos for t in self.tracks]
            # previous_pts = [t.prev_pos for t in self.tracks]
            # draw_optical_flows(self.disp_bgr, previous_pts, current_pts, status = KLT_status,
            #                     radius=4, color=Colors.magenta, draw_invalid=True)


            #****************************************************************************************************************draw_two_frame_tracking

            if draw_two_frame_only:
                current_pts = [t.pos for t in self.tracks]
                previous_pts = [t.prev_pos for t in self.tracks]
                draw_optical_flows(self.disp_bgr, previous_pts, current_pts, status = KLT_status,
                    radius=4, color=Colors.magenta, draw_invalid=True)

            lost_tracks = []
            self.prev_detection_exist = 1
        elif self.prev_detection_exist == 0:
            self.prev_gray = gray
            new_fruits = fruits
            # self.add_new_tracks(self.tracks, fruits)
            updated_tracks = []
            lost_tracks = []
            self.prev_detection_exist = 1
            print ('prev_detection_exist == 0 in fruit_tracker.py')
        elif self.tracks == []:
            updated_tracks = []
            lost_tracks = []
            self.prev_detection_exist = 0
            print ('self.tracks == [] in fruit_tracker.py')

        # TODO: removed on 8.13.2018

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
            counted_bboxes = []
            tracked_bboxes = []
            new_bboxes = []
            unobserved_tracked_bboxes = []

            # Initialize the tracked_and_new_fruits with matched_fruits_list and then add new fruits in the end of this list
            tracked_and_new_fruits = matched_fruits_list
            for new_fruit in new_fruits:
                tracked_and_new_fruits.append(copy.deepcopy(new_fruit))
            for cur_fruit, cur_track in zip(tracked_and_new_fruits,self.tracks):
                if cur_track.age >= self.min_age:
                    counted_bboxes.append(copy.deepcopy(cur_fruit))
                elif cur_track.age == 1:
                    new_bboxes.append(copy.deepcopy(cur_fruit))
                elif self.min_age>cur_track.age >1:
                    tracked_bboxes.append(copy.deepcopy(cur_fruit))


            # draw counted tracks in the previous frame
            # for cur_lost_track in lost_tracks:
            #     last_frame_track_hist = cur_lost_track.hist
            #     last_frame_bbox = shift_bbox(cur_lost_track.bbox, cur_lost_track.hist[-1])
            #     if self.disp_prev_bgr is not None:
            #         draw_bboxes(self.disp_prev_bgr, last_frame_bbox, color=Colors.cyan,
            #                     thickness=2, margin = 0)
            #         color_value = 255
            #         draw_line(self.disp_prev_bgr, last_frame_track_hist,color=Colors.cyan, thickness = 0, dash_gap=5)
            #         self.save_prev_bgr = 1


                # if len(cur_track.hist_unobserved) > 2:
                #     if cur_track.hist_unobserved[-2] == True:
                #         last_frame_track_hist = cur_track.hist[:-1]
                #         last_frame_bbox = shift_bbox(cur_track.bbox, cur_track.hist[-2])
                #         if self.disp_prev_bgr is not None:
                #             draw_bboxes(self.disp_prev_bgr, last_frame_bbox, color=Colors.cyan,
                #                         thickness=2, margin = 0)
                #             color_value = 255
                #             draw_line(self.disp_prev_bgr, last_frame_track_hist,color=Colors.cyan, thickness = 0, dash_gap=5)
                #             self.save_prev_bgr = 1


            matched_fruits_list = []

            #*****************************************************************************************************************draw_bboxes
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
                            thickness=2, margin = bbox_margin)
                draw_bboxes(self.disp_bw, new_bboxes, color=Colors.red,
                            thickness=2, margin = bbox_margin)


        # TODO: maybe time consuming, plesae edit the dash_gap to 0 if want optimal speed
        # TODO: here is visualizing the optical flow tracking in multiple concurrent frames
        if self.vis:
            if not draw_two_frame_only:
                for track in self.tracks:
                    color_value = 255
                    draw_line(self.disp_bgr, track.hist, color=[color_value,color_value,color_value], thickness = 0, dash_gap=15)
                    # draw_line(self.disp_bw, track.hist, color=[color_value,color_value,color_value], thickness = 0, dash_gap=0)
                    # draw_line(self.disp_bw, track.hist, color=[color_value,color_value,color_value], thickness = 0, dash_gap=0)

                # TODO: count the lost tracks (is valid track in previous frame but not shown in the following frames) which used to be counted tracks (min_age>=3) instead of current counted tracks!!!!!!!
        # Count fruits in lost tracks
        # TODO: add prev_bgr parameter to input previous frame(for counting)
        # RCNN
        self.count_in_tracks(lost_tracks, prev_bgr)

        self.lost_tracks_prev_fr = lost_tracks
        self.logger.info(
            "Frame/Total counts: {0}/{1}".format(self.frame_counts,
                                                 self.total_counts))

        if self.vis:
            h, w = np.shape(bw)
            # draw total count
            # draw_text(self.disp_bgr, self.total_counts, (10, h - 10), scale=1.5,
            #           color=Colors.cyan)
            # draw_text(self.disp_bw, self.total_counts, (10, h - 10), scale=1.5,
            #           color=Colors.cyan)

        # TODO: for landmark based, record current disappeard tracks (be careful: disappear tracks only have information before (k-1)th frame if lost in kth frame) instead of all tracks for direct matching
        return disappeared_tracks

#TODO: *********************************************************************************************************************main function ends here*********************************************************************************************************************




    def landmark_match_tracks(self, tracks, fruits, cur_landmarks_bboxes_shifted_to_proj_center,cur_landmarks_every_fr_pos_bboxes, cur_landmarks_every_fr_unobserved, cur_landmarks_every_fr_idxes, current_frame_index, cur_landmarks_3D_pos):
        """
        Match tracks to new detection
        :param tracks:
        :param fruits:#TODO: fruits here are a 2D array of [x y w h] of bboxes extracted in bolb_analyzer.py
        :return: matched_tracks, new_fruits, unmatched_tracks
        """

        bboxes_proj = np.array(cur_landmarks_bboxes_shifted_to_proj_center)
        # TODO: only record observed age
        bboxes_proj_ages = np.array([(len(cur_landmark_every_fr_unobserved) - sum(cur_landmark_every_fr_unobserved)) for cur_landmark_every_fr_unobserved in cur_landmarks_every_fr_unobserved])
        bboxes_detect = np.array(fruits)
        cost = landmark_based_bboxes_assignment_cost(bboxes_proj, bboxes_proj_ages, bboxes_detect, age_cost_ratio=0.4, observed_fr_thrshold=7, bbox_small_threshold = 15)
        match_inds, lost_inds, new_inds = hungarian_assignment(cost, unassigned_cost=1.2)
        # cost = landmark_based_bboxes_assignment_cost(bboxes_proj, bboxes_proj_ages, bboxes_detect, age_cost_ratio=0.01, observed_fr_thrshold=7, bbox_small_threshold = 10)
        # match_inds, lost_inds, new_inds = hungarian_assignment(cost, unassigned_cost=1.2)


        if len(new_inds) + len(match_inds) != len(fruits):
            print('check')

        # extract new tracks
        new_fruits = fruits[new_inds]
        # get unmatched tracks
        lost_landmarks_every_fr_unobserved = [cur_landmarks_every_fr_unobserved[ind] for ind in lost_inds]
        lost_cur_landmarks_every_fr_pos_bboxes = [cur_landmarks_every_fr_pos_bboxes[ind] for ind in lost_inds]
        lost_cur_landmarks_every_fr_idxes = [cur_landmarks_every_fr_idxes[ind] for ind in lost_inds]
        lost_landmark_3D_pos = [cur_landmarks_3D_pos[ind] for ind in lost_inds]

        unmatched_tracks = []
        assigned_matched_tracks_idxes = []
        assigned_all_tracks_idxes = []
        disappeared_tracks = []

        for lost_landmark_every_fr_unobserved, cur_landmark_every_fr_pos_bboxes, cur_landmark_every_fr_idxes, cur_landmark_3D_pos in zip(lost_landmarks_every_fr_unobserved,lost_cur_landmarks_every_fr_pos_bboxes, lost_cur_landmarks_every_fr_idxes, lost_landmark_3D_pos):
            #TODO:the unobserved list may be updated on the go but was not updated in the SfM results, thus find track first!

            # if lost_landmark_every_fr_unobserved[-1] == True:
            #         continue
            # else:

            track_found = 0
            min_diff = 0.01
            for idx, track in enumerate(tracks):
                if np.linalg.norm(track.prev_pos - bbox_center(cur_landmark_every_fr_pos_bboxes[-1][1])) < min_diff: #and idx not in assigned_all_tracks_idxes:
                    # assigned_all_tracks_idxes.append(idx)
                    track_found = 1
                    # print('succesfully matches track and the landmark based fruit position, they are:',track.prev_pos,bbox_center(cur_landmark_every_fr_pos_bboxes[-1][1]))
                    unmatched_tracks.append(track)
                    cur_landmark_3D_pos_tuple = (cur_landmark_3D_pos[0], cur_landmark_3D_pos[1], cur_landmark_3D_pos[2])
                    track.landmark_pos = cur_landmark_3D_pos_tuple
                    self.landmark_pos_tracks_dict[cur_landmark_3D_pos_tuple] = {'track':[copy.deepcopy(track)], 'frame number':current_frame_index}
                    break

            if track_found == 0:
                cur_landmark_3D_pos_tuple = (cur_landmark_3D_pos[0], cur_landmark_3D_pos[1], cur_landmark_3D_pos[2])
                for idx, track in enumerate(tracks):
                    if cur_landmark_3D_pos_tuple == track.landmark_pos:
                        assigned_all_tracks_idxes.append(idx)
                        track_found = 1
                        # print('succesfully matches track and the landmark based fruit position, they are:',track.prev_pos,bbox_center(cur_landmark_every_fr_pos_bboxes[-1][1]))
                        unmatched_tracks.append(track)
                        self.landmark_pos_tracks_dict[cur_landmark_3D_pos_tuple] = {'track':[copy.deepcopy(track)], 'frame number':current_frame_index}
                        break

            if track_found == 0:
                if lost_landmark_every_fr_unobserved[-1] == True:
                     continue
                else:
                    raise Exception(('no fruit track for current lost detection found, its abnormal, check the implementation!', 'the most close frame index from extracted hist is:',cur_landmark_every_fr_idxes[-1], ' current index is', current_frame_index))
                    # continue

        matched_tracks = []
        matched_fruits_list = []
        unobserved_list = []
        unobserved_tracks_matches = []
        cur_landmark_3D_pos_tuple_list = []
        for match in match_inds:
            unobserved_list.append(False)
            i_track, i_fruit = match

            matched_landmark_every_fr_unobserved = cur_landmarks_every_fr_unobserved[i_track]
            matched_landmark_every_fr_pos_bboxes = cur_landmarks_every_fr_pos_bboxes[i_track]
            matched_landmark_every_fr_idxes = cur_landmarks_every_fr_idxes[i_track]
            cur_landmark_3D_pos_tuple = (cur_landmarks_3D_pos[i_track][0], cur_landmarks_3D_pos[i_track][1], cur_landmarks_3D_pos[i_track][2])

            if cur_landmark_3D_pos_tuple in cur_landmark_3D_pos_tuple_list:
                raise Exception('duplicate landmarks')
            cur_landmark_3D_pos_tuple_list.append(cur_landmark_3D_pos_tuple)

            if matched_landmark_every_fr_unobserved[-1] == True:
                unobserved_tracks_matches.append([copy.deepcopy(match), cur_landmark_3D_pos_tuple])
            else:
                track_found = 0
                min_diff = 0.01
                for idx, track in enumerate(tracks):
                    if np.linalg.norm(track.prev_pos - bbox_center(matched_landmark_every_fr_pos_bboxes[-1][1])) < min_diff: #and idx not in assigned_all_tracks_idxes:
                        # assigned_all_tracks_idxes.append(idx)
                        assigned_matched_tracks_idxes.append(idx)
                        matched_tracks.append(track)
                        track.landmark_pos = cur_landmark_3D_pos_tuple
                        track_found = 1
                        fruit = fruits[i_fruit]
                        matched_fruits_list.append(copy.deepcopy(fruit))
                        break

                if track_found == 0:
                    raise Exception('no fruit track for current matched detection found, its abnormal, check the implementation!')


        # if len(unmatched_tracks) + len(matched_tracks) != len([track for track in tracks if track.landmark_pos != None]):
        #     raise Exception('check match_tracks')

        for idx, track in enumerate(tracks):
            if idx not in assigned_matched_tracks_idxes:
                disappeared_tracks.append(copy.deepcopy(track))
                if track.landmark_pos != None:
                    self.landmark_pos_tracks_dict[track.landmark_pos] = {'track':[copy.deepcopy(track)], 'frame number':current_frame_index}



        for unobserved_track_match in unobserved_tracks_matches:
            i_track, i_fruit = unobserved_track_match[0]
            cur_landmark_3D_pos_tuple = unobserved_track_match[1]
            matched_landmark_every_fr_unobserved = copy.deepcopy(cur_landmarks_every_fr_unobserved[i_track])
            matched_landmark_every_fr_pos_bboxes = copy.deepcopy(cur_landmarks_every_fr_pos_bboxes[i_track])
            unobserved_track = self.add_unobserved_track(tracks,matched_landmark_every_fr_pos_bboxes[-1][1], matched_landmark_every_fr_pos_bboxes, matched_landmark_every_fr_unobserved)
            unobserved_track.predict()
            unobserved_track.landmark_pos = cur_landmark_3D_pos_tuple
            matched_tracks.append(unobserved_track)
            fruit = fruits[i_fruit]
            matched_fruits_list.append(copy.deepcopy(fruit))



        if len(new_fruits) + len(matched_tracks) != len(fruits):
            raise Exception('check match_tracks')

        return matched_tracks, matched_fruits_list, unobserved_list, new_fruits, unmatched_tracks, disappeared_tracks

# TODO: landmark based:************************************************************************************************************


    # TODO: here is just the first step of kalman filter: state prediction based on prior
    def predict_tracks(self):
        """
        Predict tracks in Kalman filter
        """
        for track in self.tracks:
            track.predict()


    #TODO: in this function, both optical flow and kalman filter (for position correction) are used to generate two returns: updated track and lost track
    def landmark_update_tracks(self, gray, matched_fruits_list, unobserved_list):
        """
        Update tracks' position in Kalman filter via KLT
        :param gray: greyscale image
        :return: updated tracks, lost_tracks
        """
        prev_pts = [t.prev_pos for t in self.tracks]
        if len(matched_fruits_list) == 1:
            print('len(curr_pts.shape) == 1 in fruit_tracker.py \n')
            print('NOT SURE THE CHANGE IS CORRECT, CHECK WHETHER ERROR OCCURS! \n')
            track=self.tracks[0]
            prev_point = prev_pts[0]
            FCN_bbox = matched_fruits_list[0]
            track.landmark_update_bbox(FCN_bbox, self.flow_cov, self.bbox_cov, self.vel_cov, unobserved= False)

        else:
            for track, prev_point, FCN_bbox, unobserved in zip(self.tracks, prev_pts, matched_fruits_list, unobserved_list):
                track.landmark_update_bbox(FCN_bbox, self.flow_cov, self.bbox_cov, self.vel_cov, unobserved= False)




        # TODO: add KLT to refine position output for SfM
        init_pts = [t.pos for t in self.tracks]
        # TODO: here the status is output status vector (of unsigned chars): each element of the vector is set to 1 if the flow for the corresponding features has been found, otherwise, 0.
        # TODO: can change the window_size, max_level (levels of pyramid) to change the performace
        curr_pts, status = calc_optical_flow_for_refinement(self.prev_gray, gray,
                                             prev_pts, init_pts,
                                             self.win_size,
                                             self.max_level)
        for KLT_pos, stat, track, init_pt in zip(curr_pts, status, self.tracks, init_pts):
            if stat:
                # correction = KLT_pos - init_pt
                # corrected_pixels = np.sqrt(correction[0] ** 2 + correction[1] ** 2)
                # if corrected_pixels < 20
                track.KLT_pos = KLT_pos
                track.KLT_status.append(True)
                track.KLT_hist.append(KLT_pos)

            else:
                track.KLT_pos = init_pt
                track.KLT_status.append(False)
                track.KLT_hist.append(KLT_pos)

        print('Num of KLT estimated points:',sum(status),'Num of total points:',len(status))


        # TODO: NEWNEW
        updated_tracks = self.tracks
        self.prev_gray = gray



        return updated_tracks, status





    # TODO: count the lost tracks which used to be counted tracks (min_age>=3) instead of current counted tracks!!!!!!!
    # TODO: add prev_bgr parameter to input previous frame(for counting)

    # RCNN
    def count_in_tracks(self, tracks, prev_bgr):
        """
        Count how many fruits there are in tracks
        :param tracks: list of tracks
        """
        self.frame_counts = sum([1 for t in tracks if t.age >= self.min_age])
        print (self.frame_counts)
        self.total_counts += self.frame_counts

    # RCNN
    def finish(self, prev_bgr, current_frame_idx):
        # TODO: because this is the last frame, all tracks are counted instead of only lost tracks as we did for previous frames!!!!!!!!
        self.count_in_tracks(self.tracks, prev_bgr)
        self.logger.info("Total counts: {}".format(self.total_counts))
        print('total counts are:', self.total_counts, '\n')
        disappeared_tracks = self.tracks
        for track in self.tracks:

            if track.landmark_pos != None:
                self.landmark_pos_tracks_dict[track.landmark_pos] = {'track':[copy.deepcopy(track)], 'frame number':current_frame_idx}
        return disappeared_tracks





