from __future__ import (print_function, division, absolute_import)

import numpy as np
from scpye.track.kalman_filter import KalmanFilter
from scpye.track.bounding_box import (bbox_center, shift_bbox)


def cov2ellipse(P, ns=3):
    """
    Covariance to ellipse
    :param P: covariance
    :param ns: scale
    :return:
    """
    assert np.shape(P) == (2, 2)

    U, D, _ = np.linalg.svd(P)
    w, h = np.sqrt(D) * ns
    a = np.arctan2(U[1, 0], U[0, 0])

    return w, h, np.rad2deg(a)


class LandmarkFruitTrack(object):
    def __init__(self, bbox, init_flow, state_cov, proc_cov):
        self.bbox = bbox
        self.age = 1
        self.hist = []
        self.KLT_hist = []
        self.KLT_status = []
        self.hist_unobserved = []
        self.prev_pos = None

        # TODO:every track will initialize its own kalman filter
        # every track will initialize its own kalman filter
        init_pos = bbox_center(self.bbox)
        init_state = np.hstack((init_pos, init_flow))

        self.kf = KalmanFilter(x0=init_state, P0=state_cov, Q=proc_cov)
        self.hist.append(self.pos)
        self.KLT_hist.append(self.pos)
        self.KLT_status.append(False)
        self.hist_unobserved.append(False)
        self.landmark_pos = None
        # KLT optimized position
        self.KLT_pos = init_pos

    @property
    def pos(self):
        return self.kf.x[:2].copy()

    @property
    def vel(self):
        return self.kf.x[2:].copy()

    @property
    def pos_cov(self):
        return self.kf.P[:2, :2].copy()

    @property
    def cov_ellipse(self):
        wha = cov2ellipse(self.pos_cov)
        return np.hstack((self.pos, wha))

    def predict(self):
        """
        Predict new location of the tracking
        """
        self.prev_pos = self.pos
        self.kf.predict()
        if(np.isnan(self.pos[0])) or (np.isnan(self.pos[1])) :
            print (self.pos)
            print('isnan(self.pos[0]) in fruit_track.py')
            # pos = self.bbox[:2]
            # self.bbox = shift_bbox(self.bbox, pos)
        else:
            self.bbox = shift_bbox(self.bbox, self.pos)

    # TODO: here the bboxes are correted using optical flow and...
    def correct_flow(self, pos, flow_cov):
        """
        Correct location of the track from pos input
        :param pos:
        :param flow_cov:
        """
        self.kf.update_pos(pos, flow_cov)
        self.bbox = shift_bbox(self.bbox, self.pos)
        # print ('pos diff before and after correct flow (with optical flow raw measurements):',np.linalg.norm(self.pos-prev_pos))
        # print('pos_cov',self.pos_cov)

    # TODO: here the bboxes are correted from the original detections to detection-prediction combined result
    # def correct_bbox(self, bbox, bbox_cov):
    #     """
    #     Correct location of the track hungarian assignment
    #     :param bbox:
    #     :param bbox_cov:
    #     :return:
    #     """
    #     pos = bbox_center(bbox)
    #     self.kf.update_pos(pos, bbox_cov)
    #     # Here we shift using current bbox (fixed)
    #     self.bbox = shift_bbox(bbox, self.pos)
    #     # print ('pos diff before and after correct bbox (with FCN raw measurements):',np.linalg.norm(self.pos-prev_pos))
    #     # print('pos_cov',self.pos_cov)
    #
    #     # Increment age and add track to hist
    #     self.age += 1
    #     self.hist.append(self.pos)


    def landmark_update_bbox(self, FCN_bbox, flow_cov, bbox_cov, vel_cov, unobserved):
        FCN_pos = bbox_center(FCN_bbox)
        measurements = np.zeros(6)
        cov_all = np.zeros(6)
        # # KLT measurement
        # measurements[:2] = KLT_point
        cov_all[:2] = flow_cov
        # FCN measurement
        measurements[2:4] = FCN_pos
        cov_all[2:4] = bbox_cov
        # Velocity measurement
        # measurements[4:] = vel
        cov_all[4:] = vel_cov
        # self.kf.update_all(measurements,cov_all)


        # TODO: In landmark based method, directly update the position of bbox to the fruit position
        self.kf.x[:2] = FCN_pos
        # self.pos [FCN_pos]


        # # TODO: moving correct_bbox from match_tracks to here
        # track.correct_bbox(FCN_bbox, bbox_cov)
        # # TODO: NEWNEW update velocity
        # track.kf.update_vel(vel, vel_cov)
        # #TODO: here use kalman filter to update the position!!!
        # track.correct_flow(KLT_point, flow_cov)

        self.bbox = shift_bbox(FCN_bbox, self.pos)
        # TODO: age measures how many times the bbox is tracked
        self.age += 1
        self.hist.append(self.pos)
        self.hist_unobserved.append(unobserved)

