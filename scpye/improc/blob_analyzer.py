from __future__ import (print_function, division, absolute_import)

import logging

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from skimage.feature import peak_local_max
from skimage.morphology import watershed

from scpye.track.bounding_box import extract_bbox
from scpye.improc.image_processing import (fill_bw, scale_array, u8_from_bw,
                                           gray_from_bgr, bgr_from_gray,
                                           enhance_contrast)
from scpye.improc.contour_analysis import (contour_bounding_rect,
                                           analyze_contours_bw, Blob,
                                           find_contours)
from scpye.utils.drawing import draw_contours, Colors, imshow


def mean_blob_area(blobs):
    areas = [blob.prop['area'] for blob in blobs]
    return np.mean(areas)


class BlobAnalyzer(object):
    def __init__(self, max_aspect=1.3, min_extent=0.62,
                 min_solidity=0.91, gauss_filter_sigma=2,
                 max_filter_size=4, gray_edt_ratio=1.5, min_peak_distance=5,
                 exclude_border=True, vis=False):
        # Parameters for extracting single blob
        self.max_aspect = max_aspect  # 1.3
        self.min_extent = min_extent  # 0.62
        self.min_solidity = min_solidity  # 0.91

        # Parameters for splitting multi blob
        self.gauss_filter_sigma = gauss_filter_sigma
        self.max_filter_size = max_filter_size
        self.gray_edt_ratio = gray_edt_ratio
        self.gray_max = 100
        self.min_peak_distance = min_peak_distance
        self.exclude_border = exclude_border

        self.logger = logging.getLogger(__name__)

        # Drawing
        self.vis = vis
        self.disp_bw = None
        self.disp_bgr = None

    def analyze(self, bgr, bw):
        """
        :param bgr: color image
        :param bw: binary image
        :return: fruits
        """
        gray = gray_from_bgr(bgr)
        blobs = analyze_contours_bw(bw, min_area=50)

        # TODO: cntrs are countours, refer to contour_analysis.py, analyze_contours_bw function
        cntrs = [blob.cntr for blob in blobs]
        bw_fill = fill_bw(bw, cntrs, in_place=False)

        if self.vis:
            self.disp_bw = bgr_from_gray(bw_fill)
            good_bgr = enhance_contrast(bgr)
            self.disp_bgr = bgr_from_gray(gray_from_bgr(good_bgr))
            self.disp_bgr[bw > 0] = good_bgr[bw > 0]

        # Get single bboxes (fruits)
        # TODO: new from comp
        #single_blobs, multi_blobs = self.extract_single(blobs)
        if blobs != []:
            single_blobs, multi_blobs = self.extract_single(blobs)
        else:
            single_blobs = []
            multi_blobs = []
            print('Blob_analyzer.py single_blobs == []')

        if self.vis:
            single_cntrs = [blob.cntr for blob in single_blobs]
            draw_contours(self.disp_bgr, single_cntrs, thickness=2,
                          color=Colors.blue)

        # Split them to single bbox and add to fruits
        more_single_blobs, split_blobs = self.split_multi(multi_blobs, gray)

        if self.vis:
            more_single_cntrs = [blob.cntr for blob in more_single_blobs]
            draw_contours(self.disp_bgr, more_single_cntrs, color=Colors.cyan,
                          thickness=2)
            split_cntrs = [blob.cntr for blob in split_blobs]
            draw_contours(self.disp_bgr, split_cntrs, color=Colors.green,
                          thickness=2)

        self.logger.debug(
            "single/more/split: {}/{}/{}".format(len(single_blobs),
                                                 len(more_single_blobs),
                                                 len(split_blobs)))

        split_blobs.extend(single_blobs)
        split_blobs.extend(more_single_blobs)
        fruits = np.array([blob.bbox for blob in split_blobs])

        return fruits, bw_fill

    def is_single_blob(self, blob, mean_area):
        """
        Check if this blob is a single blob
        :param blob:
        :param mean_area:
        :return:
        """
        prop = blob.prop

        # If area is less than average then it is a single blob
        if prop['area'] < mean_area:
            return True

        # For a blob that is big enough, if it is solid then it is a single blob
        if prop['solidity'] > self.min_solidity:
            return True

        # For the rest blobs, if it is a relative filled square,
        # it is a single blob
        if prop['extent'] > self.min_extent \
                and prop['aspect'] < self.max_aspect:
            return True

        return False

    def extract_single(self, blobs):
        """
        Extract potential multi-blobs
        :param blobs:
        :return: list of potential multi-blobs
        """
        mean_area = mean_blob_area(blobs)

        single_blobs, multi_blobs = [], []
        for blob in blobs:

            if self.is_single_blob(blob, mean_area):
                single_blobs.append(blob)
            else:
                multi_blobs.append(blob)

        return single_blobs, multi_blobs

    def split_multi(self, blobs, gray):
        """
        Split potential multi-blobs into separate bounding boxes
        :param blobs:
        :param gray:
        """
        single_blobs, split_blobs = [], []

        for blob in blobs:
            gray_bbox = extract_gray(gray, blob)

            # calculate distance measure for watershed
            dist_max, markers, n_peaks = self.prepare_watershed(gray_bbox)

            if n_peaks < 2:
                single_blobs.append(blob)
            else:
                labels = watershed(-dist_max, markers, mask=gray_bbox)
                # VIS
                if self.vis:
                    imshow(labels, dist_max, markers, figsize=(12, 12),
                           interp="none", cmap=plt.cm.viridis)

                each_blobs = blobs_from_labels(labels, n_peaks, blob)
                split_blobs.extend(each_blobs)

        return single_blobs, split_blobs

    def prepare_watershed(self, gray):
        """
        Prepare for watershed
        :param gray:
        :return:
        """
        gray_blur = ndi.gaussian_filter(gray, self.gauss_filter_sigma)
        # gray will be converted to binary when performing edt
        euclid_dist = ndi.distance_transform_edt(gray)
        dist = scale_array(gray_blur, val=self.gray_max)
        # combination of intensity and distance transform
        dist += scale_array(euclid_dist,
                            val=self.gray_max / self.gray_edt_ratio)

        dist_max = ndi.maximum_filter(dist, size=self.max_filter_size,
                                      mode='constant')
        local_max = peak_local_max(dist_max,
                                   min_distance=self.min_peak_distance,
                                   indices=False,
                                   exclude_border=self.exclude_border)
        markers, n_peaks = ndi.label(local_max)

        return dist_max, markers, n_peaks


def blobs_from_labels(labels, n_peaks, blob):
    """
    Extract bboxes in labels and convert to global bboxes
    :param labels:
    :param n_peaks:
    :param blob:
    :return:
    """
    each_blobs = []

    for i in range(n_peaks):
        label_i1 = u8_from_bw(labels == i + 1)  # 0 is background
        local_bbox = contour_bounding_rect(label_i1)
        local_cntr = find_contours(label_i1)[0]
        # shift bbox from local to global
        local_bbox[:2] += blob.bbox[:2]
        local_cntr += blob.bbox[:2]
        each_blobs.append(Blob(bbox=local_bbox, prop=None, cntr=local_cntr))

    return each_blobs


def extract_gray(gray, blob):
    """
    Extract gray and binary image from rprop
    :param gray:
    :param blob:
    :return: gray bbox
    """
    bbox = blob.bbox
    cntr = blob.cntr

    gray_bbox = extract_bbox(gray, bbox, copy=True)
    # redraw contour so that we don't accidentally grab pixels from other blobs
    # and because cntr is global, we need to draw it onto full image
    bw_cntr = fill_bw(gray, [cntr])
    bw_bbox = extract_bbox(bw_cntr, bbox)
    gray_bbox[bw_bbox == 0] = 0

    return gray_bbox
