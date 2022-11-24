from __future__ import (print_function, absolute_import, division)

from collections import namedtuple
import cv2
import numpy as np
import scipy
from scipy import ndimage


"""
http://docs.opencv.org/trunk/d3/d05/tutorial_py_table_of_contents_contours.html#gsc.tab=0
"""

# TODO: new from comp -- the whole script


prop_type = [('area', float), ('aspect', float), ('extent', float),
             ('solidity', float)]
Blob = namedtuple('Blob', ('bbox', 'prop', 'cntr'))

def find_connected_components(img_bw):
    contours = []
    structure = [[1, 1, 1],
                 [1, 1, 1],
                 [1, 1, 1]]
    labeled, nr_objects = ndimage.label(img_bw, structure=structure)
    for obj_idx in np.arange(1,nr_objects+1):

        idx_temp = np.argwhere(labeled == obj_idx)
        idx = np.flip(idx_temp,1)
        contours.append(idx)
    # import matplotlib.pyplot as plt
    # plt.imshow(labeled)
    # plt.show()
    return contours


# TODO: change the min_area
def analyze_contours_bw(bw, min_area):
    """
    Same as matlab regionprops but implemented in opencv
    Prefer using this than skimage's regionprops because this return a numpy
    recarray that is very compact
    :param bw: binary image
    :param min_area:
    :return: a structured array of blobs
    """
    contours_analysis = False
    if contours_analysis:
        contours = find_contours(bw)
    else:
        contours = find_connected_components(bw)
    return analyze_contours(contours, min_area, contours_analysis)


def analyze_contours(contours, min_area, contours_analysis):
    """
    :param contours:
    :param min_area:
    :return: blobs
    """
    blobs = []
    if contours_analysis:
        for cntr in contours:
            area = contour_area(cntr)
            if area >= min_area:
                bbox = contour_bounding_rect(cntr)
                aspect = bounding_rect_aspect(bbox)
                extent = contour_extent(cntr, cntr_area=area, bbox=bbox)
                solidity = contour_solidity(cntr, cntr_area=area)
                prop = np.array((area, aspect, extent, solidity), dtype=prop_type)
                blobs.append(Blob(bbox=bbox, prop=prop, cntr=cntr))
        return blobs

    else:
        for cntr_all in contours:
            l = np.min(cntr_all,0)
            # l_idx = np.argmin(cntr_all,0)
            # l_0 = cntr_all[l_idx[0],:]
            # l_1 = cntr_all[l_idx[1],:]
            h = np.max(cntr_all,0)
            # h_idx = np.argmax(cntr_all, 0)
            # h_0 = cntr_all[h_idx[0],:]
            # h_1 = cntr_all[h_idx[1],:]

            corners = np.array([[l[0],l[1]],[l[0],h[1]],[h[0],l[1]],[h[0],h[1]]])
            bbox = contour_bounding_rect(corners)
            # raw_vertices = np.array([l_0,l_1,h_0,h_1])
            # rect = cv2.minAreaRect(raw_vertices)
            # vertices = cv2.boxPoints(rect)
            # bbox = contour_bounding_rect(vertices)

            area = bbox[2]*bbox[3]
            if area >= min_area:

                aspect = bounding_rect_aspect(bbox)
                extent = 1
                solidity = 1
                prop = np.array((area, aspect, extent, solidity), dtype=prop_type)
                blobs.append(Blob(bbox=bbox, prop=prop, cntr=corners))
        return blobs


def find_contours(bw):
    """
    :param bw: binary image
    :return: a list of contours
    """

    cv2.destroyAllWindows()
    _, cntrs, _ = cv2.findContours(bw.copy(),
                                   mode=cv2.RETR_EXTERNAL,
                                   method=cv2.CHAIN_APPROX_SIMPLE)
    return cntrs


def contour_area(cntr):
    """
    :param cntr:
    :return: contour area
    """
    return cv2.contourArea(cntr)


def contour_bounding_rect(cntr):
    """
    :param cntr:
    :return: bounding box [x, y, w, h]
    """
    return np.array(cv2.boundingRect(cntr))


def bounding_rect_aspect(bbox):
    """
    :param bbox: bounding box
    :return: aspect ratio
    """
    _, _, w, h = bbox
    aspect = float(w) / h if float(w) > h else h / w
    return aspect


def contour_extent(cntr, cntr_area=None, bbox=None):
    """
    :param cntr:
    :param cntr_area: contour area
    :param bbox: bounding box
    :return: extent
    """
    if cntr_area is None:
        cntr_area = contour_area(cntr)

    if bbox is None:
        bbox = contour_bounding_rect(cntr)

    _, _, w, h = bbox
    rect_area = w * h
    extent = float(cntr_area) / rect_area
    return extent


def contour_solidity(cntr, cntr_area=None):
    """
    :param cntr:
    :param cntr_area: contour area
    :return: solidity
    """
    if cntr_area is None:
        cntr_area = cv2.contourArea(cntr)

    hull = cv2.convexHull(cntr)
    hull_area = cv2.contourArea(hull)
    solidity = float(cntr_area) / hull_area
    return solidity


def contour_equi_diameter(cntr, cntr_area=None):
    """
    :param cntr:
    :param cntr_area: contour area
    :return: equivalent diameter
    """
    if cntr_area is None:
        cntr_area = contour_area(cntr)

    equi_diameter = np.sqrt(4 * cntr_area / np.pi)
    return equi_diameter


def moment_centroid(mmt):
    """
    Centroid of moment
    :param mmt: moment
    :return: (x, y)
    """
    return np.array((mmt['m10'] / mmt['m00'], mmt['m01'] / mmt['m00']))
