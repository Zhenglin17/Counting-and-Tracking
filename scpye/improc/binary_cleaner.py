from __future__ import (print_function, division, absolute_import)

import logging
import cv2

from scpye.improc.image_processing import u8_from_bw


class BinaryCleaner(object):
    def __init__(self, ksize=5, iters=1):
        """
        :param ksize:
        :param iters:
        """
        self.ksize = ksize
        self.iters = iters

        self.logger = logging.getLogger(__name__)

    def clean(self, bw):
        """
        Clean up binary image and generate blobs and contours
        :param bw: binary image, with 255 max
        :return:
        """
        bw = u8_from_bw(bw, val=1)
        bw_clean = clean_bw(bw, ksize=self.ksize, iters=self.iters)

        return bw_clean


def morph_closing(bw, ksize=3, iters=1):
    """
    :param bw: binary image
    :param ksize: kernel size
    :param iters: number of iterations
    :return: binary image after closing
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    bw_closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=iters)
    return bw_closed


def morph_opening(bw, ksize=3, iters=1):
    """
    :param bw: binary image
    :param ksize: kernel size
    :param iters: number of iterations
    :return: binary image after opening
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    bw_opened = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel=kernel,
                                 iterations=iters)
    return bw_opened


def clean_bw(bw, ksize=3, iters=1):
    """
    Clean binary image by doing a opening followed by a closing
    :param bw: binary image
    :param ksize: kernel size
    :param iters: number of iterations
    :return: cleaned binary image
    """
    bw = morph_opening(bw, ksize=ksize, iters=iters)
    bw = morph_closing(bw, ksize=ksize, iters=iters)
    return bw
