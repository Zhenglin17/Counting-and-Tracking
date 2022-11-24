from __future__ import (print_function, absolute_import, division)

import cv2
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d

from scpye.detect.image_transformer import (ImageTransformer, split_label)
from scpye.utils.exception import FeatureNotSupportedError


def stack_list(func):
    """
    Decorator that stacks the output if input is a list
    Currently only handles class member function
    :param func:
    """

    def func_wrapper(self, X, y=None):
        if isinstance(X, list):
            return np.vstack([func(self, each_X) for each_X in X])
        else:
            return func(self, X)

    return func_wrapper


class FeatureTransformer(ImageTransformer):
    @stack_list
    def transform(self, X, y=None):
        """
        :param X:
        :type X: MaskedData
        :param y:
        :return:
        """
        if np.ndim(X.mask) == 2:
            Xt = self._transform_mask(X)
        else:
            Xt = self._transform_labels(X)

        return np.array(Xt, np.float64)

    def _transform_mask(self, X):
        return X

    def _transform_labels(self, X):
        return X


class CspaceTransformer(FeatureTransformer):
    def __init__(self, cspace):
        self.cspace = cspace

    def cspace_transform(self, src):
        """
        :param src: bgr image
        :return: bgr image in other colorspace
        """
        # src is a nx3 vector, we need it to be nx1x3 to work with cvtColor
        if np.ndim(src) == 2:
            src = np.expand_dims(src, 1)

        if self.cspace == "hsv":
            des = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        elif self.cspace == "lab":
            des = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
        elif self.cspace == "bgr":
            des = src
        else:
            raise FeatureNotSupportedError(self.cspace)

        # squeeze des back from nx1x3 to nx3
        return np.squeeze(des)

    def _transform_mask(self, X):
        bgr, mask = X.data, X.mask
        Xt = self.cspace_transform(bgr[mask])
        return Xt

    def _transform_labels(self, X):
        bgr, mask = X.data, X.mask
        neg, pos = split_label(mask)
        Xt_neg = self.cspace_transform(bgr[neg])
        Xt_pos = self.cspace_transform(bgr[pos])
        Xt = np.vstack((Xt_neg, Xt_pos))
        return Xt


class MaskLocator(FeatureTransformer):
    @staticmethod
    def where_row_col(m):
        """
        Get locations of non-zero pixels in array
        :param m: array
        :type m: numpy.ndarray
        :return: n x 2 matrix of [x, y]
        :rtype: numpy.ndarray
        """
        assert np.ndim(m) == 2
        r, c = np.where(m)
        return np.transpose(np.vstack((r, c)))

    def _transform_mask(self, X):
        Xt = self.where_row_col(X.mask)
        return Xt

    def _transform_labels(self, X):
        neg, pos = split_label(X.mask)
        xy_neg = self.where_row_col(neg)
        xy_pos = self.where_row_col(pos)
        Xt = np.vstack((xy_neg, xy_pos))
        return Xt


class GradientTransformer(FeatureTransformer):
    def __init__(self):
        pass

    @staticmethod
    def gradient_magnitude(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        Ix = cv2.Scharr(image, cv2.CV_64F, 1, 0)
        Iy = cv2.Scharr(image, cv2.CV_64F, 0, 1)
        mag, ang = cv2.cartToPolar(Ix, Iy)
        return mag

    def _transform_mask(self, X):
        mag = self.gradient_magnitude(X.data)
        Xt = mag[X.mask]
        Xt = np.reshape(Xt, (-1, 1))
        return Xt

    def _transform_labels(self, X):
        neg, pos = split_label(X.mask)
        mag = self.gradient_magnitude(X.data)
        mag_neg = mag[neg]
        mag_pos = mag[pos]
        Xt = np.hstack((mag_neg, mag_pos))
        Xt = np.reshape(Xt, (-1, 1))
        return Xt


class PatchCreator(FeatureTransformer):
    def __init__(self, border=1):
        self.border = border
        self.size = 2 * self.border + 1

    @staticmethod
    def make_border(image, border):
        padded = cv2.copyMakeBorder(image, border, border, border, border,
                                    cv2.BORDER_REFLECT)
        return padded

    def extract_patches(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        padded = self.make_border(gray, self.border)
        patches = extract_patches_2d(padded, (self.size, self.size))
        r, c, _ = image.shape
        reshaped_patches = np.reshape(patches, (r, c, -1))
        return reshaped_patches

    def _transform_mask(self, X):
        patches = self.extract_patches(X.data)
        Xt = patches[X.mask]
        return Xt

    def _transform_labels(self, X):
        patches = self.extract_patches(X.data)
        neg, pos = split_label(X.mask)
        patches_neg = patches[neg]
        patches_pos = patches[pos]
        Xt = np.vstack((patches_neg, patches_pos))
        return Xt
