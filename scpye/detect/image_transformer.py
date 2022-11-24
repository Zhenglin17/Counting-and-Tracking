from __future__ import (absolute_import, division, print_function)

from collections import namedtuple
from functools import partial
from itertools import izip

import cv2
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator

from scpye.track.bounding_box import extract_bbox

MaskedData = namedtuple('MaskedData', ['data', 'mask'])


def forward_list(func):
    """
    Decorator that output a list if input is list
    Currently only handles class member function
    :param func:
    """

    def func_wrapper(self, X, y=None):
        if y is None:
            if isinstance(X, list):
                return [func(self, each_X) for each_X in X]
            else:
                return func(self, X)
        else:
            if isinstance(X, list):
                # Make sure y corresponds to X
                if not isinstance(y, list):
                    raise TypeError('y is not a list')
                if len(X) != len(y):
                    raise ValueError('X and y not same length')

                Xts, yts = [], []
                for each_X, each_y in izip(X, y):
                    if each_y is None:
                        raise ValueError('y is None')
                    Xt, yt = func(self, each_X, each_y)
                    Xts.append(Xt)
                    yts.append(yt)
                return Xts, yts
            else:
                return func(self, X, y)

    return func_wrapper


class ImageTransformer(BaseEstimator, TransformerMixin):
    func = None

    def fit_transform(self, X, y=None, **fit_params):
        # Because fit returns self, here fit does nothing
        if y is None:
            # fit method of arity 1 (unsupervised transformation)
            return self.fit(X, **fit_params).transform(X)
        else:
            # fit method of arity 2 (supervised transformation)
            return self.fit(X, y, **fit_params).transform(X, y)

    def fit(self, X, y=None, **fit_params):
        return self

    @forward_list
    def transform(self, X, y=None):
        """
        transform function which dispatches work to other functions
        """
        self._pre_transform(X, y)

        if y is None:
            return self._transform_X(X)
        else:
            return self._transform_Xy(X, y)

    def _pre_transform(self, X, y=None):
        """
        Pre-transform some of the data so that it can be used later
        """
        pass

    def _transform_X(self, X):
        """
        Transform X to Xt
        """
        Xt = self.func(X)
        return Xt

    def _transform_Xy(self, X, y=None):
        """
        Transform X to Xt and y to yt
        """
        Xt = self.func(X)
        yt = self.func(y)
        return Xt, yt


class ImageRotator(ImageTransformer):
    def __init__(self, ccw=-1):
        """
        Rotate image n times counter-clockwise
        :param ccw: number of counter-clockwise rotations
        :type ccw: int
        """
        self.ccw = ccw
        self.func = partial(np.rot90, k=self.ccw)


class ImageCropper(ImageTransformer):
    def __init__(self, bbox=None):
        """
        Crop image to bounding box
        :param bbox:
        """
        self.bbox = bbox
        self.func = partial(extract_bbox, bbox=self.bbox)


class ImageResizer(ImageTransformer):
    def __init__(self, k=0.5):
        """
        Resize image by k
        :param k:
        """
        self.k = k
        self.func = partial(cv2.resize, dsize=None, fx=self.k, fy=self.k,
                            interpolation=cv2.INTER_NEAREST)


class ImageSmoother(ImageTransformer):
    def __init__(self, ksize=5, sigma=1):
        """
        Smooth image with gaussian blur, only applied on image, not on label
        :param ksize: kernel size
        :param sigma: sigma
        """
        self.ksize = ksize
        self.sigma = sigma
        self.func = partial(cv2.GaussianBlur, ksize=(self.ksize, self.ksize),
                            sigmaX=self.sigma)

    # special case when y is not transformed (blurred)
    def _transform_Xy(self, X, y=None):
        Xt = self.func(X)
        yt = y
        return Xt, yt


def split_label(label):
    """
    :param label:
    :return: split label
    :rtype: numpy.ndarray
    """
    assert np.ndim(label) == 3 and np.size(label, axis=-1) == 2
    return label[:, :, 0] > 0, label[:, :, 1] > 0


class DarkRemover(ImageTransformer):
    def __init__(self, pmin=25):
        """
        Remove dark pixels in image
        :param pmin: minimum value in gray scale image
        :type pmin: int
        """
        assert 0 <= pmin < 255, "pmin should be in [0, 255)"
        self.v_min = pmin

        self.mask = None

    def _pre_transform(self, X, y=None):
        gray = cv2.cvtColor(X, cv2.COLOR_BGR2GRAY)
        self.mask = gray > self.v_min

    def _transform_X(self, X):
        return MaskedData(data=X, mask=self.mask)

    def _transform_Xy(self, X, y=None):
        neg, pos = split_label(y)
        neg_mask = self.mask & neg
        pos_mask = self.mask & pos

        y_neg = np.zeros(np.count_nonzero(neg_mask))
        y_pos = np.ones(np.count_nonzero(pos_mask))

        labels = np.dstack((neg_mask, pos_mask))
        yt = np.hstack((y_neg, y_pos))

        return MaskedData(data=X, mask=labels), yt
