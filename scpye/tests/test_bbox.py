import unittest

import numpy as np
import numpy.testing as nt

from scpye.track.bounding_box import *


class TestBbox(unittest.TestCase):
    def test_extract_bbox(self):
        # 2d
        x = np.identity(4)
        b = np.array([1, 1, 2, 2])
        e = extract_bbox(x, b)
        e0 = np.identity(2)
        nt.assert_array_equal(e, e0)

        # 3d
        x = np.dstack((x, x))
        e = extract_bbox(x, b)
        e0 = np.dstack((e0, e0))
        nt.assert_array_equal(e, e0)

        # None
        e = extract_bbox(x, None)
        nt.assert_array_equal(e, x)

    def test_bbox_center(self):
        b = np.array([1, 1, 2, 2])
        c = bbox_center(b)
        c0 = np.array([2, 2])
        nt.assert_array_equal(c, c0)

        b = np.array([1, 1, 0, 0])
        c = bbox_center(b)
        c0 = b[:2]
        nt.assert_array_equal(c, c0)

    def test_bbox_distance_squared(self):
        b0 = np.array([0, 0, 2, 2])
        d0 = bbox_distsq(b0, b0)
        self.assertEqual(d0, 0.0)

        b1 = np.array([1, 1, 2, 2])
        d1 = bbox_distsq(b0, b1)
        self.assertEqual(d1, 2.0)
