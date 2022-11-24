import cv2
import numpy as np
from skimage import exposure


def enhance_contrast(image, pmin=2.0, pmax=99.95):
    """
    Enhance image contrast
    :param image:
    :param pmin:
    :param pmax:
    :return:
    """
    vmin, vmax = np.percentile(image, (pmin, pmax))
    return exposure.rescale_intensity(image, in_range=(vmin, vmax))


def swap_channels(bgr):
    b, g, r = cv2.split(bgr)
    rgb = cv2.merge((r, g, b))
    return rgb


def gray_from_bgr(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def bgr_from_gray(gray):
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def u8_from_bw(bw, val=255):
    """
    Convert bw image from bool to uint8 if possible
    :param bw: binary image
    :param val: max_val of image
    :return: greyscale image
    """
    assert np.ndim(bw) >= 2, 'Image dimension wrong'
    return np.array(bw > 0, dtype=np.uint8) * val


def fill_bw(bw, cntrs, in_place=False):
    """
    Redraw contours of binary image
    :param bw:
    :param cntrs:
    :param in_place: draw in place
    :return: filled image
    """
    if in_place:
        bw_filled = bw
    else:
        bw_filled = np.zeros_like(bw)

    cv2.drawContours(bw_filled, cntrs, -1, color=255, thickness=-1)

    return bw_filled


def scale_array(data, val=100):
    """
    Scale array to value
    :param data:
    :param val:
    :return:
    """
    max_data = np.max(data)
    scale = float(val) / max_data
    return np.multiply(data, scale)
