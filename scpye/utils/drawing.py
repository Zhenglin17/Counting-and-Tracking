from __future__ import (print_function, division, absolute_import)

import cv2
import numpy as np
import matplotlib.pyplot as plt


class Colors:
    """
    Collection of colors
    """
    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)
    yellow = (255, 255, 0)
    magenta = (255, 0, 255)
    cyan = (0, 255, 255)
    orange = (255, 165, 0)
    purple = (160, 32, 240)
    gold = (255, 215, 0)
    white = (255, 255, 255)
    default = red


def imshow(*images, **options):
    """
    A helper function to show multiple images in one row
    :param images:
    :param options:
    :return: fig, axarr
    """
    figsize = options.pop('figsize', (10, 10))
    interp = options.pop('interp', None)
    cmap = options.pop('cmap', None)
    titles = options.pop('titles', tuple())
    hide_axes = options.pop('hide_axes', False)

    fig = plt.figure(figsize=figsize)
    naxes = len(images)
    axarr = np.empty(naxes, dtype=object)

    for i, (image, title) in enumerate(zip_longest(images, titles)):
        axarr[i] = fig.add_subplot(1, naxes, i + 1)
        axarr[i].imshow(image, interpolation=interp, cmap=cmap)
        if title is not None:
            axarr[i].set_title(title)
        axarr[i].xaxis.set_visible(not hide_axes)
        axarr[i].yaxis.set_visible(not hide_axes)

    return fig, axarr


def draw_bboxes(image, bboxes, color=Colors.default, thickness=1, margin=0):
    bboxes = np.atleast_2d(bboxes)
    for bbox in bboxes:
        x, y, w, h = np.array(bbox, dtype=int)
        cv2.rectangle(image, (x - margin, y - margin), (x + w + margin, y + h + margin), color=color,
                      thickness=thickness, lineType=cv2.LINE_AA)


def draw_bbox_centers(image, bboxes, color=Colors.default, thickness=1, r=2):
    bboxes = np.atleast_2d(bboxes)
    for bbox in bboxes:
        x, y, w, h = np.array(bbox, dtype=int)
        cv2.circle(image, (x + int(w / 2), y + int(h / 2)), r, color=color, thickness=thickness)


def draw_bbox_as_circles(image, bboxes, color=Colors.default, thickness=1):
    bboxes = np.atleast_2d(bboxes)
    for bbox in bboxes:
        x, y, w, h = np.array(bbox, dtype=int)
        cv2.circle(image, (x + int(round(w / 2.0)), y + int(round(h / 2) / 2.0)), radius=int(round((w + h) / 4.0)),
                   color=color, thickness=thickness)


def draw_circles(image, circles, color=Colors.default, thickness=1):
    circles = np.atleast_2d(circles)
    for circle in circles:
        x, y, r = np.array(circle, dtype=int)
        cv2.circle(image, (x, y), r, color=color, thickness=thickness)


def draw_points(image, points, color=Colors.default, radius=1):
    points = np.atleast_2d(points)
    for point in points:
        x, y = np.array(point, dtype=int)
        cv2.circle(image, (x, y), radius, color=color, thickness=-1)


def draw_ellipses(image, ellipses, color=Colors.default, thickness=1):
    ellipses = np.atleast_2d(ellipses)
    for ellipse in ellipses:
        x, y, ax1, ax2, ang = np.array(ellipse, dtype=int)
        cv2.ellipse(image, (x, y), (ax1, ax2), ang, 0, 360, color=color,
                    thickness=thickness)


def drawline(img, pt1, pt2, color, thickness=1, style='dotted', gap=20):
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** .5
    pts = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
        p = (x, y)
        pts.append(p)

    if style == 'dotted':
        for p in pts:
            cv2.circle(img, p, thickness, color, -1)
    elif pts != []:
        s = pts[0]
        e = pts[0]
        i = 0
        for p in pts:
            s = e
            e = p
            if i % 2 == 1:
                cv2.line(img, s, e, color, thickness)
            i += 1
        cv2.circle(img, p, 2, color=color, thickness=-1)


def draw_line(image, line, color=Colors.default, thickness=1, dash_gap=10):
    line = np.atleast_2d(np.array(line, dtype=int))
    for i in range(len(line) - 1):
        p1, p2 = line[i], line[i + 1]
        a, b = p1
        c, d = p2
        color_ratio_min = max(((i - ((len(line) - 1) - 5.0)) / 5.0, 0))
        color_ratio = min(color_ratio_min + 0.4, 1)
        if dash_gap == 0:
            cv2.line(image, (a, b), (c, d), [color_ratio * color[0], color_ratio * color[1], color_ratio * color[2]],
                     thickness=thickness)
        else:
            drawline(image, (a, b), (c, d), [color_ratio * color[0], color_ratio * color[1], color_ratio * color[2]],
                     style='dash', gap=dash_gap, thickness=thickness)


def draw_contour(image, cntr, color=Colors.default, thickness=1):
    cv2.drawContours(image, [cntr], 0, color, thickness)


def draw_contours(image, cntrs, color=Colors.default, thickness=1):
    cv2.drawContours(image, cntrs, -1, color, thickness)


def draw_text(image, text, point, color=Colors.default, scale=0.5, thickness=1):
    if type(text) is not str:
        text = str(int(text))

    x, y = np.array(point, dtype=int)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, scale,
                color=color, thickness=thickness, lineType=cv2.LINE_AA)


def draw_text_mask(image, text, text_describe, point, color=Colors.default, scale=0.5, thickness=1):
    if type(text) is not str:
        text = str(int(text))
    final_text = text_describe + text

    x, y = np.array(point, dtype=int)
    cv2.putText(image, final_text, (x, y), cv2.FONT_HERSHEY_DUPLEX, scale,
                color=color, thickness=thickness, lineType=cv2.LINE_AA)


def draw_texts(image, texts, points, color=Colors.default, scale=0.5,
               thickness=1):
    for text, point in zip(texts, points):
        draw_text(image, text, point, color=color, scale=scale,
                  thickness=thickness)


def draw_optical_flows(image, points1, points2, status=None, radius=2,
                       color=Colors.red, thickness=2, draw_invalid=False):
    points1 = np.atleast_2d(points1)
    points2 = np.atleast_2d(points2)

    if status is None:
        status = np.ones(len(points1))

    for pt1, pt2, st in zip(points1, points2, status):
        if st or draw_invalid:
            a, b = np.array(pt1.ravel(), int)
            c, d = np.array(pt2.ravel(), int)

            cv2.line(image, (a, b), (c, d), color=color, thickness=thickness,
                     lineType=cv2.LINE_AA)
            cv2.circle(image, (c, d), radius, color=color, thickness=-1)


def draw_bboxes_matches(image, matches, bboxes1, bboxes2, color, thickness=1):
    matches = np.atleast_2d(matches)
    bboxes1 = np.atleast_2d(bboxes1)
    bboxes2 = np.atleast_2d(bboxes2)

    for pair in matches:
        i1, i2 = pair
        bbox1 = bboxes1[i1]
        bbox2 = bboxes2[i2]
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        a = int(x1 + w1 / 2)
        b = int(y1 + h1 / 2)
        c = int(x2 + w2 / 2)
        d = int(y2 + h2 / 2)
        draw_bboxes(image, bbox2, color=color, thickness=thickness)
        cv2.line(image, (a, b), (c, d), color=color, thickness=thickness)


def draw_blob_analyzer(ba, disp_bgr, disp_bw):
    draw_bboxes(disp_bgr, ba.single_bboxes, color=Colors.red)
    draw_bboxes(disp_bgr, ba.multi_bboxes, color=Colors.orange)
    draw_bboxes(disp_bw, ba.single_bboxes, color=Colors.red)
    draw_bboxes(disp_bw, ba.multi_bboxes, color=Colors.orange)
