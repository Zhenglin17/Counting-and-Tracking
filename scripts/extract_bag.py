from __future__ import (absolute_import, print_function, division)

import os
import sys
import argparse as ap
import rosbag
from tqdm import tqdm


def process_bags(bagfiles):
    """
    :param bagfiles: a list of bag files
    :type bagfiles: List
    """
    for bagfile in bagfiles:
        try:
            bagfile_fixed = extract_bag(bagfile)
            print(bagfile_fixed)
        except IOError as ie:
            print(ie)
        except ValueError as ve:
            print(ve)


def add_suffix(bagfile, suffix="_fixed"):
    """
    Add suffix to a bag file name
    :param bagfile: bag file path
    :type bagfile: str
    :param suffix: suffix to add to bag file
    :type suffix: str
    :return: suffixed bag file
    :rtype str:
    """
    name, ext = os.path.splitext(bagfile)
    return name + suffix + ext


def extract_bag(bagfile, skip_topic, skip_front, skip_back):
    bagfile_fixed = add_suffix(bagfile)
    with rosbag.Bag(bagfile) as bag:
        n_messages = bag.get_message_count(skip_topic)
        print(n_messages)
        with rosbag.Bag(bagfile_fixed, 'w') as bag_fixed:
            i = 0
            for topic, msg, t in tqdm(bag.read_messages()):
                if topic == skip_topic:
                    if i < skip_front or i >= n_messages - skip_back:
                        print('skipping image {}'.format(i))
                    else:
                        bag_fixed.write(topic, msg,
                                        msg.header.stamp if msg._has_header else t)
                    i += 1
                else:
                    bag_fixed.write(topic, msg,
                                    msg.header.stamp if msg._has_header else t)
    return bagfile_fixed


def main():
    bagfile = "/home/chao/Workspace/dataset/agriculture/apple/red/slow_flash/north/bag/frame4.bag"
    skip_front = 0
    skip_back = 2
    skip_topic = '/color/image_rect_color'
    extract_bag(bagfile, skip_topic, skip_front, skip_back)


if __name__ == '__main__':
    sys.exit(main())
