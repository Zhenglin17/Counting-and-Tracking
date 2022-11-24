#!/usr/bin/env python
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
            bagfile_fixed = fix_bag(bagfile)
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


def fix_bag(bagfile):
    bagfile_fixed = add_suffix(bagfile)
    with rosbag.Bag(bagfile) as bag:
        with rosbag.Bag(bagfile_fixed, 'w') as bag_fixed:
            for topic, msg, t in tqdm(bag.read_messages()):
                bag_fixed.write(topic, msg,
                                msg.header.stamp if msg._has_header else t)
    return bagfile_fixed


def main():
    parser = ap.ArgumentParser(description='Fix timestamp of bag files')
    parser.add_argument('input', nargs='+', help='Bag file(s)')

    args = parser.parse_args()
    print(args)
    process_bags(args.input)


if __name__ == '__main__':
    sys.exit(main())
