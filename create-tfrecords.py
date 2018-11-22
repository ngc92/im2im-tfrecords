#!/usr/bin/env python

import argparse
import os
from im2im_records import make_tf_records
import logging
import re

logging.basicConfig(level=logging.INFO)


args = argparse.ArgumentParser(
    description="Generate a tfrecords dataset file from two folders of paired images")

args.add_argument("source", type=str, help="Folder for the source images.")
args.add_argument("target", type=str, help="Folder for the target images.")
args.add_argument("result", type=str, help="Filename for tfrecords file.")
args.add_argument("filter", type=str, default=r"\d+", nargs="?", help="Regexp used to extract matching files")

arguments = args.parse_args()


def make_identity(regexp):
    def identity(file_name):
        """
        Assumes a filename is built as path/basename_IDENTITY.extension and extracts IDENTITY.
        """
        _, base_name = os.path.split(file_name)
        base_name, _ = os.path.splitext(base_name)
        all = re.findall(regexp, base_name)
        if len(all) != 1:
            return ""
        else:
            return all[0]

    return identity


logging.info("Creating tfrecords file for {} -> {}".format(arguments.source, arguments.target))
make_tf_records(arguments.result, arguments.source, arguments.target, make_identity(arguments.filter))
