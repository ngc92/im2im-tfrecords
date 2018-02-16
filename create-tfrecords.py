#!/usr/bin/env python

import argparse
import os
from im2im_records import make_tf_records
import logging

logging.basicConfig(level=logging.INFO)


args = argparse.ArgumentParser(
    description="Generate a tfrecords dataset file from two folders of paired images")

args.add_argument("source", type=str, help="Folder for the source images.")
args.add_argument("target", type=str, help="Folder for the target images.")
args.add_argument("result", type=str, help="Folder for the target images.")

arguments = args.parse_args()


def identity(file_name):
    """
    Assumes a filename is built as path/basename_IDENTITY.extension and extracts IDENTITY.
    """
    _, base_name = os.path.split(file_name)
    base_name, _ = os.path.splitext(base_name)
    name, _, index = base_name.partition("_")
    return index


make_tf_records(arguments.result, arguments.source, arguments.target, identity)
