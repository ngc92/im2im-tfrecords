import PIL.Image
import tensorflow as tf
import os
import io
import logging


_logger = logging.getLogger(__name__)


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def int_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def image_example(file_path):
    filename = os.path.split(file_path)[1]
    with tf.gfile.GFile(file_path, 'rb') as fid:
        encoded = fid.read()
    image = PIL.Image.open(io.BytesIO(encoded))

    return {
        "filename": bytes_feature(filename.encode("utf8")),
        "height": int_feature(image.width),
        "width": int_feature(image.height),
        "encoded": bytes_feature(encoded)
    }


def empty_image():
    return {
        "filename": bytes_feature("".encode("utf8")),
        "height": int_feature(0),
        "width": int_feature(0),
        "encoded": bytes_feature(b"")
    }


def make_training_examples(source_folder, target_folder, identity):
    from glob import iglob

    # get files in the folders and associate file path with an identity
    files_A = {identity(name): name for name in iglob(os.path.join(source_folder, "*"))}
    files_B = {identity(name): name for name in iglob(os.path.join(target_folder, "*"))}

    all_identities = set(files_A.keys()) | set(files_B.keys())

    for example in all_identities:
        if files_A.get(example):
            A = image_example(files_A[example])
        else:
            _logger.info("Skipping training example %s for file %s because corresponding source file is missing",
                         example, files_B.get(example))
            continue

        if files_B.get(example):
            B = image_example(files_B[example])
        else:
            _logger.info("Skipping training example %s for file %s because corresponding target file is missing",
                         example, files_A.get(example))
            continue

        # we need to flatten the dict here in order to write it to tfrecords
        feature_dict = {"A/"+key: A[key] for key in A}
        feature_dict.update({"B/"+key: B[key] for key in B})
        feature_dict["key"] = bytes_feature(example.encode("utf8"))

        _logger.info("Adding training example %s for files %s and %s",
                     example, files_A[example], files_B[example])
        yield tf.train.Example(features=tf.train.Features(feature=feature_dict))


def make_tf_records(target_file, source_folder, target_folder, identity):
    """
    Make a tfrecords file for source images from `source_folder` mapped to target images in `target_folder`.
    Two images `A` and `B` are assumed to correspond if `identity(A) == identity(B)`. Images for which no
    corresponding source (or target resp) exists are ignored.
    :param target_file: Path to where the tfrecords file is saved.
    :param source_folder: Path to source images.
    :param target_folder: Path to target images.
    :param identity: Function that maps a file name to an identifier which can be used to find out which files from
                     `source_folder` correspond to those in `target_folder`.
    :return: Nothing.
    """
    writer = tf.python_io.TFRecordWriter(target_file)

    for example in make_training_examples(source_folder, target_folder, identity):
        writer.write(example.SerializeToString())

    writer.close()

