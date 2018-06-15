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


def _id_with_warning(id_fn, name):
    identity = id_fn(name)
    if identity == "":
        _logger.warning("Skipping file %s because no unique identity could be established to find matching files", name)
    return identity


def iterate_corresponding_examples(source_iter, target_iter, identity_fn):
    # get files in the folders and associate file path with an identity
    # TODO check for duplicate identities
    files_A = {_id_with_warning(identity_fn, name): name for name in source_iter}
    files_B = {_id_with_warning(identity_fn, name): name for name in target_iter}

    # if we had unidentifiable files, remove them here
    if "" in files_A:
        del files_A[""]
    if "" in files_B:
        del files_B[""]

    all_identities = set(files_A.keys()) | set(files_B.keys())

    for example in all_identities:
        if not files_A.get(example):
            _logger.info("Skipping training example %s for file %s because corresponding source file is missing",
                         example, files_B.get(example))
            continue

        if not files_B.get(example):
            _logger.info("Skipping training example %s for file %s because corresponding target file is missing",
                         example, files_A.get(example))
            continue

        yield example, files_A[example], files_B[example]


def make_training_examples(source_folder, target_folder, identity, filter_fn=None):
    from glob import iglob
    # get files in the folders and associate file path with an identity
    # TODO check for duplicate identities
    files_A = iglob(os.path.join(source_folder, "*"))
    files_B = iglob(os.path.join(target_folder, "*"))

    counter = 0

    for key, path_A, path_B in iterate_corresponding_examples(files_A, files_B, identity):
        if filter_fn and not filter_fn(key):
            continue

        A = image_example(path_A)
        B = image_example(path_B)

        # we need to flatten the dict here in order to write it to tfrecords
        feature_dict = {"A/"+key: A[key] for key in A}
        feature_dict.update({"B/"+key: B[key] for key in B})
        feature_dict["key"] = bytes_feature(key.encode("utf8"))
        feature_dict["num"] = int_feature(counter)
        counter += 1

        _logger.info("Adding training example %s for files %s and %s",
                     key, path_A, path_B)
        yield tf.train.Example(features=tf.train.Features(feature=feature_dict))


def make_tf_records(target_file, source_folder, target_folder, identity, filter_fn=None):
    """
    Make a tfrecords file for source images from `source_folder` mapped to target images in `target_folder`.
    Two images `A` and `B` are assumed to correspond if `identity(A) == identity(B)`. Images for which no
    corresponding source (or target resp) exists are ignored.
    :param target_file: Path to where the tfrecords file is saved.
    :param source_folder: Path to source images.
    :param target_folder: Path to target images.
    :param identity: Function that maps a file name to an identifier which can be used to find out which files from
                     `source_folder` correspond to those in `target_folder`.
    :param filter_fn: An optional function that can be passed and that determines (based on the example key) whether to
                    include that example in the records file.
    :return: Nothing.
    """
    writer = tf.python_io.TFRecordWriter(target_file)

    try:
        for c, example in enumerate(make_training_examples(source_folder, target_folder, identity, filter_fn)):
            writer.write(example.SerializeToString())
        _logger.info("Processed %d examples", c)
    finally:
        writer.close()

