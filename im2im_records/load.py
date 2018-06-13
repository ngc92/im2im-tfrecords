import tensorflow as tf


def both_valid(data):
    return tf.logical_and(tf.not_equal(data["A"]["width"], 0),
                          tf.not_equal(data["B"]["width"], 0))


def decode_image(image_data, channels):
    image = tf.image.decode_image(image_data, name="decode", channels=channels)
    image.set_shape([None, None, channels])
    return tf.image.convert_image_dtype(image, tf.float32)


def load_tf_records(source_file, preprocessing, shuffle=True, batch_size=32,
                    repeat_count=-1, greyscale=False, num_threads=4, cache=False):
    """
    Load a tfrecords file which contains image pairs (and was created by `make_tf_records`).
    These images can be preprocessed using the `preprocessing`function. This function gets
    passed in a dict of `key` and two dictionaries `A` and `B` that contain the images.
    Each image has the attributes `width`, `height`, `filename`, `encoded`, `image`
    where image contains the image pixels as floats in the range [0, 1].

    :param source_file: The tfrecords file.
    :param preprocessing: Preprocessing function that gets applied to all training examples.
    :param shuffle: Whether to shuffle the training examples.
    :param batch_size: Batch size.
    :param repeat_count: Number of times to iterate over the whole dataset.
    :param greyscale: Whether the images should be decoded as greyscale or colour images.
    :param num_threads: Number of threads used by preprocessing.
    :return: A `dict` of tensors.
    """
    dataset = tf.data.TFRecordDataset(source_file, buffer_size=1024*1024)

    def preproc(data):
        features = tf.parse_single_example(data,
           features={
               'A/width':    tf.FixedLenFeature([], tf.int64),
               'A/height':    tf.FixedLenFeature([], tf.int64),
               'A/filename': tf.FixedLenFeature([], tf.string),
               'A/encoded':  tf.FixedLenFeature([], tf.string),
               'B/width':    tf.FixedLenFeature([], tf.int64),
               'B/height':    tf.FixedLenFeature([], tf.int64),
               'B/filename': tf.FixedLenFeature([], tf.string),
               'B/encoded':  tf.FixedLenFeature([], tf.string),
               'key': tf.FixedLenFeature([], tf.string),
               'num': tf.FixedLenFeature([], tf.int64),
           })

        channels = 1 if greyscale else 3
        features["A/image"] = decode_image(features["A/encoded"], channels)
        features["B/image"] = decode_image(features["B/encoded"], channels)
        return preprocessing(features)

    dataset = dataset.repeat(repeat_count)
    if cache:
        dataset = dataset.cache()
    if shuffle:
        dataset = dataset.shuffle(buffer_size=256)
    dataset = dataset.map(preproc, num_parallel_calls=num_threads)
    batched = dataset.batch(batch_size)

    return batched.prefetch(10)
