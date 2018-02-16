import tensorflow as tf


def both_valid(data):
    return tf.logical_and(tf.not_equal(data["A"]["width"], 0),
                          tf.not_equal(data["B"]["width"], 0))


def decode_image(image_data, prefix, channels):
    image = tf.image.decode_image(image_data[prefix+"encoded"], name="decode", channels=channels)
    image.set_shape([None, None, channels])

    copy = {key[len(prefix):]: image_data[key] for key in image_data if key.startswith(prefix)}
    copy["image"] = tf.cast(image, tf.float32) / 255.0
    return copy


def load_tf_records(source_file, preprocessing, shuffle=True, batch_size=32,
                    repeat_count=-1, greyscale=False, num_threads=4):
    """
    Load a tfrecords file which contains image pairs (and was created by `make_tf_records`).
    These images can be preprocessed using the `preprocessing`function. This function gets
    passed in a nested dictionary that contains a unique identifier for this image pair as `"key"`
    and the data for the two images under `"A"` and `"B"`. Each image has the attributes
    `width`, `height`, `filename`, `encoded`, `image` where image contains the image
    pixels as floats in the range [0, 1].

    :param source_file: The tfrecords file.
    :param preprocessing: Preprocessing function that gets applied to all training examples.
    :param shuffle: Whether to shuffle the training examples.
    :param batch_size: Batch size.
    :param repeat_count: Number of times to iterate over the whole dataset.
    :param greyscale: Whether the images should be decoded as greyscale or colour images.
    :param num_threads: Number of threads used by preprocessing.
    :return: A `dict` of tensors.
    """
    dataset = tf.data.TFRecordDataset(source_file)

    dataset = dataset.repeat(repeat_count)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=256)

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
               'key':  tf.FixedLenFeature([], tf.string),
           })

        channels = 1 if greyscale else 3
        data = {"key": features["key"],
                "A": decode_image(features, "A/", channels),
                "B": decode_image(features, "B/", channels)}
        return preprocessing(data)

    dataset = dataset.map(preproc, num_parallel_calls=num_threads)

    batched = dataset.batch(batch_size).prefetch(10)
    return batched.make_one_shot_iterator().get_next()
