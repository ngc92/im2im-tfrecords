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
        data = {"key": features["key"]}
        data["A"] = decode_image(features, "A/", channels)
        data["B"] = decode_image(features, "B/", channels)
        return preprocessing(data)

    dataset = dataset.map(preproc, num_parallel_calls=num_threads)

    batched = dataset.batch(batch_size).prefetch(10)
    return batched.make_one_shot_iterator().get_next()
