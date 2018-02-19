import functools
import tensorflow as tf


class Preprocessor:
    def __init__(self, f):
        self.f = f

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)

    def __or__(self, g):
        f = self.f

        @functools.wraps(f)
        def chained(*args, **kwargs):
            return g(f(*args, **kwargs))
        return Preprocessor(chained)

    def apply_to(self, feature):
        return Preprocessor(apply_to_feature(self.f, feature))


def apply_to_feature(func, feature):
    @functools.wraps(func)
    def apply_to_feature_(data):
        data[feature] = func(data[feature])
        return data
    return apply_to_feature_


def random_crop(crop_size, pad=0, seed=None):
    def random_crop_(image):
        channels = image.shape[2]
        if pad != 0:
            image = tf.pad(image, [(pad, pad), (pad, pad), (0, 0)])
        cropped = tf.random_crop(image, [crop_size, crop_size, channels], seed=seed)
        return cropped
    return Preprocessor(random_crop_)


def random_flips(horizontal=True, vertical=False, seed=None):
    def augment_with_flips_(image):
        with tf.name_scope("augment_with_flips", [image]):
            if horizontal:
                image = tf.image.random_flip_left_right(image, seed=seed)
            if vertical:
                image = tf.image.random_flip_up_down(image, seed=None if seed is None else seed+1)
            return image
    return Preprocessor(augment_with_flips_)


def random_rotations(seed=None):
    def augment_with_rotations_(image):
        with tf.name_scope("augment_with_rotation", [image]):
            rotate = tf.random_uniform((), 0, 4, tf.int32, seed=seed)
            return tf.image.rot90(image, rotate)

    return Preprocessor(augment_with_rotations_)


nothing = Preprocessor(lambda x: x)
