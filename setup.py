from setuptools import setup

setup(
    name='im2im-tfrecords',
    version='0.1.0',
    packages=['im2im_records'],
    url='',
    license='',
    author='Erik Schultheis',
    author_email='',
    description='Tools for creating and loading tfrecords files form image to image models.',
    install_requires=['Pillow'],
    scripts=['create-tfrecords.py']
)
