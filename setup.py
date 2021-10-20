from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='Train a classifier on MIT AE stress data too see if AE waveforms/spectrograms can discriminate stress levels.',
    author='Nate Groebner, Strabo Analytics',
    license='MIT',
)
