from setuptools import setup, find_packages

setup(
    name='TheAmazingProc',
    url='https://github.com/andnp/TheAmazingProc.git',
    author='Andy Patterson',
    author_email='andnpatterson@gmail.com',
    packages=find_packages(exclude=['tests*']),
    install_requires=[
        'numpy<1.21.0,>=1.17',
        'numba>=0.52.0',
    ],
    version='0.0.0',
    license='MIT',
    description='',
    long_description='todo',
)
