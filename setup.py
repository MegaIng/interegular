from pathlib import Path

from setuptools import setup
import re

__version__, = re.findall('__version__ = "(.*)"', open('interegular/__init__.py').read())

with open(Path(__file__).with_name('README.md')) as f:
    long = f.read()

setup(
    name='interegular',
    version=__version__,
    packages=['interegular', 'interegular.utils'],
    package_data={'interegular': ['py.typed']},
    install_requires=['dataclasses; python_version<"3.7"'],
    python_requires=">=3.6",
    author='MegaIng',
    author_email='MegaIng <trampchamp@hotmail.de>',
    description="a regex intersection checker",
    long_description=long,
    long_description_content_type='text/markdown',
    license="MIT",
    url='https://github.com/MegaIng/regex_intersections',
    download_url='https://github.com/MegaIng/interegular/tarball/master',
    classifiers=[
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
