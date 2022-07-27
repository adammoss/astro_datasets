#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import find_packages, setup


def readfile(filename):
    with open(filename, encoding="utf-8") as fp:
        filecontents = fp.read()
    return filecontents

setup(
    name="astro_datasets",
    version="0.0.8",
    description="Astronomy and cosmology tensorflow datasets",
    author="Adam Moss",
    author_email="adam.moss@nottingham.ac.uk",
    maintainer="Adam Moss",
    maintainer_email="adam.moss@nottingham.ac.uk",
    url="https://github.com/adammoss/astro_datasets/",
    license="MIT",
    packages=find_packages(),
    provides=["astro_datasets"],
    install_requires=readfile(
        os.path.join(os.path.dirname(__file__), "requirements.txt")
    )
)
