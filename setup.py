#
# Copyright (c) 2022 Idiap Research Institute, http://www.idiap.ch/
# Written by Fabio Fehr <fabio.fehr@idiap.ch>
#

import setuptools

setuptools.setup(
    name="nvib",
    version="3.0",
    description="Nonparametric Variational Information Bottleneck",
    url="#",
    author="Fabio J Fehr",
    install_requires=["torch", "pytest"],
    author_email="fabio.fehr@idiap.ch",
    packages=setuptools.find_packages(),
    zip_safe=False,
)
