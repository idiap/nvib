#
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Fabio Fehr <fabio.fehr@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

import setuptools

setuptools.setup(
    name="nvib",
    version="0.2",
    description="Nonparametric Variational Information Bottleneck layer for Transformers",
    url="#",
    author="Fabio J Fehr",
    install_requires=["torch"],
    author_email="fabio.fehr@idiap.ch",
    packages=setuptools.find_packages(),
    zip_safe=False,
)
