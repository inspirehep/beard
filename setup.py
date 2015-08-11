# -*- coding: utf-8 -*-
#
# This file is part of Beard.
# Copyright (C) 2014, 2015 CERN.
#
# Beard is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""Setup file for Beard.

.. codeauthor:: Mateusz Susik <mateusz.susik@cern.ch>
.. codeauthor:: Jan Aage Lavik <jan.age.lavik@cern.ch>

"""

from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
import os
import re
import sys


class PyTest(TestCommand):

    """Handle ``python setup.py test``."""

    user_options = [("pytest-args=", "a", "Arguments to pass to py.test")]

    def initialize_options(self):
        """Read options from ``pytest.ini`` config file."""
        TestCommand.initialize_options(self)
        try:
            from ConfigParser import ConfigParser
        except ImportError:
            from configparser import ConfigParser
        config = ConfigParser()
        config.read("pytest.ini")
        self.pytest_args = config.get("pytest", "addopts").split(" ")

    def finalize_options(self):
        """Finalize options."""
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        """Run tests using pytest library."""
        # import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


packages = find_packages(exclude=['doc'])
# Get the version string. Cannot be done with import!
with open(os.path.join("beard", "__init__.py"), "rt") as f:
    _version = re.search(
        '__version__\s*=\s*"(?P<version>.*)"\n',
        f.read()
    ).group("version")

_classifiers = [
    # clasifiers for PyPI
    "Development Status :: 1 - Planning",
    "Environment :: Console",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: BSD License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Programming Language :: Python :: 2",
    "Programming Language :: Python :: 2.7",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.4",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Information Analysis"
]

_keywords = [
    "author disambiguation",
    "machine learning",
    "data mining"
]

_install_requires = [
    "jellyfish",
    "numpy>=1.9",
    "scikit-learn>=0.15.2",
    "scipy>=0.14",
    "six",
    "unidecode"
]

if sys.version[0] == '2':
    # fuzzy package is not available on Python 3
    _install_requires.append("fuzzy")

_tests_require = [
    "pytest",
    "pytest-cache>=1.0",
    "pytest-cov>=1.8.0",
    "pytest-pep8>=1.0.6",
    "pytest>=2.6.1",
    "coverage"
]

_parameters = {
    "author": "Invenio collaboration",
    "author_email": "beard@invenio-software.org",
    "classifiers": _classifiers,
    "cmdclass": {"test": PyTest},
    "description": "Bibliographic Entity Automatic \
        Recognition and Disambiguation",
    "install_requires": _install_requires,
    "keywords": _keywords,
    "license": "BSD",
    "long_description": open("README.rst").read(),
    "name": "beard",
    "packages": packages,
    "platforms": "any",
    "tests_require": _tests_require,
    "url": "https://github.com/inveniosoftware/beard",
    "version": _version,
}

setup(**_parameters)
