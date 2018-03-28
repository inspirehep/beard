#!/bin/bash
# This file is part of Beard.
# Copyright (C) 2016 CERN.
#
# Beard is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

set -e

check-manifest --ignore miniconda.sh
python setup.py test
