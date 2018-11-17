#!/usr/bin/env python

##############################################################################
#
#     Copyright (C) 2018 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
#
###############################################################################

from os.path import getmtime

from spip.utils.times import getCurrentTimeUnix


def get_file_age(filename):
    curr_time_unix = getCurrentTimeUnix()
    file_mtime_unix = getmtime(filename)
    age = int(curr_time_unix) - int(file_mtime_unix)
    return age
