#!/usr/bin/env python

##############################################################################
#  
#     Copyright (C) 2015 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
# 
###############################################################################

from spip.config import Config

class UWBConfig(Config):

  def __init__ (self):
    Config.__init__(self)

  def getStreamConfigFixed (self, id):

    cfg = Config.getStreamConfigFixed (self, id)

    (cfg["DATA_HOST"], cfg["DATA_PORT"]) = self.config["STREAM_UDP_" + str(id)].split(":")
    cfg["UDP_NSAMP"] = "2048"

    return cfg

