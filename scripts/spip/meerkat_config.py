#!/usr/bin/env python

##############################################################################
#  
#     Copyright (C) 2015 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
# 
###############################################################################

from spip.config import Config

class MeerKATConfig(Config):

  def __init__ (self):
    Config.__init__(self)

  def getMuxedStreamConfigFixed (self, id):

    cfg = Config.getStreamConfigFixed (self, id)

    #cfg["DATA_HOST"]  = self.config["DATA_HOST_" + id]
    #cfg["DATA_MCAST"] = self.config["DATA_MCAST_" + id]
    #cfg["DATA_PORT"]  = self.config["DATA_PORT_" + id]
    #cfg["META_HOST"]  = self.config["META_HOST_" + id]
    #cfg["META_MCAST"] = self.config["META_MCAST_" + id]
    #cfg["META_PORT"]  = self.config["META_PORT_" + id]

    (cfg["DATA_HOST_0"], cfg["DATA_HOST_1"]) = self.config["DATA_HOST"].split(",")
    (cfg["DATA_MCAST_0"], cfg["DATA_MCAST_1"]) = self.config["DATA_MCAST"].split(",")
    (cfg["DATA_PORT_0"], cfg["DATA_PORT_1"]) = self.config["DATA_PORT"].split(",")

    (cfg["META_HOST_0"], cfg["META_HOST_1"]) = self.config["META_HOST"].split(",")
    (cfg["META_MCAST_0"], cfg["META_MCAST_1"]) = self.config["META_MCAST"].split(",")
    (cfg["META_PORT_0"], cfg["META_PORT_1"]) = self.config["META_PORT"].split(",")

    cfg["ADC_SAMPLE_RATE"] = self.config["ADC_SAMPLE_RATE"]

    (freq1, bw1, nchan1) = self.config["SUBBAND_CONFIG_0"].split(":")
    (freq2, bw2, nchan2) = self.config["SUBBAND_CONFIG_1"].split(":")
 
    freq = (float(freq1) + float(freq2)) / 2
    bw = float(bw1) + float(bw2)
    nchan = int(nchan1) + int(nchan2)
    cfg["FREQ"] = str(freq)
    cfg["BW"] = str(bw)
    cfg["NCHAN"] = str(nchan)
    cfg["NPOL"] = "2"

    (start_chan1, end_chan1) = self.config["SUBBAND_CHANS_0"].split(":")
    (start_chan2, end_chan2) = self.config["SUBBAND_CHANS_1"].split(":")
    cfg["START_CHANNEL"] = start_chan1
    cfg["END_CHANNEL"]   = end_chan2
    
    return cfg

  def getStreamConfigFixed (self, id):

    cfg = Config.getStreamConfigFixed (self, id)

    (cfg["DATA_HOST_0"], cfg["DATA_HOST_1"]) = self.config["DATA_HOST"].split(",")
    (cfg["DATA_MCAST_0"], cfg["DATA_MCAST_1"]) = self.config["DATA_MCAST"].split(",")
    (cfg["DATA_PORT_0"], cfg["DATA_PORT_1"]) = self.config["DATA_PORT"].split(",")

    (cfg["META_HOST_0"], cfg["META_HOST_1"]) = self.config["META_HOST"].split(",")
    (cfg["META_MCAST_0"], cfg["META_MCAST_1"]) = self.config["META_MCAST"].split(",")
    (cfg["META_PORT_0"], cfg["META_PORT_1"]) = self.config["META_PORT"].split(",")

    cfg["ADC_SAMPLE_RATE"] = self.config["ADC_SAMPLE_RATE"]

    (freq, bw, nchan) = self.config["SUBBAND_CONFIG_" + str(id)].split(":")

    cfg["FREQ"] = freq
    cfg["BW"] = bw
    cfg["NCHAN"] = nchan
    cfg["NPOL"] = "2"

    (start_chan, end_chan) = self.config["SUBBAND_CHANS_" + str(id)].split(":")
    cfg["START_CHANNEL"] = start_chan
    cfg["END_CHANNEL"]   = end_chan

    return cfg

