#!/usr/bin/env python

##############################################################################
#  
#     Copyright (C) 2015 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
# 
###############################################################################

import os, threading, sys, time, socket, select, traceback

from spip.daemons.bases import StreamBased
from spip.daemons.daemon import Daemon
from spip.log_socket import LogSocket
from spip.config import Config
from spip_smrb import SMRBDaemon
from spip.utils.core import system_piped,system
from spip.utils.sockets import getHostNameShort
from uwb_proc import UWBProcThread, UWBProcDaemon

DAEMONIZE = True
DL        = 3

###############################################################################
# Baseband Daemon that extends the UWBProcDaemon
class UWBBasebandDaemon (UWBProcDaemon):

  def __init__ (self, name, id):
    UWBProcDaemon.__init__(self, name , id)

  # called once the header has been received, and must prepare the self.cmd to be run
  def prepare (self):

    utc_start = self.header["UTC_START"]
    self.log (1, "UTC_START=" + self.header["UTC_START"])
    self.log (1, "RESOLUTION=" + self.header["RESOLUTION"])

    # default processing commands
    self.cmd = "dada_dbnull -s -k " + self.db_key

    # check if BASEBAND mode has been requested in the header
    try:
      baseband = (self.header["PERFORM_BASEBAND"] == "1")
    except KeyError as e:
      baseband = False
    
    self.log (1, "always baseband == True")
    baseband = True

    # if no has been requested return
    if not baseband:
      return

    beam = self.cfg["BEAM_" + str(self.beam_id)]
    source = self.header["SOURCE"]

    # output directory for BASEBAND mode
    self.out_dir = self.cfg["CLIENT_RECORDING_DIR"] + "/processing/" + beam + "/" + utc_start + "/" + source + "/" + self.cfreq

    # configure the command to be run
    self.cmd = "dada_dbdisk -k " + self.db_key + " -s -z -D " + self.out_dir

    self.log_prefix = "baseband_src"

###############################################################################

if __name__ == "__main__":

  if len(sys.argv) != 2:
    print "ERROR: 1 command line argument expected"
    sys.exit(1)

  # this should come from command line argument
  stream_id = sys.argv[1]

  script = UWBBasebandDaemon ("uwb_baseband", stream_id)

  state = script.configure (DAEMONIZE, DL, "baseband", "baseband") 
  if state != 0:
    sys.exit(state)

  try:

    script.main ()

  except:

    script.log(-2, "exception caught: " + str(sys.exc_info()[0]))
    print '-'*60
    traceback.print_exc(file=sys.stdout)
    print '-'*60
    script.quit_event.set()

  script.conclude()
  sys.exit(0)

