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

DAEMONIZE = False
DL        = 3

###############################################################################
# Transients Daemon that extends the UWBProcDaemon
class UWBTransientsDaemon (UWBProcDaemon):

  def __init__ (self, name, id):
    UWBProcDaemon.__init__(self, name , id)

  # called once the header has been received, and must prepare the self.cmd to be run
  def prepare (self):

    utc_start = header["UTC_START"]
    self.log (1, "UTC_START=" + header["UTC_START"])
    self.log (1, "RESOLUTION=" + header["RESOLUTION"])

    # default processing commands
    self.cmd = "dada_dbnull -s -k " + db_key_in

    if prev_utc_start == utc_start:
      self.log (-2, "UTC_START [" + utc_start + "] repeated, ignoring observation")
      return

    # check if TRANSIENTS mode has been requested in the header
    try:
      transients = (header["PERFORM_TRANSIENTS"] == "1")
    except KeyError as e:
      transients = False

    # if no has been requested return
    if not transients:
      return

    beam = self.cfg["BEAM_" + str(self.beam_id)]

    if not float(bw) == float(header["BW"]):
      self.log (-1, "configured bandwidth ["+bw+"] != header["+header["BW"]+"]")
    if not float(cfreq) == float(header["FREQ"]):
      self.log (-1, "configured cfreq ["+cfreq+"] != header["+header["FREQ"]+"]")
    if not int(nchan) == int(header["NCHAN"]):
      self.log (-2, "configured nchan ["+nchan+"] != header["+header["NCHAN"]+"]")

    source = header["SOURCE"]

    # output directory for TRANSIENTS mode
    self.out_dir = self.cfg["CLIENT_TRANSIENTS_DIR"] + "/processing/" + beam + "/" + utc_start + "/" + source + "/" + self.cfreq

    # configure the command to be run, TODO work out configuration
    # self.cmd = ""

    self.log_prefix = "transients_src"

###############################################################################

if __name__ == "__main__":

  if len(sys.argv) != 2:
    print "ERROR: 1 command line argument expected"
    sys.exit(1)

  # this should come from command line argument
  stream_id = sys.argv[1]

  script = UWBTransientsDaemon ("uwb_transients", stream_id)

  state = script.configure (DAEMONIZE, DL, "transients", "transients") 
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

