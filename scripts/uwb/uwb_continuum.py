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
DL        = 1

###############################################################################
# Continuum Daemon that extends the UWBProcDaemon
class UWBContinuumDaemon (UWBProcDaemon):

  def __init__ (self, name, id):
    UWBProcDaemon.__init__(self, name , id)
    self.tag = "continuum"

  # called once the header has been received, and must prepare the self.cmd to be run
  def prepare (self):

    self.log (2, "UWBContinuumDaemon::prepare UTC_START=" + self.header["UTC_START"])
    self.log (2, "UWBContinuumDaemon::prepare RESOLUTION=" + self.header["RESOLUTION"])

    # default processing commands
    self.cmd = "dada_dbnull -s -k " + self.db_key

    # check if CONTINUUM mode has been requested in the header
    try:
      continuum = (self.header["PERFORM_CONTINUUM"] == "1")
    except KeyError as e:
      continuum = False

    # if no continuuming has been requested return
    if not continuum:
      return continuum

    # output directory for CONTINUUM mode
    self.out_dir = self.cfg["CLIENT_CONTINUUM_DIR"] + "/processing/" + self.beam + "/" \
                   + self.utc_start + "/" + self.source + "/" + self.cfreq

    out_tsamp = -1
    out_tsubint = -1
    out_npol = -1
    out_nchan = -1
    in_nchan = int(self.header["NCHAN"])

    try:
      out_tsamp = float(self.header["CONTINUUM_OUTTSAMP"])
    except:
      out_tsamp = 1

    try:
      out_tsubint = int(self.header["CONTINUUM_OUTTSUBINT"])
    except:
      out_tsubint = 10

    try:
      out_npol = int(self.header["CONTINUUM_OUTNPOL"])
    except:
      out_npol = 1

    try:
      out_nchan = int(self.header["CONTINUUM_OUTNCHAN"])
    except:
      out_nchan = 1024

    # configure the command to be run
    self.cmd = "uwb_continuum_pipeline " + self.db_key + " " + self.out_dir + " -d " + self.gpu_id

    # handle detection options
    if out_npol == 1 or out_npol == 2 or out_npol == 4:
      self.cmd = self.cmd + " -p " + str(out_npol)
    else:
      self.log(-1, "ignoring invalid outnstokes of " + str(outnstokes))

    # handle channelisation
    if out_nchan > in_nchan:
      if out_nchan % in_nchan == 0:
        self.cmd = self.cmd + " -n " + str(out_nchan)
      else:
        self.log(-1, "Invalid output channelisation")

    # handle temporal integration and sub-integration length
    self.cmd = self.cmd + " -t " + str(out_tsamp) + " -L " + str(out_tsubint)

    self.log(1, self.cmd)
    self.log_prefix = "continuum_src"
    
    return True

###############################################################################

if __name__ == "__main__":

  if len(sys.argv) != 2:
    print "ERROR: 1 command line argument expected"
    sys.exit(1)

  # this should come from command line argument
  stream_id = sys.argv[1]

  script = UWBContinuumDaemon ("uwb_continuum", stream_id)

  state = script.configure (DAEMONIZE, DL, "continuum", "continuum") 
  if state != 0:
    sys.exit(state)

  try:

    script.log(1, "STARTING SCRIPT")
    script.main ()

  except:

    script.log(-2, "exception caught: " + str(sys.exc_info()[0]))
    print '-'*60
    traceback.print_exc(file=sys.stdout)
    print '-'*60
    script.quit_event.set()

  script.log(1, "STOPPING SCRIPT")
  script.conclude()
  sys.exit(0)

