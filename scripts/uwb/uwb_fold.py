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
# Fold Daemon that extends the UWBProcDaemon
class UWBFoldDaemon (UWBProcDaemon):

  def __init__ (self, name, id):
    UWBProcDaemon.__init__(self, name , id)
    self.tag = "fold"

  def getEnvironment (self):
    self.log (2, "UWBFoldDaemon::getEnvironment()")
    env = UWBProcDaemon.getEnvironment (self)
    env["TMP_USER"] = "uwb" + str(self.id)
    return env

  # called once the header has been received, and must prepare the self.cmd to be run
  def prepare (self):

    self.log (2, "UWBFoldDaemon::prepare UTC_START=" + self.header["UTC_START"])
    self.log (2, "UWBFoldDaemon::prepare RESOLUTION=" + self.header["RESOLUTION"])

    # default processing commands
    self.cmd = "dada_dbnull -s -k " + self.db_key

    # check if FOLD mode has been requested in the header
    try:
      fold = (self.header["PERFORM_FOLD"] == "1")
    except KeyError as e:
      fold = False

    # if no folding has been requested return
    if not fold:
      return False

    # output directory for FOLD mode
    self.out_dir = self.cfg["CLIENT_FOLD_DIR"] + "/processing/" + self.beam + "/" \
                   + self.utc_start + "/" + self.source + "/" + self.cfreq

    # create DSPSR input file for the data block
    db_key_filename = "/tmp/spip_" + self.db_key + ".info"
    if not os.path.exists (db_key_filename):
      db_key_file = open (db_key_filename, "w")
      db_key_file.write("DADA INFO:\n")
      db_key_file.write("key " +  self.db_key + "\n")
      db_key_file.close()

    outnstokes = -1
    outtsubint = -1
    dm = -1
    outnbin = -1
    innchan = int(self.header["NCHAN"])
    outnchan = innchan

    try:
      outnstokes = int(self.header["OUTNSTOKES"])
    except:
      outnstokes = 4

    try:
      outtsub = int(self.header["OUTTSUBINT"])
    except:
      outtsub = 10

    try:
      outnbin = int(self.header["OUTNBIN"])
    except:
      outnbin = 1024

    try:
      outnchan = int(self.header["OUTNCHAN"])
    except:
      outnchan = 0
      innchan = 0

    try:
      dm = float(self.header["DM"])
    except:
      dm = -1

    # configure the command to be run
    self.cmd = "dspsr -Q " + db_key_filename + " -minram 2048 -cuda " + self.gpu_id + " -no_dyn"

    # handle detection options
    if outnstokes == 1 or outnstokes == 2 or outnstokes == 4:
      # hack for NPOL==1
      if self.header["NPOL"] == "1":
        self.cmd = self.cmd + " -d 1"
      else:
        self.cmd = self.cmd + " -d " + str(outnstokes)
    else:
      self.log(-1, "ignoring invalid outnstokes of " + str(outnstokes))

    # handle channelisation
    if outnchan > innchan:
      if outnchan % innchan == 0:
        self.cmd = self.cmd + " -F " + str(outnchan) + ":D"
      else:
        self.log(-1, "output channelisation was not a multiple of input channelisation")
    else:
      self.log(-2, "requested output channelisation [" + str(outnchan) + "] " + \
               "less than input channelisation [" + str(innchan) + "]")

    # handle output binning
    if outnbin > 0:
      self.cmd = self.cmd + " -b " + str(outnbin)

    # subint is required
    self.cmd = self.cmd + " -L " + str(outtsub)
    mintsub = outtsub - 1
    if mintsub > 0:
      self.cmd = self.cmd + " -Lmin " + str(mintsub)

    # handle a custom DM
    if dm >= 0:
      self.cmd = self.cmd + " -D " + str(dm)

    # set a minimum kernel length : TODO remove this via CUDA benchmark
    self.cmd = self.cmd + " -x 2048"

    self.log_prefix = "fold_src"

    return True

###############################################################################

if __name__ == "__main__":

  if len(sys.argv) != 2:
    print "ERROR: 1 command line argument expected"
    sys.exit(1)

  # this should come from command line argument
  stream_id = sys.argv[1]

  script = UWBFoldDaemon ("uwb_fold", stream_id)

  state = script.configure (DAEMONIZE, DL, "fold", "fold") 
  if state != 0:
    sys.exit(state)

  script.log(1, "STARTING SCRIPT")

  try:

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

