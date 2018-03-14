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
    innchan = self.header["NCHAN"]
    outnchan = innchan
    outnstokes = 1
    outnbit = 8

    try:
      outtsubint = int(self.header["OUTTSUBINT"])
    except:
      outtsubint = 10

    try:
      outnstokes = int(self.header["OUTNSTOKES"])
    except:
      outnstokes = 1

    try:
      outnbit = int(self.header["OUTNBIT"])
    except:
      outnbit = 8

    try:
      outtdec = int(self.header["OUTTDEC"])
    except:
      outtdec = 32

    try:
      outnchan = int(self.header["OUTNCHAN"])
    except:
      outnchan = 0
      innchan = 0

    try:
      dm = float(self.header["DM"])
    except:
      dm = -1

    nsblk = 1024

    # configure the command to be run
    self.cmd = "digifits -Q " + db_key_filename + " -cuda " + self.gpu_id + " -nsblk " + str(nsblk)

    # handle detection options
    if outnstokes == 1 or outnstokes == 2 or outnstokes == 4:
      self.cmd = self.cmd + " -p " + str(outnstokes)
    else:
      self.log(-1, "ignoring invalid outnstokes of " + str(outnstokes))

    # handle channelisation
    if outnchan > innchan:
      if outnchan % innchan == 0:
        self.cmd = self.cmd + " -F " + str(outnchan) + ":D"
      else:
        self.log(-1, "Invalid output channelisation")

    # handle output digitization
    if outnbit == 1 or outnbit == 2 or outnbit == 4 or outnbit == 8:
      self.cmd = self.cmd + " -b " + str(outnbit)

    # handle temporal integration
    out_tsamp = (float(self.header["TSAMP"]) * outtdec) / 1000000
    self.cmd = self.cmd +  " -t " + str(out_tsamp)

    # handle output sub-int length, need lots of precision
    block_length_seconds = out_tsamp * nsblk
    blocks_per_subint = int(math.floor(outtsubint / block_length_seconds))
    subint_length_seconds = block_length_seconds * blocks_per_subint
    self.cmd = self.cmd + " -L " + format(subint_length_seconds, ".16f")

    # handle a custom DM
    if dm >= 0:
      self.cmd = self.cmd + " -do_dedisp true -D " + str(dm)
      # set a minimum kernel length
      self.cmd = self.cmd + " -x 2048"

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

    script.main ()

  except:

    script.log(-2, "exception caught: " + str(sys.exc_info()[0]))
    print '-'*60
    traceback.print_exc(file=sys.stdout)
    print '-'*60
    script.quit_event.set()

  script.conclude()
  sys.exit(0)

