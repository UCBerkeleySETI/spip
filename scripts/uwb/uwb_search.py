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
DL        = 2

###############################################################################
# Search Daemon that extends the UWBProcDaemon
class UWBSearchDaemon (UWBProcDaemon):

  def __init__ (self, name, id):
    UWBProcDaemon.__init__(self, name , id)

  # called once the header has been received, and must prepare the self.cmd to be run
  def prepare (self):

    self.log (1, "UTC_START=" + self.header["UTC_START"])
    self.log (1, "RESOLUTION=" + self.header["RESOLUTION"])

    # default processing commands
    self.cmd = "dada_dbnull -s -k " + self.db_key

    # check if SEARCH mode has been requested in the header
    try:
      search = (self.header["PERFORM_SEARCH"] == "1")
    except KeyError as e:
      search = False

    # if no searching has been requested return
    if not search:
      return

    # output directory for SEARCH mode
    self.out_dir = self.cfg["CLIENT_SEARCH_DIR"] + "/processing/" + self.beam + "/" \
                   + self.utc_start + "/" + self.source + "/" + self.cfreq

    # create DSPSR input file for the data block
    db_key_filename = "/tmp/spip_" + self.db_key + ".info"
    if not os.path.exists (db_key_filename):
      db_key_file = open (db_key_filename, "w")
      db_key_file.write("DADA INFO:\n")
      db_key_file.write("key " +  self.db_key + "\n")
      db_key_file.close()

    # configure the command to be run
    self.cmd = "digifits -Q " + db_key_filename + " -F 256:D -cuda " + self.gpu_id

    self.log_prefix = "search_src"

###############################################################################

if __name__ == "__main__":

  if len(sys.argv) != 2:
    print "ERROR: 1 command line argument expected"
    sys.exit(1)

  # this should come from command line argument
  stream_id = sys.argv[1]

  script = UWBSearchDaemon ("uwb_search", stream_id)

  state = script.configure (DAEMONIZE, DL, "search", "search") 
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

