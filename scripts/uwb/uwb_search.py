#!/usr/bin/env python

##############################################################################
#  
#     Copyright (C) 2015 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
# 
###############################################################################

import os, threading, sys, time, socket, select, traceback, math

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
# Search Daemon that extends the UWBProcDaemon
class UWBSearchDaemon (UWBProcDaemon):

  def __init__ (self, name, id):
    UWBProcDaemon.__init__(self, name , id)
    self.tag = "search"

  # called once the header has been received, and must prepare the self.cmd to be run
  def prepare (self):

    self.log (2, "UWBSearchDaemon::prepare UTC_START=" + self.header["UTC_START"])
    self.log (2, "UWBSearchDaemon::prepare RESOLUTION=" + self.header["RESOLUTION"])

    # default processing commands
    self.cmd = "dada_dbnull -s -k " + self.db_key

    # check if SEARCH mode has been requested in the header
    try:
      search = (self.header["PERFORM_SEARCH"] == "1")
    except KeyError as e:
      search = False

    # if no searching has been requested return
    if not search:
      return search

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

    out_npol = -1
    out_tsubint = -1
    out_tsamp = -1
    dm = -1
    in_nchan = int(self.header["NCHAN"])
    out_nchan = in_nchan
    out_npol = 1
    out_nbit = 8

    try:
      out_tsubint = int(self.header["SEARCH_OUTTSUBINT"])
    except:
      out_tsubint = 10

    try:
      out_npol = int(self.header["SEARCH_OUTNPOL"])
    except:
      out_npol = 1

    try:
      out_nbit = int(self.header["SEARCH_OUTNBIT"])
    except:
      out_nbit = 8

    try:
      out_tsamp = int(self.header["SEARCH_OUTTSAMP"])
    except:
      out_tsamp = 64

    try:
      out_nchan = int(self.header["SEARCH_OUTNCHAN"])
    except:
      out_nchan = 0
      in_nchan = 0

    try:
      dm = float(self.header["SEARCH_DM"])
    except:
      dm = -1

    nsblk = 1024

    # configure the command to be run
    self.cmd = "digifits -Q " + db_key_filename + " -cuda " + self.gpu_id + " -nsblk " + str(nsblk)

    # handle detection options
    if out_npol == 1 or out_npol == 2 or out_npol == 4:
      # hack for NPOL==1
      if self.header["NPOL"] == "1":
        self.cmd = self.cmd + " -p 1"
      else:
        self.cmd = self.cmd + " -p " + str(out_npol)
    else:
      self.log(-1, "ignoring invalid out_npol of " + str(out_npol))

    # handle channelisation
    if out_nchan > in_nchan:
      if out_nchan % in_nchan == 0:
        self.cmd = self.cmd + " -F " + str(out_nchan) + ":1024"
      else:
        self.log(-1, "Invalid output channelisation")

    # handle output digitization
    if out_nbit == 1 or out_nbit == 2 or out_nbit == 4 or out_nbit == 8:
      self.cmd = self.cmd + " -b " + str(out_nbit)

    # handle temporal integration
    out_tsamp_secs = float(out_tsamp) / 1000000
    self.cmd = self.cmd +  " -t " + str(out_tsamp_secs)

    # handle output sub-int length, need lots of precision
    block_length_seconds = out_tsamp_secs * nsblk
    blocks_per_subint = int(math.floor(out_tsubint / block_length_seconds))
    subint_length_seconds = block_length_seconds * blocks_per_subint
    self.cmd = self.cmd + " -L " + format(subint_length_seconds, ".16f")

    # handle a custom DM
    if dm >= 0:
      self.cmd = self.cmd + " -do_dedisp true -D " + str(dm)
      # set a minimum kernel length
      self.cmd = self.cmd + " -x 2048"

    self.log(1, self.cmd)
    self.log_prefix = "search_src"
    
    return True

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

