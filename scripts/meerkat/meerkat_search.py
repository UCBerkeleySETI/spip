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
from meerkat_proc import MEERKATProcThread, MEERKATProcDaemon

DAEMONIZE = True
DL        = 1

###############################################################################
# Search Daemon that extends the MEERKATProcDaemon
class MEERKATSearchDaemon (MEERKATProcDaemon):

  def __init__ (self, name, id):
    MEERKATProcDaemon.__init__(self, name , id)

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
      return False

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

    outnpol = -1
    outtsubint = -1
    dm = -1
    innchan = int(self.header["NCHAN"])
    outnchan = innchan
    outnbit = -1

    try:
      outtsubint = int(self.header["SEARCH_OUTTSUBINT"])
    except:
      outtsubint = 10
      self.log(-1, "SEARCH_OUTTSUBINT not present in header, assuming " + str(outtsubint))
      self.header["SEARCH_OUTTSUBINT"] = str(outtsubint)

    try:
      outnpol = int(self.header["SEARCH_OUTNPOL"])
    except:
      outnpol = 1
      self.log(-1, "SEARCH_OUTNPOL not present in header, assuming " + str(outnpol))
      self.header["SEARCH_OUTNPOL"] = str(outnpol)

    try:
      outnbit = int(self.header["SEARCH_OUTNBIT"])
    except:
      outnbit = 8
      self.log(-1, "SEARCH_OUTNBIT not present in header, assuming " + str(outnbit))
      self.header["SEARCH_OUTNBIT"] = str(outnbit)

    try:
      outtdec = int(self.header["SEARCH_OUTTDEC"])
    except:
      outtdec = 32
      self.log(-1, "SEARCH_OUTTDEC not present in header, assuming " + str(outtdec))
      self.header["SEARCH_OUTTDEC"] = str(outtdec)

    try:
      outnchan = int(self.header["SEARCH_OUTNCHAN"]) / 2
    except: 
      outnchan = innchan
      self.log(-1, "SEARCH_OUTNCHAN not present in header, assuming " + str(outnchan))
      self.header["SEARCH_OUTNCHAN"] = str(outnchan)

    try:
      dm = float(self.header["SEARCH_DM"])
    except:
      dm = -1
      self.log(-1, "SEARCH_DM not present in header, assuming " + str(dm))
      self.header["SEARCH_DM"] = str(dm)

    # this seems to be a good default
    nsblk = 1024

    # configure the command to be run
    self.cmd = "digifits -Q " + db_key_filename + " -cuda " + self.gpu_id + " -nsblk " + str(nsblk)
 
    # handle detection options
    if outnpol == 1 or outnpol == 2 or outnpol == 4:
      self.cmd = self.cmd + " -p " + str(outnpol)
    else:
      self.log(-1, "ignoring invalid outnpol of " + str(outnpol))

    # handle channelisation
    if outnchan > innchan:
      if outnchan % innchan == 0:
        self.cmd = self.cmd + " -F " + str(outnchan) + ":D"

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

      # use a relatively small kernel length 
      bw = float(self.header["BW"])
      freq = float(self.header["FREQ"])
      (rval, opt_kernel_length) = self.get_optimal_kernel_length (dm, innchan, bw, freq)
      if rval == 0:
        self.log(1, "opt_kernel_length=" + str(opt_kernel_length))
        if opt_kernel_length < 1024:
          self.log(1, "overriding kernel length to 1024")
          opt_kernel_length = 1024
        self.cmd = self.cmd + " -x " + str(opt_kernel_length)
      else:
        self.log(-1, "get_optimal_kernel_length failed, using default kernel length")

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

  script = MEERKATSearchDaemon ("meerkat_search", stream_id)
  script.cpu_list = "-1"

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

