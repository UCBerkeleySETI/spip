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
from meerkat_proc import MEERKATProcThread, MEERKATProcDaemon

DAEMONIZE = True
DL = 1

###############################################################################
# Fold Daemon that extends the MEERKATProcDaemon
class MEERKATFoldDaemon (MEERKATProcDaemon):

  def __init__ (self, name, id):
    MEERKATProcDaemon.__init__(self, name , id)
    self.tag = "fold"

  def getEnvironment (self):
    self.log (2, "MEERKATFoldDaemon::getEnvironment()")
    env = MEERKATProcDaemon.getEnvironment (self)
    env["TMP_USER"] = "meerkat" + str(self.id)
    return env

  # called once the header has been received, and must prepare the self.cmd to be run
  def prepare (self):

    self.log (1, "UTC_START=" + self.header["UTC_START"])
    self.log (1, "RESOLUTION=" + self.header["RESOLUTION"])

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

    outnpol = -1
    outtsubint = -1
    dm = -1
    outnbin = -1
    innchan = int(self.header["NCHAN"])
    outnchan = innchan

    try:
      outnpol = int(self.header["FOLD_OUTNPOL"])
    except:
      outnpol = 4
      self.log(-1, "FOLD_OUTNPOL not present in header, assuming " + str(outnpol))
      self.header["FOLD_OUTNPOL"] = str(outnpol)

    try:
      outtsub = int(self.header["FOLD_OUTTSUBINT"])
    except:
      outtsub = 8
      self.log(-1, "FOLD_OUTTSUBINT not present in header, assuming " + str(outtsubint))
      self.header["FOLD_OUTTSUBINT"] = str(outtsubint)

    try:
      outnbin = int(self.header["FOLD_OUTNBIN"])
    except:
      outnbin = 1024
      self.log(-1, "FOLD_OUTNBIN not present in header, assuming " + str(outnbin))
      self.header["FOLD_OUTNBIN"] = str(outnbin)

    try:
      # TODO deal with the 2 subbands better
      outnchan = int(self.header["FOLD_OUTNCHAN"]) / 2
    except:
      outnchan = innchan
      self.log(-1, "FOLD_OUTNCHAN not present in header, assuming " + str(outnchan))
      self.header["FOLD_OUTNCHAN"] = str(outnchan)

    try:
      dm = float(self.header["FOLD_DM"])
    except:
      dm = -1
      self.log(-1, "FOLD_OUTDM not present in header, assuming " + str(dm))
      self.header["FOLD_OUTDM"] = str(dm)

    # configure the command to be run
    self.cmd = "dspsr -Q " + db_key_filename + " -minram 512 -cuda " + self.gpu_id + " -no_dyn"

    # handle detection options
    if outnpol == 1 or outnpol == 2 or outnpol == 3 or outnpol == 4:
      self.cmd = self.cmd + " -d " + str(outnpol)
    else:
      self.log(-1, "ignoring invalid outnpol of " + str(outnpol))

    # handle channelisation
    if outnchan > innchan:
      if outnchan % innchan == 0:
        self.cmd = self.cmd + " -F " + str(outnchan) + ":D"

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
    else:
      if self.header["MODE"] == "PSR":
        (rval, dm) = self.get_dm (self.header["SOURCE"])
        if rval != 0:
          self.log(-1, "get_dm(" + self.header["SOURCE"] + ") failed, using 0")
          dm = 0
      else:
        dm = 0

    self.log(1, "dm=" + str(dm))

    # use a relatively small kernel length 
    if dm > 0:
      bw = float(self.header["BW"])
      self.log(1, "bw=" + str(bw))
      freq = float(self.header["FREQ"])
      self.log(1, "freq=" + str(freq))
      (rval, opt_kernel_length) = self.get_optimal_kernel_length (dm, innchan, bw, freq)
      if rval == 0:
        self.log(1, "opt_kernel_length=" + str(opt_kernel_length))
        if opt_kernel_length < 1024:
          self.log(1, "overriding kernel length to 1024")
          opt_kernel_length = 1024
        self.cmd = self.cmd + " -x " + str(opt_kernel_length)
      else:
        self.log(-1, "get_optimal_kernel_length failed, using default kernel length")

      #(rval, min_kernel_length) = self.get_minimum_kernel_length (dm, innchan, bw, freq)
      #self.log(1, "min_kernel_length=" + str(min_kernel_length))
      #if rval == 0:
      #  if min_kernel_length < 2048:
      #    kernel_length = 2048
      #  else:
      #    kernel_length = 2 * min_kernel_length
      #  self.cmd = self.cmd + " -x " + str(kernel_length)
      #else:
      #  self.log(-1, "get_minimum_kernel_length failed, using default kernel length")

    self.log_prefix = "fold_src"

    return True

###############################################################################

if __name__ == "__main__":

  if len(sys.argv) != 2:
    print "ERROR: 1 command line argument expected"
    sys.exit(1)

  # this should come from command line argument
  stream_id = sys.argv[1]

  script = MEERKATFoldDaemon ("meerkat_fold", stream_id)
  script.cpu_list = "-1"
  state = script.configure (DAEMONIZE, DL, "fold", "fold") 
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

