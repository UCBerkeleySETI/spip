#!/usr/bin/env python

##############################################################################
#  
#     Copyright (C) 2015 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
# 
###############################################################################

import os, threading, sys, time, socket, select, traceback, copy

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
    self.wideband_predictor = False

  def getEnvironment (self):
    self.info("UWBFoldDaemon::getEnvironment()")
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
      fold = (self.header["PERFORM_FOLD"] in ["1", "true"])
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

    # create DSPSR viewing file for the data block
    view_key_filename = "/tmp/spip_" + self.db_key + ".viewer"
    if not  os.path.exists (view_key_filename):
      view_key_file = open (view_key_filename, "w")
      view_key_file.write("DADA INFO:\n")
      view_key_file.write("key " +  self.db_key + "\n")
      view_key_file.write("viewer\n")
      view_key_file.close()

    outnstokes = -1
    outtsubint = -1
    dm = -1
    outnbin = -1
    innchan = int(self.header["NCHAN"])
    outnchan = innchan
    sk = False
    sk_threshold = -1
    sk_nsamps = -1
    mode = "PSR"

    try:
      outnstokes = int(self.header["FOLD_OUTNSTOKES"])
    except:
      outnstokes = 4

    try:
      outtsub = int(self.header["FOLD_OUTTSUBINT"])
    except:
      outtsub = 10

    try:
      outnbin = int(self.header["FOLD_OUTNBIN"])
    except:
      outnbin = 1024

    try:
      outnchan = int(self.header["FOLD_OUTNCHAN"])
    except:
      outnchan = 0
      innchan = 0

    try:
      mode = self.header["MODE"]
    except:
      mode = "PSR"

    try:
      dm = float(self.header["DM"])
    except:
      dm = -1

    try:
      sk = self.header["FOLD_SK"] == "1"
    except:
      sk = False

    try:
      sk_threshold = int(self.header["FOLD_SK_THRESHOLD"])
    except:
      sk_threshold = 3

    try:
      sk_nsamps = int(self.header["FOLD_SK_NSAMPS"])
    except:
      sk_nsamps = 1024


    # configure the command to be run
    self.cmd = "dspsr -Q " + db_key_filename + " -minram 2048 -cuda " + self.gpu_id + " -no_dyn"

    # handle detection options
    if outnstokes >= 1 or outnstokes <= 4:
      # hack for NPOL==1
      if self.header["NPOL"] == "1":
        self.cmd = self.cmd + " -d 1"
      else:
        self.cmd = self.cmd + " -d " + str(outnstokes)
    elif outnstokes == -1:
      self.log(2, "using stokes IQUV default for DSPSR")
    else:
      self.log(-1, "ignoring invalid outnstokes of " + str(outnstokes))

    # handle channelisation
    if outnchan > innchan:
      if outnchan % innchan == 0:
        if mode == "PSR":
          self.cmd = self.cmd + " -F " + str(outnchan) + ":D"
        else:
          self.cmd = self.cmd + " -F " + str(outnchan) + ":" + str(outnchan*4)
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
    mintsub = outtsub - 5
    if mintsub > 0:
      self.cmd = self.cmd + " -Lmin " + str(mintsub)

    # if observing a puslar
    if mode == "PSR":

      # handle a custom DM
      if dm >= 0:
        self.cmd = self.cmd + " -D " + str(dm)

      # if the SK options are active
      if sk:
        self.cmd = self.cmd + " -skz -skz_no_fscr"

        if sk_threshold != -1:
          self.cmd = self.cmd + " -skzs " + str(sk_threshold)

        if sk_nsamps != -1:
          self.cmd = self.cmd + " -skzm " + str(sk_nsamps)

      # if we are trialing the wideband predictor mode
      if self.wideband_predictor:

        # create a copy of the header to modify
        fullband_header = copy.deepcopy (self.header)

        nchan_total = 0
        freq_low = 1e12
        freq_high = -1e12

        # now update the key parameters of the header
        for i in range(int(self.cfg["NUM_STREAM"])):
          (cfreq, bw, nchan) = self.cfg["SUBBAND_CONFIG_" + str(i)].split(":")
          nchan_total += int(nchan)
          half_chan_bw = abs(float(bw))
          freq_low_subband = float(cfreq) - half_chan_bw
          freq_high_subband = float(cfreq) + half_chan_bw
          if freq_low_subband < freq_low:
            freq_low = freq_low_subband
          if freq_high_subband > freq_high:
            freq_high = freq_high_subband

        bw = (freq_high - freq_low)
        fullband_header["NCHAN"] = str(nchan_total)
        fullband_header["BW"] = str(bw)
        fullband_header["FREQ"] = str(freq_low + bw/2)
        self.info("fullband predictor: NCHAN=" + fullband_header["NCHAN"] +
                  " BW=" + fullband_header["BW"] + " FREQ=" +
                  fullband_header["FREQ"])

        # create the output directory
        if not os.path.exists (self.out_dir):
          os.makedirs (self.out_dir, 0755)

        # write the sub-bands header to the out_dir
        dummy_file = self.out_dir + "/obs.dummy"
        Config.writeDictToCFGFile (fullband_header, dummy_file, prepend='DUMMY')
 
        # generate an ephemeris file
        ephemeris_file = self.out_dir + "/pulsar.eph"
        cmd = "psrcat -all -e " + self.header["SOURCE"] + " > " + ephemeris_file
        rval, lines = self.system(cmd, 1)

        # generate the tempo2 predictor
        cmd = "t2pred " + ephemeris_file + " " + dummy_file
        rval, lines = self.system(cmd, 1, False, self.getEnvironment())

        # copy the predictor file to the out_dir
        predictor_file = self.out_dir + "/pulsar.pred"
        cmd = "cp /tmp/tempo2/uwb" + str(self.id) + "/t2pred.dat " + predictor_file
        rval, lines = self.system(cmd, 1)

        # append the ephemeris and predictor to DSPSR command line
        self.cmd = self.cmd + " -E " + ephemeris_file + " -P " + predictor_file


    # set the optimal filterbank kernel length
    self.cmd = self.cmd + " -fft-bench"

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

