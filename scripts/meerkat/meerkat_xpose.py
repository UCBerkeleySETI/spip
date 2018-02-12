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

DAEMONIZE = False
DL = 2

#################################################################
# thread for executing processing commands
class MeerKATXposeThread (threading.Thread):

  def __init__ (self, cmd, pipe, dl):
    threading.Thread.__init__(self)
    self.cmd = cmd
    self.pipe = pipe
    self.dl = dl

  def run (self):
    rval = system_piped (self.cmd, self.pipe, self.dl <= DL)
    return rval

###############################################################################
# Proc Daemon Proper
class MeerKATXposeDaemon (Daemon, StreamBased):

  def __init__ (self, name, id):
    Daemon.__init__(self, name, str(id))
    StreamBased.__init__(self, id, self.cfg)

  def configure_child (self):

    self.stream_id = self.id

    # get the data block keys
    self.db_prefix  = self.cfg["DATA_BLOCK_PREFIX"]
    self.db_id_in   = self.cfg["RECEIVING_DATA_BLOCK"]
    self.db_id_out  = self.cfg["PROCESSING_DATA_BLOCK"]
    self.num_stream = self.cfg["NUM_STREAM"]
    self.cpu_core   = self.cfg["STREAM_PROC_CORE_" + self.stream_id]

    # share memory keys
    self.db_key_in1  = SMRBDaemon.getDBKey (self.db_prefix, 0, self.num_stream, self.db_id_in)
    self.db_key_in2  = SMRBDaemon.getDBKey (self.db_prefix, 1, self.num_stream, self.db_id_in)
    self.db_key_out = SMRBDaemon.getDBKey (self.db_prefix, self.stream_id, self.num_stream, self.db_id_out)

    self.log (0, "MeerKATProcDaemon::configure db_key=" + self.db_key)

    # GPU to use for signal processing
    self.gpu_id = self.cfg["GPU_ID_" + str(self.id)]

    (host, self.beam_id, self.subband_id) = self.cfg["STREAM_" + self.stream_id].split(":")
    (self.cfreq, self.bw, self.nchan) = self.cfg["SUBBAND_CONFIG_" + self.subband_id].split(":")
    self.beam = self.cfg["BEAM_" + str(self.beam_id)]

    self.log (0, "MeerKATProcDaemon::configure done")

  # prepare the processing command once the header is ready
  def prepare (self):

    # default processing commands
    self.cmd = "dada_dbnull -s -k " + self.db_key

    # output directory for FOLD mode
    self.out_dir = self.cfg["CLIENT_DIR"] 

    # the

    # configure the command to be run
    self.cmd = "meerkat_polsubxpose " \
        + " " + self.db_key_in1 \
        + " " + self.db_key_in2 \
        + " " + self.db_key_ouy \
        + " " + self.subband_id \ 
        + " -d " + self.gpu_id + " -b " + self.cpu_core

    self.log_prefix = "xpose_src"


  def wait_for_smrb (self):

    # wait up to 10s for the SMRB to be created
    smrb_wait = 10
    cmd_in = "dada_dbmetric -k " + self.db_key_in
    cmd_out = "dada_dbmetric -k " + self.db_key_out

    smrb_ready = False

    while not smrb_ready and smrb_wait > 0 and not self.quit_event.isSet():
    
      # assume they are not ready
      smrb_ready = True

      # check input SMRB
      self.log (2, "MeerKATProcDaemon::wait_for_smrb " + cmd_in)
      rval, lines = self.system (cmd_in)
      if rval:
        smrb_ready = False

      # check output SMRB
      self.log (2, "MeerKATProcDaemon::wait_for_smrb " + cmd_out)
      rval, lines = self.system (cmd_out)
      if rval:
        smrb_ready = False

      if not smrb_ready:
        time.sleep(1)
        smrb_wait -= 1

    if not smrb_ready:
      self.log(-2, "smrb["+str(self.id)+"] no valid SMRB with " +
                  "keys=" + self.db_key_in + " or " + self.db_key_out)
      self.quit_event.set()

  # main method 
  def main (self):

    self.log (2, "MeerKATProcDaemon::main configure_child()")
    self.configure_child()

    self.log (2, "MeerKATProcDaemon::main wait_for_smrb()")
    self.wait_for_smrb()

    if self.quit_event.isSet():
      self.log (-1, "MeerKATProcDaemon::main quit event was set after waiting for SMRB creation")
      return

    # continuously run the main command waiting on the SMRB
    while (not self.quit_event.isSet()):

      tag = "meerkat_xpose_" + self.stream_id

      # wait for the header to determine if folding is required
      cmd = "dada_header -t " + tag + " -k " + self.db_key
      self.log(1, cmd)
      self.binary_list.append (cmd)
      rval, lines = self.system (cmd)
      self.binary_list.remove (cmd)

      # if the command returned ok and we have a header
      if rval != 0:
        if self.quit_event.isSet():
          self.log (2, "MeerKATProcDaemon::main " + cmd + " failed, but quit_event true")
        else:
          self.log (-2, "MeerKATProcDaemon::main " + cmd + " failed")
          self.quit_event.set()

      elif len(lines) == 0:
        
        self.log (-2, "MeerKATProcDaemon::main header was empty")
        self.quit_event.set()
        
      else:

        self.log (2, "MeerKATProcDaemon::main parsing header")
        self.header = Config.parseHeader (lines)

        if not float(self.bw) == float(self.header["BW"]):
          self.log (-1, "configured bandwidth ["+bw+"] != self.header["+self.header["BW"]+"]")
        if not float(self.cfreq) == float(self.header["FREQ"]):
          self.log (-1, "configured cfreq ["+cfreq+"] != self.header["+self.header["FREQ"]+"]")
        if not int(self.nchan) == int(self.header["NCHAN"]):
          self.log (-2, "configured nchan ["+nchan+"] != self.header["+self.header["NCHAN"]+"]")

        # configure the output pipe
        self.log (2, "MeerKATProcDaemon::main configuring output log pipe")
        log_host = self.cfg["SERVER_HOST"]
        log_port = int(self.cfg["SERVER_LOG_PORT"])
        log_pipe = LogSocket (self.log_prefix, self.log_prefix,
                              str(self.id), "stream",
                              log_host, log_port, int(DL))
        log_pipe.connect()

        # add the binary command to the kill list
        self.binary_list.append (self.cmd)

        # create processing threads
        self.log (1, "MeerKATProcDaemon::main creating processing threads")      
        proc_thread = MeerKATProcThread (self.cmd, log_pipe.sock, 1)

        # start processing threads
        self.log (1, "MeerKATProcDaemon::main starting processing thread")
        proc_thread.start()

        # join processing threads
        self.log (2, "MeerKATProcDaemon::main waiting for proc thread to terminate")
        rval = proc_thread.join() 
        self.log (2, "MeerKATProcDaemon::main proc thread joined")

        # remove the binary command from the list
        self.binary_list.remove (self.cmd)

        if rval:
          self.log (-2, "MeerKATProcDaemon::main proc thread failed")
          quit_event.set()

        log_pipe.close()

      self.log (1, "MeerKATProcDaemon::main processing completed")
