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
from os import environ

DL        = 2

#################################################################
# thread for executing processing commands
class UWBProcThread (threading.Thread):

  def __init__ (self, script, cmd, pipe, env, dl):
    threading.Thread.__init__(self)
    self.script = script
    self.cmd = cmd
    self.pipe = pipe
    self.dir = script.out_dir
    self.env = env
    self.dl = dl

  def run (self):
    cmd = "cd " + self.dir + "; " + self.cmd
    self.script.log (2, "UWBProcThread::run " + self.cmd)
    rval = self.script.system_piped (cmd, self.pipe, 2, self.env)
    self.script.log (2, "UWBProcThread::run rval=" + str(rval))
    
    if rval == 0:
      rval2, lines = system ("touch " + self.dir + "/obs.finished")
    else:
      rval2, lines = system ("touch " + self.dir + "/obs.failed")

    return rval

###############################################################################
# Proc Daemon Proper
class UWBProcDaemon (Daemon, StreamBased):

  def __init__ (self, name, id):
    Daemon.__init__(self, name, str(id))
    StreamBased.__init__(self, id, self.cfg)
    self.tag = "unset"

  def configure_child (self):

    self.stream_id = self.id

    # get the data block keys
    self.db_prefix  = self.cfg["DATA_BLOCK_PREFIX"]
    self.db_id      = self.cfg["PROCESSING_DATA_BLOCK"]
    self.num_stream = self.cfg["NUM_STREAM"]
    self.cpu_core   = self.cfg["STREAM_PROC_CORE_" + self.stream_id]

    self.db_key = SMRBDaemon.getDBKey (self.db_prefix, self.stream_id, self.num_stream, self.db_id)

    self.log (2, "UWBProcDaemon::configure db_key=" + self.db_key)

    # GPU to use for signal processing
    self.gpu_id = self.cfg["GPU_ID_" + str(self.id)]

    (host, self.beam_id, self.subband_id) = self.cfg["STREAM_" + self.stream_id].split(":")
    (self.cfreq, self.bw, self.nchan) = self.cfg["SUBBAND_CONFIG_" + self.subband_id].split(":")
    self.beam = self.cfg["BEAM_" + str(self.beam_id)]

    self.log (2, "UWBProcDaemon::configure done")

  def wait_for_smrb (self):

    # wait up to 10s for the SMRB to be created
    smrb_wait = 10
    cmd = "dada_dbmetric -k " + self.db_key
    self.binary_list.append (cmd)

    rval = 1
    while rval and smrb_wait > 0 and not self.quit_event.isSet():

      self.log (2, "UWBProcDaemon::wait_for_smrb " + cmd)
      rval, lines = self.system (cmd)
      if rval:
        time.sleep(1)
      smrb_wait -= 1

    if rval:
      self.log(-2, "smrb["+str(self.id)+"] no valid SMRB with " +
                  "key=" + self.db_key)
      self.quit_event.set()

  def getEnvironment (self):
    self.log(2, "UWBProcDaemon::getEnvironment()")
    return environ.copy()

  # main method 
  def main (self):

    self.log (2, "UWBProcDaemon::main configure_child()")
    self.configure_child()

    self.log (2, "UWBProcDaemon::main wait_for_smrb()")
    SMRBDaemon.waitForSMRB(self.db_key, self)

    if self.quit_event.isSet():
      self.log (-1, "UWBProcDaemon::main quit event was set after waiting for SMRB creation")
      return

    # continuously run the main command waiting on the SMRB
    while (not self.quit_event.isSet()):

      # wait for the header to determine if folding is required
      cmd = "dada_header -k " + self.db_key + " -t " + self.tag
      self.log(2, "UWBProcDaemon::main " + cmd)
      self.binary_list.append (cmd)
      rval, lines = self.system (cmd, 2, True)
      self.binary_list.remove (cmd)

      # if the command returned ok and we have a header
      if rval != 0:
        time.sleep(0.1)
        if self.quit_event.isSet():
          self.log (2, "UWBProcDaemon::main " + cmd + " failed, but quit_event true")
        else:
          self.log (-2, "UWBProcDaemon::main " + cmd + " failed")
          self.quit_event.set()

      elif len(lines) == 0:
        
        self.log (-2, "UWBProcDaemon::main header was empty")
        self.quit_event.set()
        
      else:

        self.log (2, "UWBProcDaemon::main parsing header")
        self.header = Config.parseHeader (lines)

        # account for lower to upper sideband conversion
        if not abs(float(self.bw)) == float(self.header["BW"]):
          self.log (-1, "configured bandwidth ["+self.bw+"] != self.header["+self.header["BW"]+"]")
        if not float(self.cfreq) == float(self.header["FREQ"]):
          self.log (-1, "configured cfreq ["+self.cfreq+"] != self.header["+self.header["FREQ"]+"]")
        if not int(self.nchan) == int(self.header["NCHAN"]):
          self.log (-2, "configured nchan ["+self.nchan+"] != self.header["+self.header["NCHAN"]+"]")

        self.source = self.header["SOURCE"]
        self.utc_start = self.header["UTC_START"]

        # call the child class prepare method
        self.log (2, "UWBProcDaemon::main prepare()")
        valid = self.prepare()

        if valid:

          # ensure the output directory exists
          self.log (2, "UWBProcDaemon::main creating out_dir: " + self.out_dir)
          if not os.path.exists (self.out_dir):
            os.makedirs (self.out_dir, 0755)

          # write the sub-bands header to the out_dir
          header_file = self.out_dir + "/obs.header"
          self.log (2, "UWBProcDaemon::main writing obs.header to out_dir")
          Config.writeDictToCFGFile (self.header, header_file)
    
          # configure the output pipe
          self.log (2, "UWBProcDaemon::main configuring output log pipe")
          log_host = self.cfg["SERVER_HOST"]
          log_port = int(self.cfg["SERVER_LOG_PORT"])
          log_pipe = LogSocket (self.log_prefix, self.log_prefix,
                                str(self.id), "stream",
                                log_host, log_port, int(DL))
          log_pipe.connect()

          # get any modifications to the environment
          env = self.getEnvironment()

          # add the binary command to the kill list
          self.binary_list.append (self.cmd)

          # create processing threads
          self.log (2, "UWBProcDaemon::main creating processing threads")      
          cmd = "numactl -C " + self.cpu_core + " -- " + self.cmd
          proc_thread = UWBProcThread (self, cmd, log_pipe.sock, env, 1)

          # start processing threads
          self.log (2, "UWBProcDaemon::main starting processing thread")
          proc_thread.start()

          self.log (1, "START " + self.cmd)

          # join processing threads
          self.log (2, "UWBProcDaemon::main waiting for proc thread to terminate")
          rval = proc_thread.join() 
          self.log (2, "UWBProcDaemon::main proc thread joined")

          self.log (1, "END   " + self.cmd)

          # remove the binary command from the list
          self.binary_list.remove (self.cmd)

          if rval:
            self.log (-2, "UWBProcDaemon::main proc thread failed")
            quit_event.set()

          log_pipe.close()

          # good practise in case the proc thread always fails
          time.sleep(1)

        else:

          self.log (2, "MEERKATProcDaemon::main skip this processing")
          time.sleep(10)

      self.log (2, "UWBProcDaemon::main processing loop completed")

