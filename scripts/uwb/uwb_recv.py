#!/usr/bin/env python

###############################################################################
#  
#     Copyright (C) 2015 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
# 
###############################################################################

import sys, traceback
from time import sleep

from spip_recv import RecvDaemon,ConfiguringThread
from uwb_config import UWBConfig

DAEMONIZE = True
DL = 1


###############################################################################
# 
class UWBRecvDaemon(RecvDaemon):

  def __init__ (self, name, id):
    RecvDaemon.__init__(self, name, id)

  def getConfiguration (self):
    uwb_config = UWBConfig()
    local_config = uwb_config.getStreamConfigFixed (self.id)
    return local_config

  def getEnvironment (self):
    env = RecvDaemon.getEnvironment (self)
    env["LD_PRELOAD"] = "libvma.so"
    env["VMA_MTU"] = "9216"
    env["VMA_RING_ALLOCATION_LOGIC_RX"] = "0"
    env["VMA_THREAD_MODE"] = "0"
    env["VMA_INTERNAL_THREAD_AFFINITY"] = str(int(self.cpu_core) + 1)
    env["VMA_MEM_ALLOC_TYPE"] = "1"
    env["VMA_TRACELEVEL"] = "WARNING"
    return env

  def getCommand (self, config_file):

    # get the beam name for the stream
    (host, self.beam_id, self.subband_id) = self.cfg["STREAM_" + self.id].split(":")
    beam = self.cfg["BEAM_" + str(self.beam_id)]

    cmd = self.cfg["STREAM_BINARY"] + " -k " + self.db_key \
            + " -b " + self.cpu_core \
            + " -c " + self.ctrl_port \
            + " -D " + self.cfg["CLIENT_STATS_DIR"] + "/" + beam \
            + " -s " + str(self.id) \
            + " -f dualvdif" \
            + " " + config_file

    #cmd = self.cfg["STREAM_BINARY"] + " -k " + self.db_key \
    #        + " -b " + self.cpu_core \
    #        + " -c " + self.ctrl_port \
    #        + " -f dualvdif" \
    #        + " " + config_file

    return cmd

###############################################################################
# main

if __name__ == "__main__":

  if len(sys.argv) != 2:
    print "ERROR: 1 command line argument expected"
    sys.exit(1)

  # this should come from command line argument
  stream_id = sys.argv[1]

  script = UWBRecvDaemon ("uwb_recv", stream_id)

  # ensure the recv daemons can bind as they see fit
  script.cpu_list = "-1"
  state = script.configure (DAEMONIZE, DL, "recv", "recv")
  if state != 0:
    sys.exit(state)

  script.log(1, "STARTING SCRIPT")

  try:

    configuring_thread = ConfiguringThread (script, stream_id)
    configuring_thread.start()

    script.main ()

    configuring_thread.join()


  except:
    script.quit_event.set()
    script.log(-2, "exception caught: " + str(sys.exc_info()[0]))
    print '-'*60
    traceback.print_exc(file=sys.stdout)
    print '-'*60

  script.log(1, "STOPPING SCRIPT")
  script.conclude()
  sys.exit(0)
