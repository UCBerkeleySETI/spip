#!/usr/bin/env python

###############################################################################
#  
#     Copyright (C) 2015 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
# 
###############################################################################

import sys, traceback
from time import sleep

from spip_smrb import SMRBDaemon
from spip_recv import RecvDaemon,ConfiguringThread
from spip.meerkat_config import MeerKATConfig

DAEMONIZE = True
DL = 1


###############################################################################
# 
class MeerKATRecvDaemon(RecvDaemon):

  def __init__ (self, name, id):
    RecvDaemon.__init__(self, name, id)

  def getConfiguration (self):
    self.log (2, "MeerKATRecvDaemon::getConfiguration()")
    meerkat_config = MeerKATConfig()
    local_config = meerkat_config.getMuxedStreamConfigFixed (self.id)
    return local_config

  def getEnvironment (self):
    self.log (2, "MeerKATRecvDaemon::getEnvironment()")
    env = RecvDaemon.getEnvironment (self)
    env["LD_PRELOAD"] = "libvma.so"
    env["VMA_MTU"] = "4200"
    env["VMA_RING_ALLOCATION_LOGIC_RX"] = "10"
    env["VMA_INTERNAL_THREAD_AFFINITY"] = "6"
    env["VMA_TRACELEVEL"] = "WARNING"
    return env

  def getCommand (self, config_file):
    self.log (2, "MeerKATRecvDaemon::getCommand()")

    db_id = self.cfg["RECEIVING_DATA_BLOCK"]
    db_prefix = self.cfg["DATA_BLOCK_PREFIX"]
    num_stream = self.cfg["NUM_STREAM"]
    key1 = SMRBDaemon.getDBKey (db_prefix, 0, num_stream, db_id)
    key2 = SMRBDaemon.getDBKey (db_prefix, 1, num_stream, db_id)

    cmd = self.cfg["STREAM_BINARY"] + " " + config_file + " " + key1 + " " + key2 \
            + " -b " + self.cpu_core \
            + " -c " + self.ctrl_port \
            + " -f spead"

    # hack for sub-band mode
    if self.id != "0":
      cmd = "meerkat_dummyserver -b " + self.cpu_core + " -c " + self.ctrl_port

    return cmd

###############################################################################
# main

if __name__ == "__main__":

  if len(sys.argv) != 2:
    print "ERROR: 1 command line argument expected"
    sys.exit(1)

  # this should come from command line argument
  stream_id = sys.argv[1]

  script = MeerKATRecvDaemon ("meerkat_recv", stream_id)
  state = script.configure (DAEMONIZE, DL, "recv", "recv")
  if state != 0:
    sys.exit(state)

  script.log(2, "STARTING SCRIPT")

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

  script.log(2, "STOPPING SCRIPT")
  script.conclude()
  sys.exit(0)
