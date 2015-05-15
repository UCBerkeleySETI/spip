#!/usr/bin/env python

##############################################################################
#  
#     Copyright (C) 2015 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
# 
###############################################################################

import os, threading, sys, time, socket, select, signal, traceback
import spip

SCRIPT = "spip_recv"
DL     = 2

def signal_handler(signal, frame):
  print 'You pressed Ctrl+C!'
  global quit_event
  quit_event.set()

#################################################################
#
# main
#

# this should come from command line argument
id = 0

# read configuration file
cfg = spip.getConfig()

control_thread = []

log_file  = cfg["SERVER_LOG_DIR"] + "/" + SCRIPT + ".log"
pid_file  = cfg["SERVER_CONTROL_DIR"] + "/" + SCRIPT + ".pid"
quit_file = cfg["SERVER_CONTROL_DIR"] + "/"  + SCRIPT + ".quit"

if os.path.exists(quit_file):
  sys.stderr.write("quit file existed at launch: " + quit_file)
  sys.exit(1)

# become a daemon
spip.daemonize(pid_file, log_file)

try:

  spip.logMsg(1, DL, "STARTING SCRIPT")

  quit_event = threading.Event()

  signal.signal(signal.SIGINT, signal_handler)

  # start a control thread to handle quit requests
  control_thread = spip.controlThread(quit_file, pid_file, quit_event, DL)
  control_thread.start()

  log_host = cfg["LOG_HOST"]
  log_port = cfg["LOG_PORT"]

  log_pipe = spip.openSocket (DL, log_host, log_port, 3)

  cmd = cfg["PWC_BINARY"] \
        + " -b " + cfg["RECV_CORE_" + id] \
        + " -c " + cfg["RECV_PORT_" + id] \
        + " -i " + cfg["RECV_UDP_IP_" + id]
        + " -l " + cfg["RECV_LOGPORT_" + id] \
        + " -p " + cfg["RECV_UDP_PORT_" + id]

  spip.logMsg(2, DL, "main: " + command)
  rval = spip.system_piped (command, pipe, 2 < DL)
  spip.logMsg(2, DL, "main: rval=" + str(rval))

  # now we want to start a popen command that pipes stderr and stdout 
  # to a redirectable socket

  # script body
  time.sleep (2)

except:
  spip.logMsg(-2, DL, "main: exception caught: " + str(sys.exc_info()[0]))
  print '-'*60
  traceback.print_exc(file=sys.stdout)
  print '-'*60
  quit_event.set()

# join threads
spip.logMsg(2, DL, "main: joining control thread")
if (control_thread):
  control_thread.join()

spip.logMsg(1, DL, "STOPPING SCRIPT")

sys.exit(0)
