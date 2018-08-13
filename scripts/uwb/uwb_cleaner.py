#!/usr/bin/env python

###############################################################################
#  
#     Copyright (C) 2017 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
# 
###############################################################################

import logging, sys, os, traceback, socket
from time import sleep

from spip.daemons.bases import StreamBased,ServerBased
from spip.daemons.daemon import Daemon
from spip.config import Config

DAEMONIZE = True
DL = 1

###############################################################################
#  Generic Implementation
class UWBCleanerDaemon(Daemon):

  def __init__ (self, name, id):
    Daemon.__init__(self, name, str(id))
    self.beams = []

  def configure (self, become_daemon, dl, source, dest):
    Daemon.configure (self, become_daemon, dl, source, dest)
    return 0

  def main (self):

    self.log(2, "UWBCleanerDaemon::main starting main loop")    

    while not self.quit_event.isSet():

      # for each directory that has a completed dir
      for proc_type in self.proc_types:

        self.log(2, "UWBCleanerDaemon::main proc_type=" + proc_type)

        # for each configured beam (there is only 1 for UWB)
        for beam in self.beams:

          self.log(2, "UWBCleanerDaemon::main beam=" + beam)

          if self.quit_event.isSet():
            self.log(1, "UWBCleanerDaemon::main quit_event true [1]")
            continue

          # directory containing transferred observations
          sent_dir = self.sent_dirs[proc_type] + "/" + beam
          finished_dir = self.finished_dirs[proc_type] + "/" + beam

          if not os.path.exists(sent_dir):
            self.log(-1, "sent_dir [" + sent_dir + "] did not exist")
            os.makedirs(sent_dir, 0755)

          # look for observations that have been located in the sent dir and are more than 2 hours old
          cmd = "find " + sent_dir + " -type f -path '*/" + self.cfreq + "/obs.finished' -mmin +120 | sort"
          rval, fin_files = self.system(cmd, 2)
          if rval:
            self.log (-1, "main find command failed: " + fin_files[0])
            sleep(1)
            continue
      
          self.log(2, "UWBCleanerDaemon::main assessing obs.finished observations") 
          # transfer the completed directory to herschel
          for path in fin_files:

            if self.quit_event.isSet():
              self.log(1, "UWBCleanerDaemon::main quit_event true [2]")
              continue

            # strip dir prefix
            subpath = path [(len(sent_dir)+1):] 

            # extract the the beam, utc, source and cfreq
            (utc, source, cfreq, file) = subpath.split("/")

            self.log(2, "UWBCleanerDaemon::main deleting sent dir")
            dir = sent_dir + "/" + utc + "/" + source + "/" + cfreq
            cmd = "rm -rf " + dir
            self.log(2, "UWBCleanerDaemon::main " + cmd)
            rval, lines= self.system(cmd, 2)
            if rval:
              self.log (-1, "failed to delete " + dir + ": " + str(lines))

            # count the number of other sub-band directories
            subdir_count = len(os.listdir(sent_dir + "/" + utc + "/" + source))
            self.log (2, "UWBArchiverDaemon::main subdir_count=" + str(subdir_count))
            if subdir_count == 0:
              os.rmdir(sent_dir + "/" + utc + "/" + source)
              os.rmdir (sent_dir + "/" + utc)
         
            # convert the atnf UTC to a regular one
            (year, mon, day, hh, mm, ss) = utc.split("-")
            regular_utc = year + "-" + mon + "-" + day + "-" + hh + ":" + mm + ":" + ss
            self.log(2, "UWBCleanerDaemon::main converted utc: " + utc + " -> " + regular_utc) 
 
            # delete the finished directory
            dir = finished_dir + "/" + regular_utc + "/" + source + "/" + cfreq
            if os.path.exists(dir):
              cmd = "rm -rf " + dir
              self.log(2, "UWBCleanerDaemon::main " + cmd)
              rval, lines= self.system(cmd, 2)
              if rval:
                self.log (-1, "failed to delete " + fin_dir + ": " + str(lines))

              # count the number of other sub-band directories
              subdir_count = len(os.listdir(finished_dir + "/" + regular_utc + "/" + source))
              self.log (2, "UWBArchiverDaemon::main subdir_count=" + str(subdir_count))
              if subdir_count == 0:
                os.rmdir(finished_dir + "/" + regular_utc + "/" + source)
                os.rmdir(finished_dir + "/" + regular_utc)

            self.log (1, proc_type + " " + utc + "/" + source + "/" + cfreq + " deleted")
            
            sleep (1)

      to_sleep = 60
      self.log (2, "UWBCleanerDaemon::main sleeping " + str(to_sleep) + " seconds")
      while to_sleep > 0 and not self.quit_event.isSet():
        sleep(1)
        to_sleep -= 1


###############################################################################
# Server based implementation
class UWBCleanerServerDaemon(UWBCleanerDaemon, ServerBased):

  def __init__ (self, name):
    UWBCleanerDaemon.__init__(self, name, "-1")
    ServerBased.__init__(self, self.cfg)

  def configure (self,become_daemon, dl, source, dest):

    UWBCleanerDaemon.configure (self, become_daemon, dl, source, dest)

    self.proc_types = ["fold", "search", "continuum"]
    self.sent_dirs = {}

    self.finished_dirs["fold"]      = self.cfg["SERVER_FOLD_DIR"] + "/finished"
    self.finished_dirs["search"]    = self.cfg["SERVER_SEARCH_DIR"] + "/finished"
    self.finished_dirs["continuum"] = self.cfg["SERVER_CONTINUUM_DIR"] + "/finished"

    self.sent_dirs["fold"]      = self.cfg["SERVER_FOLD_DIR"] + "/sent"
    self.sent_dirs["search"]    = self.cfg["SERVER_SEARCH_DIR"] + "/sent"
    self.sent_dirs["continuum"] = self.cfg["SERVER_CONTINUUM_DIR"] + "/sent"

    for i in range(int(self.cfg["NUM_BEAM"])):
      bid = self.cfg["BEAM_" + str(i)]
      self.beams.append(bid)
    return 0

    self.cfreq = ""


###############################################################################
# Stream based implementation
class UWBCleanerStreamDaemon (UWBCleanerDaemon, StreamBased):

  def __init__ (self, name, id):
    UWBCleanerDaemon.__init__(self, name, str(id))
    StreamBased.__init__(self, str(id), self.cfg)

  def configure (self, become_daemon, dl, source, dest):

    self.log(1, "UWBCleanerStreamDaemon::configure()")
    UWBCleanerDaemon.configure(self, become_daemon, dl, source, dest)

    self.proc_types = ["fold", "search", "continuum"]
    self.finished_dirs    = {}
    self.sent_dirs = {}

    self.finished_dirs["fold"]      = self.cfg["CLIENT_FOLD_DIR"] + "/finished"
    self.finished_dirs["search"]    = self.cfg["CLIENT_SEARCH_DIR"] + "/finished"
    self.finished_dirs["continuum"] = self.cfg["CLIENT_CONTINUUM_DIR"] + "/finished"

    self.sent_dirs["fold"]      = self.cfg["CLIENT_FOLD_DIR"] + "/sent"
    self.sent_dirs["search"]    = self.cfg["CLIENT_SEARCH_DIR"] + "/sent"
    self.sent_dirs["continuum"] = self.cfg["CLIENT_CONTINUUM_DIR"] + "/sent"

    # determine the beam name
    (host, beam_id, subband_id) = self.cfg["STREAM_" + self.id].split(":")
    beam_name = self.cfg["BEAM_" + beam_id]
    self.beams.append(beam_name)

    # determine the subband config to find the centre frequency
    (cfreq, bw, nchan) = self.cfg["SUBBAND_CONFIG_" + subband_id].split(":")
    self.cfreq = cfreq

    return 0
    
###############################################################################
# main

if __name__ == "__main__":

  # logging.basicConfig(filename='example.log', filemode='w', level=logging.DEBUG)

  if len(sys.argv) != 2:
    print "ERROR: 1 command line argument expected"
    sys.exit(1)

  # this should come from command line argument
  beam_id = sys.argv[1]

  script = []
  if int(beam_id) == -1:
    print "ERROR: server mode disabled for now"
    sys.exit(1)
    # script = UWBCleanerServerDaemon ("uwb_cleaner")
  else:
    script = UWBCleanerStreamDaemon ("uwb_cleaner", beam_id)

  state = script.configure (DAEMONIZE, DL, "cleaner", "cleaner")
  if state != 0:
    sys.exit(state)

  script.log(1, "STARTING SCRIPT")

  try:
    script.main ()

  except:
    script.quit_event.set()
    script.log(-2, "exception caught: " + str(sys.exc_info()[0]))
    print '-'*60
    traceback.print_exc(file=sys.stdout)
    print '-'*60

  script.log(1, "STOPPING SCRIPT")
  script.conclude()
  sys.exit(0)
