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

DAEMONIZE = False
DL = 2

class CleanerDaemon(Daemon):

  def __init__ (self, name, id):
    Daemon.__init__(self, name, str(id))
    self.beams = []

  def configure (self, become_daemon, dl, source, dest):
    Daemon.configure (self, become_daemon, dl, source, dest)
    return 0

  def get_out_cfreq (self, obs_dir):

    # read obs.info file to find the subbands
    if not os.path.exists (obs_dir + "/obs.info"):
      self.log(0, "CleanerDaemon::get_out_cfreq obs.info file did not exist")
      return (False, 0)

    info = Config.readCFGFileIntoDict (obs_dir + "/obs.info")
    num_streams = info["NUM_STREAM"]
    freq_low  = float(1e12)
    freq_high = float(-1e12)

    for i in range(int(num_streams)):
      (freq, bw, beam) = info["SUBBAND_" + str(i)].split(":")
      freq_low  = min (freq_low, float(freq) - (float(bw)/2.0))
      freq_high = max (freq_high, float(freq) + (float(bw)/2.0))

    cfreq = int(freq_low + ((freq_high - freq_low) / 2.0))
    self.log(2, "CleanerDaemon::get_out_cfreq low=" + str(freq_low) + " high=" + str(freq_high) + " cfreq=" + str(cfreq))

    return cfreq

###############################################################################
#  Generic Implementation
class MeerKATCleanerDaemon(CleanerDaemon):

  def __init__ (self, name, id):
    CleanerDaemon.__init__(self, name, str(id))

    # delete observations that are 6 months old
    self.clean_age_days = str(30 * 6)

  def main (self):

    self.log(2, "MeerKATCleanerDaemon::main starting main loop")    

    while not self.quit_event.isSet():

      # for each directory that has a completed dir
      for proc_type in self.proc_types:

        self.log(2, "MeerKATCleanerDaemon::main proc_type=" + proc_type)

        # for each configured beam (there is only 1 for MeerKAT)
        for beam in self.beams:

          self.log(2, "MeerKATCleanerDaemon::main beam=" + beam)

          if self.quit_event.isSet():
            self.log(1, "MeerKATCleanerDaemon::main quit_event true [1]")
            continue

          # directory containing transferred observations
          sent_dir = self.sent_dirs[proc_type] + "/" + beam
          finished_dir = self.finished_dirs[proc_type] + "/" + beam
          deleted_dir = self.deleted_dirs[proc_type] + "/" + beam

          if not os.path.exists(sent_dir):
            self.log(-1, "sent_dir [" + sent_dir + "] did not exist")
            os.makedirs(sent_dir, 0755)

          # look for observations that have been located in the sent dir and are old enough
          cmd = "find " + sent_dir + " -type f -name 'obs.finished' -mtime +" + self.clean_age_days + " | sort"
          rval, fin_files = self.system(cmd, 2)
          if rval:
            self.log (-1, "main find command failed: " + fin_files[0])
            sleep(1)
            continue

          self.log(1, "MeerKATCleanerDaemon::main files=" + str(fin_files))
          self.quit_event.set()
          continue

          self.log(2, "MeerKATCleanerDaemon::main assessing obs.finished observations") 
          # transfer the completed directory to herschel
          for path in fin_files:

            if self.quit_event.isSet():
              self.log(1, "MeerKATCleanerDaemon::main quit_event true [2]")
              continue

            # strip dir prefix
            subpath = path [(len(sent_dir)+1):] 

            # extract the the beam, utc, source and cfreq
            (utc, source, cfreq, file) = subpath.split("/")

            # delete the sent dir
            self.log(2, "MeerKATCleanerDaemon::main deleting sent dir")
            dir = sent_dir + "/" + utc + "/" + source + "/" + cfreq
            cmd = "rm -rf " + dir
            self.log(2, "MeerKATCleanerDaemon::main " + cmd)
            rval, lines= self.system(cmd, 2)
            if rval:
              self.log (-1, "failed to delete " + dir + ": " + str(lines))

            # count the number of other sub-band directories
            subdir_count = len(os.listdir(sent_dir + "/" + utc + "/" + source))
            self.log (2, "MeerKATArchiverDaemon::main subdir_count=" + str(subdir_count))
            if subdir_count == 0:
              os.rmdir(sent_dir + "/" + utc + "/" + source)
              os.rmdir (sent_dir + "/" + utc)
         
            # move finished dir to deleted for long term storage 
            fin_dir = finished_dir + "/" + utc
            del_dir = deleted_dir + "/" + utc

            if os.path.exists(fin_dir) and not os.path.exists(del_dir):

              cmd = "mv " + fin_dir + " " + del_dir
              self.log(2, "MeerKATCleanerDaemon::main " + cmd)
              rval, lines = self.system(cmd, 2)
              if rval:
                self.log (-1, "failed to move " + fin_dir + " to deleted: " + str(lines))

              self.log (1, proc_type + " " + utc + "/" + source + "/" + cfreq + " deleted")
            
            sleep (1)

      to_sleep = 60
      self.log (2, "MeerKATCleanerDaemon::main sleeping " + str(to_sleep) + " seconds")
      while to_sleep > 0 and not self.quit_event.isSet():
        sleep(1)
        to_sleep -= 1


###############################################################################
# Server based implementation
class MeerKATCleanerServerDaemon(MeerKATCleanerDaemon, ServerBased):

  def __init__ (self, name):
    MeerKATCleanerDaemon.__init__(self, name, "-1")
    ServerBased.__init__(self, self.cfg)

  def configure (self,become_daemon, dl, source, dest):

    MeerKATCleanerDaemon.configure (self, become_daemon, dl, source, dest)

    self.proc_types = ["fold", "search"]
    self.deleted_dirs = {}
    self.finished_dirs = {}
    self.sent_dirs = {}

    self.deleted_dirs["fold"]      = self.cfg["SERVER_FOLD_DIR"] + "/deleted"
    self.deleted_dirs["search"]    = self.cfg["SERVER_SEARCH_DIR"] + "/deleted"

    self.finished_dirs["fold"]      = self.cfg["SERVER_FOLD_DIR"] + "/finished"
    self.finished_dirs["search"]    = self.cfg["SERVER_SEARCH_DIR"] + "/finished"

    self.sent_dirs["fold"]      = self.cfg["SERVER_FOLD_DIR"] + "/sent"
    self.sent_dirs["search"]    = self.cfg["SERVER_SEARCH_DIR"] + "/sent"

    for i in range(int(self.cfg["NUM_BEAM"])):
      bid = self.cfg["BEAM_" + str(i)]
      self.beams.append(bid)
    return 0


###############################################################################
# Stream based implementation
class MeerKATCleanerStreamDaemon (MeerKATCleanerDaemon, StreamBased):

  def __init__ (self, name, id):
    MeerKATCleanerDaemon.__init__(self, name, str(id))
    StreamBased.__init__(self, str(id), self.cfg)

  def configure (self, become_daemon, dl, source, dest):

    self.log(1, "MeerKATCleanerStreamDaemon::configure()")
    MeerKATCleanerDaemon.configure(self, become_daemon, dl, source, dest)

    self.proc_types = ["fold", "search"]
    self.deleted_dirs = {}
    self.finished_dirs = {}
    self.sent_dirs = {}

    self.deleted_dirs["fold"]      = self.cfg["CLIENT_FOLD_DIR"] + "/deleted"
    self.deleted_dirs["search"]    = self.cfg["CLIENT_SEARCH_DIR"] + "/deleted"

    self.finished_dirs["fold"]      = self.cfg["CLIENT_FOLD_DIR"] + "/finished"
    self.finished_dirs["search"]    = self.cfg["CLIENT_SEARCH_DIR"] + "/finished"

    self.sent_dirs["fold"]      = self.cfg["CLIENT_FOLD_DIR"] + "/sent"
    self.sent_dirs["search"]    = self.cfg["CLIENT_SEARCH_DIR"] + "/sent"

    # determine the beam name
    (host, beam_id, subband_id) = self.cfg["STREAM_" + self.id].split(":")
    beam_name = self.cfg["BEAM_" + beam_id]
    self.beams.append(beam_name)

    # determine the subband config to find the centre frequency
    (cfreq, bw, nchan) = self.cfg["SUBBAND_CONFIG_" + subband_id].split(":")

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
    script = MeerKATCleanerServerDaemon ("meerkat_cleaner")
  else:
    print "ERROR: stream mode disabled for now"
    sys.exit(1)
    script = MeerKATCleanerStreamDaemon ("meerkat_cleaner", beam_id)

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

