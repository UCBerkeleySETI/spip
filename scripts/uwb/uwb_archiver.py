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

###############################################################################
#  Generic Implementation
class UWBArchiverDaemon(Daemon):

  def __init__ (self, name, id):
    Daemon.__init__(self, name, str(id))
    self.beams = []

    self.rsync_server = "herschel.atnf.csiro.au"
    self.rsync_username = "swinburne"
    self.rsync_module = "uwl_incoming"
    self.rsync_options =  "-a --relative --stats --no-g --no-l -chmod go-ws " + \
                          "--password-file=/home/uwb/.ssh/herschel_pw --bwlimit=10240"

  def configure (self, become_daemon, dl, source, dest):
    Daemon.configure (self, become_daemon, dl, source, dest)
    return 0

  def main (self):

    self.log(2, "UWBArchiverDaemon::main starting main loop")    

    while not self.quit_event.isSet():

      # for each directory that has a completed dir
      for proc_type in self.proc_types:

        self.log(2, "UWBArchiverDaemon::main proc_type=" + proc_type)

        # for each configured beam (there is only 1 for UWB)
        for beam in self.beams:

          self.log(2, "UWBArchiverDaemon::main beam=" + beam)

          if self.quit_event.isSet():
            continue

          # the input and output directories
          completed_dir = self.completed_dirs[proc_type] + "/" + beam
          sent_dir      = self.sent_dirs[proc_type] + "/" + beam

          if not os.path.exists(completed_dir):
            self.log(-1, "completed_dir [" + completed_dir + "] did not exist")
            cmd = "mkdir -p " + completed_dir
            rval, lines = self.system(cmd, 2)
            if rval:
              self.log(-2, "Could not create completed dir [" + completed_dir + "]")
              return 1

          if not os.path.exists(sent_dir):
            self.log(-1, "sent_dir [" + sent_dir + "] did not exist")
            cmd = "mkdir -p " + sent_dir
            rval, lines = self.system(cmd, 2)
            if rval:
              self.log(-2, "Could not create sent dir [" + sent_dir + "]")
              return 1

          # look for observations that have been completed and have / BEAM / utc / source / CFREQ
          self.log(2, "UWBArchiverDaemon::main looking for obs.finished in " + completed_dir + "/UTC/SOURCE/" + self.cfreq)
          cmd = "find " + completed_dir + " -type f -path '*/" + self.cfreq + "/obs.finished' -mmin +5 | sort"
          rval, fin_files = self.system(cmd, 2)
          if rval:
            self.log (-1, "UWBArchiverDaemon::main find command failed: " + fin_files[0])
            sleep(1)
            continue
      
          self.log(2, "UWBArchiverDaemon::main assessing obs.finished observations") 
          # transfer the completed directory to herschel
          for path in fin_files:

            # strip dir prefix
            subpath = path [(len(completed_dir)+1):] 

            # extract the the beam, utc, source and cfreq
            (utc, source, cfreq, file) = subpath.split("/")
            utc_source = utc + "/" + source

            self.log (2, "UWBArchiverDaemon::main found obs to transfer " + utc_source)

            # finished and completed directories
            finished_subdir = utc_source
            completed_subdir = utc_source + "/" + cfreq

            # determine the size of the data to be transferred
            cmd = "du -sb " + completed_dir + "/" + completed_subdir + " | awk '{print $1}'"
            rval, size = self.system(cmd, 2)
            if rval:
              self.log (-1, "failed to determine size of " + completed_subdir)
            else:
              self.log (2, "UWBArchiverDaemon::main transferring " + (str(float(size[0])/1048576)) + " MB")
      
            # change to the beam directory
            os.chdir (completed_dir)

            # build the rsync command TODO handle fold/search/etc
            cmd = "rsync ./" + completed_subdir + " " + \
                  self.rsync_username + "@" + self.rsync_server + "::" + self.rsync_module + "/" + proc_type + "/ " + \
                  self.rsync_options

            # run the rsync command
            rval, size = self.system (cmd, 2)
            if rval:
              self.log (-1, "failed to transfer " + completed_subdir)

            # create a parent directory in the transferred dir
            cmd = "mkdir -p " + sent_dir + "/" + utc_source
            rval, lines = self.system(cmd, 2)

            # now move this observation from completed to transferred
            cmd = "mv " + completed_dir + "/" + utc_source + "/" + cfreq + " " + sent_dir + "/" + utc_source
            rval, lines = self.system(cmd, 2)

            self.log (1, utc_source + " transferred")

            self.quit_event.set()

      to_sleep = 10
      self.log (2, "UWBArchiverDaemon::main sleeping " + str(to_sleep) + " seconds")
      while to_sleep > 0 and not self.quit_event.isSet():
        sleep(1)
        to_sleep -= 1



###############################################################################
# Server based implementation
class UWBArchiverServerDaemon(UWBArchiverDaemon, ServerBased):

  def __init__ (self, name):
    UWBArchiverDaemon.__init__(self, name, "-1")
    ServerBased.__init__(self, self.cfg)

  def configure (self,become_daemon, dl, source, dest):

    UWBArchiverDaemon.configure (self, become_daemon, dl, source, dest)

    self.proc_types = ["fold", "search"]
    self.completed_dirs   = {}
    self.finished_dirs    = {}
    self.sent_dirs = {}

    self.completed_dirs["fold"]   = self.cfg["SERVER_FOLD_DIR"] + "/archived"
    self.completed_dirs["search"] = self.cfg["SERVER_SEARCH_DIR"] + "/archived"

    self.finished_dirs["fold"]   = self.cfg["SERVER_FOLD_DIR"] + "/finished"
    self.finished_dirs["search"] = self.cfg["SERVER_SEARCH_DIR"] + "/finished"

    self.sent_dirs["fold"]   = self.cfg["SERVER_FOLD_DIR"] + "/sent"
    self.sent_dirs["search"] = self.cfg["SERVER_SEARCH_DIR"] + "/sent"

    for i in range(int(self.cfg["NUM_BEAM"])):
      bid = self.cfg["BEAM_" + str(i)]
      self.beams.append(bid)
    return 0

    self.cfreq = ""


###############################################################################
# Stream based implementation
class UWBArchiverStreamDaemon (UWBArchiverDaemon, StreamBased):

  def __init__ (self, name, id):
    UWBArchiverDaemon.__init__(self, name, str(id))
    StreamBased.__init__(self, str(id), self.cfg)

  def configure (self, become_daemon, dl, source, dest):

    self.log(1, "UWBArchiverStreamDaemon::configure()")
    UWBArchiverDaemon.configure(self, become_daemon, dl, source, dest)

    self.proc_types = ["fold", "search"]
    self.completed_dirs   = {}
    self.finished_dirs    = {}
    self.sent_dirs = {}

    self.completed_dirs["fold"]   = self.cfg["CLIENT_FOLD_DIR"] + "/archived"
    self.completed_dirs["search"] = self.cfg["CLIENT_SEARCH_DIR"] + "/archived"

    self.finished_dirs["fold"]   = self.cfg["CLIENT_FOLD_DIR"] + "/finished"
    self.finished_dirs["search"] = self.cfg["CLIENT_SEARCH_DIR"] + "/finished"

    self.sent_dirs["fold"]   = self.cfg["CLIENT_FOLD_DIR"] + "/sent"
    self.sent_dirs["search"] = self.cfg["CLIENT_SEARCH_DIR"] + "/sent"

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
    script = UWBArchiverServerDaemon ("uwb_archiver")
  else:
    script = UWBArchiverStreamDaemon ("uwb_archiver", beam_id)

  state = script.configure (DAEMONIZE, DL, "archiver", "archiver")
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
