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
class UWBArchiverDaemon(Daemon):

  def __init__ (self, name, id):
    Daemon.__init__(self, name, str(id))
    self.beams = []

    self.rsync_username = "swinburne"
    self.rsync_server = "herschel.atnf.csiro.au"
    self.rsync_module = "uwl_incoming"
    self.rsync_options =  "-ah --relative --stats --no-g --no-l --omit-dir-times " + \
                          "--password-file=/home/uwb/.ssh/herschel_pw --bwlimit=122070"

    self.num_rsync = 4
    self.rsync_servers = ["10.17.10.1", "10.17.10.1", "10.17.10.2", "10.17.10.2"]
    self.rsync_modules = ["uwl_incoming_0", "uwl_incoming_1", "uwl_incoming_2", "uwl_incoming_3"]


  def configure (self, become_daemon, dl, source, dest):
    Daemon.configure (self, become_daemon, dl, source, dest)
    return 0


  def get_rsync_from_stream(self, stream_id):

    (freq_str, bw_str, beam_str) = self.cfg["SUBBAND_CONFIG_" + str(stream_id)].split(":")

    freq = float(freq_str)

    # based on Lawrence's preferences
    if freq > 704 and freq < 1344:
      rsync_id = 0
    elif freq > 1344 and freq < 2368:
      rsync_id = 1
    elif freq > 2368 and freq < 4032:
      rsync_id = 2
    else:
      rsync_id = 3

    server = self.rsync_servers[rsync_id]
    module = self.rsync_modules[rsync_id]

    return (self.rsync_username, server, module)

  def main (self):

    self.debug("starting main loop")

    while not self.quit_event.isSet():

      # for each directory that has a completed dir
      for proc_type in self.proc_types:

        self.debug("proc_type=" + proc_type)

        # for each configured beam (there is only 1 for UWB)
        for beam in self.beams:

          self.debug("beam=" + beam)

          if self.quit_event.isSet():
            self.info("quit_event true [1]")
            continue

          # the input and output directories
          send_dir = self.send_dirs[proc_type] + "/" + beam
          junk_dir = self.junk_dirs[proc_type] + "/" + beam
          sent_dir = self.sent_dirs[proc_type] + "/" + beam

          if not os.path.exists(send_dir):
            self.warn("send_dir [" + send_dir + "] did not exist")
            os.makedirs(send_dir, 0755)

          if not os.path.exists(sent_dir):
            self.warn("sent_dir [" + sent_dir + "] did not exist")
            os.makedirs(sent_dir, 0755)

          if not os.path.exists(junk_dir):
            self.warn("sent_dir [" + junk_dir + "] did not exist")
            os.makedirs(junk_dir, 0755)

          # look for observations that have been completed and have / BEAM / utc / source / CFREQ
          self.debug("looking for obs.finished in " + send_dir + "/<UTC>/<SOURCE>/" + self.cfreq)
          cmd = "find " + send_dir + " -type f -path '*/" + self.cfreq + "/obs.finished' -mmin +1 | sort"
          rval, fin_files = self.system(cmd, 2)
          if rval:
            self.warn("find command failed: " + fin_files[0])
            sleep(1)
            continue
      
          self.debug("assessing obs.finished observations") 
          # transfer the completed directory to herschel
          for path in fin_files:

            if self.quit_event.isSet():
              self.info("quit_event true [2]")
              continue

            # strip dir prefix
            subpath = path [(len(send_dir)+1):] 

            # extract the the beam, utc, source and cfreq
            (utc, source, cfreq, file) = subpath.split("/")
            utc_source = utc + "/" + source

            self.debug("found obs to transfer " + utc_source)

            # finished and completed directories
            completed_subdir = utc_source + "/" + cfreq

            # determine the size of the data to be transferred
            cmd = "du -sb " + send_dir + "/" + completed_subdir + " | awk '{print $1}'"
            rval, size = self.system(cmd, 2)
            if rval:
              self.warn("failed to determine size of " + completed_subdir)
            else:
              self.debug("transferring " + (str(float(size[0])/1048576)) + " MB")
      
            # change to the beam directory
            os.chdir (send_dir)

            transfer = True
            # check the header file
            header_file = send_dir + "/" + completed_subdir + "/obs.header"
            if os.path.exists (header_file):
              header = Config.readCFGFileIntoDict (header_file)
              self.debug("utc=" + utc + " source=" + source + " pid=" + header["PID"])
              if header["PID"] == "P999":
                transfer = False

            if transfer:
            
              self.debug("get_rsync_from_stream (" + str(self.id) + ")")
              (username, server, module) = self.get_rsync_from_stream (self.id)
              self.debug("rsync stream=" + str(self.id)+ " user=" + username + " server=" + server + " module=" + module)

              # build the rsync command TODO handle fold/search/etc
              cmd = "rsync ./" + completed_subdir + " " + \
                    username + "@" + server + "::" + module + "/" + proc_type + "/ " + \
                    self.rsync_options + " --exclude='obs.finished'"
  
              # run the rsync command
              transfer_rate = ""
              transfer_success = True
              rval, lines = self.system (cmd, 2)
              if rval:
                transfer_success = False
                self.warn("failed to transfer " + completed_subdir)
                # TODO add support for terminating the transfer early

              else:

                # parse the transfer speed
                for line in lines:
                  if line.find ("bytes/sec") != -1:
                    transfer_rate = line
  
                # transfer the obs.finished file
                cmd = "rsync ./" + completed_subdir + "/obs.finished " + \
                      username + "@" + server + "::" + \
                      module + "/" + proc_type + "/ " + self.rsync_options
  
                # run the rsync command
                rval, size = self.system (cmd, 2)
                if rval:
                  transfer_success = False
                  self.warn("failed to transfer " + completed_subdir + "/obs.finished")
             
                if transfer_success:
                  # create a parent directory in the transferred dir
                  try:
                    os.makedirs(sent_dir + "/" + utc_source, 0755)
                  except OSError, e:
                    self.debug(str(e))

                  # now move this observation from send to sent
                  cmd = "mv " + send_dir + "/" + utc_source + "/" + cfreq + " " + sent_dir + "/" + utc_source
                  rval, lines = self.system(cmd, 2)

                  self.clean_utc_source_dir (send_dir + "/" + utc_source)
                  self.info(proc_type + " " + utc_source + "/" + cfreq + " transferred to " + module + ": " + transfer_rate)
                else:
                  self.info(proc_type + " " + utc_source + "/" + cfreq + " failed to transfer")
            else:

              # create a parent directory in the transferred dir
              try:
                os.makedirs(junk_dir + "/" + utc_source, 0755)
              except OSError, e:
                self.debug(str(e))

              # now move this observation from send to junk
              cmd = "mv " + send_dir + "/" + utc_source + "/" + cfreq + " " + junk_dir + "/" + utc_source + "/"
              rval, lines = self.system(cmd, 2)

              self.clean_utc_source_dir (send_dir + "/" + utc_source)
              self.info(proc_type + " " + utc_source + "/" + cfreq + " junked")

      to_sleep = 10
      self.debug("sleeping " + str(to_sleep) + " seconds")
      while to_sleep > 0 and not self.quit_event.isSet():
        sleep(1)
        to_sleep -= 1

  def clean_utc_source_dir (self, source_dir):

    self.debug("cleaning " + source_dir)
    if os.path.exists(source_dir):
      subdir_count = len(os.listdir(source_dir))
      self.debug("subdir_count=" + str(subdir_count))
      if subdir_count == 0:
        try:
          os.rmdir(source_dir)
          utc_dir = os.path.dirname (source_dir)
          try:
            os.rmdir(utc_dir)
          except OSError, e:
            self.info("clean_utc_source_dir os.rmdir(" + utc_dir + ") " + str(e))
        except OSError, e:
          self.info("clean_utc_source_dir os.rmdir(" + source_dir + ") " + str(e))



###############################################################################
# Server based implementation
class UWBArchiverServerDaemon(UWBArchiverDaemon, ServerBased):

  def __init__ (self, name):
    UWBArchiverDaemon.__init__(self, name, "-1")
    ServerBased.__init__(self, self.cfg)

  def configure (self,become_daemon, dl, source, dest):

    UWBArchiverDaemon.configure (self, become_daemon, dl, source, dest)

    self.proc_types = ["fold", "search"]
    self.send_dirs   = {}
    self.junk_dirs    = {}
    self.sent_dirs = {}

    self.send_dirs["fold"]   = self.cfg["SERVER_FOLD_DIR"] + "/send"
    self.send_dirs["search"] = self.cfg["SERVER_SEARCH_DIR"] + "/send"

    self.junk_dirs["fold"]   = self.cfg["SERVER_FOLD_DIR"] + "/junk"
    self.junk_dirs["search"] = self.cfg["SERVER_SEARCH_DIR"] + "/junk"

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

    self.info("()")
    UWBArchiverDaemon.configure(self, become_daemon, dl, source, dest)

    self.proc_types = ["fold", "search", "continuum", "preproc"]
    self.send_dirs   = {}
    self.junk_dirs    = {}
    self.sent_dirs = {}

    self.send_dirs["fold"]   = self.cfg["CLIENT_FOLD_DIR"] + "/send"
    self.send_dirs["search"] = self.cfg["CLIENT_SEARCH_DIR"] + "/send"
    self.send_dirs["continuum"] = self.cfg["CLIENT_CONTINUUM_DIR"] + "/send"
    self.send_dirs["preproc"] = self.cfg["CLIENT_PREPROC_DIR"] + "/send"

    self.junk_dirs["fold"]   = self.cfg["CLIENT_FOLD_DIR"] + "/junk"
    self.junk_dirs["search"] = self.cfg["CLIENT_SEARCH_DIR"] + "/junk"
    self.junk_dirs["continuum"] = self.cfg["CLIENT_CONTINUUM_DIR"] + "/junk"
    self.junk_dirs["preproc"] = self.cfg["CLIENT_PREPROC_DIR"] + "/junk"

    self.sent_dirs["fold"]   = self.cfg["CLIENT_FOLD_DIR"] + "/sent"
    self.sent_dirs["search"] = self.cfg["CLIENT_SEARCH_DIR"] + "/sent"
    self.sent_dirs["continuum"] = self.cfg["CLIENT_CONTINUUM_DIR"] + "/sent"
    self.sent_dirs["preproc"] = self.cfg["CLIENT_PREPROC_DIR"] + "/sent"

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
