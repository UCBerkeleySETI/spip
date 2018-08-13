#!/usr/bin/env python

##############################################################################
#  
#     Copyright (C) 2015 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
# 
###############################################################################
#
#     UWB Repack Search:
#       updates header parameters [PSRFITS]
#       copies output from a sub-band to web server
#


import os, sys, socket, select, signal, traceback, time, threading, copy, string, shutil

from time import sleep

from spip.daemons.bases import StreamBased,ServerBased
from spip.daemons.daemon import Daemon
from spip.threads.reporting_thread import ReportingThread
from spip.utils import times,sockets
from spip.config import Config

DAEMONIZE = True
DL        = 1

#############################################################################
# RepackSearchDaemon
class RepackSearchDaemon(Daemon):

  #############################################################################
  # RepackSearchDaemon::__init__
  def __init__ (self, name, id):
    Daemon.__init__(self, name, str(id))

    self.beam = ""
    self.subbands = []
    self.results = {}

  #################################################################
  # RepackSearchDaemon::main
  def main (self):

    archives_glob = "*.?f"

    self.log (2, "main: beam=" + str(self.beam))

    # archives stored in directory structure
    #  beam / utc_start / source / cfreq

    # summary data stored in
    #  beam / utc_start / source / freq.sum
    # out_cfreq = 0

    if not os.path.exists(self.processing_dir):
      os.makedirs(self.processing_dir, 0755) 
    if not os.path.exists(self.finished_dir):
      os.makedirs(self.finished_dir, 0755) 
    if not os.path.exists(self.send_dir):
      os.makedirs(self.send_dir, 0755) 

    self.log (2, "main: stream_id=" + str(self.id))

    while (not self.quit_event.isSet()):

      processed_this_loop = 0

      # check each beam for searched archives to process    
      beam_dir = self.processing_dir + "/" + self.beam
      self.log (2, "main: beam=" + self.beam + " beam_dir=" + beam_dir)

      if not os.path.exists(beam_dir):
        os.makedirs(beam_dir, 0755)

      # get a list of all the processing observations for any utc / source for this cfreq
      cmd = "find " + beam_dir + " -mindepth 3 -maxdepth 3 -type d -name '" + self.cfreq + "'"
      rval, observations = self.system (cmd, 2)

      # for each observation
      for observation in observations:
 
        # strip prefix 
        observation = observation[(len(beam_dir)+1):]

        (utc, source, cfreq) = observation.split("/")
        self.log (2, "main: found processing obs utc=" + utc + " source=" + source + " cfreq=" + cfreq)

        # skip any stats sources
        if source == "stats":
          continue

        in_dir = beam_dir + "/" + observation

        # find searched archives in the processing directory
        cmd = "find " + in_dir + " -mindepth 1 -maxdepth 1 " + \
              "-type f -name '" + archives_glob + "' -printf '%f\\n'"
        rval, files = self.system (cmd, 3)
        self.log (2, "main: utc=" + utc + " found " + str(len(files)) + " files")

        # processing in chronological order
        files.sort()

        for file in files:

          processed_this_loop += 1
          self.log (1, observation + ": processing " + file)

          self.log (2, "main: process_archive("+utc+", "+source+", "+file)
          (rval, response) = self.process_archive (utc, source, file)
          if rval:
            self.log (-1, "failed to process " + file + ": " + response)

        # check if the observation has been marked as failed by spip_proc
        filename = in_dir + "/obs.failed"
        if os.path.exists(filename):
          self.log (1, observation + ": processing -> failed")

          self.log (2, "main: fail_observation ("+utc+","+source+")")
          (rval, response) = self.fail_observation (utc, source)
          if rval:
            self.log (-1, "failed to finalise observation: " + response)

          # terminate this loop
          continue

        # check if the observation has been marked as finished by spip_proc
        filename = in_dir + "/obs.finished"
        if os.path.exists(filename):
          # if the file was created at least 20s ago, then there are unlikely to be
          # any more sub-ints to process
          self.log (2, observation + " mtime=" + str(os.path.getmtime(filename)) + " time=" + str(time.time()))
          if os.path.getmtime(filename) + 20 < time.time():
            self.log (1, observation + ": processing -> finished")
            self.log (2, "main: finalise_observation("+utc+","+source+")")
            (rval, response) = self.finalise_observation (utc, source)
            if rval:
              self.log (-1, "failed to finalise observation: " + response)

      if processed_this_loop == 0:
        to_sleep = 5
        while to_sleep > 0 and not self.quit_event.isSet():
          self.log (3, "time.sleep(1)")
          time.sleep(1)
          to_sleep -= 1

  #############################################################################
  # convert a UTC in ????-??-??-??:??:?? to ????-??-??-??-??-??
  def atnf_utc (self, instr):
    return instr.replace(":","-")

  #############################################################################
  # patch missing information into the PSRFITS header 
  def patch_psrfits_header (self, input_dir, input_file):

    header_file = input_dir + "/obs.header"
    self.log(3, "patch_psrfits_header: header_file="+header_file)

    header = Config.readCFGFileIntoDict (input_dir + "/obs.header")

    return (0, "")

  #############################################################################
  # create a remote directory for the observation
  def create_remote_dir (self, utc, source):

    in_dir = self.processing_dir + "/" + self.beam + "/" + utc + "/" + source + "/" + self.cfreq
    rem_dir = self.cfg["SERVER_SEARCH_DIR"] + "/processing/" + self.beam + "/" + utc + "/" + source + "/" + self.cfreq
    ctrl_file = in_dir + "/remdir.created"

    # use a local control file to know if remote directory exists
    if not os.path.exists(ctrl_file):
      cmd = "ssh " + self.rsync_user + "@" + self.rsync_server + " 'mkdir -p " + rem_dir + "'";
      rval, lines = self.system(cmd, 2)
      if rval:
        return (rval, "failed to create remote dir " + str(lines[0]))

      cmd = "touch " + ctrl_file
      rval, lines = self.system(cmd, 2)
      if rval:
        return (rval, "failed to create control file " + str(lines[0]))

      # copy the header to the remote directory
      header_file = in_dir + "/obs.header"
      cmd = "rsync -a " + header_file + " " + self.rsync_user + "@" + \
            self.rsync_server + "::" + self.rsync_module + "/search/processing/" + self.beam + "/" + utc + \
            "/" + source + "/" + self.cfreq + "/"

      # we dont have an rsync module setup on the server yet
      cmd = "rsync -a " + header_file + " " + self.rsync_user + "@" + \
            self.rsync_server + ":" + self.cfg["SERVER_SEARCH_DIR"] + "/processing/" + self.beam + "/" + utc + \
            "/" + source + "/" + self.cfreq + "/"

      rval, lines = self.system (cmd, 2)
      if rval:
        return (rval, "failed to rsync files to the server")

      return (0, "remote directory created")
    else:
      return (0, "remote directory existed")

  #############################################################################
  # process and file in the directory, adding file to 
  def process_archive (self, utc, source, file): 

    utc_out = self.atnf_utc(utc)
    in_dir = self.processing_dir + "/" + self.beam + "/" + utc + "/" + source + "/" + self.cfreq
    out_dir = self.send_dir + "/" + self.beam + "/" + utc_out +  "/" + source + "/" + self.cfreq

    # ensure required directories exist
    required_dirs = [out_dir]
    for dir in required_dirs:
      if not os.path.exists(dir):
        os.makedirs(dir, 0755)

    # create a remote directory on the server [should no longer be necessary]
    rval, message = self.create_remote_dir (utc, source)
    if rval:
      self.log (-2, "RepackSearchDaemon::process_archive create_remote_dir failed: " + message)
      return (rval, "failed to create remote directory: " + message)

    input_file = in_dir + "/" + file
    header_file = in_dir + "/obs.header"

    self.log (2, "process_archive() input_file=" + input_file + " header_file=" + header_file)
    if not os.path.exists(out_dir + "/obs.header"):
      cmd = "cp " + in_dir + "/obs.header " + out_dir + "/obs.header"
      rval, lines = self.system(cmd, 2)
      if rval:
        return (rval, "failed to copy obs.header to out_dir: " + str(lines[0]))

    # rename the psrfits file to match ATNF standards
    atnf_psrfits_file = out_dir + "/" + self.atnf_utc(os.path.basename (input_file))
    try:
      os.rename (input_file, atnf_psrfits_file)
    except OSError, e:
      self.log (0, "fail_observation: failed to rename " + psrfits_file + " to " + atnf_psrfits_file)
      return (1, "failed to rename psrfits file")

    return (0, "")

  #############################################################################
  # transition observation from processing to finished
  def finalise_observation (self, utc, source):

    utc_out = self.atnf_utc (utc)
    in_dir = self.processing_dir + "/" + self.beam + "/" + utc + "/" + source + "/" + self.cfreq
    fin_dir = self.finished_dir + "/" + self.beam + "/" + utc + "/" + source + "/" + self.cfreq
    send_dir = self.send_dir + "/" + self.beam + "/" + utc_out + "/" + source + "/" + self.cfreq

    # ensure required directories exist
    required_dirs = [fin_dir, send_dir]
    for dir in required_dirs:
      if not os.path.exists(dir):
        os.makedirs(dir, 0755)

    # touch and obs.finished file in the archival directory
    cmd = "touch " + send_dir + "/obs.finished"
    rval, lines = self.system (cmd, 3)
    if rval:
      return (1, "finalise_observation: failed to create " + send_dir + "/obs.finished")

    # create remote directory on server
    self.log (3, "RepackSearchDaemon::finalise_observation create_remote_dir(" + utc + ", " + source + ")")
    (rval, message) = self.create_remote_dir (utc, source)
    if rval:
      return (1, "finalise_observation: failed to create remote directory: " + message)

    # touch obs.finished file in the remote directory
    rem_cmd = "touch " + self.cfg["SERVER_SEARCH_DIR"] + "/processing/" + self.beam + \
          "/" + utc + "/" + source + "/" + self.cfreq + "/obs.finished"
    cmd = "ssh " + self.rsync_user + "@" + self.rsync_server + " '" + rem_cmd + "'"
    rval, lines = self.system(cmd, 2)
    if rval:
      return (rval, "failed to touch remote obs.finished file: " + str(lines[0]))

    # simply move the observation from processing to finished
    try:
      self.log (3, "finalise_observation: rename(" + in_dir + "," + fin_dir + ")")
      os.rename (in_dir, fin_dir)
    except OSError, e:
      self.log (0, "finalise_observation: failed to rename in_dir: " + str(e))
      return (1, "finalise_observation failed to rename in_dir to fin_dir")

    # count the number of other sub-band directories
    src_dir = os.path.dirname (in_dir)
    self.log (2, "RepackSearchDaemon::finalise_observation src_dir=" + src_dir + " list=" + str(os.listdir(src_dir)))
    subdir_count = len(os.listdir(src_dir))
    self.log (2, "RepackSearchDaemon::finalise_observation src_dir subdir_count=" + str(subdir_count))
    if subdir_count == 0:
      os.rmdir(src_dir)
      utc_dir = os.path.dirname (src_dir)
      os.rmdir(utc_dir)

    return (0, "")

  #############################################################################
  # transition an observation from processing to failed
  def fail_observation (self, utc, source):

    utc_out = self.atnf_utc (utc)
    in_dir = self.processing_dir + "/" + self.beam + "/" + utc + "/" + source + "/" + self.cfreq
    fail_dir = self.failed_dir + "/" + self.beam + "/" + utc + "/" + source + "/" + self.cfreq
    send_dir = self.send_dir + "/" + self.beam + "/" + utc_out + "/" + source + "/" + self.cfreq

    # ensure required directories exist
    required_dirs = [fail_dir, send_dir]
    for dir in required_dirs:
      if not os.path.exists(dir):
        os.makedirs(dir, 0755)

    # touch obs.failed file in the send directory
    cmd = "touch " + send_dir + "/obs.failed"
    rval, lines = self.system (cmd, 3)

    # create remote directory on the server [this should no longer be required]
    self.log (3, "RepackSearchDaemon::fail_observation create_remote_dir(" + utc + ", " + source + ")")
    (rval, message) = self.create_remote_dir (utc, source)
    if rval:
      return (1, "fail_observation: failed to create remote directory: " + message)

    # touch obs.failed file in the remote processing directory
    rem_cmd = "touch " + self.cfg["SERVER_SEARCH_DIR"] + "/processing/" + self.beam + \
          "/" + utc + "/" + source + "/" + self.cfreq + "/obs.failed"
    cmd = "ssh " + self.rsync_user + "@" + self.rsync_server + " '" + rem_cmd + "'"
    rval, lines = self.system(cmd, 2)
    if rval:
      return (rval, "failed to touch remote obs.failed file")

    # simply move the observation to the failed directory
    try:
      fail_parent_dir = os.path.dirname(fail_dir)
      if not os.path.exists(fail_parent_dir):
        os.mkdir (fail_parent_dir, 0755)
      os.rename (in_dir, fail_dir)
    except OSError, e:
      self.log (0, "fail_observation: failed to rename " + in_dir + " to " + fail_dir)
      self.log (0, str(e))
      return (1, "failed to rename in_dir to fail_dir")

    # count the number of other sub-band directories
    src_dir = os.path.dirname (in_dir)
    self.log (2, "RepackSearchDaemon::fail_observation src_dir=" + src_dir)
    subdir_count = len(os.listdir(src_dir))
    self.log (2, "RepackSearchDaemon::fail_observation src_dir subdir_count=" + str(subdir_count))
    if subdir_count == 0:
      os.rmdir(src_dir)
      utc_dir = os.path.dirname (src_dir)
      os.rmdir(utc_dir)

    return (0, "")

##########################a#######################################################
# RepackSearchStreamDaemon
class RepackSearchStreamDaemon (RepackSearchDaemon, StreamBased):

  #############################################################################
  # RepackSearchStreamDaemon::__init__
  def __init__ (self, name, id):
    RepackSearchDaemon.__init__(self, name, str(id))
    StreamBased.__init__(self, str(id), self.cfg)

  #############################################################################
  # RepackSearchStreamDaemon::configure
  def configure (self, become_daemon, dl, source, dest):
 
    self.log(2, "RepackSearchStreamDaemon::configure()")
    Daemon.configure(self, become_daemon, dl, source, dest)

    self.log(3, "RepackSearchStreamDaemon::configure stream_id=" + self.id)
    self.log(3, "RepackSearchStreamDaemon::configure beam_id=" + self.beam_id)
    self.log(3, "RepackSearchStreamDaemon::configure subband_id=" + self.subband_id)
  
    # beam_name
    self.beam = self.cfg["BEAM_" + str(self.beam_id)]

    # base directories for search data products
    self.processing_dir = self.cfg["CLIENT_SEARCH_DIR"] + "/processing"
    self.finished_dir   = self.cfg["CLIENT_SEARCH_DIR"] + "/finished"
    self.send_dir       = self.cfg["CLIENT_SEARCH_DIR"] + "/send"
    self.failed_dir     = self.cfg["CLIENT_SEARCH_DIR"] + "/failed"

    # get the properties for this subband
    (cfreq , bw, nchan) = self.cfg["SUBBAND_CONFIG_" + str(self.subband_id)].split(":")
    self.cfreq = cfreq
    self.bw = bw
    self.nchan = nchan

    # not sure is this is needed
    self.out_cfreq = cfreq

    # Rsync parameters
    self.rsync_user = "uwb"
    self.rsync_server = "medusa-srv0.atnf.csiro.au"
    self.rsync_module = "TBD"

    return 0

###############################################################################

if __name__ == "__main__":

  if len(sys.argv) != 2:
    print "ERROR: 1 command line argument expected"
    sys.exit(1)

  # this should come from command line argument
  stream_id = sys.argv[1]

  if int(stream_id) < 0:
    print "ERROR: RepackSearchDaemon can only operate on streams"
    sys.exit(1)
  else:
    script = RepackSearchStreamDaemon ("uwb_repack_search", stream_id)

  state = script.configure (DAEMONIZE, DL, "uwb_repack_search", "uwb_repack_search") 
  if state != 0:
    script.quit_event.set()
    sys.exit(state)

  script.log(1, "STARTING SCRIPT")

  try:

    script.main ()

  except:

    script.log(-2, "exception caught: " + str(sys.exc_info()[0]))
    formatted_lines = traceback.format_exc().splitlines()
    script.log(0, '-'*60)
    for line in formatted_lines:
      script.log(0, line)
    script.log(0, '-'*60)

    print '-'*60
    traceback.print_exc(file=sys.stdout)
    print '-'*60
    script.quit_event.set()

  script.log(1, "STOPPING SCRIPT")
  script.conclude()
  sys.exit(0)

