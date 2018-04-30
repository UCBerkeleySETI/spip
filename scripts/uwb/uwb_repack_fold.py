#!/usr/bin/env python

##############################################################################
#  
#     Copyright (C) 2015 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
# 
###############################################################################
#
#     UWB Repack Fold:
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
from spip.plotting import SNRPlot

DAEMONIZE = True
DL        = 1

class RepackFoldDaemon(Daemon):

  def __init__ (self, name, id):
    Daemon.__init__(self, name, str(id))

    self.valid_plots = ["freq_vs_phase", "flux_vs_phase", "time_vs_phase", "bandpass", "snr_vs_time"]
    self.beam = ""
    self.subbands = []
    self.results = {}
    self.snr_history = {}

    self.snr_plot = SNRPlot()

  #################################################################
  # main
  #       id >= 0   process folded archives from a stream
  #       id == -1  process folded archives from all streams
  def main (self):

    archives_glob = "*.ar"

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
    if not os.path.exists(self.archived_dir):
      os.makedirs(self.archived_dir, 0755) 

    self.log (2, "main: stream_id=" + str(self.id))

    while (not self.quit_event.isSet()):

      processed_this_loop = 0

      # check each beam for folded archives to process    
      beam_dir = self.processing_dir + "/" + self.beam
      self.log (2, "main: beam=" + self.beam + " beam_dir=" + beam_dir)

      if not os.path.exists(beam_dir):
        os.makedirs(beam_dir, 0755)

      # get a list of all the processing observations for any utc / source for this cfreq
      cmd = "find " + beam_dir + " -mindepth 3 -maxdepth 3 -type d -name '" + self.cfreq + "'"
      rval, observations = self.system (cmd, 3)

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

        # find folded archives in the processing directory
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

# 
# patch missing information into the PSRFITS header 
#
  def patch_psrfits_header (self, input_dir, input_file):

    header_file = input_dir + "/obs.header"
    self.log(3, "patch_psrfits_header: header_file="+header_file)

    header = Config.readCFGFileIntoDict (input_dir + "/obs.header")

    new = {}
    new["obs:observer"] = header["OBSERVER"] 
    new["obs:projid"]   = header["PID"]
    new["be:nrcvr"]     = "2"

    # need to know what these mean!
    new["be:phase"]     = "+1"    # Phase convention of backend
    new["be:tcycle"]    = "8"     # Correlator cycle time
    new["be:dcc"]       = "0"     # Downconversion conjugation corrected
    new["sub:nsblk"]    = "1"     # Samples/row (SEARCH mode, else 1)
  
    # this needs to come from CAM, hack for now
    new["ext:trk_mode"] = "TRACK" # Tracking mode
    new["ext:bpa"]      = "0" # Beam position angle [?]
    new["ext:bmaj"]     = "0" # Beam major axis [degrees]
    new["ext:bmin"]     = "0" # Beam minor axis [degrees]

    new["ext:obsfreq"]  = header["FREQ"]
    new["ext:obsbw"]    = header["BW"]
    new["ext:obsnchan"] = header["NCHAN"]

    new["ext:stp_crd1"] = header["RA"]
    new["ext:stp_crd2"] = header["DEC"]
    new["ext:stt_date"] = header["UTC_START"][0:10]
    new["ext:stt_time"] = header["UTC_START"][11:19]

    # create the psredit command necessary to apply "new"
    cmd = "psredit -m -c " + ",".join(['%s=%s' % (key, value) for (key, value) in new.items()]) + " " + input_file
    rval, lines = self.system(cmd, 2)
    if rval:
      return rval, lines[0]
    return 0, ""

  #
  # create a remote directory for the observation
  #
  def create_remote_dir (self, utc, source):

    in_dir = self.processing_dir + "/" + self.beam + "/" + utc + "/" + source + "/" + self.cfreq
    rem_dir = self.cfg["SERVER_FOLD_DIR"] + "/processing/" + self.beam + "/" + utc + "/" + source + "/" + self.cfreq
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

      return (0, "remote directory created")
    else:
      return (0, "remote directory existed")

  #
  # process and file in the directory, adding file to 
  #
  def process_archive (self, utc, source, file): 

    in_dir = self.processing_dir + "/" + self.beam + "/" + utc + "/" + source + "/" + self.cfreq
    out_dir = self.archived_dir + "/" + self.beam + "/" + utc + "/" + source + "/" + self.cfreq

    # ensure required directories exist
    out_source_dir = os.path.dirname(out_dir)
    out_utc_dir = os.path.dirname(out_source_dir)
    required_dirs = [out_utc_dir, out_source_dir, out_dir]
    for dir in required_dirs:
      if not os.path.exists(dir):
        os.makedirs(dir, 0755)

    rval, message = self.create_remote_dir (utc, source)
    if rval:
      self.log (-2, "RepackFoldDaemon::process_archive create_remote_dir failed: " + message)
      return (rval, "failed to create remote directory: " + message)

    input_file = in_dir + "/" + file
    header_file = in_dir + "/obs.header"

    self.log (2, "process_archive() input_file=" + input_file + " header_file=" + header_file)
    if not os.path.exists(out_dir + "/obs.header"):
      cmd = "cp " + in_dir + "/obs.header " + out_dir + "/obs.header"
      rval, lines = self.system(cmd, 2)
      if rval:
        return (rval, "failed to copy obs.header to out_dir: " + str(lines[0]))

    # convert the timer archive file to psrfits in the output directory
    cmd = "psrconv " + input_file + " -O " + out_dir
    rval, lines = self.system (cmd, 2)
    if rval:
      return (rval, "failed to copy processed file to archived dir")
    psrfits_file = out_dir + "/" + os.path.basename (input_file)
    self.log (2, "process_archive() psrfits_file=" + psrfits_file)

    # update the header parameters in the psrfits file
    (rval, message) = self.patch_psrfits_header (out_dir, psrfits_file)
    if rval:
      return (rval, "failed to convert psrfits file")

    # auto-zap bad channels
    cmd = "zap.psh -m " + input_file
    rval, lines = self.system (cmd, 2)
    if rval:
      return (rval, "failed to zap known bad channels")

    # scrunch to 128 channels for the Fsum
    cmd = "pam --setnchn 128 -m " + input_file
    rval, lines = self.system (cmd, 2)
    if rval:
      return (rval, "could not fscrunch to 128 channels")

    # rsync the zapped input file to the server
    cmd = "rsync -a " + header_file + " " + input_file + " " + self.rsync_user + "@" + \
          self.rsync_server + "::" + self.rsync_module + "/fold/processing/" + self.beam + "/" + utc + \
          "/" + source + "/" + self.cfreq + "/"

    # we dont have an rsync module setup on the server yet
    cmd = "rsync -a " + header_file + " " + input_file + " " + self.rsync_user + "@" + \
          self.rsync_server + ":" + self.cfg["SERVER_FOLD_DIR"] + "/processing/" + self.beam + "/" + utc + \
          "/" + source + "/" + self.cfreq + "/"

    rval, lines = self.system (cmd, 2)
    if rval:
      return (rval, "failed to rsync files to the server")

    cmd = "rm -f " + input_file
    rval, lines = self.system (cmd, 2)
    if rval:
      return (rval, "failed to delete local files")

    return (0, "")

  def fail_observation (self, utc, source):

    in_dir = self.processing_dir + "/" + self.beam + "/" + utc + "/" + source + "/" + self.cfreq
    fail_dir = self.failed_dir + "/" + self.beam + "/" + utc + "/" + source + "/" + self.cfreq
    arch_dir = self.archived_dir + "/" + self.beam + "/" + utc + "/" + source + "/" + self.cfreq

    # ensure required directories exist
    fail_source_dir = os.path.dirname(fail_dir)
    fail_utc_dir = os.path.dirname(fail_source_dir)
    required_dirs = [fail_utc_dir, fail_source_dir, arch_dir]
    for dir in required_dirs:
      if not os.path.exists(dir):
        os.makedirs(dir, 0755)

    # touch obs.failed file in the archival directory
    cmd = "touch " + arch_dir + "/obs.failed"
    rval, lines = self.system (cmd, 3)

    self.log (3, "RepackFoldDaemon::fail_observation create_remote_dir(" + utc + ", " + source + ")")
    (rval, message) = self.create_remote_dir (utc, source)
    if rval:
      return (1, "fail_observation: failed to create remote directory: " + message)

    # touch obs.failed file in the remote processing directory
    rem_cmd = "touch " + self.cfg["SERVER_FOLD_DIR"] + "/processing/" + self.beam + \
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
    parent_dir = os.path.dirname (in_dir)
    self.log (2, "RepackFoldDaemon::fail_observation parent_dir=" + parent_dir)
    subdir_count = len(os.listdir(parent_dir))
    self.log (2, "RepackFoldDaemon::fail_observation parent_dir subdir_count=" + str(subdir_count))
    if subdir_count == 0:
      os.rmdir(parent_dir)

    return (0, "")


  # transition observation from processing to finished
  def finalise_observation (self, utc, source):

    in_dir = self.processing_dir + "/" + self.beam + "/" + utc + "/" + source + "/" + self.cfreq
    fin_dir = self.finished_dir + "/" + self.beam + "/" + utc + "/" + source + "/" + self.cfreq
    arch_dir = self.archived_dir + "/" + self.beam + "/" + utc + "/" + source + "/" + self.cfreq

    # ensure required directories exist
    fin_source_dir = os.path.dirname(fin_dir)
    fin_utc_dir = os.path.dirname(fin_source_dir)
    required_dirs = [fin_utc_dir, fin_source_dir, arch_dir]
    for dir in required_dirs:
      if not os.path.exists(dir):
        os.makedirs(dir, 0755)

    # touch and obs.finished file in the archival directory
    cmd = "touch " + arch_dir + "/obs.finished"
    rval, lines = self.system (cmd, 3)
    if rval:
      return (1, "finalise_observation: failed to create " + arch_dir + "/obs.finished")

    self.log (3, "RepackFoldDaemon::finalise_observation create_remote_dir(" + utc + ", " + source + ")")
    (rval, message) = self.create_remote_dir (utc, source)
    if rval:
      return (1, "finalise_observation: failed to create remote directory: " + message)

    # touch obs.finished file in the remote directory
    rem_cmd = "touch " + self.cfg["SERVER_FOLD_DIR"] + "/processing/" + self.beam + \
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
    parent_dir = os.path.dirname (in_dir)
    self.log (2, "RepackFoldDaemon::finalise_observation parent_dir=" + parent_dir)
    subdir_count = len(os.listdir(parent_dir))
    self.log (2, "RepackFoldDaemon::finalise_observation parent_dir subdir_count=" + str(subdir_count))
    if subdir_count == 0:
      os.rmdir(parent_dir)

    return (0, "")

class RepackFoldStreamDaemon (RepackFoldDaemon, StreamBased):

  def __init__ (self, name, id):
    RepackFoldDaemon.__init__(self, name, str(id))
    StreamBased.__init__(self, str(id), self.cfg)

  def configure (self, become_daemon, dl, source, dest):
 
    self.log(2, "RepackFoldStreamDaemon::configure()")
    Daemon.configure(self, become_daemon, dl, source, dest)

    self.log(3, "RepackFoldStreamDaemon::configure stream_id=" + self.id)
    self.log(3, "RepackFoldStreamDaemon::configure beam_id=" + self.beam_id)
    self.log(3, "RepackFoldStreamDaemon::configure subband_id=" + self.subband_id)
  
    # beam_name
    self.beam = self.cfg["BEAM_" + str(self.beam_id)]

    # base directories for fold data products
    self.processing_dir = self.cfg["CLIENT_FOLD_DIR"] + "/processing"
    self.finished_dir   = self.cfg["CLIENT_FOLD_DIR"] + "/finished"
    self.archived_dir   = self.cfg["CLIENT_FOLD_DIR"] + "/archived"
    self.failed_dir     = self.cfg["CLIENT_FOLD_DIR"] + "/failed"

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

    self.log(3, "RepackFoldStreamDaemon::configure done")

    return 0

###############################################################################

if __name__ == "__main__":

  if len(sys.argv) != 2:
    print "ERROR: 1 command line argument expected"
    sys.exit(1)

  # this should come from command line argument
  stream_id = sys.argv[1]

  if int(stream_id) < 0:
    print "ERROR: RepackFoldDaemon can only operate on streams"
    sys.exit(1)
  else:
    script = RepackFoldStreamDaemon ("uwb_repack_fold", stream_id)

  state = script.configure (DAEMONIZE, DL, "uwb_repack_fold", "uwb_repack_fold") 
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

