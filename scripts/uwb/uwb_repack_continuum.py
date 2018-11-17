#!/usr/bin/env python

##############################################################################
#  
#     Copyright (C) 2015 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
# 
###############################################################################
#
#     UWB Repack Continuum:
#       StreamBased: copies output from a sub-band to server
#       ServerBased: creates monitoring plots for web server
#

import os, sys, socket, select, signal, traceback, time, threading, copy, string, shutil
import numpy as np

from time import sleep

from spip.daemons.bases import StreamBased,ServerBased
from spip.daemons.daemon import Daemon
from spip.threads.reporting_thread import ReportingThread
from spip.utils import times,sockets
from spip.config import Config
from spip.plotting import BandpassPlot

DAEMONIZE = True
DL        = 1

###############################################################################
# RepackReportingThread
# 
class RepackReportingThread(ReportingThread):

  #############################################################################
  # RepackReportingThread::__init__
  # 
  def __init__ (self, script, id):

    host = sockets.getHostNameShort()
    port = int(script.cfg["BEAM_REPACK_CONTINUUM_PORT"])

    if int(id) >= 0:
      port += int(id)
    ReportingThread.__init__(self, script, host, port)

    with open (script.cfg["WEB_DIR"] + "/spip/images/blankimage.gif", mode='rb') as file:
      self.no_data = file.read()

    self.script.log (1, "RepackReportingThread::ReportingThread listening on " + host + ":" + str(port))

  #############################################################################
  # RepackReportingThread::parse_message
  # 
  def parse_message (self, request):

    self.script.log (2, "RepackReportingThread::parse_message: " + str(request))

    xml = ""
    req = request["repack_request"]

    if req["type"] == "state":

      self.script.log (3, "RepackReportingThread::parse_message: preparing state response")
      xml = "<repack_state>"

      for beam in self.script.beams:

        self.script.log (2, "RepackReportingThread::parse_message: preparing state for beam: " + beam)

        self.script.results[beam]["lock"].acquire()

        xml += "<beam name='" + str(beam) + "' active='" + str(self.script.results[beam]["valid"]) + "'>"

        self.script.log (3, "RepackReportingThread::parse_message: keys="+str(self.script.results[beam].keys()))

        # always show the most recent result
        if self.script.results[beam]["valid"]:
          self.script.log (3, "RepackReportingThread::parse_message: beam " + str(beam) + " is valid!")

        xml += "<source>"
        xml += "<name epoch='J2000'>" + self.script.results[beam]["source"] + "</name>"
        xml += "</source>"

        xml += "<observation>"
        xml += "<start units='datetime'>" + self.script.results[beam]["utc_start"] + "</start>"
        xml += "<integrated units='seconds'>" + self.script.results[beam]["length"] + "</integrated>"
        xml += "</observation>"

        # allow for all possible configured streams
        for i in range(int(self.script.cfg["NUM_STREAM"])):
          (freq, bw, beam) = self.script.cfg["SUBBAND_CONFIG_" + str(i)].split(":")
          xml += "<plot type='bandpass" + freq +"' timestamp='" + self.script.results[beam]["timestamp"] + "'/>"

        xml += "</beam>"

        self.script.results[beam]["lock"].release()

      xml += "</repack_state>"
      self.script.log (2, "RepackReportingThread::parse_message: returning " + str(xml))

      return True, xml + "\r\n"

    elif req["type"] == "plot":

      self.script.log (2, "RepackReportingThread::parse_message: results[" + req["beam"] +"][lock].acquire()")

      self.script.results[req["beam"]]["lock"].acquire()
      self.script.log (2, "RepackReportingThread::parse_message: beam=" + \
                        req["beam"] + " plot=" + req["plot"])

      if req["plot"] in self.script.results[req["beam"]].keys() and len(self.script.results[req["beam"]][req["plot"]]) > 32:
        bin_data = copy.deepcopy(self.script.results[req["beam"]][req["plot"]])
        self.script.log (2, "RepackReportingThread::parse_message: beam=" + req["beam"] + " valid, image len=" + str(len(bin_data)))
        self.script.results[req["beam"]]["lock"].release()
        return False, bin_data
      else:
        if not req["plot"] in self.script.results[req["beam"]].keys():
          self.script.log (1, "RepackReportingThread::parse_message " + req["plot"] + " did not exist in results[" + req["beam"] + "].keys()")
        else:
          self.script.log (1, "RepackReportingThread::parse_message len(plot)= " + str(len(self.script.results[req["beam"]][req["plot"]])))

        self.script.log (2, "RepackReportingThread::parse_message beam was not valid")

        self.script.results[req["beam"]]["lock"].release()
        # still return if the timestamp is recent
        return False, self.no_data
    
    else:

      xml += "<repack_state>"
      xml += "<error>Invalid request</error>"
      xml += "</repack_state>\r\n"

      return True, xml

###################################################################
# RepackContinuumDaemon
# 
class RepackContinuumDaemon (Daemon):

  #################################################################
  # RepackContinuumDaemon::__init__
  # 
  def __init__ (self, name, id):
    Daemon.__init__(self, name, str(id))

    self.beams = []
    self.subbands = []
    self.results = {}
    self.freq_match = ""

  #################################################################
  # RepackContinuumDaemon::configure
  # 
  def configure (self, become_daemon, dl, source, dest):

    self.log (2, "RepackContinuumDaemon::configure")
    Daemon.configure(self, become_daemon, dl, source, dest)

    keys = ["source", "utc_start", "timestamp", "length", "freq", "bw"]
    for i in range(int(self.cfg["NUM_BEAM"])):
      beam = self.cfg["BEAM_" + str(i)]
      self.beams.append(beam)
      self.results[beam] = {}
      self.results[beam]["valid"] = False
      self.results[beam]["lock"] = threading.Lock()
      self.results[beam]["cond"] = threading.Condition(self.results[beam]["lock"])
      for key in keys:
        self.results[beam][key] = ""

    # the ploting class
    self.bp_plot = BandpassPlot()
    log = False
    zap = False
    transpose = False
    self.bp_plot.configure(log, zap, transpose)


  #################################################################
  # RepackContinuumDaemon::main
  # 
  def main (self):

    # data files stored in directory structure
    #   beam / utc_start / source / cfreq

    self.log (2, "RepackContinuumDaemon::main beams=" + str(self.beams))

    if not os.path.exists(self.processing_dir):
      os.makedirs(self.processing_dir, 0755) 
    if not os.path.exists(self.finished_dir):
      os.makedirs(self.finished_dir, 0755) 
    if not os.path.exists(self.send_dir):
      os.makedirs(self.send_dir, 0755) 

    self.log (2, "RepackContinuumDaemon::main stream_id=" + str(self.id))

    self.load_finished ()

    while (not self.quit_event.isSet()):

      processed_this_loop = 0

      # check each beam for continuum output files to process    
      for beam in self.beams:
        beam_dir = self.processing_dir + "/" + beam
        self.log (2, "RepackContinuumDaemon::main beam=" + beam + " beam_dir=" + beam_dir)

        if not os.path.exists(beam_dir):
          os.makedirs(beam_dir, 0755)

        # get a list of all the processing observations for any utc / source for the freq_match
        cmd = "find " + beam_dir + " -mindepth 3 -maxdepth 3 -type d -name '" + self.freq_match + "' -printf '%h\n' | sort | uniq"
        self.log (2, "RepackContinuumDaemon::main cmd=" + cmd)
        rval, observations = self.system (cmd, 3)
        self.log (2, "RepackContinuumDaemon::main observations=" + str(observations))

        # for each observation
        for observation in observations:

          if self.quit_event.isSet():
            continue

          # strip prefix 
          observation = observation[(len(beam_dir)+1):]
          (utc, source) = observation.split("/")
          self.log (2, "main: found processing obs utc=" + utc + " source=" + source)

          # find input data files in the processing directory
          files = self.find_input_files (beam_dir + "/" +  observation)
          self.log (2, "main: files=" + str(files))

          # process each file in the Stream/Server way
          for file in files:

            if self.quit_event.isSet():
              continue

            self.log (1, observation + ": processing " + file)

            self.log (2, "RepackContinuumDaemon::main process_file("+beam+","+utc+", "+source+", "+file)
            (rval, response) = self.process_file (beam, utc, source, file)
            if rval:
              self.log (-1, "failed to process " + file + ": " + response)
            else:
              processed_this_loop += 1

          if self.check_any_state (beam_dir + "/" +  observation, "failed"):
            (rval, response) = self.fail_observation (beam, utc, source)
            if rval:
              self.log (-1, "failed to fail observation: " + response)
            # terminate this loop
            continue

          if self.check_all_state (beam_dir + "/" +  observation, "finished"):
            (rval, response) = self.finalise_observation (beam, utc, source)
            if rval:
              self.log (-1, "failed to finalise observation: " + response)

        if processed_this_loop == 0:
          to_sleep = 5
          while to_sleep > 0 and not self.quit_event.isSet():
            self.log (3, "RepackContinuumDaemon::main time.sleep(1)")
            time.sleep(1)
            to_sleep -= 1

  #############################################################################
  # RepackContinuumDaemon::atnf_utc
  # convert a UTC in ????-??-??-??:??:?? to ????-??-??-??-??-??
  def atnf_utc (self, instr):
    return instr.replace(":","-")

  #################################################################
  # RepackContinuumDaemon::fail_observation
  # 
  def fail_observation (self, beam, utc, source):

    self.log (1, utc + "/" + source + ": processing -> failed")
    self.log (2, "RepackContinuumDaemon::fail_obsevation beam="+beam+" utc="+utc+"source="+source)

    utc_out = self.atnf_utc(utc)
    in_dir = self.processing_dir + "/" + beam + "/" + utc + "/" + source
    fail_dir = self.failed_dir + "/" + beam + "/" + utc + "/" + source
    send_dir = self.send_dir + "/" + beam + "/" + utc_out + "/" + source

    if not self.cfreq == "":
      in_dir = in_dir + "/" + self.cfreq
      fail_dir = fail_dir + "/" + self.cfreq
      send_dir = send_dir + "/" + self.cfreq

    self.log (2, "RepackContinuumDaemon::fail_obsevation in_dir=" + in_dir)
    self.log (2, "RepackContinuumDaemon::fail_obsevation fail_dir=" + fail_dir)
    self.log (2, "RepackContinuumDaemon::fail_obsevation send_dir=" + send_dir)

    # ensure required directories exist
    required_dirs = [fail_dir, send_dir]
    for dir in required_dirs:
      if not os.path.exists(dir):
        os.makedirs(dir, 0755)

    # ensure this observation is marked as failed
    self.mark_state (beam, utc, source, "failed")

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
    self.log (2, "RepackContinuumDaemon::fail_observation src_dir=" + src_dir)
    subdir_count = len(os.listdir(src_dir))
    self.log (2, "RepackContinuumDaemon::fail_observation src_dir subdir_count=" + str(subdir_count))
    if subdir_count == 0:
      os.rmdir(src_dir)
      utc_dir = os.path.dirname (src_dir)
      os.rmdir(utc_dir)

    return (0, "")


  #################################################################
  # RepackContinuumDaemon::finalise_observation
  # 
  def finalise_observation (self, beam, utc, source):

    utc_out = self.atnf_utc(utc)
    in_dir = self.processing_dir + "/" + beam + "/" + utc + "/" + source
    fin_dir = self.finished_dir + "/" + beam + "/" + utc + "/" + source
    send_dir = self.send_dir + "/" + beam + "/" + utc_out + "/" + source

    # if the file was created at least 20s ago, then finalise
    obs_finished_files = self.find_files (in_dir, "obs.finished")
    age = time.time() - os.path.getmtime(obs_finished_files[0])
    self.log (2, utc + " obs.finished age=" + str(age))
    if age < 30:
      return (0, "finalise_observation: not old enough yet")

    stream_daemon = (self.cfreq != "")

    # observation is old enough, append cfreq if requieted
    if stream_daemon:
      in_dir = in_dir + "/" + self.cfreq
      fin_dir = fin_dir + "/" + self.cfreq
      send_dir = send_dir + "/" + self.cfreq

    self.log (2, "RepackContinuumDaemon::finalise_obsevation in_dir=" + in_dir)
    self.log (2, "RepackContinuumDaemon::finalise_obsevation fin_dir=" + fin_dir)
    self.log (2, "RepackContinuumDaemon::finalise_obsevation send_dir=" + send_dir)

    # ensure required directories exist
    required_dirs = [fin_dir, send_dir]
    self.log (3, "RepackContinuumDaemon::finalise_obsevation required_dirs=" + str(required_dirs))
    for dir in required_dirs:
      if not os.path.exists(dir):
        self.log (3, "RepackContinuumDaemon::finalise_obsevation os.makedirs(" + str(dir))
        os.makedirs(dir, 0755)

    # mark as ready to finalise 
    self.log (1, utc + ": processing -> finished")

    # touch and obs.finished file in the send directory
    (rval, message) = self.mark_state (beam, utc, source, "finished")
    if rval:
      return (1, "finalise_observation: mark_state failed: " + message)

    # simply move the observation from processing to finished
    # for streams this is moving the cfreq subdir, for servers, it is the src dir
    try:
      self.log (2, "finalise_observation: rename(" + in_dir + "," + fin_dir + ")")
      os.rename (in_dir, fin_dir)
    except OSError, e:
      self.log (0, "finalise_observation: failed to rename in_dir: " + str(e))
      return (1, "finalise_observation failed to rename in_dir to fin_dir")

    if stream_daemon:
      src_dir = os.path.dirname (in_dir)
      self.log (2, "RepackContinuumDaemon::finalise_observation src_dir=" + src_dir)
      subdir_count = len(os.listdir(src_dir))
      self.log (2, "RepackContinuumDaemon::finalise_observation src_dir subdir_count=" + str(subdir_count))
      if subdir_count == 0:
        os.rmdir(src_dir)
        utc_dir = os.path.dirname (src_dir)
        os.rmdir(utc_dir)
    else:
      utc_dir = os.path.dirname (in_dir)
      os.rmdir(utc_dir)

    return (0, "")

  #############################################################################
  # RepackContinuumDaemon::check_any_state
  #   true if the required obs.state files exist 
  def check_any_state (self, in_dir, state):
    files = self.find_files (in_dir, "obs." + state)
    return len(files) > 0


###################################################################
# RepackContinuumServerDaemon
# 
class RepackContinuumServerDaemon (RepackContinuumDaemon, ServerBased):

  #################################################################
  # RepackContinuumServerDaemon::__init__
  # 
  def __init__ (self, name, id):
    RepackContinuumDaemon.__init__(self, name, "-1")
    ServerBased.__init__(self, self.cfg)
    self.files_glob = "????-??-??-??:??:??_*.*.dada.scr"

  #############################################################################
  # RepackContinuumServerDaemon::configure
  # 
  def configure (self, become_daemon, dl, source, dest):

    self.log(2, "RepackContinuumServerDaemon::configure()")
    RepackContinuumDaemon.configure(self, become_daemon, dl, source, dest)

    self.log(3, "RepackContinuumServerDaemon::configure stream_id=" + self.id)
    self.log(3, "RepackContinuumServerDaemon::configure subband_id=" + str(self.subband_id))

    # base directories for continuum data products
    self.processing_dir = self.cfg["SERVER_CONTINUUM_DIR"] + "/processing"
    self.finished_dir   = self.cfg["SERVER_CONTINUUM_DIR"] + "/finished"
    self.send_dir   = self.cfg["SERVER_CONTINUUM_DIR"] + "/send"
    self.failed_dir     = self.cfg["SERVER_CONTINUUM_DIR"] + "/failed"

    # server daemon matches on all input frequencies
    self.freq_match = "*"

    self.total_channels = 0
    for isubband in range(int(self.cfg["NUM_SUBBAND"])):
      (cfreq , bw, nchan) = self.cfg["SUBBAND_CONFIG_" + str(isubband)].split(":")
      self.subbands.append({ "cfreq": cfreq, "bw": bw, "nchan": nchan })
      self.total_channels += int(nchan)

    freq_low  = float(self.subbands[0]["cfreq"])  - (float(self.subbands[0]["bw"]) / 2.0)
    freq_high = float(self.subbands[-1]["cfreq"]) + (float(self.subbands[-1]["bw"]) / 2.0)
    self.out_cfreq = int(freq_low + ((freq_high - freq_low) / 2.0))

    self.cfreq = ""

    self.log(3, "RepackContinuumServerDaemon::configure done")

    return 0

  #################################################################
  # RepackContinuumServerDaemon::load_finished
  #   load the most recently finished observation
  def load_finished (self):

    # read the most recently finished observations
    for beam in self.beams:
      beam_dir = self.finished_dir + "/" + beam

      cmd = "find " + beam_dir + " -mindepth 2 -maxdepth 2 -type d | sort | tail -n 1"
      rval, observation = self.system (cmd, 3)

      self.log (2, "load_finished: observation=" + str(observation))
      if len(observation) < 1:
        return 0, ""

      # strip prefix 
      observation = observation[0][(len(beam_dir)+1):]

      self.log (2, "load_finished: " + observation)
      (utc, source) = observation.split("/")

      # find the band.last file in each subband
      cmd = "find " + beam_dir + "/" + utc + "/" + source + " -mindepth 2 -maxdepth 2 -type f -name 'band.last'"
      rval, files = self.system (cmd, 3)
      if rval:
        return (1, "load_finished: failed to find band.last files for " + utc + "/" + source)

      for file in files:
        self.log (2, "load_finished: plot_file(" + beam + "," + utc + "," + source + "," + file + ")")
        (rval, response) = self.plot_file (beam, utc, source, file)
        if rval:
          return (1, "load_finished: failed to plot " + beam + "/" + utc + "/" + source + "/" + file)

  #############################################################################
  # RepackContinuumServerDaemon::conclude
  #
  def acquire_obs_header (self, in_dir):
    """Generate the obs.header file for the whole band from sub-bands."""

    # test if already exists
    if os.path.exists (in_dir + "/obs.header"):
      self.log(2, "RepackContinuumServerDaemon::acquire_obs_header obs.header file already existed")
      return (0, "")

    # read obs.info file to find the subbands
    if not os.path.exists (in_dir + "/obs.info"):
      self.log(0, "RepackContinuumServerDaemon::acquire_obs_header obs.info file did not exist")
      return (0, "")

    info = Config.readCFGFileIntoDict (in_dir + "/obs.info")
    num_streams = info["NUM_STREAM"]
    subband_freqs = []
    for i in range(int(num_streams)):
      (freq, bw, beam) = info["SUBBAND_" + str(i)].split(":")
      subband_freqs.append(freq)
    
    # start with header file from first sub-band
    if not os.path.exists (in_dir + "/" + subband_freqs[0] + "/obs.header"):
      self.log(2, "RepackContinuumServerDaemon::acquire_obs_header first sub-band obs.header did not exist")
      return (1, "first sub-band header file did not exist")

    self.log (2, "RepackContinuumServerDaemon::acquire_obs_header header_file[0]=" + in_dir + "/" + subband_freqs[0] + "/obs.header")
    header = Config.readCFGFileIntoDict (in_dir + "/" + subband_freqs[0] + "/obs.header")

    # merge the headers from the other sub-bands
    for i in range(1,len(subband_freqs)):
      subband_header_file = in_dir + "/" + subband_freqs[i] + "/obs.header"
      self.log (2, "RepackContinuumServerDaemon::acquire_obs_header header_file[" + str(i)+ "]=" + subband_header_file)
      if os.path.exists (subband_header_file):
        header_sub = Config.readCFGFileIntoDict (subband_header_file)
        header = Config.mergeHeaderFreq (header, header_sub)
      else:
          return (1, "not all sub-band header files present: missing " + subband_header_file)

    # write the combined header
    self.log (2, "RepackContinuumServerDaemon::acquire_obs_header writing header to " + in_dir + "/" + "obs.header")
    Config.writeDictToCFGFile (header, in_dir + "/" + "obs.header")

    return (0, "")

  #############################################################################
  # RepackContinuumServerDaemon::check_any_state
  # 
  def check_all_state (self, in_dir, state):
    files = self.find_files (in_dir, "obs." + state)

    # read obs.info file to find the subbands
    if not os.path.exists (in_dir + "/obs.info"):
      self.log(0, "RepackContinuumServerDaemon::check_all_state obs.info file did not exist")
      return (0, "")

    info = Config.readCFGFileIntoDict (in_dir + "/obs.info")
    return len(files) == int(info["NUM_STREAM"])

  #############################################################################
  # RepackContinuumServerDaemon::conclude
  # 
  def conclude (self):
    for i in range(int(self.cfg["NUM_BEAM"])):
      beam = self.cfg["BEAM_" + str(i)]
      self.results[beam]["lock"].acquire()
      self.results[beam]["lock"].release()
    Daemon.conclude(self)

  #############################################################################
  # RepackContinuumServerDaemon::find_input_files
  # 
  def find_input_files (self, in_dir):
    cmd = "find " + in_dir + " -mindepth 2 -maxdepth 2 " + \
          "-type f -name '" + self.files_glob + "' -printf '%f\\n' | sort | uniq"
    rval, files = self.system (cmd, 2)

    self.log (2, "RepackContinuumServerDaemon::find_input_files found " + str(len(files)) + " files")
    return files

  #############################################################################
  # RepackContinuumServerDaemon::find_files
  # 
  def find_files (self, in_dir, glob):
    cmd = "find " + in_dir + " -mindepth 2 -maxdepth 2 " + "-type f -name '" + glob + "' | sort"
    rval, files = self.system (cmd, 2)
    self.log (2, "RepackContinuumServerDaemon::find_files found " + str(len(files)) + " files")
    return files

  #############################################################################
  # RepackContinuumServerDaemon::process_file_sum
  # process and file in the directory, adding file to 
  def process_file (self, beam, utc, source, file):

    out_utc = self.atnf_utc(utc)
    in_dir = self.processing_dir + "/" + beam + "/" + utc + "/" + source
    out_dir = self.send_dir + "/" + beam + "/" + out_utc + "/" + source

    # ensure required directories exist
    required_dirs = [out_dir]
    for dir in required_dirs:
      if not os.path.exists(dir):
        os.makedirs(dir, 0755)

    # process the data from sub-bands in the processing dir to a single file in the send dir
    input_file = in_dir + "/*/" + file
    header_file = in_dir + "/obs.header"

    # ensure that the obs.header file (merged from all sub-bands) is present
    self.log (2, "RepackContinuumServerDaemon::process_file acquire_obs_header(" + in_dir + ")") 
    rval, message = self.acquire_obs_header (in_dir)
    if rval:
      self.log (2, "RepackContinuumServerDaemon::process_file " + message)
      return (rval, message)

    self.log (2, "RepackContinuumServerDaemon::process_file input_file=" + \
                 input_file + " header_file=" + header_file)
    if not os.path.exists (out_dir + "/obs.header"):
      cmd = "cp " + in_dir + "/obs.header " + out_dir + "/obs.header"
      rval, lines = self.system(cmd, 2)
      if rval:
        return (rval, "failed to copy obs.header to out_dir: " + str(lines[0]))

    # get sub-band files
    cmd = "find " + in_dir + " -mindepth 2 -maxdepth 2 -type f -name '" + file + "'"
    rval, files = self.system (cmd, 3)
    if rval:
      return (rval, "failed to get list of sub-band files for " + file)

    for input_file in files:

      # plot this sub-band file, storing the result in self.results[beam]
      # self.plot_file (beam, utc, source, input_file)

      # remove in_dir from file
      in_dir_len = len(in_dir) + 1
      freq_file = input_file[in_dir_len:]

      self.log (2, "RepackContinuumServerDaemon::process_file freq_file=" + freq_file)

      # split into freq and filename
      (freq, filename) = freq_file.split("/")

      band_last_file = in_dir + "/" + freq + "/band.last"
      band_sum_file  = in_dir + "/" + freq + "/band.sum"

      # save the results
      try:
        # save this file in the directory for this sub-band
        if os.path.exists (band_last_file):
          os.unlink (band_last_file)
        os.rename (input_file, band_last_file)

        # add this file to the tscrunched sum
        if os.path.exists (band_sum_file):
          cmd = "uwb_continuum_tadd " + band_sum_file + " " + band_last_file + " " + band_sum_file
        else:
          cmd = "cp " + band_last_file + " " + band_sum_file

        rval, lines = self.system (cmd, 2)
        if rval:
          return (rval, "failed to add band_last to band_sum: " + str(lines))
      except OSError, e:
        self.log (0, "RepackContinuumServerDaemon::process_file failed to move input " + \
                     "file [" + input_flie + " to output file [" + output_file + "]")
        return (1, "process_file: failed to rename input file to output file")

    # plot the files once they have all been added
    band_lasts = self.find_files (in_dir, 'band.sum')
    for band_last in band_lasts:

      # plot this sub-band file, storing the result in self.results[beam]
      self.plot_file (beam, utc, source, band_last)

    return (0, "")

  #############################################################################
  # RepackContinuumServerDaemon::plot_file
  #   process a continuum dada file, generating a quick plot
  def plot_file (self, beam, utc, source, input_file):

    # read the ascii header in the input file
    fptr = open (input_file, "r")
    header = Config.readDictFromString(fptr.read (4096))
    fptr.close()

    fptr = open (input_file, "rb")

    # see past the header
    fptr.seek (int(header["HDR_SIZE"]))

    # NDAT should always be 1 on this averaged data
    ndat  = 1
    nsig  = 1
    subband = int(header["STREAM_SUBBAND_ID"])
    nbit  = int(header["NBIT"])
    nbin  = int(header["NBIN"])
    npol  = int(header["NPOL"])
    nchan = int(header["NCHAN"])
    freq = float(header["FREQ"])
    bw = float(header["BW"])
    order = header["ORDER"]
    bytes_per_second = float(header["BYTES_PER_SECOND"])
    obs_offset = (header["OBS_OFFSET"])

    if npol == 1:
      labels = ["AA+BB"]
    elif npol == 2:
      labels = ["AA","BB"]
    elif npol == 4:
      labels = ["AA","BB","Re(AB*)", "Im(AB*)"]
    else:
      labels = []

    nval = nbin * npol * nchan

    raw = np.fromfile (fptr, dtype=np.float32, count=nval)
    fptr.close()
    data = []
    freq_axis = -1
    if order == "TSFP":
      data = raw.reshape((ndat, nsig, nchan, npol))
      freq_axis = 2
    elif order == "TSPF":
      data = raw.reshape((ndat, nsig, npol, nchan))
      freq_axis = 3
    elif order == "TSPFB":
      data = raw.reshape((ndat, nsig, npol, nchan, nbin))
      freq_axis = 3
    else:
      return (1, "unsupported input data order")

    # generate plots of the combined spectrum
    self.results[beam]["lock"].acquire()

    self.results[beam]["utc_start"] = utc
    self.results[beam]["source"] = source
    self.results[beam]["timestamp"] = times.getCurrentTimeFromUnix(os.path.getmtime(input_file))

    prefix = "bandpass" + str(int(freq))
    self.log (2, "RepackContinuumServerDaemon::plot_file creating plots for " + prefix)
    self.bp_plot.plot_npol (800, 300, True, nchan, freq, bw, data[0][0], labels)
    self.results[beam][prefix + "_lo"] = self.bp_plot.getRawImage()
    self.bp_plot.plot_npol (1024, 768, False, nchan, freq, bw, data[0][0], labels)
    self.results[beam][prefix + "_hi"] = self.bp_plot.getRawImage()

    self.results[beam]["valid"] = True
    self.results[beam]["lock"].release()

    return (0, "")


  #############################################################################
  # RepackContinuumServerDaemon::process_file_sum
  # process and file in the directory, adding file to 
  def process_file_sum (self, beam, utc, source, file):

    in_dir = self.processing_dir + "/" + beam + "/" + utc + "/" + source 
    out_dir = self.send_dir + "/" + beam + "/" + utc + "/" + source

    # ensure required directories exist
    required_dirs = [out_dir]
    for dir in required_dirs:
      if not os.path.exists(dir):
        os.makedirs(dir, 0755)

    # process the data from sub-bands in the processing dir to a single file in the send dir
    input_file = in_dir + "/*/" + file
    header_file = in_dir + "/obs.header"

    # ensure that the obs.header file (merged from all sub-bands) is present
    rval, message = self.acquire_obs_header (in_dir)
    if rval:
      self.log (0, "RepackContinuumServerDaemon::process_file " + message)
      return rval, message

    self.log (2, "RepackContinuumServerDaemon::process_file input_file=" + \
                 input_file + " header_file=" + header_file)
    if not os.path.exists (out_dir + "/obs.header"):
      cmd = "cp " + in_dir + "/obs.header " + out_dir + "/obs.header"
      rval, lines = self.system(cmd, 2)
      if rval:
        return (rval, "failed to copy obs.header to out_dir: " + str(lines[0]))

    # get sub-band files
    cmd = "find " + in_dir + " -mindepth 2 -maxdepth 2 -type f -name '" + file + "'"
    rval, files = self.system (cmd, 3)
    if rval:
      return (rval, "failed to get list of sub-band files for " + file)    

    nfiles = len(files)

    if nfiles != int(self.cfg["NUM_SUBBAND"]):
      return (0, "not all sub-band files present")

    data  = {}
    freqs  = np.zeros (nfiles, dtype=[("sub-band", int), ("freq", float)])
    bws    = np.zeros (nfiles, dtype=float)
    nchans = np.zeros (nfiles, dtype=int)

    total_nchan = 0
    total_bw = 0
    freq_low = 1e9
    freq_high = -1e9
    bytes_per_second = 0
    obs_offset = 0 

    for i in range(len(files)):

      filename = files[i]
      fptr = open (filename, "r")
      header = Config.readDictFromString(fptr.read (4096))
      fptr.close()

      fptr = open (filename, "rb")

      # see past the header
      fptr.seek (int(header["HDR_SIZE"]))

      # NDAT should always be 1 on this averaged data
      ndat  = 1
      nsig  = 1
      nbit  = int(header["NBIT"])
      nbin  = int(header["NBIN"])
      npol  = int(header["NPOL"])
      nchan = int(header["NCHAN"])
      freq = float(header["FREQ"])
      bw = float(header["BW"])
      order = header["ORDER"]
      bytes_per_second = float(header["BYTES_PER_SECOND"])
      obs_offset = (header["OBS_OFFSET"])

      total_nchan = total_nchan + nchan
      total_bw = total_bw + bw
      if freq - bw/2 < freq_low:
        freq_low = freq - bw/2
      if freq + bw/2 > freq_high:
        freq_high = freq + bw/2

      if (nbit != 32):
        return (1, filename + " NBIT=" + str(nbit), ", but 32-bit floating point input required")

      if npol == 1:
        labels = ["AA+BB"]
      elif npol == 2:
        labels = ["AA","BB"]
      elif npol == 4:
        labels = ["AA","BB","Re(AB*)", "Im(AB*)"]
      else:
        labels = [] 

      nval = nbin * npol * nchan

      # store this for sorting later
      freqs[i] = (i, freq)
      bws[i] = bw
      nchans[i] = nchan

      raw = np.fromfile (fptr, dtype=np.float32, count=nval)
      freq_axis = -1
      if order == "TSFP": 
        data[i] = raw.reshape((ndat, nsig, nchan, npol))
        freq_axis = 2
      elif order == "TSPF":
        data[i] = raw.reshape((ndat, nsig, npol, nchan))
        freq_axis = 3
      elif order == "TSPFB":
        data[i] = raw.reshape((ndat, nsig, npol, nchan, nbin))
        freq_axis = 3
      else:
        return (1, "unsupported input data order")

    # sort the inputs on centre frequency
    freqs = np.sort(freqs, order='freq')

    # build a joined array
    full_data = []
    for (i, freq) in freqs:
      if full_data == []:
        full_data = data[i]
      else:
        full_data = np.append(full_data, data[i], axis=freq_axis)

    total_freq = freq_low + total_bw / 2

    # generate plots of the combined spectrum
    self.results[beam]["lock"].acquire()

    self.results[beam]["utc_start"] = utc
    self.results[beam]["source"] = source
    self.results[beam]["timestamp"] = times.getCurrentTimeFromUnix(os.path.getmtime(files[0]))

    # for each input subband
    for (i, freq) in freqs:
      prefix = "bandpass" + str(i)
      self.bp_plot.plot_npol (160, 120, False, nchans[i], freq, bws[i], data[i][0][0], labels)
      self.results[beam][prefix + "_lo"] = self.bp_plot.getRawImage()
      self.bp_plot.plot_npol (1024, 768, False, nchans[i], freq, bws[i], data[i][0][0], labels)
      self.results[beam][prefix + "_hi"] = self.bp_plot.getRawImage()

    # and then for the whole band
    self.bp_plot.plot_npol (900, 400, True, total_nchan, total_freq, total_bw, full_data[0][0], labels)
    self.results[beam]["bandpass_lo"] = self.bp_plot.getRawImage()
    self.bp_plot.plot_npol (1024, 768, False, total_nchan, total_freq, total_bw, full_data[0][0], labels)
    self.results[beam]["bandpass_hi"] = self.bp_plot.getRawImage()

    self.results[beam]["valid"] = True
    self.results[beam]["lock"].release()

    for input_file in files:

      # remove in_dir from file
      in_dir_len = len(in_dir) + 1
      freq_file = input_file[in_dir_len:]

      self.log (2, "RepackContinuumServerDaemon::process_file freq_file=" + freq_file)

      # split into freq and filename
      (freq, filename) = freq_file.split("/")

      self.log (2, "RepackContinuumServerDaemon::process_file freq=" + freq)
      # ensure the output directory exists
      if not os.path.exists(out_dir + "/" + freq):
        os.makedirs(out_dir + "/" + freq, 0755)

      output_file = out_dir + "/" + freq + "/" + filename

      self.log (2, "RepackContinuumServerDaemon::process_file input_file=" + input_file + " output_file=" + output_file)

      # rename/move the input file to the output file
      try:
        os.rename (input_file, output_file)
      except OSError, e:
        self.log (0, "RepackContinuumServerDaemon::process_file failed to move input file [" + input_flie + " to output file [" + output_file + "]")
        return (1, "process_file: failed to rename input file to output file")

    return (0, "")


  #############################################################################
  # RepackContinuumServerDaemon::process_file
  # process and file in the directory, adding file to 
  def mark_state (self, beam, utc, source, state):

    self.log (2, "RepackContinuumStreamDaemon::mark_state utc=" + utc + \
                 " source=" + source + " state=" + state)
    utc_out = self.atnf_utc(utc)
    # output directory
    send_dir = self.send_dir + "/" + beam + "/" + utc_out + "/" + source

    # touch and obs.finished file in the send directory
    cmd = "touch " + send_dir + "/obs." + state
    rval, lines = self.system (cmd, 3)
    if rval:
      return (1, "mark_state: failed to create " + send_dir + "/obs." + state)

    return (0, "")


#############################################################################
# RepackContinuumStreamDaemon
# 
class RepackContinuumStreamDaemon (RepackContinuumDaemon, StreamBased):

  #############################################################################
  # RepackContinuumStreamDaemon::__init__
  # 
  def __init__ (self, name, id):
    RepackContinuumDaemon.__init__(self, name, str(id))
    StreamBased.__init__(self, str(id), self.cfg)
    self.files_glob = "????-??-??-??:??:??_*.*.dada"

  #############################################################################
  # RepackContinuumStreamDaemon::configure
  # 
  def configure (self, become_daemon, dl, source, dest):
 
    self.log(2, "RepackContinuumStreamDaemon::configure()")
    RepackContinuumDaemon.configure(self, become_daemon, dl, source, dest)

    self.log(3, "RepackContinuumStreamDaemon::configure stream_id=" + self.id)
    self.log(3, "RepackContinuumStreamDaemon::configure subband_id=" + str(self.subband_id))
  
    # beam_name
    beam_name = self.cfg["BEAM_" + str(self.beam_id)]
    if not beam_name in self.beams:
      self.beams.append(beam_name)

    # base directories for continuum data products
    self.processing_dir = self.cfg["CLIENT_CONTINUUM_DIR"] + "/processing"
    self.finished_dir   = self.cfg["CLIENT_CONTINUUM_DIR"] + "/finished"
    self.send_dir   = self.cfg["CLIENT_CONTINUUM_DIR"] + "/send"
    self.failed_dir     = self.cfg["CLIENT_CONTINUUM_DIR"] + "/failed"

    # get the properties for this subband
    (cfreq , bw, nchan) = self.cfg["SUBBAND_CONFIG_" + str(self.subband_id)].split(":")
    self.cfreq = cfreq
    self.bw = bw
    self.nchan = nchan

    # stream daemon matches on just one centre frequency
    self.freq_match = str(cfreq)

    # not sure is this is needed
    self.out_cfreq = cfreq

    # Rsync parameters
    self.rsync_user = "uwb"
    self.rsync_server = "medusa-srv0.atnf.csiro.au"
    self.rsync_module = "TBD"

    self.log(3, "RepackContinuumStreamDaemon::configure done")

    return 0


  #############################################################################
  # RepackContinuumStreamDaemon::acquire_obs_header
  # 
  def acquire_obs_header (self, in_dir):
    return (0, "")

  #############################################################################
  # RepackContinuumStreamDaemon::check_all_state
  #   true if the required obs.state files exist 
  def check_all_state (self, in_dir, state):
    files = self.find_files (in_dir, "obs." + state)
    return len(files) == 1

  #############################################################################
  # RepackContinuumStreamDaemon::load_finished
  # 
  def load_finished (self):
    return (0, "")

  #############################################################################
  # RepackContinuumStreamDaemon::find_input_files
  #   return input files in the specified dir, matching the glob 
  def find_input_files (self, in_dir):
    cmd = "find " + in_dir + "/" + self.cfreq + " -mindepth 1 -maxdepth 1 " + \
          "-type f -name '" + self.files_glob + "' -printf '%f\\n' | sort | uniq"
    rval, files = self.system (cmd, 2)
    self.log (2, "RepackContinuumStreamDaemon::get_input_files found " + str(len(files)) + " files")
    return files

  #############################################################################
  # RepackContinuumStreamDaemon::find_files
  # 
  def find_files (self, in_dir, glob):
    cmd = "find " + in_dir + "/" + self.cfreq + " -mindepth 1 -maxdepth 1 " + "-type f -name '" + glob + "' | sort"
    rval, files = self.system (cmd, 2)
    self.log (2, "RepackContinuumStreamDaemon::find_files found " + str(len(files)) + " files")
    return files

  #############################################################################
  # RepackContinuumStreamDaemon::process_file
  # process and file in the directory, adding file to 
  def process_file (self, beam, utc, source, file):

    out_utc = self.atnf_utc(utc)
    in_dir = self.processing_dir + "/" + beam + "/" + utc + "/" + source + "/" + self.cfreq
    out_dir = self.send_dir + "/" + beam + "/" + out_utc + "/" + source + "/" + self.cfreq
    rem_dir = self.cfg["SERVER_CONTINUUM_DIR"] + "/processing/" + beam + "/" + utc + "/" + source + "/" + self.cfreq 

    # ensure required local directories exist
    required_dirs = [out_dir]
    for dir in required_dirs:
      if not os.path.exists(dir):
        os.makedirs(dir, 0755)

    # ensure required remote directories exist [should be able to remove this]
    rval, message = self.create_remote_dir (beam, utc, source)
    if rval:
      self.log (-2, "RepackContinuumStreamDaemon::process_file create_remote_dir failed: " + message)
      return (rval, "failed to create remote directory: " + message)

    header_file = in_dir + "/obs.header"
    input_file = in_dir + "/" + file
    output_file = out_dir + "/" + self.atnf_utc(file)
    scrunched_file = in_dir + "/" + file + ".scr"

    # copy the header file to the output directory
    if not os.path.exists(out_dir + "/obs.header"):
      cmd = "cp " + in_dir + "/obs.header " + out_dir + "/obs.header"
      rval, lines = self.system(cmd, 2)
      if rval:
        return (rval, "failed to copy obs.header to out_dir: " + str(lines[0]))

    # create a scrunched version of the input file
    cmd = "uwb_continuum_scrunch -c 1024 " + input_file + " " + scrunched_file
    rval, lines = self.system (cmd, 2)
    if rval:
      return (rval, "failed to scrunch input file: " + str(lines))

    # rsync the tscrunched input file to the server
    # cmd = "rsync -a " + header_file + " " + tscrunched_file + " " + self.rsync_user + "@" + \
    #       self.rsync_server + "::" + self.rsync_module + "/continuum/processing/" + beam + "/" + utc + \
    #       "/" + source + "/" + self.cfreq + "/"

    # we dont have an rsync module setup on the server yet
    cmd = "rsync -a " + header_file + " " + scrunched_file + " " + self.rsync_user + "@" + \
          self.rsync_server + ":" + rem_dir + "/" 

    rval, lines = self.system (cmd, 2)
    if rval:
      return (rval, "failed to rsync files to the server")

    # delete the scrunched version of the file
    os.unlink (scrunched_file)

    # rename/move the input file to the output file
    try:
      os.rename (input_file, output_file)
    except OSError, e:
      self.log (0, "process_file: failed to move input file [" + input_file + " to output file [" + output_file + "]")
      return (1, "process_file: failed to rename input file to output file")
  
    return (0, "")

  #############################################################################
  # RepackContinuumStreamDaemon::plot_continuum
  #   not used in the StreamDaemon
  def plot_continuum (self, beam, utc, source, tscrunched_file):

    # read requir
    fptr = open (tscrunched_file, "r")
    header = Config.readDictFromString(fptr.read (4096))
    fptr.close()

    fptr = open (tscrunched_file, "rb")

    # see past the header
    fptr.seek (int(header["HDR_SIZE"]))

    # NDAT should always be 1 on this averaged data
    ndat  = 1
    nsig  = 1
    subband = int(header["STREAM_SUBBAND_ID"])
    nbit  = int(header["NBIT"])
    nbin  = int(header["NBIN"])
    npol  = int(header["NPOL"])
    nchan = int(header["NCHAN"])
    freq = float(header["FREQ"])
    bw = float(header["BW"])
    order = header["ORDER"]
    bytes_per_second = float(header["BYTES_PER_SECOND"])
    obs_offset = (header["OBS_OFFSET"])

    if npol == 1:
      labels = ["AA+BB"]
    elif npol == 2:
      labels = ["AA","BB"]
    elif npol == 4:
      labels = ["AA","BB","Re(AB*)", "Im(AB*)"]
    else:
      labels = []

    nval = nbin * npol * nchan

    raw = np.fromfile (fptr, dtype=np.float32, count=nval)
    data = []
    freq_axis = -1
    if order == "TSFP":
      data = raw.reshape((ndat, nsig, nchan, npol))
      freq_axis = 2
    elif order == "TSPF":
      data = raw.reshape((ndat, nsig, npol, nchan))
      freq_axis = 3
    elif order == "TSPFB":
      data = raw.reshape((ndat, nsig, npol, nchan, nbin))
      freq_axis = 3
    else:
      return (1, "unsupported input data order")

    # generate plots of the combined spectrum
    self.results[beam]["lock"].acquire()

    self.results[beam]["utc_start"] = utc
    self.results[beam]["source"] = source
    self.results[beam]["bw"] = bw
    self.results[beam]["freq"] = freq
    self.results[beam]["timestamp"] = times.getCurrentTimeFromUnix(os.path.getmtime(tscrunched_file))

    prefix = "bandpass" + str(subband)
    self.bp_plot.plot_npol (160, 120, False, nchan, freq, bw, data[0][0], labels)
    self.results[beam][prefix + "_lo"] = self.bp_plot.getRawImage()
    self.bp_plot.plot_npol (1024, 768, False, nchan, freq, bw, data[0][0], labels)
    self.results[beam][prefix + "_hi"] = self.bp_plot.getRawImage()

    self.results[beam]["valid"] = True
    self.results[beam]["lock"].release()

  #############################################################################
  # RepackContinuumStreamDaemon::mark_file
  #
  def mark_state (self, beam, utc, source, state):

    utc_out = self.atnf_utc(utc)
    self.log (2, "RepackContinuumStreamDaemon::mark_state utc=" + utc + \
                 " source=" + source + " state=" + state)

    # output directory
    send_dir = self.send_dir + "/" + beam + "/" + utc_out + "/" + source + "/" + self.cfreq

    # touch and obs.finished file in the send directory
    cmd = "touch " + send_dir + "/obs." + state
    rval, lines = self.system (cmd, 3)
    if rval:
      return (1, "mark_state: failed to create " + send_dir + "/obs." + state)

    self.log (3, "RepackContinuumStreamDaemon::mark_state create_remote_dir(" + utc + ", " + source + ")")
    (rval, message) = self.create_remote_dir (beam, utc, source)
    if rval:
      return (1, "mark_state: failed to create remote directory: " + message)

    # touch obs.finished file in the remote directory
    rem_cmd = "touch " + self.cfg["SERVER_CONTINUUM_DIR"] + "/processing/" + beam + \
          "/" + utc + "/" + source + "/" + self.cfreq + "/obs.finished"
    cmd = "ssh " + self.rsync_user + "@" + self.rsync_server + " '" + rem_cmd + "'"
    rval, lines = self.system(cmd, 2)
    if rval:
      return (rval, "failed to touch remote obs.finished file: " + str(lines[0]))

    return (0, "")

  #############################################################################
  # RepackContinuumStreamDaemon::create_remote_dir
  # 
  def create_remote_dir (self, beam, utc, source):

    in_dir = self.processing_dir + "/" + beam + "/" + utc + "/" + source + "/" + self.cfreq
    self.log (3, "RepackContinuumStreamDaemon::create_remote_dir in_dir=" + in_dir)

    rem_dir = self.cfg["SERVER_CONTINUUM_DIR"] + "/processing/" + beam + \
              "/" + utc + "/" + source + "/" + self.cfreq
    self.log (3, "RepackContinuumStreamDaemon::create_remote_dir rem_dir=" + rem_dir)

    ctrl_file = in_dir + "/remdir.created"
    self.log (3, "RepackContinuumStreamDaemon::create_remote_dir ctrl_file=" + ctrl_file)

    # use a local control file to know if remote directory exists
    if not os.path.exists(ctrl_file):

      self.log (2, "RepackContinuumStreamDaemon::create_remote_dir creating remote dir")
      cmd = "ssh " + self.rsync_user + "@" + self.rsync_server + " 'mkdir -p " + rem_dir + "'";
      rval, lines = self.system(cmd, 2)
      if rval:
        return (rval, "failed to create remote dir " + str(lines[0]))

      self.log (2, "RepackContinuumStreamDaemon::create_remote_dir " + cmd)
      cmd = "touch " + ctrl_file
      rval, lines = self.system(cmd, 2)
      if rval:
        return (rval, "failed to create control file " + str(lines[0]))

      return (0, "remote directory created")
    else:
      return (0, "remote directory existed")


###############################################################################

if __name__ == "__main__":

  if len(sys.argv) != 2:
    print "ERROR: 1 command line argument expected"
    sys.exit(1)

  # this should come from command line argument
  stream_id = int(sys.argv[1])

  if stream_id < 0:
    script = RepackContinuumServerDaemon ("uwb_repack_continuum", stream_id)
  else:
    script = RepackContinuumStreamDaemon ("uwb_repack_continuum", stream_id)

  state = script.configure (DAEMONIZE, DL, "repack_continuum", "repack_continuum") 
  if state != 0:
    script.quit_event.set()
    sys.exit(state)

  script.log(1, "STARTING SCRIPT")

  try:

    if stream_id < 0:
      reporting_thread = RepackReportingThread(script, stream_id)
      reporting_thread.start()

    script.main ()

    if stream_id < 0:
      reporting_thread.join()

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

