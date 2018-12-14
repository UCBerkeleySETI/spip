#!/usr/bin/env python

##############################################################################
#  
#     Copyright (C) 2015 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
# 
###############################################################################

import os, sys, socket, select, signal, traceback, time, threading, copy, string, shutil

from time import sleep
from shutil import copyfile

from spip.daemons.bases import BeamBased,ServerBased
from spip.daemons.daemon import Daemon
from spip.threads.reporting_thread import ReportingThread
from spip.utils import times,sockets,files
from spip.config import Config
from spip.plotting import SNRPlot

DAEMONIZE = True
DL        = 1

class RepackReportingThread(ReportingThread):

  def __init__ (self, script, id):
    host = sockets.getHostNameShort()
    port = int(script.cfg["BEAM_REPACK_FOLD_PORT"])
    if int(id) >= 0:
      port += int(id)
    ReportingThread.__init__(self, script, host, port)

    with open (script.cfg["WEB_DIR"] + "/spip/images/blankimage.gif", mode='rb') as file:
      self.no_data = file.read()

    self.script.log (2, "RepackReportingThread::ReportingThread listening on " + host + ":" + str(port))

  def parse_message (self, request):

    self.script.log (3, "RepackReportingThread::parse_message: " + str(request))

    xml = ""
    req = request["repack_request"]

    if req["type"] == "state":

      self.script.log (3, "RepackReportingThread::parse_message: preparing state response")
      xml = "<repack_state>"

      for beam in self.script.beams:

        self.script.log (3, "RepackReportingThread::parse_message: preparing state for beam: " + beam)

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
        xml += "<snr>" + self.script.results[beam]["snr"] + "</snr>"
        xml += "</observation>"

        xml += "<plot type='flux_vs_phase' timestamp='" + self.script.results[beam]["timestamp"] + "'/>"
        xml += "<plot type='freq_vs_phase' timestamp='" + self.script.results[beam]["timestamp"] + "'/>"
        xml += "<plot type='time_vs_phase' timestamp='" + self.script.results[beam]["timestamp"] + "'/>"
        xml += "<plot type='bandpass' timestamp='" + self.script.results[beam]["timestamp"] + "'/>"
        xml += "<plot type='snr_vs_time' timestamp='" + self.script.results[beam]["timestamp"] + "'/>"

        xml += "</beam>"

        self.script.results[beam]["lock"].release()

      xml += "</repack_state>"
      self.script.log (3, "RepackReportingThread::parse_message: returning " + str(xml))
    
      return True, xml + "\r\n"

    elif req["type"] == "plot":
     
      if req["plot"] in self.script.valid_plots:

        self.script.log (3, "RepackReportingThread::parse_message: results[" + req["beam"] +"][lock].acquire()")

        self.script.results[req["beam"]]["lock"].acquire()
        self.script.log (3, "RepackReportingThread::parse_message: beam=" + \
                          req["beam"] + " plot=" + req["plot"]) 

        #if self.script.results[req["beam"]]["valid"]:
        if req["plot"] in self.script.results[req["beam"]].keys() and len(self.script.results[req["beam"]][req["plot"]]) > 32:
          bin_data = copy.deepcopy(self.script.results[req["beam"]][req["plot"]])
          self.script.log (3, "RepackReportingThread::parse_message: beam=" + req["beam"] + " valid, image len=" + str(len(bin_data)))
          self.script.results[req["beam"]]["lock"].release()
          return False, bin_data
        else:
          if not req["plot"] in self.script.results[req["beam"]].keys():
            self.script.log (1, "RepackReportingThread::parse_message " + req["plot"] + " did not exist in results[" + req["beam"] + "].keys()")
          else:
            self.script.log (1, "RepackReportingThread::parse_message len(plot)= " + str(len(self.script.results[req["beam"]][req["plot"]])))
           
          self.script.log (3, "RepackReportingThread::parse_message beam was not valid")

          self.script.results[req["beam"]]["lock"].release()
          # still return if the timestamp is recent
          return False, self.no_data

      else:
        # still return if the timestamp is recent
        return False, self.no_data

      xml += "<repack_state>"
      xml += "<error>Invalid request</error>"
      xml += "</repack_state>\r\n"

      return True, xml

class RepackDaemon(Daemon):

  def __init__ (self, name, id):
    Daemon.__init__(self, name, str(id))

    self.plot_types = ["freq_vs_phase", "flux_vs_phase", "time_vs_phase", "bandpass", "snr_vs_time"]
    self.plot_resses = ["lo", "hi"]
    self.valid_plots = []
    for plot_type in self.plot_types:
      for plot_res in self.plot_resses:
        self.valid_plots.append(plot_type + "_" + plot_res)

    self.zap_psh_script = "zap.psh"
    self.convert_psrfits = True
    self.psrfits_rf_filename = False
    self.nchan_plot = 1024
    self.beams = []
    self.subbands = []
    self.results = {}
    self.snr_history = {}

    self.snr_plot = SNRPlot()

    self.suffixes = { "flux_vs_phase_lo": "flux.lo.png",
                      "flux_vs_phase_hi": "flux.hi.png",
                      "freq_vs_phase_lo": "freq.lo.png",
                      "freq_vs_phase_hi": "freq.hi.png",
                      "time_vs_phase_lo": "time.lo.png",
                      "time_vs_phase_hi": "time.hi.png",
                      "snr_vs_time_lo":   "snrt.lo.png",
                      "snr_vs_time_hi":   "snrt.hi.png",
                      "bandpass_lo":      "band.lo.png",
                      "bandpass_hi":      "band.hi.png" }


  #################################################################
  # main
  #       id >= 0   process folded archives from a stream
  #       id == -1  process folded archives from all streams
  def main (self):

    archives_glob = "*.ar"

    self.log (2, "main: beams=" + str(self.beams))

    # archives stored in directory structure
    #  beam / utc_start / source / cfreq / "fold"

    # summary data stored in
    #  beam / utc_start / source / freq.sum

    if not os.path.exists(self.processing_dir):
      os.makedirs(self.processing_dir, 0755) 
    if not os.path.exists(self.finished_dir):
      os.makedirs(self.finished_dir, 0755) 
    if not os.path.exists(self.archived_dir):
      os.makedirs(self.archived_dir, 0755) 

    self.log (2, "main: stream_id=" + str(self.id))

    self.load_finished ()

    while (not self.quit_event.isSet()):

      processed_this_loop = 0

      # check each beam for folded archives to process    
      for beam in self.beams:

        beam_dir = self.processing_dir + "/" + beam
        self.log (3, "main: beam=" + beam + " beam_dir=" + beam_dir)

        if not os.path.exists(beam_dir):
          os.makedirs(beam_dir, 0755)

        # get a list of all the recent observations
        cmd = "find " + beam_dir + " -mindepth 2 -maxdepth 2 -type d"
        rval, observations = self.system (cmd, 3)

        # for each observation      
        for observation in observations:

          processed_this_obs = 0
   
          # strip prefix 
          observation = observation[(len(beam_dir)+1):]

          (utc, source) = observation.split("/")

          if source == "stats":
            continue

          obs_dir = beam_dir + "/" + observation

          if not os.path.exists(obs_dir + "/obs.info"):
            continue

          info_age = files.get_file_age (obs_dir + "/obs.info")
          if info_age < 5:
            continue

          # read the subbands in this observation
          subbands = self.get_subbands (obs_dir)
          (ok, out_cfreq) = self.get_out_cfreq (obs_dir)
          if not ok:
            self.log (-1, "could not determine out cfreq for " + obs_dir)
            continue

          out_dir = self.archived_dir + "/" + beam + "/" + utc + "/" + source + "/" + str(out_cfreq)
          if not os.path.exists(out_dir):
            os.makedirs(out_dir, 0755)

          # if we have only 1 sub-band, then files can be processed immediately
          archives = {}
          for subband in subbands:
            self.log (3, "processing subband=" + str(subband))

            # if this sub-band directory exists yet
            if os.path.exists(obs_dir + "/" + subband):
            
              cmd = "find " + obs_dir + "/" + subband + " -mindepth 1 -maxdepth 1 " + \
                    "-type f -name '" + archives_glob + "' -printf '%f\\n'"
              rval, filelist = self.system (cmd, 3)

              for file in filelist:
                if not file in archives:
                  archives[file] = 0
                archives[file] += 1

          # if a file meets the subband count it is ripe for processing
          filelist = archives.keys()
          filelist.sort()

          for file in filelist:

            self.log (2, observation + ": " + file + " has " + str(archives[file]) \
                      + " of " + str(len(subbands)) + " present")

            if archives[file] == len(subbands):

              self.log (1, observation + ": processing " + file)

              processed_this_obs += 1
              if len(subbands) > 1:
                self.log (2, "main: process_subband()")
                (rval, response) = self.process_subband (obs_dir, out_dir, source, file)
                if rval:
                  self.log (-1, "failed to process sub-bands for " + file + ": " + response)
              else:
                input_file  = obs_dir  + "/" + subbands[0] + "/" + file
                self.log (2, "main: process_archive() "+ input_file)
                (rval, response) = self.process_archive (obs_dir, input_file, out_dir, source)
                if rval:
                  self.log (-1, "failed to process " + file + ": " + response)

          # if any individual file (from sub-bands) were found, update the monitoring plots etc
          if processed_this_obs > 0:
            # now process the sum files to produce plots etc
            self.log (2, "main: process_observation("+beam+","+utc+","+source+","+obs_dir+")")
            (rval, response) = self.process_observation (beam, utc, source, obs_dir)
            if rval:
              self.log (-1, "failed to process observation: " + response)

          # if the proc has marked this observation as finished
          any_failed = False
          all_done = True

          # perhaps a file was produced whilst the previous list was being processed,
          # do another pass
          if len(filelist) > 0:
            all_done = False

          for subband in subbands:
            fin_file = obs_dir + "/" + subband + "/obs.finished"
            fail_file = obs_dir + "/" + subband + "/obs.failed"

            # all sub-bands need a file produced
            if not (os.path.exists(fin_file) or os.path.exists(fail_file)):
              all_done = False
            else:
              # finished file exists and is 10s old
              if os.path.exists (fin_file):
                if os.path.getmtime(fin_file) + 10 < time.time():
                  self.log (3, "main: found finished subband: " + fin_file)
                else:
                  all_done = False
              # fail file exists and is 30s old
              if os.path.exists (fail_file):
                if os.path.getmtime(fail_file) + 30 < time.time():
                  self.log (2, "found failed subband: " + fail_file)
                  any_failed = True
                else:
                  self.log (3, "found failed subband that was too young")
                  all_done = False

          self.log (2, "obs_dir=" + obs_dir + " all_done=" + str(all_done) + " any_failed=" + str(any_failed))

          # check that at least 1 archive was produced
          # if all_done and not any_failed:
          #   cmd = "find " + out_dir + " -type f -name '????-??-??-??:??:??.ar' | wc -l"
          #   rval, lines = self.system(cmd, 3)
          #   if rval or  lines[0] == "0":
          #     self.log (-1, "main: no archives have been produced in " + out_dir)
          #     all_done = False

          # the observation is completed
          if all_done:

            # and failed
            if any_failed:
              self.log (1, observation + ": processing -> failed")
              fail_parent_dir = self.failed_dir + "/" + beam + "/" + utc
              if not os.path.exists(fail_parent_dir):
                os.makedirs(fail_parent_dir, 0755)
              fail_dir = self.failed_dir + "/" + beam + "/" + utc + "/" + source
              self.log (2, "main: fail_observation("+beam+","+obs_dir+","+fail_dir+","+out_dir+")")
              (rval, response) = self.fail_observation (beam, obs_dir, fail_dir, out_dir)
              if rval:
                self.log (-1, "failed to finalise observation: " + response)

            # and finished
            else:
              self.log (1, observation + ": processing -> finished")

              fin_parent_dir = self.finished_dir + "/" + beam + "/" + utc
              if not os.path.exists(fin_parent_dir):
                os.makedirs(fin_parent_dir, 0755)

              fin_dir = self.finished_dir + "/" + beam + "/" + utc + "/" + source
              self.log (2, "main: finalise_observation("+beam + "," + obs_dir + "," + fin_dir + "," + out_dir + ")")
              (rval, response) = self.finalise_observation (beam, obs_dir, fin_dir, out_dir)
              if rval:
                self.log (-1, "failed to finalise observation: " + response)
              else:

                # ensure the merged obs.header file exists
                (rval, response) = self.acquire_obs_header (fin_dir)
                if rval:
                  self.log(-1, "main: could not acquire obs.header from " + fin_dir)
                else:
                  # copy the merged header file to the output directory
                  self.log (2, "main: copying header to " + out_dir + "/" + "obs.header")
                  shutil.copyfile (fin_dir + "/obs.header", out_dir + "/obs.header")

                  # remove sub-band files and directories that are now superfluous
                  for i in range(len(subbands)):
                    os.remove (fin_dir + "/" + subbands[i] + "/obs.finished")
                    os.remove (fin_dir + "/" + subbands[i] + "/obs.header")
                    os.rmdir (fin_dir + "/" + subbands[i])

          processed_this_loop += processed_this_obs
          self.log (2, "main: finished processing loop for " + observation)
          
      if processed_this_loop == 0:
        self.log (3, "time.sleep(1)")
        time.sleep(1)

  def get_subbands (self, obs_dir):

    subband_freqs = []

    # read obs.info file to find the subbands
    if not os.path.exists (obs_dir + "/obs.info"):
      self.log(0, "RepackDaemon::get_subbands " + obs_dir + "/obs.info file did not exist")
      return subband_freqs

    info = Config.readCFGFileIntoDict (obs_dir + "/obs.info")
    num_streams = info["NUM_STREAM"]
    for i in range(int(num_streams)):
      (freq, bw, beam) = info["SUBBAND_" + str(i)].split(":")
      subband_freqs.append(freq)
    self.log(2, "RepackDaemon::get_subbands subband_freqs=" + str(subband_freqs))
    return subband_freqs

  def get_out_cfreq (self, obs_dir):

    # read obs.info file to find the subbands
    if not os.path.exists (obs_dir + "/obs.info"):
      self.log(0, "RepackDaemon::get_out_cfreq obs.info file did not exist")
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
    self.log(2, "RepackDaemon::get_out_cfreq low=" + str(freq_low) + " high=" + str(freq_high) + " cfreq=" + str(cfreq))
    return (True, cfreq)

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

      self.log (1, "load_finished: " + observation)
      (utc, source) = observation.split("/")

      obs_dir = self.finished_dir + "/" + beam + "/" + utc + "/" + source

      time_file = obs_dir + "/time.sum"
      freq_file = obs_dir + "/freq.sum"
      timestamp = ""

      cmd = "find  " + obs_dir + " -mindepth 1 -maxdepth 1 -type f -name '????-??-??-??:??:??.*.png' | sort"
      rval, pngs = self.system (cmd, 3)

      self.results[beam]["lock"].acquire()

      for png in pngs:
        for key in self.suffixes.keys():
          suffix = self.suffixes[key]
          if png.find(suffix) >= 0:
            fptr = open (png, "rb")
            self.results[beam][key] = fptr.read()
            fptr.close()

        if timestamp == "":
          filename = png.split("/")[-1]
          timestamp = filename.split(".")[0]
          self.log (2, "load_finished: png=" + png + " filename=" + filename + " timestamp=" + timestamp)

      self.results[beam]["utc_start"] = utc
      self.results[beam]["source"] = source
      self.log (2, "load_finished: utc_start=" + utc + " source=" + source)
    
      cmd = "psrstat -jFDp -c snr " + freq_file + " | awk -F= '{printf(\"%f\",$2)}'"
      rval, lines = self.system (cmd, 3)
      if rval < 0:
        return (rval, "failed to extract snr from freq.sum")
      snr = lines[0]

      cmd = "psrstat -c length " + time_file + " | awk -F= '{printf(\"%f\",$2)}'"
      rval, lines = self.system (cmd, 3)
      if rval < 0:
        return (rval, "failed to extract time from time.sum")
      length = lines[0]

      self.results[beam]["timestamp"] = timestamp
      self.results[beam]["snr"] = snr
      self.results[beam]["length"] = length
      self.results[beam]["valid"] = False 

      self.log (2, "load_finished: self.results["+beam+"][valid]=False")
      self.results[beam]["lock"].release()

    return 0, ""


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

    new["be:nrcvr"]     = header["NPOL"]
    
    try:
      val               = header["RCVR_HAND"]
      new["rcvr:hand"]  = val
    except KeyError as e:
      self.log(2, "patch_psrfits_header: RCVR_HAND not set in header")

    try:
      val               = header["BACKEND_PHASE"] # Phase convention of backend
      new["be:phase"]   = val
    except KeyError as e:
      self.log(2, "patch_psrfits_header: BACKEND_PHASE not set in header")

    try:
      val               = header["FOLD_OUTTSUBINT"] # Correlator cycle time
      new["be:tcycle"]  = val
    except KeyError as e:
      self.log(2, "patch_psrfits_header: FOLD_OUTTSUBINT not set in header")

    new["be:dcc"]       = "0"     # Downconversion conjugation corrected
    new["sub:nsblk"]    = "1"     # Samples/row (SEARCH mode, else 1)
  
    # this needs to come from CAM, hack for now
    new["ext:trk_mode"] = "TRACK" # Tracking mode
    new["ext:bpa"]      = "0" # Beam position angle [?]
    new["ext:bmaj"]     = "0" # Beam major axis [degrees]
    new["ext:bmin"]     = "0" # Beam minor axis [degrees]
    
    self.log(3, "RepackDaemon::patch_psrfits_header freq=" + str(header["FREQ"]))

    new["ext:obsfreq"]  = header["FREQ"]
    new["ext:obsbw"]    = header["BW"]
    new["ext:obsnchan"] = header["NCHAN"]

    new["ext:stp_crd1"] = header["RA"]
    new["ext:stp_crd2"] = header["DEC"]
    new["ext:stt_date"] = header["UTC_START"][0:10]
    new["ext:stt_time"] = header["UTC_START"][11:19]

    # build psredit command, in-place modification 
    cmd = "psredit -m"

    try:
      itrf = header["ITRF"]
      (x, y, z) = itrf.split(",")
      new["itrf:ant_x"] = x
      new["itrf:ant_y"] = y
      new["itrf:ant_z"] = z
      cmd = cmd + " -a itrf"

    except KeyError as e:
      self.log(2, "patch_psrfits_header: ITRF not set in header")

    # create the psredit command necessary to apply "new"
    cmd = cmd + " -c " + ",".join(['%s=%s' % (key, value) for (key, value) in new.items()]) + " " + input_file
    rval, lines = self.system(cmd, 2)
    if rval:
      return rval, lines[0]
    return 0, ""

  #
  # process and file in the directory, adding file to 
  #
  def process_archive (self, in_dir, input_file, out_dir, source):

    (rval, response) = self.acquire_obs_header (in_dir)
    if rval:
      return (-1, "process_archive: could not acquire obs.header from " + in_dir)

    self.log (2, "process_archive() input_file=" + input_file)

    freq_file   = in_dir + "/freq.sum"
    time_file   = in_dir + "/time.sum"
    band_file   = in_dir + "/band.last"

    if self.convert_psrfits:

      # convert the timer archive file to psrfits in the output directory
      cmd = "psrconv " + input_file + " -O " + out_dir
      rval, lines = self.system (cmd, 2)
      if rval:
        return (rval, "failed to copy processed file to archived dir")
      psrfits_file = out_dir + "/" + os.path.basename (input_file)

      if self.psrfits_rf_filename:
        psrfits_file = string.replace(psrfits_file, ".ar", ".rf")

        self.log (2, "process_archive() psrfits_file=" + psrfits_file)

        # rename the output file
        try:
          shutil.move (input_file, psrfits_file)
        except OSError, e:
          return (-1, "failed to rename input_file to psrfits_file: " + str(e))

      # update the header parameters in the psrfits file
      (rval, message) = self.patch_psrfits_header (in_dir, psrfits_file)
      if rval:
        return (rval, "failed to convert psrfits file")

    # auto-zap bad channels
    cmd = self.zap_psh_script + " -m " + input_file
    rval, lines = self.system (cmd, 2)
    if rval:
      return (rval, "failed to zap known bad channels")

    # bscrunch to 4 bins for the bandpass plot
    cmd = "pam --setnbin 4  -e bscr " + input_file
    rval, lines = self.system (cmd, 2)
    if rval:
      return (rval, "could not bscrunch to 4 bins")

    # copy/overwrite the bandpass file
    input_band_file = string.replace(input_file, ".ar", ".bscr")
    if os.path.exists (band_file):
      os.remove (band_file)
    cmd = "mv " + input_band_file + " " + band_file
    rval, lines = self.system (cmd, 2)
    if rval:
      return (rval, "failed to copy recent band file")
    if not os.path.exists (band_file):
      self.log (1, "process_archive: " + band_file + " did not exist after copy...")

    # get the number of channels
    cmd = "psredit -q -c nchan " + input_file + " | awk -F= '{print $2}'"
    rval, lines = self.system (cmd, 2)
    if rval:
      return (rval, "failed to get number of channels")
    nchan = int(lines[0])
    
    while nchan > self.nchan_plot:
      nchan = nchan / 2

    # pscrunch and fscrunch to < self.nchan_plot channels for the Fsum
    cmd = "pam -p --setnchn " + str(nchan) + " -m " + input_file
    rval, lines = self.system (cmd, 2)
    if rval:
      return (rval, "could not fscrunch to " + str(nchan) + " channels")

    # add the archive to the freq sum, tscrunching it
    if not os.path.exists(freq_file):
      try:
        shutil.copyfile (input_file, freq_file)
      except IOError, e:
        return (rval, "failed to copy archive to freq.sum: " + str(e))

    else:
      cmd = "psradd -T -o " + freq_file + " " + freq_file + " " + input_file
      rval, lines = self.system (cmd, 2)
      if rval:
        return (rval, "failed add archive to freq.sum")

    # fscrunch the archive 
    cmd = "pam -m -F " + input_file
    rval, lines = self.system (cmd, 2)
    if rval:
      return (rval, "failed add Fscrunch archive")

    # add it to the time sum
    if not os.path.exists (time_file):
      try:
        shutil.copyfile (input_file, time_file)
      except IOError, e:
        return (-1, "failed rename Fscrunched archive to time.sum: " + str(e))
    else:
      cmd = "psradd -o " + time_file + " " + time_file + " " + input_file
      rval, lines = self.system (cmd, 2)
      if rval:
        return (rval, "failed add Fscrunched archive to time.sum")
  
    # now delete the working input file
    try:
      os.remove (input_file)
    except OSError, e:
      return (-1, "failed remove Fscrunched archive")

    return (0, "")

  def acquire_obs_header (self, in_dir):
    """Generate the obs.header file for the whole band from sub-bands."""

    # test if already exists
    if os.path.exists (in_dir + "/obs.header"):
      self.log(2, "RepackDaemon::acquire_obs_header obs.header file already existed")
      return (0, "")
  
    subband_freqs = self.get_subbands (in_dir)

    # start with header file from first sub-band
    if not os.path.exists (in_dir + "/" + subband_freqs[0] + "/obs.header"):
      self.log(2, "RepackDaemon::acquire_obs_header first sub-band obs.header did not exist")
      return (1, "first sub-band header file did not exist")

    self.log (2, "RepackDaemon::acquire_obs_header header_file[0]=" + in_dir + "/" + subband_freqs[0] + "/obs.header")
    header = Config.readCFGFileIntoDict (in_dir + "/" + subband_freqs[0] + "/obs.header")

    # merge the headers from the other sub-bands
    for i in range(1,len(subband_freqs)):
      subband_header_file = in_dir + "/" + subband_freqs[i] + "/obs.header"
      self.log (2, "RepackDaemon::acquire_obs_header header_file[" + str(i)+ "]=" + subband_header_file)
      if os.path.exists (subband_header_file):
        header_sub = Config.readCFGFileIntoDict (subband_header_file)
        header = Config.mergeHeaderFreq (header, header_sub)
      else:
        return (1, "not all sub-band header files present")

    # write the combined header
    self.log (2, "RepackDaemon::acquire_obs_header writing header to " + in_dir + "/" + "obs.header")
    Config.writeDictToCFGFile (header, in_dir + "/" + "obs.header")

    return (0, "")


  #
  # process all sub-bands for the same archive
  #
  def process_subband (self, in_dir, out_dir, source, file):

    interim_file = "/dev/shm/" + file
    input_files = in_dir + "/*/" + file

    cmd = "psradd -R -o " + interim_file + " " + input_files
    rval, observations = self.system (cmd, 3)
    if rval:
      return (rval, "failed to add sub-band archives to interim file")

    (rval, response) = self.process_archive (in_dir, interim_file, out_dir, source)
    if rval:
      return (rval, "process_archive failed: " + response)
    
    # remove in the input sub-banded files
    cmd = "rm -f " + input_files
    rval, lines = self.system (cmd, 3)
    if rval:
      return (rval, "failed to delete input files")

    return (0, "")

  def process_observation (self, beam, utc, source, in_dir):

    self.log (2, "process_observation: beam="+beam+" utc="+utc+" source="+source) 

    freq_file   = in_dir + "/freq.sum"
    time_file   = in_dir + "/time.sum"
    band_file   = in_dir + "/band.last"

    timestamp = times.getCurrentTime() 
    lo_res = " -g 240x180 -c x:view=0:1 -c y:view=0:1"
    hi_res = " -g 1024x768"
    opts = " -c above:l= -c above:c=  -D -/png"

    # create the freq plots
    cmd = "psrplot -p freq " + freq_file + " -jDp -j 'r 0.5'" + lo_res + opts
    self.log (2, "process_observation: " + cmd)
    rval, freq_lo_raw = self.system_raw (cmd, 3)
    if rval < 0:
      return (rval, "failed to create freq plot")

    cmd = "psrplot -p freq " + freq_file + " -jDp -j 'r 0.5'" + hi_res + opts
    self.log (2, "process_observation: " + cmd)
    rval, freq_hi_raw = self.system_raw (cmd, 3)
    if rval < 0:
      return (rval, "failed to create freq plot")

    cmd = "psrplot -p time " + time_file + " -jDp -j 'r 0.5'" + lo_res + opts
    self.log (2, "process_observation: " + cmd)
    rval, time_lo_raw = self.system_raw (cmd, 3)
    if rval < 0:
      return (rval, "failed to create time plot")

    cmd = "psrplot -p time " + time_file + " -jDp -j 'r 0.5'" + hi_res + opts
    self.log (2, "process_observation: " + cmd)
    rval, time_hi_raw = self.system_raw (cmd, 3)
    if rval < 0:
      return (rval, "failed to create time plot")

    cmd = "psrplot -p flux " + freq_file + " -jFDp -j 'r 0.5'" + lo_res + opts
    self.log (2, "process_observation: " + cmd)
    rval, flux_lo_raw = self.system_raw (cmd, 3)
    if rval < 0:
      return (rval, "failed to create flux plot")

    cmd = "psrplot -p flux " + freq_file + " -jFDp -j 'r 0.5'" + hi_res + opts
    self.log (2, "process_observation: " + cmd)
    rval, flux_hi_raw = self.system_raw (cmd, 3)
    if rval < 0:
      return (rval, "failed to create flux plot")

    # get the number of polarisations
    cmd = "psredit -q -c npol " + band_file + " | awk -F= '{print $2}'"
    rval, lines = self.system (cmd, 2)
    if rval:
      return (rval, "failed to get number of polarisations")
    npol = int(lines[0])

    # plot the bandpass
    basecmd = "psrplot -p b -x "
    if npol >= 2:
      basecmd = basecmd + "-lpol=0,1 -O -c log=1 "
  
    cmd = basecmd + band_file + lo_res + opts
    self.log (2, "process_observation: " + cmd)
    rval, bandpass_lo_raw = self.system_raw (cmd, 2)
    if rval < 0:
      return (rval, "failed to create bandpass plot")

    cmd = basecmd + band_file + hi_res + opts
    self.log (2, "process_observation: " + cmd)
    rval, bandpass_hi_raw = self.system_raw (cmd, 2)
    if rval < 0:
      return (rval, "failed to create bandpass plot")

    cmd = "psrstat -jFDp -c snr " + freq_file + " | awk -F= '{printf(\"%f\",$2)}'"
    rval, lines = self.system (cmd, 3)
    if rval < 0:
      return (rval, "failed to extract snr from freq.sum")
    snr = lines[0]

    cmd = "psrstat -c length " + time_file + " | awk -F= '{printf(\"%f\",$2)}'"
    rval, lines = self.system (cmd, 3)
    if rval < 0:
      return (rval, "failed to extract time from time.sum")
    length = lines[0]

    self.log (2, "process_observation: self.results[" + beam + "][lock].acquire")
    self.results[beam]["lock"].acquire() 

    self.results[beam]["utc_start"] = utc
    self.results[beam]["source"] = source
    self.results[beam]["freq_vs_phase_lo"] = copy.deepcopy(freq_lo_raw)
    self.results[beam]["flux_vs_phase_lo"] = copy.deepcopy(flux_lo_raw)
    self.results[beam]["time_vs_phase_lo"] = copy.deepcopy(time_lo_raw)
    self.results[beam]["bandpass_lo"]      = copy.deepcopy(bandpass_lo_raw)
    self.results[beam]["freq_vs_phase_hi"] = copy.deepcopy(freq_hi_raw)
    self.results[beam]["flux_vs_phase_hi"] = copy.deepcopy(flux_hi_raw)
    self.results[beam]["time_vs_phase_hi"] = copy.deepcopy(time_hi_raw)
    self.results[beam]["bandpass_hi"]      = copy.deepcopy(bandpass_hi_raw)
    self.results[beam]["timestamp"] = timestamp
    self.results[beam]["snr"] = snr
    self.results[beam]["length"] = length
    self.results[beam]["valid"] = True
    self.log (2, "process_observation: self.results[" + str(beam) + "][valid] = True")
    
    t1 = int(times.convertLocalToUnixTime(timestamp))
    t2 = int(times.convertUTCToUnixTime(utc))
    delta_time = t1 - t2
    self.log (2, "process_observation: timestamp=" + timestamp + " utc=" + utc + 
              " t1=" + str(t1) + " t2=" + str(t2) + " delta_time=" + str(delta_time))
    self.snr_history[beam]["times"].append(delta_time)
    self.snr_history[beam]["snrs"].append(snr)

    self.snr_plot.configure()
    self.snr_plot.plot (240, 180, True, self.snr_history[beam]["times"], self.snr_history[beam]["snrs"])
    self.results[beam]["snr_vs_time_lo"] = self.snr_plot.getRawImage()
    self.log (3, "process_observation: snr_plot len=" + str(len(self.snr_plot.getRawImage())))

    self.snr_plot.plot (1024, 768, False, self.snr_history[beam]["times"], self.snr_history[beam]["snrs"])
    self.results[beam]["snr_vs_time_hi"] = self.snr_plot.getRawImage()
    self.log (3, "process_observation: snr_plot len=" + str(len(self.snr_plot.getRawImage())))

    self.results[beam]["lock"].release() 

    return (0, "")

  def fail_observation (self, beam, obs_dir, fail_dir, arch_dir):

    self.log(3, "fail_observation()")

    self.end_observation (beam, obs_dir, fail_dir, arch_dir)

    # ensure the archived / beam / utc_start / source directory exists
    cmd = "mkdir -p " + arch_dir
    rval, lines = self.system (cmd, 3)
    if rval:
      return (1, "finalise_observation: failed to create " + arch_dir)

    # touch obs.failed file in the archival directory
    cmd = "touch " + arch_dir + "/obs.failed"
    rval, lines = self.system (cmd, 3)

    # create the failed / beam / utc_start / source directory
    cmd = "mkdir -p " + fail_dir
    rval, lines = self.system (cmd, 3)
    if rval:
      return (1, "finalise_observation: failed to create " + fail_dir)

    # move any subdirs
    subdirs = [item for item in os.listdir(obs_dir) if os.path.isdir(os.path.join(obs_dir, item))]
    for s in subdirs:
      try:
        shutil.move (obs_dir + "/" + s, fail_dir + "/" + s)
      except OSError, e:
        self.log (0, "fail_observation failed to rename " + \
                  obs_dir + "/" + s + " to " + \
                  fail_dir + "/" + s)
        return (1, "failed to rename obs_dir to fail_dir")

    # move each png file from obs_dir to fail_dir
    pngs = [file for file in os.listdir(obs_dir) if file.lower().endswith(".png")]
    for p in pngs:
      try:
        shutil.move (obs_dir + "/" + p, fail_dir + "/" + p)
      except OSError, e:
        self.log (0, "fail_observation failed to move " + \
                  obs_dir + "/" + p + " to " + \
                  fail_dir + "/" + p)
        return (1, "failed to move png from obs_dir to fail_dir")

    # move the other files from obs_dir to fail_dir
    files = [file for file in os.listdir(obs_dir) if file.lower().endswith(".sum") or file.lower().startswith("obs.") or file.lower().startswith("band")]
    for f in files:
      try:
        shutil.move (obs_dir + "/" + f, fail_dir + "/" + f)
      except OSError, e:
        self.log (0, "fail_observation failed to move " + \
                  obs_dir + "/" + f + " to " + \
                  fail_dir + "/" + f)
        return (1, "failed to move file from obs_dir to fail_dir")

    # check if directory empty
    cmd = "find " + obs_dir + " -mindepth 1 | wc -l"
    rval, lines = self.system (cmd, 3)
    if rval == 0 and lines[0] == "0":
      # delete the beam / utc_start / source directory
      os.rmdir (obs_dir)
      # delete the beam / utc_start directory
      os.rmdir ( os.path.dirname (obs_dir))
      return (0, "")
    else:
      return (0, "")


  # transition observation from processing to finished
  def finalise_observation (self, beam, proc_dir, fin_dir, arch_dir):

    self.end_observation (beam, proc_dir, fin_dir, arch_dir)

    cmd = "mkdir -p " + arch_dir
    rval, lines = self.system (cmd, 3)
    if rval:
      return (1, "finalise_observation: failed to create " + arch_dir)

    # touch and obs.finished file in the archival directory
    cmd = "touch " + arch_dir + "/obs.finished"
    rval, lines = self.system (cmd, 3)
    if rval:
      return (1, "finalise_observation: failed to create " + arch_dir + "/obs.finished")

    # simply move the observation from processing to finished
    try:
      self.log (3, "finalise_observation: rename(" + proc_dir + "," + fin_dir + ")")
      os.rename (proc_dir, fin_dir)
    except OSError, e:
      self.log (0, "finalise_observation: failed to rename proc_dir: " + str(e))
      return (1, "finalise_observation failed to rename proc_dir to fin_dir")

    # delete the parent directory of proc_dir
    parent_dir = os.path.dirname (proc_dir)
    self.log (3, "finalise_observation: rmdir(" + parent_dir + ")") 
    os.rmdir(parent_dir)
    return (0, "")

  def end_observation (self, beam, obs_dir, fin_dir, arch_dir):

    # write the most recent images disk for long term storage
    timestamp = times.getCurrentTime()
    
    self.results[beam]["lock"].acquire()

    self.log (2, "end_observation: beam=" + beam + " timestamp=" + \
              timestamp + " valid=" + str(self.results[beam]["valid"]))

    if (self.results[beam]["valid"]):

      # write out image files
      for key in self.suffixes.keys():

        extension = self.suffixes[key]
    
        filename = obs_dir + "/" + timestamp + "." + extension

        fptr = open (filename, "wb")
        fptr.write(self.results[beam][key])
        fptr.close()

      self.snr_history[beam]["times"] = []
      self.snr_history[beam]["snrs"] = []

      # indicate that the beam is no longer valid now that the 
      # observation has finished
      self.results[beam]["valid"] = False
      self.log (2, "end_observation: self.results["+beam+"][valid]=False")

    self.results[beam]["lock"].release()


class RepackServerDaemon (RepackDaemon, ServerBased):

  def __init__ (self, name):
    RepackDaemon.__init__(self,name, "-1")
    ServerBased.__init__(self, self.cfg)

  def configure (self,become_daemon, dl, source, dest):

    Daemon.configure (self, become_daemon, dl, source, dest)

    self.processing_dir = self.cfg["SERVER_FOLD_DIR"] + "/processing"
    self.finished_dir   = self.cfg["SERVER_FOLD_DIR"] + "/finished"
    self.archived_dir   = self.cfg["SERVER_FOLD_DIR"] + "/archived"
    self.failed_dir     = self.cfg["SERVER_FOLD_DIR"] + "/failed"

    for i in range(int(self.cfg["NUM_BEAM"])):
      bid = self.cfg["BEAM_" + str(i)]
      self.beams.append(bid)
      self.results[bid] = {}
      self.results[bid]["valid"] = False
      self.results[bid]["lock"] = threading.Lock()
      self.results[bid]["cond"] = threading.Condition(self.results[bid]["lock"])

      keys = ["source", "utc_start", "timestamp", "snr", "length"]
      for key in keys:
        self.results[bid][key] = ""

      self.snr_history[bid] = {}
      self.snr_history[bid]["times"] = []
      self.snr_history[bid]["snrs"] = []

    return 0

  def conclude (self):
    for i in range(int(self.cfg["NUM_BEAM"])):
      bid = self.cfg["BEAM_" + str(i)]
      self.results[bid]["lock"].acquire()
      self.results[bid]["lock"].release()
    Daemon.conclude(self)

class RepackBeamDaemon (RepackDaemon, BeamBased):

  def __init__ (self, name, id):
    RepackDaemon.__init__(self, name, str(id))
    BeamBased.__init__(self, str(id), self.cfg)

  def configure (self, become_daemon, dl, source, dest):
 
    self.log(1, "RepackBeamDaemon::configure()")
    Daemon.configure(self, become_daemon, dl, source, dest)
 
    self.processing_dir = self.cfg["CLIENT_FOLD_DIR"] + "/processing"
    self.finished_dir   = self.cfg["CLIENT_FOLD_DIR"] + "/finished"

    self.archived_dir   = self.cfg["CLIENT_FOLD_DIR"] + "/archived"
    self.failed_dir     = self.cfg["CLIENT_FOLD_DIR"] + "/failed"

    bid = self.cfg["BEAM_" + str(self.beam_id)]

    self.beams.append(bid)
    self.results[bid] = {}
    self.results[bid]["valid"] = False
    self.results[bid]["lock"] = threading.Lock()
    self.results[bid]["cond"] = threading.Condition(self.results[bid]["lock"])
    keys = ["source", "utc_start", "timestamp", "snr", "length"]
    for key in keys:
      self.results[bid][key] = ""

    self.snr_history[bid] = {}
    self.snr_history[bid]["times"] = []
    self.snr_history[bid]["snrs"] = []

    self.log(1, "RepackBeamDaemon::configure done")

    return 0

###############################################################################

if __name__ == "__main__":

  if len(sys.argv) != 2:
    print "ERROR: 1 command line argument expected"
    sys.exit(1)

  # this should come from command line argument
  beam_id = sys.argv[1]

  script = []
  if int(beam_id) == -1:
    script = RepackServerDaemon ("spip_repack")
  else:
    script = RepackBeamDaemon ("spip_repack", beam_id)

  state = script.configure (DAEMONIZE, DL, "repack", "repack") 
  if state != 0:
    script.quit_event.set()
    sys.exit(state)

  script.log(1, "STARTING SCRIPT")

  try:

    reporting_thread = RepackReportingThread(script, beam_id)
    reporting_thread.start()

    script.main ()

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

