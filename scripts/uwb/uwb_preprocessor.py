#!/usr/bin/env python

##############################################################################
#  
#     Copyright (C) 2015 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
# 
###############################################################################

import os, sys, socket, select, signal, traceback, time, threading, copy
import numpy as np

from spip.daemons.bases import StreamBased
from spip.daemons.daemon import Daemon
from spip.threads.reporting_thread import ReportingThread
from spip.log_socket import LogSocket
from spip.utils import times,sockets
from spip.utils.core import system_piped
from spip.plotting import HistogramPlot,FreqTimePlot,BandpassPlot,TimeseriesPlot

from spip_smrb import SMRBDaemon

DAEMONIZE = True
DL        = 1

class PreprocessorReportingThread(ReportingThread):

  def __init__ (self, script, id):
    host = sockets.getHostNameShort()
    port = int(script.cfg["STREAM_STAT_PORT"])
    if id >= 0:
      port += int(id)
    ReportingThread.__init__(self, script, host, port)

    with open (script.cfg["WEB_DIR"] + "/spip/images/blankimage.gif", mode='rb') as file:
      self.no_data = file.read()

  def parse_message (self, request):
    self.script.log (2, "PreprocessorReportingThread::parse_message: " + str(request))

    xml = ""
    req = request["preproc_request"]

    if req["type"] == "state":

      self.script.log (3, "PreprocessorReportingThread::parse_message: preparing state response")
      xml = "<preproc_state>"

      self.script.results["lock"].acquire()
      xml += "<stream id='" + str(self.script.id) + "' beam_name='" + self.script.beam_name + "' active='" + str(self.script.results["valid"]) + "'>"

      self.script.log (3, "PreprocessorReportingThread::parse_message: keys="+str(self.script.results.keys()))

      if self.script.results["valid"]:

        if "pref_chan" in req:
          self.script.pref_freq = int(req["pref_chan"])
     
        self.script.log (3, "PreprocessorReportingThread::parse_message: stream is valid!")

        npol = self.script.results["hg_npol"]
        ndim = self.script.results["hg_ndim"]
        dims = { 0: "real", 1: "imag" }

        for ipol in range(npol):

          xml += "<polarisation name='" + str(ipol)+ "'>"
       
          for idim in range(ndim): 

            suffix = str(ipol) + "_" + dims[idim]
            
            xml += "<dimension name='" + dims[idim] + "'>"
            xml += "<histogram_mean>" + str(self.script.results["hg_mean_" + suffix]) + "</histogram_mean>"
            xml += "<histogram_stddev>" + str(self.script.results["hg_stddev_" + suffix]) + "</histogram_stddev>"
            # xml += "<plot type='histogram' timestamp='" + str(self.script.results["timestamp"]) + "'/>"
            xml += "</dimension>"

          xml += "<dimension name='none'>"
          xml += "<plot type='timeseries' timestamp='" + str(self.script.results["timestamp"]) + "'/>"
          xml += "<plot type='bandpass' timestamp='" + str(self.script.results["timestamp"]) + "'/>"
          xml += "<plot type='histogram' timestamp='" + str(self.script.results["timestamp"]) + "'/>"
          xml += "</dimension>"

          xml += "</polarisation>"

      xml += "</stream>"

      self.script.results["lock"].release()

      xml += "</preproc_state>"
      self.script.log (2, "PreprocessorReportingThread::parse_message: returning " + str(xml))

      return True, xml + "\r\n"

    elif req["type"] == "plot":

      if req["plot"] in self.script.valid_plots:

        self.script.results["lock"].acquire()
        self.script.log (2, "PreprocessorReportingThread::parse_message: " + \
                         " plot=" + req["plot"] + " pol=" + req["pol"] + " dim=" + req["dim"]) 

        if self.script.results["valid"]:
          plot = req["plot"] + "_" + req["pol"] + "_" + req["dim"]

          if req["res"] == "hi":
            plot += "_hires"

          self.script.log (2, "PreprocessorReportingThread::parse_message plot=" + plot)
          if plot in self.script.results.keys():
            if len (self.script.results[plot]) > 64:
              bin_data = copy.deepcopy(self.script.results[plot])
              self.script.log (3, "PreprocessorReportingThread::parse_message: " + plot + " valid, image len=" + str(len(bin_data)))
              self.script.results["lock"].release()
              return False, bin_data
            else:
              self.script.log (1, "PreprocessorReportingThread::parse_message image length=" + str(len (self.script.results[plot])) + " <= 64")
          else:
            self.script.log (1, "PreprocessorReportingThread::parse_message plot ["+plot+"] not in keys [" + str(self.script.results.keys()))
        else:
          self.script.log (1, "PreprocessorReportingThread::parse_message results not valid")

        # return empty plot
        self.script.log (1, "PreprocessorReportingThread::parse_message [returning NO DATA YET]")
        self.script.results["lock"].release()
        return False, self.no_data

      else:
        
        self.script.log (1, "PreprocessorReportingThread::parse_message invalid plot, " + req["plot"] + " not in " + str(self.script.valid_plots))
        self.script.log (1, "PreprocessorReportingThread::parse_message returning 'no_data' of size " + str(len(self.no_data)))
        return False, self.no_data

      xml += "<preproc_state>"
      xml += "<error>Invalid request</error>"
      xml += "</preproc_state>\r\n"

      return True, xml

#################################################################
# thread for executing processing commands
class preprocThread (threading.Thread):

  def __init__ (self, cmd, dir, pipe, dl):
    threading.Thread.__init__(self)
    self.cmd = cmd
    self.pipe = pipe
    self.dir = dir
    self.dl = dl

  def run (self):
    cmd = self.cmd
    rval = system_piped (cmd, self.pipe, self.dl <= DL, work_dir=self.dir)
    return rval

class PreprocessorDaemon(Daemon,StreamBased):

  def __init__ (self, name, id):
    Daemon.__init__(self, name, str(id))
    StreamBased.__init__(self, id, self.cfg)

    self.processing_dir = self.cfg["CLIENT_PREPROC_DIR"]
    self.valid_plots = []
    self.results = {}

    self.results["lock"] = threading.Lock()
    self.results["valid"] = False

    (host, beam_id, subband_id) = self.cfg["STREAM_" + id].split(":")
    self.beam_name = self.cfg["BEAM_" + beam_id]

    (cfreq, bw, nchan) = self.cfg["SUBBAND_CONFIG_" + subband_id].split(":")
    self.cfreq = cfreq

    self.dirty_valid = False
    self.clean_valid = False
    self.gain_valid = False

  #################################################################
  # main
  #       id >= 0   process folded archives from a stream
  #       id == -1  process folded archives from all streams
  def main (self):

    self.dirty_plot = BandpassPlot()
    self.clean_plot = BandpassPlot()
    self.gains_plot = TimeSeriesPlot()

    self.valid_plots = ["dirty", "clean", "gainstime"]

    # stats files are stored in flat directory structure
    # stats_dir / beam / cfreq

    if not os.path.exists(self.processing_dir):
      os.makedirs(self.processing_dir, 0755) 

    # get the data block keys
    db_prefix  = self.cfg["DATA_BLOCK_PREFIX"]
    num_stream = self.cfg["NUM_STREAM"]
    stream_id  = str(self.id)
    self.log (2, "PreprocessorDaemon::main stream_id=" + str(self.id))

    # 4 data blocks
    in_id = self.cfg["PROCESSING_DATA_BLOCK"]
    cal_id = self.cfg["CALIBRATION_DATA_BLOCK"]
    trans_id = self.cfg["TRANSIENTS_DATA_BLOCK"]
    out_id = self.cfg["GPUPROCESSING_DATA_BLOCK"]

    # 4 data block keys
    in_key = SMRBDaemon.getDBKey (db_prefix, stream_id, num_stream, in_id)
    cal_key = SMRBDaemon.getDBKey (db_prefix, stream_id, num_stream, cal_id)
    trans_key = SMRBDaemon.getDBKey (db_prefix, stream_id, num_stream, trans_id)
    out_key = SMRBDaemon.getDBKey (db_prefix, stream_id, num_stream, out_id)

    # start dbstats in a separate thread
    preproc_dir = self.processing_dir + "/" + self.beam_name + "/" + self.cfreq

    if not os.path.exists(preproc_dir):
      os.makedirs(preproc_dir, 0755)

    log = False
    zap = False
    transpose = False

    # configure the bandpass plots
    self.dirty_plot.configure (log, zap, transpose)
    self.clean_plot.configure (log, zap, transpose)
    self.gains_plot.configure (TBD)

    log_host = self.cfg["SERVER_HOST"]
    log_port = int(self.cfg["SERVER_LOG_PORT"])

    self.log(2, "main: self.waitForSMRB()")
    smrb_exists = self.waitForSMRB()

    if not smrb_exists:
      self.log(-2, "smrb["+str(self.id)+"] no valid SMRB with " +
                  "key=" + self.db_key)
      self.quit_event.set()
      return

    # determine the number of channels to be processed by this stream
    (cfreq, bw, nchan) = self.cfg["SUBBAND_CONFIG_" + stream_id].split(":")

    # this stat command will not change from observation to observation
    preproc_cmd = "uwb_preprocessing_pipeline " + in_key + " " + cal_key + \
                  " " + trans_key + " " + out_key + " -d " + \
                  self.cfg["GPU_ID_" + stream_id]

    tag = "preproc" + stream_id

    # enter the main loop
    while (not self.quit_event.isSet()):

      # wait for the header to acquire the processing parameters
      cmd = "dada_header -k " + in_key + " -t " + tag
      self.debug (cmd)
      self.binary_list.append (cmd)
      rval, lines = self.system (cmd, 2, True)
      self.binary_list.remove (cmd)

      self.debug("parsing header")
      header = Config.parseHeader (lines)

      cmd = preproc_cmd

      run_adaptive_filter = (header["ADAPTIVE_FILTER"] == "1")
      rfi_reference_pol = int(self.cfg["NPOL"]) - 1
      run_calibration = (header["CAL_SIGNAL"] == "1")
      run_transients = (header["TRANSIENTS"] == "1")

      if run_adaptive_filter:
        cmd = cmd + " -a -r " + str(rfi_reference_pol)

      if run_calibration:
        cmd = cmd + " -c 10"

      if run_transients:
        cmd = cmd + " -f " + header["TRANS_TSAMP"]
              
      # AJ todo check the channelisation limits with Nuer
      if run_adaptive_filter or run_calibration or run_transients:
        cmd = cmd + " -n 128"
      
      # create a log pipe for the stats command
      preproc_log_pipe   = LogSocket ("preproc_src", "preproc_src", str(self.id), "stream",
                                      log_host, log_port, int(DL))

      # connect up the log file output
      preproc_log_pipe.connect()

      # add this binary to the list of active commands
      self.binary_list.append ("uwb_preprocessing_pipeline " + in_key)

      self.info("START " + cmd)

       # initialize the threads
      preproc_thread = preprocThread (cmd, preproc_dir, preproc_log_pipe.sock, 2)

      self.debug("cmd=" + _cmd)

      self.debug("starting preproc thread")
      preproc_thread.start()
      self.debug("preproc thread started")

      while preproc_thread.is_alive() and not self.quit_event.isSet():

        # get a list of all the files in preproc_dir
        files = os.listdir (preproc_dir)

        self.log (2, "PreprocessorDaemon::main found " + str(len(files)) + " in " + preproc_dir)

        # if stat files exist in the directory
        if len(files) > 0:
          self.process_clean(preproc_dir)
          self.process_dirty(preproc_dir)
          self.process_gains(preproc_dir)

          self.results["lock"].acquire()
          self.results["timestamp"] = times.getCurrentTime()
          self.results["valid"] = self.clean_valid and self.dirty_valid and self.gains_valid
          self.results["lock"].release()

        time.sleep(5)

      self.debug("joining stat thread")
      rval = preproc_thread.join()
      self.debug("stat thread joined")

      self.info("END   " + cmd)

      if rval:
        self.error("stat thread failed")
        self.quit_event.set()

  def process_clean (self, preproc_dir):

    # find the most recent HG stats file
    files = [file for file in os.listdir(preproc_dir) if file.lower().endswith(".clean")]
    self.trace ("files=" + str(files))

    if len(files) > 0:
      # process only the most recent file
      clean_file = files[-1]

      # read the data from file into a numpy array as per example

      # acquire the results lock
      self.results["lock"].acquire()

      # generate plots for each polarisation
      for ipol in range(npol):

        prefix = "bandpass_" + str(ipol) + "_none"
  
        self.clean_plot.plot (160, 120, True, nfreq, freq, bw, bp_data[ipol])
        self.results[prefix] = self.clean_plot.getRawImage()
        self.clean_plot.plot (1024, 768, False, nfreq, freq, bw, bp_data[ipol])
        self.results[prefix + "_hires"] = self.clean_plot.getRawImage()
  
      self.clean_valid = True
  
      self.results["lock"].release()
  
      # maybe keep these files? or move them to the observation output directory
      # need to think about this
      for file in files:
        os.remove (preproc_dir + "/" + file)


  def process_dirty (self, preproc_dir):
    # nuer to write
 
  def process_gains (self, preproc_dir):
    # nuer to write

  # wait for the SMRB to be created
  def waitForSMRB (self):

    db_id = self.cfg["PROCESSING_DATA_BLOCK"]
    db_prefix = self.cfg["DATA_BLOCK_PREFIX"]
    num_stream = self.cfg["NUM_STREAM"]
    self.db_key = SMRBDaemon.getDBKey (db_prefix, self.id, num_stream, db_id)

    # port of the SMRB daemon for this stream
    smrb_port = SMRBDaemon.getDBMonPort(self.id)

    # wait up to 30s for the SMRB to be created
    smrb_wait = 60

    smrb_exists = False
    while not smrb_exists and smrb_wait > 0 and not self.quit_event.isSet():

      self.log(2, "trying to open connection to SMRB")
      smrb_sock = sockets.openSocket (DL, "localhost", smrb_port, 1)
      if smrb_sock:
        smrb_sock.send ("smrb_status\r\n")
        junk = smrb_sock.recv (65536)
        smrb_sock.close()
        smrb_exists = True
      else:
        time.sleep (1)
        smrb_wait -= 1

    return smrb_exists

  #############################################################################
  # PreprocessorDaemon::process_ts
  def process_ts (self, preproc_dir):
    """ Process Timeseries stats files. """
    self.log (2, "PreprocessorDaemon::process_ts("+preproc_dir+")")

    # read the most recent time series stats file
    files = [file for file in os.listdir(preproc_dir) if file.lower().endswith(".ts.stats")]
    self.log (2, "PreprocessorDaemon::process_ts files=" + str(files))
    if len(files) > 0:

      ts_file = files[-1]
      self.log (3, "PreprocessorDaemon::process_ts ts_file=" + str(ts_file))

      ts_fptr = open (preproc_dir + "/" + str(ts_file), "rb")
      npol = np.fromfile(ts_fptr, dtype=np.uint32, count=1)[0]
      ntime = np.fromfile(ts_fptr, dtype=np.uint32, count=1)[0]
      tsamp = np.fromfile(ts_fptr, dtype=np.float64, count=1)[0]
      self.log (3, "PreprocessorDaemon::process_ts npol=" + str(npol) + " ntime=" + str(ntime) + " tsamp=" + str(tsamp))

      xvals = np.arange (0, ntime*tsamp, tsamp, dtype=float)
      ts_data = {}
      ntype = 3
      labels = ["min", "mean", "max"]
      colors = ["green", "black", "red"]
      for ipol in range(npol):
        ts_data[ipol] = np.fromfile (ts_fptr, dtype=np.float32, count=ntype*ntime)
        ts_data[ipol].shape = (ntype, ntime)
      ts_fptr.close()

      self.log (3, "PreprocessorDaemon::process_ts plotting")
      self.results["lock"].acquire()

      self.results["ts_npol"] = npol
      self.results["ts_ntype"] = ntype

      for ipol in range(npol):
        prefix = "timeseries_" + str(ipol) + "_none"
        self.ts_plot.plot (160, 120, True, xvals, ts_data[ipol], labels, colors)
        self.results[prefix] = self.ts_plot.getRawImage()
        self.ts_plot.plot (1024, 768, False, xvals, ts_data[ipol], labels, colors)
        self.results[prefix + "_hires"] = self.ts_plot.getRawImage()

      for file in files:
        os.remove (preproc_dir + "/" + file)

      self.ts_valid = True
      self.results["lock"].release()
    
###############################################################################

if __name__ == "__main__":

  if len(sys.argv) != 2:
    print "ERROR: 1 command line argument expected"
    sys.exit(1)

  # this should come from command line argument
  stream_id = sys.argv[1]

  script = []
  script = PreprocessorDaemon ("uwb_preproc", stream_id)

  state = script.configure (DAEMONIZE, DL, "preproc", "preproc")

  if state != 0:
    sys.exit(state)

  script.log(1, "STARTING SCRIPT")

  try:

    reporting_thread = PreprocessorReportingThread(script, stream_id)
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

