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

class StatReportingThread(ReportingThread):

  def __init__ (self, script, id):
    host = sockets.getHostNameShort()
    port = int(script.cfg["STREAM_STAT_PORT"])
    if id >= 0:
      port += int(id)
    ReportingThread.__init__(self, script, host, port)

    with open (script.cfg["WEB_DIR"] + "/spip/images/blankimage.gif", mode='rb') as file:
      self.no_data = file.read()

  def parse_message (self, request):
    self.script.log (2, "StatReportingThread::parse_message: " + str(request))

    xml = ""
    req = request["stat_request"]

    if req["type"] == "state":

      self.script.log (3, "StatReportingThread::parse_message: preparing state response")
      xml = "<stat_state>"

      self.script.results["lock"].acquire()
      xml += "<stream id='" + str(self.script.id) + "' beam_name='" + self.script.beam_name + "' active='" + str(self.script.results["valid"]) + "'>"

      self.script.log (3, "StatReportingThread::parse_message: keys="+str(self.script.results.keys()))

      if self.script.results["valid"]:

        if "pref_chan" in req:
          self.script.pref_freq = int(req["pref_chan"])
     
        self.script.log (3, "StatReportingThread::parse_message: stream is valid!")

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

      xml += "</stat_state>"
      self.script.log (2, "StatReportingThread::parse_message: returning " + str(xml))

      return True, xml + "\r\n"

    elif req["type"] == "plot":

      if req["plot"] in self.script.valid_plots:

        self.script.results["lock"].acquire()
        self.script.log (2, "StatReportingThread::parse_message: " + \
                         " plot=" + req["plot"] + " pol=" + req["pol"] + " dim=" + req["dim"]) 

        if self.script.results["valid"]:
          plot = req["plot"] + "_" + req["pol"] + "_" + req["dim"]

          if req["res"] == "hi":
            plot += "_hires"

          self.script.log (2, "StatReportingThread::parse_message plot=" + plot)
          if plot in self.script.results.keys():
            if len (self.script.results[plot]) > 64:
              bin_data = copy.deepcopy(self.script.results[plot])
              self.script.log (3, "StatReportingThread::parse_message: " + plot + " valid, image len=" + str(len(bin_data)))
              self.script.results["lock"].release()
              return False, bin_data
            else:
              self.script.log (1, "StatReportingThread::parse_message image length=" + str(len (self.script.results[plot])) + " <= 64")
          else:
            self.script.log (1, "StatReportingThread::parse_message plot ["+plot+"] not in keys [" + str(self.script.results.keys()))
        else:
          self.script.log (1, "StatReportingThread::parse_message results not valid")

        # return empty plot
        self.script.log (1, "StatReportingThread::parse_message [returning NO DATA YET]")
        self.script.results["lock"].release()
        return False, self.no_data

      else:
        
        self.script.log (1, "StatReportingThread::parse_message invalid plot, " + req["plot"] + " not in " + str(self.script.valid_plots))
        self.script.log (1, "StatReportingThread::parse_message returning 'no_data' of size " + str(len(self.no_data)))
        return False, self.no_data

      xml += "<stat_state>"
      xml += "<error>Invalid request</error>"
      xml += "</stat_state>\r\n"

      return True, xml

#################################################################
# thread for executing processing commands
class dbstatsThread (threading.Thread):

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

class StatDaemon(Daemon,StreamBased):

  def __init__ (self, name, id):
    Daemon.__init__(self, name, str(id))
    StreamBased.__init__(self, id, self.cfg)

    self.processing_dir = self.cfg["CLIENT_STATS_DIR"]
    self.valid_plots = []
    self.results = {}

    self.results["lock"] = threading.Lock()
    self.results["valid"] = False

    (host, beam_id, subband_id) = self.cfg["STREAM_" + id].split(":")
    self.beam_name = self.cfg["BEAM_" + beam_id]

    (cfreq, bw, nchan) = self.cfg["SUBBAND_CONFIG_" + subband_id].split(":")
    self.cfreq = cfreq

    self.gen_timeseries = False
    self.gen_freqtime = True
    self.gen_bandpass = True
    self.gen_histogram = True

    self.hg_valid = False
    self.ft_valid = False
    self.bp_valid = False
    self.ms_valid = False
    self.ts_valid = False

    self.pref_freq = 0
    self.histogram_abs_xmax = 128

  #################################################################
  # main
  #       id >= 0   process folded archives from a stream
  #       id == -1  process folded archives from all streams
  def main (self):

    if self.gen_histogram:
      self.hg_plot = HistogramPlot()
      self.valid_plots.appen("histogram")

    if self.gen_bandpass:
      self.bp_plot = BandpassPlot()
      self.valid_plots.appen("bandpass")

    if self.gen_timeseries:
      self.ts_plot = TimeseriesPlot()
      self.valid_plots.appen("timeseries")

    if self.gen_freqtime:
      self.ft_plot = FreqTimePlot()
      self.valid_plots.appen("freqtime")

    # stats files are stored in flat directory structure
    # stats_dir / beam / cfreq

    if not os.path.exists(self.processing_dir):
      os.makedirs(self.processing_dir, 0755) 

    # get the data block keys
    db_prefix  = self.cfg["DATA_BLOCK_PREFIX"]
    db_id      = self.cfg["PROCESSING_DATA_BLOCK"]
    num_stream = self.cfg["NUM_STREAM"]
    stream_id  = str(self.id)
    self.log (2, "StatDaemon::main stream_id=" + str(self.id))
    db_key     = SMRBDaemon.getDBKey (db_prefix, stream_id, num_stream, db_id)
    self.log (2, "StatDaemon::main db_key=" + db_key)

    # start dbstats in a separate thread
    stat_dir = self.processing_dir + "/" + self.beam_name + "/" + self.cfreq

    if not os.path.exists(stat_dir):
      os.makedirs(stat_dir, 0755)

    # configure the histogram plot with all channels included
    self.hg_plot.configure (-1, self.histogram_abs_xmax)

    log = False
    zap = False
    transpose = False
    # configure the freq v time plot
    if self.gen_freqtime:
      self.ft_plot.configure (log, zap, transpose)

    # configure the bandpass plot
    log = True
    if self.gen_bandpass:
      self.bp_plot.configure (log, zap, transpose)

    log_host = self.cfg["SERVER_HOST"]
    log_port = int(self.cfg["SERVER_LOG_PORT"])

    # stat will use the stream config file created for the recv command
    stream_config_file = "/tmp/spip_stream_" + str(self.id) + ".cfg"
    while (not os.path.exists(stream_config_file)):
      self.log (2, "StatDaemon::main waiting for stream_config file [" + stream_config_file +"] to be created by recv")
      time.sleep(1)    

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
    stat_cmd = self.cfg["STREAM_STATS_BINARY"] + " -k " + db_key + \
               " " + stream_config_file + " -D  " + stat_dir + " -n " + nchan

    while (not self.quit_event.isSet()):

      # create a log pipe for the stats command
      stat_log_pipe   = LogSocket ("stat_src", "stat_src", str(self.id), "stream",
                                   log_host, log_port, int(DL))

      # connect up the log file output
      stat_log_pipe.connect()

      # add this binary to the list of active commands
      self.binary_list.append (self.cfg["STREAM_STATS_BINARY"] + " -k " + db_key)

      self.log (1, "START " + stat_cmd)

       # initialize the threads
      stat_thread = dbstatsThread (stat_cmd, stat_dir, stat_log_pipe.sock, 2)

      self.log (2, "StatDaemon::main cmd=" + stat_cmd)

      self.log (2, "StatDaemon::main starting stat thread")
      stat_thread.start()
      self.log (2, "StatDaemon::main stat thread started")

      pref_freq = 0

      while stat_thread.is_alive() and not self.quit_event.isSet():

        # get a list of all the files in stat_dir
        files = os.listdir (stat_dir)

        self.log (2, "StatDaemon::main found " + str(len(files)) + " in " + stat_dir)

        # if stat files exist in the directory
        if len(files) > 0:
          if self.gen_histogram:
            self.process_hg (stat_dir, pref_freq)
          if self.gen_bandpass:
            self.process_bp (stat_dir, pref_freq)
          if self.gen_freqtime:
            self.process_ft (stat_dir, pref_freq)
          if self.gen_timeseries:
            self.process_ts (stat_dir)
          self.process_ms (stat_dir)

          self.results["lock"].acquire()

          pref_freq = self.pref_freq
          self.results["timestamp"] = times.getCurrentTime()
          self.results["valid"] = self.ms_valid
          if self.gen_histogram:
            self.results["valid"] |= self.hg_valid
          if self.gen_timeseries:
            self.results["valid"] |= self.ts_valid
          if self.gen_freqtime:
            self.results["valid"] |= self.ft_valid
          if self.gen_bandpass:
            self.results["valid"] |= self.bp_valid

          self.results["lock"].release()

        time.sleep(5)

      self.log (2, "StatDaemon::main joining stat thread")
      rval = stat_thread.join()
      self.log (2, "StatDaemon::main stat thread joined")

      self.log (1, "END   " + stat_cmd)

      if rval:
        self.log (-2, "stat thread failed")
        self.quit_event.set()

  
  def process_hg (self, stat_dir, ifreq=-1):

    # find the most recent HG stats file
    files = [file for file in os.listdir(stat_dir) if file.lower().endswith(".hg.stats")]

    if len(files) > 0:

      self.log (3, "StatDaemon::process_hg files=" + str(files))
      hg_file = files[-1]
      self.log (2, "StatDaemon::process_hg hg_file=" + str(hg_file))

      # only 1 channel in the histogram
      hg_fptr = open (stat_dir + "/" + str(hg_file), "rb")
      npol = np.fromfile(hg_fptr, dtype=np.uint32, count=1)[0]
      nfreq = np.fromfile(hg_fptr, dtype=np.uint32, count=1)[0]
      ndim = np.fromfile(hg_fptr, dtype=np.uint32, count=1)[0]
      nbin = np.fromfile(hg_fptr, dtype=np.uint32, count=1)[0]

      self.log (2, "StatDaemon::process_hg npol=" + str(npol) + " ndim=" + str(ndim) + " nbin=" + str(nbin) + " nfreq=" + str(nfreq))
      hg_data = {}
      for ipol in range(npol):
        hg_data[ipol] = {}
        for idim in range(ndim):
          hg_data[ipol][idim] = np.fromfile (hg_fptr, dtype=np.uint32, count=nfreq*nbin)
          hg_data[ipol][idim].shape = (nfreq, nbin)
      hg_fptr.close()

      self.results["lock"].acquire()

      self.results["hg_npol"] = npol 
      self.results["hg_ndim"] = ndim

      dims = {0: "real", 1: "imag"}

      for ipol in range(npol):

        # dual-dim histogram
        if nfreq == 1:
          ifreq = 0
        if ifreq == -1:
          ifreq = nfreq / 2

        prefix = "histogram_" + str(ipol) + "_none"
        
        chan_real = hg_data[ipol][0][ifreq]
        chan_imag = hg_data[ipol][1][ifreq]
        self.hg_plot.plot_binned_dual (160, 120, True, chan_real, chan_imag, nbin)
        self.results[prefix] = self.hg_plot.getRawImage()
        self.hg_plot.plot_binned_dual (1024, 768, False, chan_real, chan_imag, nbin)
        self.results[prefix + "_hires"] = self.hg_plot.getRawImage()

        #self.hg_plot.plot_binned4 (160, 120, True, hg_data[0][0][ifreq,:], hg_data[0][1][ifreq,:], hg_data[1][0][ifreq,:], hg_data[1][1][ifreq,:], nbin)
        #self.results["histogram_s_none"] = self.hg_plot.getRawImage()

      self.hg_valid = True
      self.results["lock"].release()

      for file in files:
        os.remove (stat_dir + "/" + file)

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
  # StatDaemon::process_bp
  def process_bp (self, stat_dir, ifreq=-1):
    """ Process Bandpass stats files. """
    self.log (2, "StatDaemon::process_bp("+stat_dir+")")

    # read the most recent freq_vs_time stats file
    files = [file for file in os.listdir(stat_dir) if file.lower().endswith(".bp.stats")]
    if len(files) > 0:

      self.log (3, "StatDaemon::process_bp files=" + str(files))
      bp_file = files[-1]
      self.log (2, "StatDaemon::process_bp bp_file=" + str(bp_file))

      bp_fptr = open (stat_dir + "/" + str(bp_file), "rb")
      npol = np.fromfile(bp_fptr, dtype=np.uint32, count=1)[0]
      nfreq = np.fromfile(bp_fptr, dtype=np.uint32, count=1)[0]
      freq = np.fromfile(bp_fptr, dtype=np.float64, count=1)[0]
      bw = np.fromfile(bp_fptr, dtype=np.float64, count=1)[0]
      self.log (3, "StatDaemon::process_bp npol=" + str(npol) + " nfreq=" + \
                str(nfreq) + " freq=" + str(freq) + " bw=" + str(bw))

      bp_data = {}
      for ipol in range(npol):
        bp_data[ipol] = np.fromfile (bp_fptr, dtype=np.float32, count=nfreq)
      bp_fptr.close()

      self.log (3, "StatDaemon::process_bp plotting")
      self.results["lock"].acquire()

      self.results["bp_npol"] = npol

      # also configure the bandpass plot
      for ipol in range(npol):

        prefix = "bandpass_" + str(ipol) + "_none"

        self.bp_plot.plot (160, 120, True, nfreq, freq, bw, bp_data[ipol])
        self.results[prefix] = self.bp_plot.getRawImage()
        self.bp_plot.plot (1024, 768, False, nfreq, freq, bw, bp_data[ipol])
        self.results[prefix + "_hires"] = self.bp_plot.getRawImage()

      self.bp_valid = True

      self.results["lock"].release()

      for file in files:
        os.remove (stat_dir + "/" + file)


  #############################################################################
  # StatDaemon::process_ft
  def process_ft (self, stat_dir, ifreq=-1):
    """ Process Freq vs Time stats files. """ 
    self.log (2, "StatDaemon::process_ft("+stat_dir+")")

    # read the most recent freq_vs_time stats file
    files = [file for file in os.listdir(stat_dir) if file.lower().endswith(".ft.stats")]
    if len(files) > 0:

      self.log (3, "StatDaemon::process_ft files=" + str(files))
      ft_file = files[-1]
      self.log (2, "StatDaemon::process_ft ft_file=" + str(ft_file))

      ft_fptr = open (stat_dir + "/" + str(ft_file), "rb")
      npol = np.fromfile(ft_fptr, dtype=np.uint32, count=1)[0]
      ndim = 2
      nfreq = np.fromfile(ft_fptr, dtype=np.uint32, count=1)[0]
      ntime = np.fromfile(ft_fptr, dtype=np.uint32, count=1)[0]
      freq = np.fromfile(ft_fptr, dtype=np.float64, count=1)[0]
      bw = np.fromfile(ft_fptr, dtype=np.float64, count=1)[0]
      tsamp = np.fromfile(ft_fptr, dtype=np.float64, count=1)[0]
      self.log (3, "StatDaemon::process_ft npol=" + str(npol) + " nfreq=" + \
                str(nfreq) + " ntime=" + str(ntime) + " freq=" + str(freq) + " bw=" + str(bw) + " tsamp=" + str(tsamp))

      ft_data = {}
      for ipol in range(npol):
        ft_data[ipol] = np.fromfile (ft_fptr, dtype=np.float32, count=nfreq*ntime)
        ft_data[ipol].shape = (nfreq, ntime)
      ft_fptr.close()

      self.log (3, "StatDaemon::process_ft plotting")
      self.results["lock"].acquire()

      self.results["ft_npol"] = npol 
      self.results["ft_ndim"] = ndim

      for ipol in range(npol):
        prefix = "freq_vs_time_" + str(ipol) + "_none"
        self.ft_plot.plot (160, 120, True, ft_data[ipol], nfreq, freq, bw, tsamp, ntime)
        self.results[prefix] = self.ft_plot.getRawImage()
        self.ft_plot.plot (1024, 768, False, ft_data[ipol], nfreq, freq, bw, tsamp, ntime)
        self.results[prefix + "_hires"] = self.ft_plot.getRawImage()
 
      ft_summed = []
      if npol == 2:
        ft_summed = np.add(ft_data[0], ft_data[1])
      else:
        ft_summed = ft_data[0]

      self.ft_plot.plot (160, 120, True, ft_summed, nfreq, freq, bw, tsamp, ntime)
      self.results["freq_vs_time_s_none"] = self.ft_plot.getRawImage()
      self.ft_plot.plot (1024, 768, False, ft_summed, nfreq, freq, bw, tsamp, ntime)
      self.results["freq_vs_time_s_none_hires"] = self.ft_plot.getRawImage()
      
      self.ft_valid = True

      self.results["lock"].release()

      for file in files:
        os.remove (stat_dir + "/" + file)

  #############################################################################
  # StatDaemon::process_ts
  def process_ts (self, stat_dir):
    """ Process Timeseries stats files. """
    self.log (2, "StatDaemon::process_ts("+stat_dir+")")

    # read the most recent time series stats file
    files = [file for file in os.listdir(stat_dir) if file.lower().endswith(".ts.stats")]
    self.log (2, "StatDaemon::process_ts files=" + str(files))
    if len(files) > 0:

      ts_file = files[-1]
      self.log (3, "StatDaemon::process_ts ts_file=" + str(ts_file))

      ts_fptr = open (stat_dir + "/" + str(ts_file), "rb")
      npol = np.fromfile(ts_fptr, dtype=np.uint32, count=1)[0]
      ntime = np.fromfile(ts_fptr, dtype=np.uint32, count=1)[0]
      tsamp = np.fromfile(ts_fptr, dtype=np.float64, count=1)[0]
      self.log (3, "StatDaemon::process_ts npol=" + str(npol) + " ntime=" + str(ntime) + " tsamp=" + str(tsamp))

      xvals = np.arange (0, ntime*tsamp, tsamp, dtype=float)
      ts_data = {}
      ntype = 3
      labels = ["min", "mean", "max"]
      colors = ["green", "black", "red"]
      for ipol in range(npol):
        ts_data[ipol] = np.fromfile (ts_fptr, dtype=np.float32, count=ntype*ntime)
        ts_data[ipol].shape = (ntype, ntime)
      ts_fptr.close()

      self.log (3, "StatDaemon::process_ts plotting")
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
        os.remove (stat_dir + "/" + file)

      self.ts_valid = True
      self.results["lock"].release()
    
  #############################################################################
  # StatDaemon::process_ms
  def process_ms (self, stat_dir):
    """ Process Mean and Standard Deviation stats files. """
    self.log (2, "StatDaemon::process_ms("+stat_dir+")")

    # find the most recent Mean/Stddev stats files
    files = [file for file in os.listdir(stat_dir) if file.lower().endswith(".ms.stats")]
    if len(files) > 0:

      self.log (3, "StatDaemon::process_ms files=" + str(files))
      ms_file = files[-1]
      self.log (2, "StatDaemon::process_ms ms_file=" + str(ms_file))
      ms_fptr = open (stat_dir + "/" + str(ms_file), "rb")
      npol = np.fromfile(ms_fptr, dtype=np.uint32, count=1)[0]
      ndim = np.fromfile(ms_fptr, dtype=np.uint32, count=1)[0]

      means = np.fromfile (ms_fptr, dtype=np.float32, count=npol*ndim)
      stddevs = np.fromfile (ms_fptr, dtype=np.float32, count=npol*ndim)
      ms_fptr.close()

      self.results["lock"].acquire()

      self.results["ms_npol"] = npol
      self.results["ms_ndim"] = ndim
      
      dims = {0: "real", 1: "imag"}
      for ipol in range(npol):
        for idim in range(ndim):
          self.results["hg_mean_" + str(ipol) + "_" + dims[idim]] = means[ipol*ndim+idim]
          self.results["hg_stddev_" + str(ipol) + "_" + dims[idim]] = stddevs[ipol*ndim+idim]
 
      self.ms_valid = True
      self.results["lock"].release()

      for file in files:
        os.remove (stat_dir + "/" + file)


###############################################################################

if __name__ == "__main__":

  if len(sys.argv) != 2:
    print "ERROR: 1 command line argument expected"
    sys.exit(1)

  # this should come from command line argument
  stream_id = sys.argv[1]

  script = []
  script = StatDaemon ("spip_stat", stream_id)

  state = script.configure (DAEMONIZE, DL, "stat", "stat")

  if state != 0:
    sys.exit(state)

  script.log(1, "STARTING SCRIPT")

  try:

    reporting_thread = StatReportingThread(script, stream_id)
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

