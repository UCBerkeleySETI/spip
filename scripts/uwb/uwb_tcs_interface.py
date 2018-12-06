#!/usr/bin/env python

###############################################################################
#  
#     Copyright (C) 2018 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
# 
###############################################################################

import os, threading, sys, socket, select, signal, traceback, xmltodict, copy
import errno, time

from spip.config import Config
from spip.daemons.bases import ServerBased
from spip.daemons.daemon import Daemon
from spip.log_socket import LogSocket
from spip.utils import sockets,times

DAEMONIZE = True
DL = 1

###############################################################
# TCS daemon
class TCSInterfaceDaemon(Daemon):

  def __init__ (self, name, id):
    Daemon.__init__(self, name, str(id))
    self.beam_cfg = {}
    self.host = sockets.getHostNameShort()

  def configure_beam_state(self):

    beam_xml = \
        "<beam_configuration>" + \
        "<nbeam key='NBEAM'>1</nbeam>" + \
        "<beam_state_0 key='BEAM_STATE_0' name='1'>1</beam_state_0>" + \
        "</beam_configuration>"

    source_xml = \
        "<source_parameters>" + \
        "<name key='SOURCE' epoch='J2000'>None</name>" + \
        "<ra key='RA' units='hh:mm:ss'>None</ra>" + \
        "<dec key='DEC' units='dd:mm:ss'>None</dec>" + \
        "</source_parameters>"

    observation_xml = \
        "<observation_parameters>" + \
        "<observer key='OBSERVER'>None</observer>" + \
        "<project_id key='PID'>None</project_id>" + \
        "<tobs key='TOBS'>None</tobs>" + \
        "<calfreq key='CALFREQ'>None</calfreq>" + \
        "<utc_start key='UTC_START'>None</utc_start>" + \
        "<utc_stop key='UTC_STOP'>None</utc_stop>" + \
        "</observation_parameters>"

    custom_xml = \
        "<custom_parameters>" + \
        "<adaptive_filter key='ADAPTIVE_FILTER'>0</adaptive_filter>" + \
        "<adaptive_filter_epsilon key='ADAPTIVE_FILTER_EPSILON'>0.1</adaptive_filter_epsilon>" + \
        "<adaptive_filter_nchan key='ADAPTIVE_FILTER_NCHAN'>128</adaptive_filter_nchan>" + \
        "<adaptive_filter_nsamp key='ADAPTIVE_FILTER_NSAMP'>1024</adaptive_filter_nsamp>" + \
        "<schedule_block_id key='SCHED_BLOCK_ID'>0</schedule_block_id>" + \
        "<scan_id key='SCAN_ID'>0</scan_id>" + \
        "</custom_parameters>"

    calibration_xml = \
        "<calibration_parameters>" + \
        "<signal key='CAL_SIGNAL'>0</signal>" + \
        "<freq key='CAL_FREQ'>1</freq>" + \
        "<phase key='CAL_PHASE'>0.0</phase>" + \
        "<duty_cycle key='CAL_DUTY_CYCLE'>0.5</duty_cycle>" + \
        "<epoch key='CAL_EPOCH'>Unknown</epoch>" + \
        "<tsys_avg_time key='TSYS_AVG_TIME' units='seconds'>10</tsys_avg_time>" + \
        "<tsys_freq_resolution key='TSYS_FREQ_RES' units='MHz'>1</tsys_freq_resolution>" + \
        "</calibration_parameters>"

    modes_xml = \
        "<processing_modes>" + \
        "<fold key='PERFORM_FOLD'>1</fold>" + \
        "<search key='PERFORM_SEARCH'>0</search>" + \
        "<continuum key='PERFORM_CONTINUUM'>0</continuum>" + \
        "<spectral_line key='PERFORM_SPECTRAL_LINE'>0</spectral_line>" + \
        "<vlbi key='PERFORM_VLBI'>0</vlbi>" + \
        "<baseband key='PERFORM_BASEBAND'>0</baseband>" + \
        "</processing_modes>"

    fold_xml = \
        "<fold_processing_parameters>" + \
        "<output_nchan key='FOLD_OUTNCHAN'>128</output_nchan>" + \
        "<custom_dm key='FOLD_DM'>-1</custom_dm>" + \
        "<output_nbin key='FOLD_OUTNBIN'>1024</output_nbin>" + \
        "<output_tsubint key='FOLD_OUTTSUBINT'>10</output_tsubint>" + \
        "<output_npol key='FOLD_OUTNPOL'>4</output_npol>" + \
        "<mode key='MODE'>PSR</mode>" + \
        "<sk key='FOLD_SK'>0</sk>" + \
        "<sk_threshold key='FOLD_SK_THRESHOLD'>3</sk_threshold>" + \
        "<sk_nsamps key='FOLD_SK_NSAMPS'>1024</sk_nsamps>" + \
        "<append_output key='FOLD_APPEND_OUTPUT'>1</append_output>" + \
        "<custom_period key='FOLD_CUSTOM_PERIOD'>-1</custom_period>" + \
        "</fold_processing_parameters>"

    search_xml = \
        "<search_processing_parameters>" + \
        "<output_nchan key='SEARCH_OUTNCHAN'>128</output_nchan>" + \
        "<custom_dm key='SEARCH_DM'>-1</custom_dm>" + \
        "<output_nbit key='SEARCH_OUTNBIT'>8</output_nbit>" + \
        "<output_tsubint key='SEARCH_OUTTSUBINT'>10</output_tsubint>" + \
        "<output_npol key='SEARCH_OUTNPOL'>4</output_npol>" + \
        "<output_tsamp key='SEARCH_OUTTSAMP'>64</output_tsamp>" + \
        "<coherent_dedispersion key='SEARCH_COHERENT_DEDISPERSION'>0</coherent_dedispersion>" + \
        "</search_processing_parameters>"

    continuum_xml = \
        "<continuum_processing_parameters>" + \
        "<output_nchan key='CONTINUUM_OUTNCHAN'>1024</output_nchan>" + \
        "<output_tsubint key='CONTINUUM_OUTTSUBINT'>10</output_tsubint>" + \
        "<output_npol key='CONTINUUM_OUTNPOL'>4</output_npol>" + \
        "<output_tsamp key='CONTINUUM_OUTTSAMP'>1</output_tsamp>" + \
        "</continuum_processing_parameters>"

    stream_xml = "<stream_configuration><nstream key='NSTREAM'>" + str(self.cfg["NUM_STREAM"]) + "</nstream>"
    for i in range(int(self.cfg["NUM_STREAM"])):
      stream_xml += "<active_" + str(i) + " key='ACTIVE_" + str(i) + "'>1</active_" + str(i) + ">"
    stream_xml += "</stream_configuration>"

    xml = "<obs_cmd>"
    xml += beam_xml
    xml += stream_xml
    xml += source_xml
    xml += observation_xml
    xml += calibration_xml

    for i in range(int(self.cfg["NUM_STREAM"])):
      xml += "<stream" + str(i) + ">" + \
        custom_xml + \
        modes_xml + \
        fold_xml + \
        search_xml + \
        continuum_xml + \
      "</stream" + str(i) + ">"

    xml += "</obs_cmd>"

    self.beam_cfg = xmltodict.parse (xml)

  def main (self, id):

    self.log(2, "main: TCS listening on " + self.host + ":" + str(self.interface_port))
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    sock.bind((self.host, self.interface_port))
    sock.listen(1)

    can_read = [sock]
    can_write = []
    can_error = []

    while not self.quit_event.isSet():

      timeout = 1

      did_read = []
      did_write = []
      did_error = []

      try:
        # wait for some activity on the socket
        self.log(3, "main: select")
        did_read, did_write, did_error = select.select(can_read, can_write, can_error, timeout)
        self.log(3, "main: read="+str(len(did_read))+" write="+
                  str(len(did_write))+" error="+str(len(did_error)))

      except select.error as e:
        if e[0] == errno.EINTR:
          self.log(0, "SIGINT received during select, exiting")
          self.quit_event.set()

      if (len(did_read) > 0):
        for handle in did_read:
          if (handle == sock):
            (new_conn, addr) = sock.accept()
            self.log(1, "main: accept connection from "+repr(addr))

            # add the accepted connection to can_read
            can_read.append(new_conn)

          # an accepted connection must have generated some data
          else:
            try:
              raw = handle.recv(131072)

              message = raw.strip()

              lines = message.split ("\n")

              for line in lines:

                if len(line) > 0:

                  self.log(2, "TCSInterfaceDaemon::commandThread line='" + line+"'")
                  self.log(1, "<- " + str(line))

                  # Parse XML for correctness
                  (valid, command, reply) = self.parse_obs_cmd (line, id)

                  self.log(2, "TCSInterfaceDaemon::commandThread valid=" + str(valid) \
                           + " command=" + command + " reply=" + str(reply))

                  if valid :
                    if command == "start":
                      self.log(1, "-> ok")
                      handle.send ("ok\r\n")
                      self.log(2, "TCSInterfaceDaemon::commandThread issue_start_cmd line="+line)
                      reply = self.issue_start_cmd (line)
                    elif command == "stop":
                      self.log(2, "TCSInterfaceDaemon::commandThread issue_stop_cmd line="+line)
                      reply = self.issue_stop_cmd (line)
                    elif command == "configure":
                      self.log(2, "TCSInterfaceDaemon::commandThread no action for configure command")
                    else:
                      self.log(-1, "Unrecognized command [" + command + "]")

                  else:
                    self.log(-1, "failed to parse line: " + reply)

                  self.log(1, "-> " + str(reply))
                  handle.send (reply + "\r\n")

                else:
                  self.log(1, "TCSInterfaceDaemon::commandThread closing socket on 0 byte message")
                  handle.close()
                  for i, x in enumerate(can_read):
                    if (x == handle):
                      del can_read[i]
      
            except socket.error as e:
              if e.errno == errno.ECONNRESET:
                self.log(1, "TCSInterfaceDaemon::commandThread closing connection")
                handle.close()
                for i, x in enumerate(can_read):
                  if (x == handle):
                    del can_read[i]
              else:
                raise

  ###############################################################################
  # set singular parameter
  def set_param (self, mode, param, value):
    self.beam_cfg["obs_cmd"][mode][param]["#text"] = str(value)

  ###############################################################################
  # set singular parameter
  def get_param (self, mode, param):
    return str(self.beam_cfg["obs_cmd"][mode][param]["#text"])

  ###############################################################################
  # set mode parameters for all streams
  def set_stream_params (self, nstream, mode, param, value):
    for istream in range(nstream):
      self.beam_cfg["obs_cmd"]["stream" + str(istream)][mode][param]["#text"] = str(value)

  ###############################################################################
  # parse an XML command for correctness
  def parse_obs_cmd (self, message, id):

    command = ""
    reply = ""
    result = True

    parts = message.split()
    nstream = int(self.cfg["NUM_STREAM"])

    if len(parts) == 1:
      if parts[0] == "start":
        command = "start"
        reply = "ok"
      elif parts[0] == "stop":
        command = "stop"
        reply = "ok"
      else:
        self.log(1, "TCSInterfaceDaemon::parse_obs_cmd unrecognized command: " + command)
        result = False
        reply = "bad command"

    elif len(parts) == 2:

      key = parts[0]
      value = parts[1]

      (key, value) = message.split()

      try:

        if key == "OBSERVER":
          self.set_param ("observation_parameters", "observer", value)
        if key == "PID":
          self.set_param ("observation_parameters", "project_id", value)
        if key == "CALFREQ":
          self.set_param ("observation_parameters", "calfreq", value)
          self.set_param ("calibration_parameters", "freq", value)
        if key == "OBSVAL":
          self.set_param ("observation_parameters", "tobs", value)
        if key == "OBSUNIT":
          self.log (0, "TCSInterfaceDaemon::parse_obs_cmd ignoring OBSUNIT")
        if key == "SOURCE":
          self.set_param ("source_parameters", "name", value)
        if key == "RA":
          self.set_param ("source_parameters", "ra", value)
        if key == "DEC":
          self.set_param ("source_parameters", "dec", value)
        if key == "MODE":
          self.set_stream_params (nstream, "fold_processing_parameters", "mode", value)
          if value == "CAL":
            self.set_param ("calibration_parameters", "signal", 1)
          else:
            self.set_param ("calibration_parameters", "signal", 0)
          self.set_stream_params (nstream, "fold_processing_parameters", "mode", value)
        if key == "PERFORM_FOLD":
          self.set_stream_params (nstream, "processing_modes", "fold", value)
        if key == "PERFORM_SRCH":
          self.set_stream_params (nstream, "processing_modes", "search", value)
        if key == "PERFORM_CONT":
          self.set_stream_params (nstream, "processing_modes", "continuum", value)
        if key == "FOLD_OUTNCHAN":
          self.set_stream_params (nstream, "fold_processing_parameters", "output_nchan", value)
        if key == "FOLD_OUTNBIN":
          self.set_stream_params (nstream, "fold_processing_parameters", "output_nbin", value)
        if key == "FOLD_OUTTSUBINT":
          self.set_stream_params (nstream, "fold_processing_parameters", "output_tsubint", value)
        if key == "FOLD_OUTNPOL":
          self.set_stream_params (nstream, "fold_processing_parameters", "output_npol", value)
        if key == "FOLD_SK":
          self.set_stream_params (nstream, "fold_processing_parameters", "sk", value)
        if key == "FOLD_SK_THRESHOLD":
          self.set_stream_params (nstream, "fold_processing_parameters", "sk_threshold", value)
        if key == "FOLD_SK_NSAMPS":
          self.set_stream_params (nstream, "fold_processing_parameters", "sk_nsamps", value)
        if key == "SEARCH_DM":
          self.set_stream_params (nstream, "search_processing_parameters", "custom_dm", value)
        if key == "SEARCH_OUTNCHAN":
          self.set_stream_params (nstream, "search_processing_parameters", "output_nchan", value)
        if key == "SEARCH_OUTNBIT":
          self.set_stream_params (nstream, "search_processing_parameters", "output_nbit", value)
        if key == "SEARCH_OUTTSUBINT":
          self.set_stream_params (nstream, "search_processing_parameters", "output_tsubint", value)
        if key == "SEARCH_OUTNPOL":
          self.set_stream_params (nstream, "search_processing_parameters", "output_npol", value)
        if key == "SEARCH_OUTTSAMP":
          self.set_stream_params (nstream, "search_processing_parameters", "output_tsamp", value)
        if key == "CONTINUUM_OUTNCHAN":
          self.set_stream_params (nstream, "continuum_processing_parameters", "output_nchan", value)
        if key == "CONTINUUM_OUTTSAMP":
          self.set_stream_params (nstream, "continuum_processing_parameters", "output_tsamp", value)
        if key == "CONTINUUM_OUTTSUBINT":
          self.set_stream_params (nstream, "continuum_processing_parameters", "output_tsubint", value)
        if key == "CONTINUUM_OUTNPOL":
          self.set_stream_params (nstream, "continuum_processing_parameters", "output_npol", value)

        command = "configure"
        reply = "ok"

      except KeyError as e:
        self.log (0, "TCSInterfaceDaemon::parse_obs_cmd KeyError exception: " + str(e))
        return (False, "none", "Could not find key " + str(e))

    else:
      self.log(1, "TCSInterfaceDaemon::parse_obs_cmd command was not key or key + value")
      result = False
      command = "none"
      reply = "bad input line"

    return (result, command, reply)

  def range_inclusive (self, lower, upper, value):

    if value < lower:
      return (False, str(value) + " was below lower limit [" + str(lower) + "]")
    if value > upper:
      return (False, str(value) + " was above upper limit [" + str(upper) + "]")
    return (True, "")

  def power_of_two (self, value):
    if value == math.pow(math.log2(value),2):
      return (False, str(value) + " was not a lower of two")
    return (True, "")

  def power_of_two_range_inclusive (self, lower, upper, value):

    (ok, message) = self.range_inclusive (lower, upper, value)
    if not ok:
      return (ok, message)
    (ok, message) = self.power_of_two (value)
    if not ok:
      return (ok, message)
    return (True, "")

  def in_list (self, valid_values, value):
    if value in valid_values:
      return (True, "")
    else:
      return (False, str(value) + " was not in the allowed list " + str(valid_values))


  ###############################################################################
  # issue_start_cmd
  def issue_start_cmd (self, line):

    self.log(2, "issue_start_cmd()")

    # check calibration parameters to see if we can enable the CAL processing
    if self.get_param("calibration_parameters","signal") == "1":
      # read the cal freq
      calfreq_str = float(self.get_param("calibration_parameters","freq"))
      self.info("calibration_parmaeters::freq=" + str(calfreq_str))
      valid_cal = False
      try:
        calfreq_int = int(calfreq_str)
        if calfreq_int > 0 and calfreq_str == float(calfreq_int):
          valid_cal = True
          self.info("calibration_parmaeters::freq [" + str(calfreq_str) + "] was an integer as [" + str(calfreq_int) + "]")
        else:
          self.info("calibration_parmaeters::freq [" + str(calfreq_str) + "] was not > 0")
      except:
        self.info("calibration_parmaeters::freq [" + str(calfreq_str) + "] was not an integer")

      # if the calfreq is a positive integer, assume 0.5 duty cycle and 0.0 phase
      if valid_cal:
        self.set_param("calibration_parameters","duty_cycle", "0.5")
        self.set_param("calibration_parameters","phase", "0.5")
        self.set_param("calibration_parameters","tsys_avg_time", "10")
        self.set_param("calibration_parameters","tsys_freq_resolution", "1")
        # generate a fake epoch for the CAL
        fake_cal_epoch = times.getUTCTime()
        self.set_param ("calibration_parameters", "epoch", fake_cal_epoch)
      else:
        self.set_param("calibration_parameters","signal", "0")
        self.set_param("calibration_parameters","epoch", "Unknown")

    script.beam_cfg["obs_cmd"]["command"] = "configure"
    xml = xmltodict.unparse (script.beam_cfg)
    self.log(2, "issue_start_cmd xml=" + str(xml))
    sock = sockets.openSocket (DL, self.host, self.spip_tcs_port, 1)
    if sock:
      self.log(1, "UWB_TCS <- configure")
      sock.send (xml + "\r\n")
      xml_reply = sock.recv(131072)
      sock.close ()
      reply = xmltodict.parse (xml_reply)
      self.log(1, "UWB_TCS -> " + reply["tcs_response"])
      if reply["tcs_response"] != "OK":
        self.log(-1, "TCSInterfaceDaemon::issue_start_cmd: bad configuration: " + reply["tcs_response"])
        return reply["tcs_response"]
    else:
      self.log(-1, "TCSInterfaceDaemon::issue_start_cmd could not connect to uwb_tcs")
      return "FAIL: could not connect to Medusa's TCS service"
    
    # the configure command did work, start!

    script.beam_cfg["obs_cmd"]["command"] = "start"
    utc_start = times.getUTCTime(10)
    self.set_param ("observation_parameters", "utc_start", utc_start)

    xml = xmltodict.unparse (script.beam_cfg)
    # convert dict into XML to send to spip_tcs

    sock = sockets.openSocket (DL, self.host, self.spip_tcs_port, 1)
    if sock:
      self.log(1, "UWB_TCS <- start")
      sock.send (xml + "\r\n")
      xml_reply = sock.recv(131072)
      sock.close ()
      reply = xmltodict.parse (xml_reply)
      self.log(1, "UWB_TCS -> " + reply["tcs_response"])
      if reply["tcs_response"] != "OK":
        self.log(-1, "TCSInterfaceDaemon::issue_start_cmd: bad configuration: " + reply["tcs_response"])
        return reply["tcs_response"]
      else:
        return "start_utc " + utc_start
    else:
      self.log(1, "TCSInterfaceDaemon::issue_start_cmd could not connect to spip_tcs")
      return "internal medusa error"


  ###############################################################################
  # issue_stop_cmd
  def issue_stop_cmd (self, xml):

    self.log(2, "issue_stop_cmd()")
    script.beam_cfg["obs_cmd"]["command"] = "stop"
    xml = xmltodict.unparse (script.beam_cfg)
    # convert dict into XML to send to spip_tcs

    sock = sockets.openSocket (DL, self.host, self.spip_tcs_port, 1)
    if sock:
      sock.send (xml + "\r\n")
      sock.close ()
    else:
      self.log(1, "TCSInterfaceDaemon::issue_stop_cmd could not conenct to spip_tcs")

    # reset the XML configuration to a default value
    self.configure_beam_state()

    time.sleep(1)

    return "ok"

class TCSServerDaemon (TCSInterfaceDaemon, ServerBased):

  def __init__ (self, name):
    TCSInterfaceDaemon.__init__(self, name, "-1")
    ServerBased.__init__(self, self.cfg)

    self.host = sockets.getHostNameShort()
    self.interface_port = int(self.cfg["TCS_INTERFACE_PORT"]) - 1
    self.spip_tcs_port = int(self.cfg["TCS_INTERFACE_PORT"])

    # beam_states maintains info about last observation for beam
    self.beam_cfg = {}

###############################################################################
#
if __name__ == "__main__":

  if len(sys.argv) != 2:
    print "ERROR: at most 1 command line argument expected"
    sys.exit(1)

  # this should come from command line argument
  beam_id = sys.argv[1]

  # if the beam_id is < 0, then there is a single TCS for 
  # all beams, otherwise, 1 per beam
    
  if int(beam_id) == -1:
    script = TCSServerDaemon ("uwb_tcs_interface")
    beam_id = 0
  else:
    print "ERROR: only server mode supported"
    sys.exit(1)


  state = script.configure (DAEMONIZE, DL, "tcs_inteface", "tcs_interface")
  if state != 0:
    sys.exit(state)

  script.log(1, "STARTING SCRIPT")

  try:

    # prepare the beam_cfg dict
    script.configure_beam_state ()
    script.main (beam_id)

  except:

    script.quit_event.set()

    script.log(-2, "exception caught: type=" + str(sys.exc_info()[0]) + " value="+str(sys.exc_info()[1]))
    script.log(0, "-----------------------------------------")
    formatted_lines = traceback.format_exc().splitlines()
    for formatted_line in formatted_lines:
      script.log(0, formatted_line)
    script.log(0, "-----------------------------------------")

  script.log(1, "STOPPING SCRIPT")
  script.conclude()
  sys.exit(0)
