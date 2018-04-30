#!/usr/bin/env python

###############################################################################
#  
#     Copyright (C) 2015 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
# 
###############################################################################

import os, threading, sys, socket, select, signal, traceback, xmltodict, copy
import errno

from spip.config import Config
from spip.daemons.bases import ServerBased,BeamBased
from spip.daemons.daemon import Daemon
from spip.log_socket import LogSocket
from spip.utils import sockets,times
from spip.threads.reporting_thread import ReportingThread

DAEMONIZE = True
DL = 1

###############################################################
# thread for reporting state of 
class TCSReportingThread (ReportingThread):

  def __init__ (self, script, id):
    script.log(0, "TCSReportingThread::__init__ reporting on " + \
                    script.host + ":" + str(script.report_port))
    ReportingThread.__init__(self, script, script.host, script.report_port)
    self.beam_states = script.beam_states
    self.script.log(2, "TCSReportingThread::__init__ beam_states=" + str(self.beam_states))
    self.beams = self.beam_states.keys()
    self.script.log(2, "TCSReportingThread::__init__ beams=" + str(self.beams))

  def parse_message (self, xml):

    self.script.log (2, "TCSReportingThread::parse_message: " + str(xml))

    xml  = "<tcs_state>"
    for beam in self.beams:

      self.beam_states[beam]["lock"].acquire()

      c = self.beam_states[beam]["config"]

      xml += "<beam name='" + str(beam) + "' state='" + self.beam_states[beam]["state"] + "'>"

      s = c["source_parameters"]
      xml += "<source>"
      xml += "<name epoch='" + s["name"]["@epoch"] + "'>" + s["name"]["#text"] + "</name>"
      xml += "<ra units='" + s["ra"]["@units"] + "'>" + s["ra"]["#text"] + "</ra>"
      xml += "<dec units='" + s["dec"]["@units"] + "'>" + s["dec"]["#text"] + "</dec>"
      xml += "</source>"        

      o = c["observation_parameters"]
      xml += "<observation_parameters>"
      xml += "<observer>" + o["observer"]["#text"] + "</observer>"
      xml += "<project_id>" + o["project_id"]["#text"] + "</project_id>"

      utc_start = ""
      utc_stop = ""
      elapsed_time = ""

      if len(o["utc_start"]["#text"]) and o["utc_start"]["#text"] != None and  o["utc_start"]["#text"] != "None":
        utc_start = o["utc_start"]["#text"]
        elapsed_time = str(times.diffUTCTime(utc_start))

        if len(o["utc_stop"]["#text"]) > 0 and o["utc_stop"]["#text"] != None and o["utc_stop"]["#text"] != "None":
          utc_stop = o["utc_stop"]["#text"]
          elapsed_time = str(times.diffUTCTimes(utc_start, utc_stop))
        else:
          utc_stop = ""

      xml += "<utc_start>" + utc_start + "</utc_start>"
      xml += "<utc_stop>" + utc_stop + "</utc_stop>"
      xml += "<elapsed_time units='seconds'>" + elapsed_time + "</elapsed_time>"
      xml += "<expected_length units='seconds'>" + o["tobs"]["#text"] + "</expected_length>"
      xml += "</observation_parameters>"

      try:
        modes = c["processing_modes"]
        for k in modes.keys():
          key = modes[k]["@key"]
          val = modes[k]["#text"]

          mode_xml = ""
          # inject processing parameters into header
          if val == "true" or val == "1":
            mode_xml += "<" + k + "_processing_parameters>"
            p = c[k + "_processing_parameters"]
            for l in p.keys():
              try:
                pval = p[l]["#text"]
              except KeyError as e:
                val = ''
              mode_xml += "<" + l + ">" + pval + "</" + l + ">"
            mode_xml += "</" + k + "_processing_parameters>"
          xml += mode_xml
      except:
        self.script.log (2, "TCSReportingThread::parse_message exception")
      xml += "</beam>"
      
      self.beam_states[beam]["lock"].release()

    xml += "</tcs_state>\r\n"

    self.script.log (3, "TCSReportingThread::parse_message: returning " + str(xml))

    return True, xml

###############################################################
# TCS daemon
class TCSDaemon(Daemon):

  def __init__ (self, name, id):
    Daemon.__init__(self, name, str(id))
    self.beam_states = {}
    self.host = sockets.getHostNameShort()

  def load_finished (self):

    # read the most recently finished observations
    for b in self.beam_states.keys():

      # TODO check this for SERVER / BEAM
      beam_dir = self.fold_dir + "/finished/" + b

      cmd = "find " + beam_dir + " -mindepth 2 -maxdepth 2 -type d | sort | tail -n 1"
      rval, observation = self.system (cmd, 3)

      # strip prefix 
      observation = observation[0][(len(beam_dir)+1):]

      self.log (1, "main: " + observation)
      (utc, source) = observation.split("/")

      obs_dir = beam_dir + "/" + utc + "/" + source

      self.log(2, "load_finished: reading configuration for " + b + "/" + utc + "/" + source)

      if os.path.exists (obs_dir + "/obs.header"):
        header = Config.readCFGFileIntoDict(obs_dir + "/obs.header")
        self.beam_states[b]["lock"].acquire()

        self.beam_states[b]["config"]["source_parameters"]["name"]["#text"] = header["SOURCE"]
        self.beam_states[b]["config"]["source_parameters"]["name"]["@epoch"] = "J2000"

        self.beam_states[b]["config"]["source_parameters"]["ra"]["#text"] = header["RA"]
        self.beam_states[b]["config"]["source_parameters"]["ra"]["@units"] = "hh:mm:ss"
        self.beam_states[b]["config"]["source_parameters"]["dec"]["#text"] = header["DEC"]
        self.beam_states[b]["config"]["source_parameters"]["dec"]["@units"] = "dd:mm:ss"

        self.beam_states[b]["config"]["observation_parameters"]["observer"]["#text"] = header["OBSERVER"]
        self.beam_states[b]["config"]["observation_parameters"]["project_id"]["#text"] = header["PID"]
        self.beam_states[b]["config"]["observation_parameters"]["mode"]["#text"] = header["MODE"]
        self.beam_states[b]["config"]["observation_parameters"]["calfreq"]["#text"] = header["CALFREQ"]
        self.beam_states[b]["config"]["observation_parameters"]["tobs"]["#text"] = header["TOBS"]
        self.beam_states[b]["config"]["observation_parameters"]["utc_start"]["#text"] = header["UTC_START"]
        self.beam_states[b]["config"]["observation_parameters"]["utc_stop"]["#text"] = ""
        #self.beam_states[b]["config"]["observation_parameters"]["utc_stop"]["#text"] = header["UTC_STOP"]

        self.beam_states[b]["state"] = "Idle"

        self.beam_states[b]["lock"].release()

  def main (self, id):

    self.load_finished()

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
              raw = handle.recv(4096)

              message = raw.strip()

              if len(message) > 0:
                self.log(2, "TCSDaemon::commandThread message='" + message+"'")
                xml = xmltodict.parse(message)
                self.log(3, "<- " + str(xml))

                # Parse XML for correctness
                (valid, command, error) = self.parse_obs_cmd (xml, id)

                self.log(1, "TCSDaemon::commandThread valid=" + str(valid) \
                         + " command=" + command + " error=" + str(error))

                if valid :
                  if command == "start":
                    self.log(1, "TCSDaemon::commandThread issue_start_cmd message="+message)
                    self.issue_start_cmd (xml)
                  elif command == "stop":
                    self.log(1, "TCSDaemon::commandThread issue_stop_cmd message="+message)
                    self.issue_stop_cmd (xml)
                  elif command == "configure":
                    self.log(1, "TCSDaemon::commandThread no action for configure command")
                  else:
                    self.log(-1, "Unrecognized command [" + command + "]")
           
                else:
                  self.log(-1, "failed to parse xml: " + error)

                response = "OK"
                self.log(3, "-> " + response)
                xml_response = "<?xml version='1.0' encoding='ISO-8859-1'?>" + \
                               "<tcs_response>" + response + "</tcs_response>"
                handle.send (xml_response + "\r\n")

              else:
                self.log(1, "TCSDaemon::commandThread closing socket on 0 byte message")
                handle.close()
                for i, x in enumerate(can_read):
                  if (x == handle):
                    del can_read[i]
      
            except socket.error as e:
              if e.errno == errno.ECONNRESET:
                self.log(1, "TCSDaemon::commandThread closing connection")
                handle.close()
                for i, x in enumerate(can_read):
                  if (x == handle):
                    del can_read[i]
              else:
                raise


  ###############################################################################
  # parse an XML command for correctness
  def parse_obs_cmd (self, xml, id):

    command = ""

    try:

      command = xml['obs_cmd']['command']

      # determine which beams this command corresponds to
      for ibeam in range(int(xml['obs_cmd']['beam_configuration']['nbeam']['#text'])):
        state = xml['obs_cmd']['beam_configuration']['beam_state_' + str(ibeam)]['#text']
        if state == "on" or state == "1":
          b = xml['obs_cmd']['beam_configuration']['beam_state_' + str(ibeam)]['@name']

          # if the beam in the XML command is one of the beams managed by
          # this instance of spip_tcs
          if b in self.beam_states.keys():

            if command == "configure":

              self.log(2, "TCSDaemon::parse_obs_cmd received configuration for beam " + b)
              self.beam_states[b]["lock"].acquire()
              self.beam_states[b]["config"] = copy.deepcopy(xml['obs_cmd'])
              self.log (2, "TCSDaemon::parse_obs_cmd command==configure UTC_START=" + self.beam_states[b]["config"]["observation_parameters"]["utc_start"]["#text"])
              self.beam_states[b]["state"]     = "Configured"
              self.beam_states[b]["lock"].release()

            elif command == "start":

              self.beam_states[b]["lock"].acquire()
              self.beam_states[b]["state"] = "Starting"
              self.log (2, "TCSDaemon::parse_obs_cmd command==start config UTC_START=" + self.beam_states[b]["config"]["observation_parameters"]["utc_start"]["#text"])
              utc_start = xml['obs_cmd']['observation_parameters']['utc_start']['#text']
              self.beam_states[b]["config"]["observation_parameters"]["utc_start"]["#text"] = utc_start
              self.log (2, "TCSDaemon::parse_obs_cmd command==start config UTC_START=" + self.beam_states[b]["config"]["observation_parameters"]["utc_start"]["#text"])
              self.beam_states[b]["lock"].release()

            elif command == "stop":
  
              self.beam_states[b]["lock"].acquire()
              self.beam_states[b]["state"] = "Stopping"
              utc_stop = xml['obs_cmd']['observation_parameters']['utc_stop']['#text']
              self.beam_states[b]["config"]["observation_parameters"]["utc_stop"]["#text"] = utc_stop
              self.beam_states[b]["lock"].release()

            else:
              self.log(-1, "TCSDaemon::parse_obs_cmd unrecognized command " + command)

    except KeyError as e:
      self.log (0, "TCSDaemon::parse_obs_cmd KeyError exception: " + str(e))
      return (False, "none", "Could not find key " + str(e))

    return (True, command, "")

  ###############################################################################
  # issue_start_cmd
  def issue_start_cmd (self, xml):

    self.log(2, "TCSDaemon::issue_start_cmd nbeam=" + xml['obs_cmd']['beam_configuration']['nbeam']['#text'])

    # determine which beams this command corresponds to
    for ibeam in range(int(xml['obs_cmd']['beam_configuration']['nbeam']['#text'])):
      state = xml['obs_cmd']['beam_configuration']['beam_state_' + str(ibeam)]['#text']
      self.log(2, "TCSDaemon::issue_start_cmd beam state=" + state)
      if state == "1" or state == "on":
        b = xml['obs_cmd']['beam_configuration']['beam_state_' + str(ibeam)]['@name']
        self.log(2, "TCSDaemon::issue_start_cmd beam name=" + b)
        if b in self.beam_states.keys():
          obs = {}

          self.beam_states[b]["lock"].acquire()

          # add sour parameters from XML to header
          s = self.beam_states[b]["config"]["source_parameters"]
          for k in s.keys():
            key = s[k]["@key"]
            val = s[k]["#text"]
            obs[key] = val
 
          # inject the observation parameters         
          o = self.beam_states[b]["config"]["observation_parameters"]

          self.log(1, "TCSDaemon::issue_start_cmd o=" + str(o))
          self.log(1, "TCSDaemon::issue_start_cmd checking value of supplied UTC start: [" + o["utc_start"]["#text"] + "]" )

          # if no UTC_START has been specified, set it to +5 seconds
          if o["utc_start"]["#text"] == "None":
            local_current = times.getCurrentTime()
            utc_current = times.getUTCTime()

            local_offset = times.getCurrentTime(5)
            utc_offset = times.getUTCTime(5)

            self.log(1, "TCSDaemon::issue_start_cmd local_current=" + local_current)
            self.log(1, "TCSDaemon::issue_start_cmd   utc_current=" +   utc_current)
            self.log(1, "TCSDaemon::issue_start_cmd  local_offset=" + local_offset)
            self.log(1, "TCSDaemon::issue_start_cmd    utc_offset=" +   utc_offset)

            o["utc_start"]["#text"] = utc_offset
            self.log(1, "TCSDaemon::issue_start_cmd utc_start=" + o["utc_start"]["#text"])

          else:
            self.log(1, "TCSDaemon::issue_start_cmd utc_start already set to =" + o["utc_start"]["#text"])

          for k in o.keys():
            key = o[k]["@key"]
            try:
              val = o[k]["#text"]
            except KeyError as e:
              val = ''
            obs[key] = val
            self.log(1, key + "=" + val)

          # inject custom fields into header
          c = self.beam_states[b]["config"]["custom_parameters"]
          for k in c.keys():
            key = c[k]["@key"]
            try:
              val = c[k]["#text"]
            except KeyError as e:
              val = ''
            obs[key] = val
            self.log(1, key + "=" + val)

          modes = self.beam_states[b]["config"]["processing_modes"]
          for k in modes.keys():
            key = modes[k]["@key"]
            val = modes[k]["#text"] 
            obs[key] = val
            self.log(1, key + "=" + val)
            
            # inject processing parameters into header
            if val == "true" or val == "1":
              self.log (1, "TCSDaemon::issue_start_cmd mode=" + k)
              p = self.beam_states[b]["config"][k + "_processing_parameters"]
              for l in p.keys():
                pkey = p[l]["@key"]
                try:
                  pval = p[l]["#text"]
                except KeyError as e:
                  val = ''
                obs[pkey] = pval
                self.log(1, pkey + "=" + pval)

          # ensure the start command is set
          obs["COMMAND"] = "START"
          obs["OBS_OFFSET"] = "0"

          self.beam_states[b]["lock"].release()

          # convert to a single ascii string
          obs_header = Config.writeDictToString (obs)

          # work out which streams correspond to these beams
          for istream in range(int(self.cfg["NUM_STREAM"])):
            (host, beam_idx, subband) = self.cfg["STREAM_"+str(istream)].split(":")
            beam = self.cfg["BEAM_" + beam_idx]
            self.log(2, "TCSDaemon::issue_start_cmd host="+host+" beam="+beam+" subband="+subband)

            # connect to streams for this beam only
            if beam == b:

              # control port the this recv stream 
              ctrl_port = int(self.cfg["STREAM_CTRL_PORT"]) + istream

              self.log(1, host + ":"  + str(ctrl_port) + " <- start")

              # connect to recv agent and provide observation configuration
              self.log(2, "TCSDaemon::issue_start_cmd openSocket("+host+","+str(ctrl_port)+")")
              recv_sock = sockets.openSocket (DL, host, ctrl_port, 5)
              if recv_sock:
                self.log(3, "TCSDaemon::issue_start_cmd sending obs_header")
                recv_sock.send(obs_header)
                self.log(3, "TCSDaemon::issue_start_cmd header sent")
                recv_sock.close()
                self.log(3, "TCSDaemon::issue_start_cmd socket closed")
              else:
                self.log(-2, "TCSDaemon::issue_start_cmd failed to connect to "+host+":"+str(ctrl_port))

              # connect to spip_gen and issue start command for UTC
              # assumes gen host is the same as the recv host!
              # gen_port = int(self.cfg["STREAM_GEN_PORT"]) + istream
              # sock = sockets.openSocket (DL, host, gen_port, 1)
              # if sock:
              #   sock.send(obs_header)
              #   sock.close()

          # update the dict of observing info for this beam
          self.beam_states[b]["lock"].acquire()
          self.beam_states[b]["state"] = "Recording"
          self.beam_states[b]["lock"].release()

  ###############################################################################
  # issue_stop_cmd
  def issue_stop_cmd (self, xml):

    self.log(2, "issue_stop_cmd()")

    # determine which beams this command corresponds to
    for ibeam in range(int(xml['obs_cmd']['beam_configuration']['nbeam']['#text'])):
      state = xml['obs_cmd']['beam_configuration']['beam_state_' + str(ibeam)]['#text']
      if state == "1" or state == "on":
        b = xml['obs_cmd']['beam_configuration']['beam_state_' + str(ibeam)]['@name']
        if b in self.beam_states.keys():

          self.log(1, "issue_stop_cmd: beam=" + b)
          obs = {}

          self.beam_states[b]["lock"].acquire()
          self.beam_states[b]["state"] = "Stopping"
          obs["COMMAND"] = "STOP"

          # inject the observation parameters         
          o = self.beam_states[b]["config"]["observation_parameters"]

          # if no UTC_STOP has been specified, set it to now
          if o["utc_stop"]["#text"] == "None":
            o["utc_stop"]["#text"] = times.getUTCTime()
          obs["UTC_STOP"] = o["utc_stop"]["#text"]
          self.beam_states[b]["lock"].release()

          # convert to a single ascii string
          obs_header = Config.writeDictToString (obs)

          # work out which streams correspond to these beams
          for istream in range(int(self.cfg["NUM_STREAM"])):
            (host, beam_idx, subband) = self.cfg["STREAM_"+str(istream)].split(":")
            beam = self.cfg["BEAM_" + beam_idx]
            self.log(2, "issue_stop_cmd: host="+host+" beam="+beam+" subband="+subband)

            # connect to streams for this beam only
            if beam == b:

              # control port the this recv stream 
              ctrl_port = int(self.cfg["STREAM_CTRL_PORT"]) + istream

              # connect to recv agent and provide observation configuration
              self.log(3, "issue_stop_cmd: openSocket("+host+","+str(ctrl_port)+")")
              sock = sockets.openSocket (DL, host, ctrl_port, 1)
              if sock:
                self.log(3, "issue_stop_cmd: sending obs_header")
                sock.send(obs_header)
                self.log(3, "issue_stop_cmd: command sent")
                sock.close()
                self.log(3, "issue_stop_cmd: socket closed")

              # connect to spip_gen and issue stop command for UTC
              # assumes gen host is the same as the recv host!
              # gen_port = int(self.cfg["STREAM_GEN_PORT"]) + istream
              # sock = sockets.openSocket (DL, host, gen_port, 1)
              # if sock:
              #   sock.send(obs_header)
              #  sock.close()

          # update the dict of observing info for this beam
          self.beam_states[b]["lock"].acquire()
          self.beam_states[b]["state"] = "Idle"
          self.beam_states[b]["lock"].release()


class TCSServerDaemon (TCSDaemon, ServerBased):

  def __init__ (self, name):
    TCSDaemon.__init__(self, name, "-1")
    ServerBased.__init__(self, self.cfg)
    self.interface_port = int(self.cfg["TCS_INTERFACE_PORT"])
    self.report_port = int(self.cfg["TCS_REPORT_PORT"])
    self.fold_dir = self.cfg["SERVER_FOLD_DIR"]

    # beam_states maintains info about last observation for beam
    for i in range(int(self.cfg["NUM_BEAM"])):
      b = self.cfg["BEAM_"+str(i)]
      self.beam_states[b] = {}
      self.beam_states[b]["config"] = {}

      self.beam_states[b]["config"]["source_parameters"] = {}
      self.beam_states[b]["config"]["observation_parameters"] = {}
      self.beam_states[b]["config"]["custom_parameters"] = {}
      self.beam_states[b]["config"]["processing_modes"] = {}

      self.beam_states[b]["config"]["source_parameters"]["name"] = {"@key":"SOURCE", "@epoch":"J2000", "#text":""}
      self.beam_states[b]["config"]["source_parameters"]["ra"] = {"@key":"RA", "@units":"hhmmss", "#text":""}
      self.beam_states[b]["config"]["source_parameters"]["dec"] = {"@key":"DEC", "@units":"ddmmss", "#text":""}
      self.beam_states[b]["config"]["observation_parameters"]["project_id"] = {"@key":"PID", "#text":""}
      self.beam_states[b]["config"]["observation_parameters"]["observer"] = {"@key":"OBSERVER", "#text":""}
      self.beam_states[b]["config"]["observation_parameters"]["utc_start"] = {"@key":"UTC_START", "#text":"None"}
      self.beam_states[b]["config"]["observation_parameters"]["utc_stop"] = {"@key":"UTC_STOP", "#text":"None"}
      self.beam_states[b]["config"]["observation_parameters"]["tobs"] = {"@key":"TOBS", "#text":""}
      self.beam_states[b]["config"]["observation_parameters"]["mode"] = {"@key":"MODE", "#text":""}
      self.beam_states[b]["config"]["observation_parameters"]["calfreq"] = {"@key":"CALFREQ", "#text":""}

      self.beam_states[b]["state"] = "Idle"
      self.beam_states[b]["lock"] = threading.Lock()

class TCSBeamDaemon (TCSDaemon, BeamBased):

  def __init__ (self, name, id):
    TCSDaemon.__init__(self, name, str(id))
    BeamBased.__init__(self, str(id), self.cfg)
    self.interface_port = int(self.cfg["TCS_INTERFACE_PORT_" + str(id)])
    self.report_port = int(self.cfg["TCS_REPORT_PORT_" + str(id)])
    self.fold_dir = self.cfg["CLIENT_FOLD_DIR"]

    b = self.cfg["BEAM_"+str(id)]
    self.beam_states[b] = {}
    self.beam_states[b]["config"] = {}
    self.beam_states[b]["config"]["source_parameters"] = {}
    self.beam_states[b]["config"]["observation_parameters"] = {}
    self.beam_states[b]["config"]["custom_parameters"] = {}
    self.beam_states[b]["config"]["processing_parameters"] = {}

    self.beam_states[b]["config"]["source_parameters"]["name"] = {"@key":"SOURCE", "@epoch":"J2000", "#text":""}
    self.beam_states[b]["config"]["source_parameters"]["ra"] = {"@key":"RA", "@units":"hhmmss", "#text":""}
    self.beam_states[b]["config"]["source_parameters"]["dec"] = {"@key":"DEC", "@units":"ddmmss", "#text":""}
    self.beam_states[b]["config"]["observation_parameters"]["project_id"] = {"@key":"PID", "#text":""}
    self.beam_states[b]["config"]["observation_parameters"]["observer"] = {"@key":"OBSERVER", "#text":""}
    self.beam_states[b]["config"]["observation_parameters"]["utc_start"] = {"@key":"UTC_START", "#text":"None"}
    self.beam_states[b]["config"]["observation_parameters"]["utc_stop"] = {"@key":"UTC_STOP", "#text":"None"}
    self.beam_states[b]["config"]["observation_parameters"]["tobs"] = {"@key":"TOBS", "#text":""}
    self.beam_states[b]["config"]["observation_parameters"]["mode"] = {"@key":"MODE", "#text":""}
    self.beam_states[b]["config"]["observation_parameters"]["calfreq"] = {"@key":"CALFREQ", "#text":""}

    self.beam_states[b]["state"] = "Idle"
    self.beam_states[b]["lock"] = threading.Lock()

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
    script = TCSServerDaemon ("spip_tcs")
    beam_id = 0
  else:
    script = TCSBeamDaemon ("spip_tcs", beam_id)

  state = script.configure (DAEMONIZE, DL, "tcs", "tcs")
  if state != 0:
    sys.exit(state)

  script.log(1, "STARTING SCRIPT")

  try:
   
    reporting_thread = TCSReportingThread(script, beam_id)
    reporting_thread.start()

    script.main (beam_id)

    reporting_thread.join()

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
