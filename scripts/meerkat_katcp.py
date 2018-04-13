#!/usr/bin/env python

###############################################################################
#  
#     Copyright (C) 2015 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
# 
###############################################################################

import os, threading, sys, socket, select, signal, traceback, xmltodict
import errno, time, random, re

from xmltodict import parse
from xml.parsers.expat import ExpatError

from spip.daemons.bases import BeamBased
from spip.daemons.daemon import Daemon
from spip.log_socket import LogSocket
from spip.utils import sockets,times
from spip.utils import catalog
from spip.threads.reporting_thread import ReportingThread
from spip.threads.socketed_thread import SocketedThread
from meerkat.pubsub import PubSubThread
from meerkat.device_server import KATCPServer

DAEMONIZE = True
DL = 1

###############################################################
# KATCP daemon
class KATCPDaemon(Daemon):

  def __init__ (self, name, id):
    Daemon.__init__(self, name, str(id))
    self.beam_state = {}
    self.cam_config = {}
    self.beam_config = {}
    self.tcs_host = "None"
    self.tcs_port = -1
    self.beam_name = "None"
    self.lmc = "None"
    self.repack = "None"
    self.beam = str(id)
    self.host = []
    self.katcp = []
    self.pubsub = []
    self.snr = 0
    self.stddev = 0

    if not (self.cfg["INDEPENDENT_BEAMS"] == "true"):
      raise Exception ("KATCPDaemon incompatible with INDEPENDENT_BEAMS != true")

    self.reset_cam_config()

    # get a list of all LMCs
    for i in range(int(self.cfg["NUM_STREAM"])):
      (host, beam, subband) = self.cfg["STREAM_" + str(i)].split(":")
      if (beam == self.beam):
        # if not self.lmc == "None":
        #   raise Exception ("KATCPDaemon more than 1 matching lmc")
        self.lmc = host +  ":" + self.cfg["LMC_PORT"]
        self.host = host

    self.lmc_cmd = "<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?>" + \
                   "<lmc_cmd>" + \
                   "<requestor>katcp daemon</requestor>" + \
                   "<command>host_status</command>" + \
                   "</lmc_cmd>"

    self.cpu_temp_pattern  = re.compile("cpu[0-9]+_temp")
    self.fan_speed_pattern = re.compile("fan[0-9,a-z]+")
    self.power_supply_pattern = re.compile("ps[0-9]+_status")

    # get the repacking host:port for this beam
    for i in range(int(self.cfg["NUM_STREAM"])):
      (host, beam_id, subband_id) = self.cfg["STREAM_" + str(i)].split(":")
      # if the beam resides on this host
      if beam == self.beam:
        # if the current beam is not configured
        if self.repack == "None":
          self.repack = host + ":" + str(int(self.cfg["BEAM_REPACK_PORT"]) + int(beam))
    
    if self.repack == "None":
      raise Exception ("KATCPDaemon could not find beam repack daemon")

    self.repack_cmd = "<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?>" + \
                      "<repack_request>" + \
                      "<requestor>katcp daemon</requestor>" + \
                      "<type>state</type>" + \
                      "</repack_request>"

  def reset_cam_config (self):
    self.cam_config["ADC_SYNC_TIME"] = "0"
    self.cam_config["PRECISETIME_FRACTION_POLV"] = "0.0"
    self.cam_config["PRECISETIME_FRACTION_POLH"] = "0.0"
    self.cam_config["PRECISETIME_UNCERTAINTY_POLV"] = "0.0"
    self.cam_config["PRECISETIME_UNCERTAINTY_POLH"] = "0.0"
    self.cam_config["TFR_GNSS_KTT"] = "0.0"
    self.cam_config["NCHAN_PER_STREAM"] = "1"
    self.cam_config["TARGET"] = "None"
    self.cam_config["RA"] = "None"
    self.cam_config["DEC"] = "None"
    self.cam_config["OBSERVER"] = "None"
    self.cam_config["BANDWIDTH"] = "0"
    self.cam_config["CFREQ"] = "0"
    self.cam_config["TSUBINT"] = "10"
    self.cam_config["ANTENNAE"] = "None"
    self.cam_config["SCHEDULE_BLOCK_ID"] = "None"
    self.cam_config["EXPERIMENT_ID"] = "None"
    self.cam_config["PROPOSAL_ID"] = "None"
    self.cam_config["DESCRIPTION"] = "None"
    self.cam_config["POOL_RESOURCES"] = "None"
    self.cam_config["SUBARRAY_STATE"] = "None"

  # reset the sepcified beam configuration
  def reset_beam_config (self):
    self.beam_config["lock"].acquire()
    self.beam_config["SOURCE"] = "None"
    self.beam_config["RA"] = "None"
    self.beam_config["DEC"] = "None"
    self.beam_config["PID"] = "None"
    self.beam_config["OBSERVER"] = "None"
    self.beam_config["TOBS"] = "None"
    self.beam_config["MODE"] = "PSR"
    self.beam_config["CALFREQ"] = "0"
    self.beam_config["ADC_SYNC_TIME"] = "0"
    self.beam_config["PRECISETIME_FRACTION_POLV"] = "0.0"
    self.beam_config["PRECISETIME_FRACTION_POLH"] = "0.0"
    self.beam_config["PRECISETIME_UNCERTAINTY_POLV"] = "0.0"
    self.beam_config["PRECISETIME_UNCERTAINTY_POLH"] = "0.0"
    self.beam_config["PRECISETIME_UNCERTAINTY_POLH"] = "0.0"
    self.beam_config["TFR_GNSS_KTT"] = "0.0"
    self.beam_config["NCHAN_PER_STREAM"] = "0.0"
    self.beam_config["PERFORM_FOLD"] = "1"
    self.beam_config["PERFORM_SEARCH"] = "0"
    self.beam_config["ANTENNAE"] = "0"
    self.beam_config["SCHEDULE_BLOCK_ID"] = "None"
    self.beam_config["EXPERIMENT_ID"] = "None"
    self.beam_config["PROPOSAL_ID"] = "None"
    self.beam_config["DESCRIPTION"] = "None"
    self.beam_config["POOL_RESOURCES"] = "None"
    self.beam_config["DM"] = "-1"
    self.beam_config["OUTNCHAN"] = "4096"
    self.beam_config["OUTNBIN"] = "1024"
    self.beam_config["OUTTSUBINT"] = "8"
    self.beam_config["OUTNPOL"] = "4"
    self.beam_config["OUTNBIT"] =  "8"
    self.beam_config["OUTTDEC"] = "32"
    self.beam_config["lock"].release()

  #############################################################################
  # configure fixed sensors (for lifetime of script)
  def set_katcp (self, server):
    self.katcp = server

    self.katcp._beam_name.set_value (self.beam_name)
    self.katcp._host_name.set_value (self.host)

  def set_pubsub (self, pubsub):
      self.pubsub = pubsub

  def get_pubsub (self):
    return self.pubsub

  #############################################################################
  # script core
  def main (self, id):

    # connect to the various scripts running to collect the information
    # to be provided by the KATCPServer instance as sensors

    time.sleep(2)

    while not self.quit_event.isSet():

      self.log(3, "KATCPDaemon::main loop start")
      # the primary function of the KATCPDaemon is to update the 
      # sensors in the DeviceServer periodically

      # TODO compute overall device status
      self.katcp._device_status.set_value("ok")

      # connect to SPIP_LMC to retreive temperature information
      if self.quit_event.isSet():
        self.log(2, "KATCPDaemon::main quit_event was set, exiting main 1")
        return

      (host, port) = self.lmc.split(":")
      self.log(3, "KATCPDaemon::main updating sensors from LMC")
      try:
        self.log(3, "KATCPDaemon::main openSocket("+host+","+port+")")
        sock = sockets.openSocket (DL, host, int(port), 1)
        self.log(3, "KATCPDaemon::main socket opened")
        if sock:
          sock.settimeout(1.0)
          sock.send(self.lmc_cmd)
          lmc_reply = sock.recv (65536)
          xml = xmltodict.parse(lmc_reply)
          sock.close()

          if self.quit_event.isSet():
            self.log(2, "KATCPDaemon::main quit_event was set, exiting main 2")
            return
          self.log(3, "KATCPDaemon::main update_lmc_sensors("+host+",[xml])")
          self.update_lmc_sensors(host, xml)

      except socket.error as e:
        self.log(2, "KATCPDaemon::main socket error on LMC sensor read")
        if e.errno == errno.ECONNRESET:
          self.log(1, "lmc connection was unexpectedly closed")
          sock.close()

      except:
        self.log(2, "KATCPDaemon::main other exception on LMC sensor read")
        sock.close()

      self.log(3, "KATCPDaemon::main received LMC data")

      # connect to SPIP_REPACK to retrieve Pulsar SNR performance
      if self.quit_event.isSet():
        return
      self.log(3, "KATCPDaemon::main pulsar SNR sensor from REPACK")
      (host, port) = self.repack.split(":")
      try:
        sock = sockets.openSocket (DL, host, int(port), 1)
        if sock:
          sock.send (self.repack_cmd)
          repack_reply = sock.recv (65536)
          xml = xmltodict.parse(repack_reply)
          sock.close()

          if self.quit_event.isSet():
            return
          self.log(3, "KATCPDaemon::main update_repack_sensors("+host+",[xml])")
          self.update_repack_sensors(host, xml)

      except socket.error as e:
        if e.errno == errno.ECONNRESET:
          self.log(1, "repack connection was unexpectedly closed")
          sock.close()

      except:
        self.log(2, "KATCPDaemon::main other exception on repack sensor read")
        sock.close()


      # connect to STAT (TBD) to retrieve beam-former power levels
      host = "None"
      xml = "None"
      self.update_stat_sensors (host, xml)

      self.log(3, "KATCPDaemon::main sleeping for 5 seconds")

      to_sleep = 5
      while not self.quit_event.isSet() and to_sleep > 0:
        to_sleep -= 1
        time.sleep (1) 

      if not self.katcp.running():
        self.log (-2, "KATCP server was not running, exiting")
        self.quit_event.set()

  #############################################################################
  # return valid XML configuration 
  def get_xml_config (self):

    xml =  "<?xml version='1.0' encoding='ISO-8859-1'?>"
    xml += "<obs_cmd>"
    xml +=   "<command>configure</command>"
    xml +=   "<beam_configuration>"
    xml +=     "<nbeam key='NBEAM'>1</nbeam>"
    xml +=     "<beam_state_0 key='BEAM_STATE_0' name='" + self.beam_name + "'>on</beam_state_0>"
    xml +=   "</beam_configuration>"

    xml +=   "<source_parameters>"
    xml +=     "<name key='SOURCE' epoch='J2000'>" + self.beam_config["SOURCE"] + "</name>"

    ra = self.beam_config["RA"]
    if ra == "None":
      (reply, message) = catalog.get_psrcat_param (self.beam_config["SOURCE"], "raj")
      if reply == "ok":
        ra = message

    dec = self.beam_config["DEC"]
    if dec == "None":
      (reply, message) = catalog.get_psrcat_param (self.beam_config["SOURCE"], "decj")
      if reply == "ok":
        dec = message

    xml +=     "<ra key='RA' units='hh:mm:ss'>" + ra + "</ra>"
    xml +=     "<dec key='DEC' units='dd:mm:ss'>" + dec+ "</dec>"
    xml +=   "</source_parameters>"

    xml +=   "<observation_parameters>"
    xml +=     "<observer key='OBSERVER'>" + self.beam_config["OBSERVER"] + "</observer>"
    xml +=     "<project_id key='PID'>" + self.beam_config["PID"] + "</project_id>"
    xml +=     "<tobs key='TOBS' units='seconds'>" + self.beam_config["TOBS"] + "</tobs>"
    xml +=     "<calfreq key='CALFREQ'>" + self.beam_config["CALFREQ"] + "</calfreq>"
    xml +=     "<utc_start key='UTC_START'>None</utc_start>"
    xml +=     "<utc_stop key='UTC_STOP'>None</utc_stop>"
    xml +=   "</observation_parameters>"

    xml +=   "<custom_parameters>"
    xml +=     "<adc_sync_time key='ADC_SYNC_TIME'>" + self.beam_config["ADC_SYNC_TIME"] + "</adc_sync_time>"
    xml +=     "<antennae key='ANTENNAE'>" + self.beam_config["ANTENNAE"] + "</antennae>"
    xml +=     "<schedule_block_id key='SCHEDULE_BLOCK_ID'>" + self.beam_config["SCHEDULE_BLOCK_ID"] + "</schedule_block_id>"
    xml +=     "<experiment_id key='EXPERIMENT_ID'>" + self.beam_config["EXPERIMENT_ID"] + "</experiment_id>"
    xml +=     "<proposal_id key='PROPOSAL_ID'>" + self.beam_config["PROPOSAL_ID"] + "</proposal_id>"
    xml +=     "<program_block_id key='PROGRAM_BLOCK_ID'>" + "TBD" + "</program_block_id>"
    xml +=     "<description key='DESCRIPTION'>" + self.beam_config["DESCRIPTION"] + "</description>"
    xml +=     "<precisetime_fraction_polv key='PRECISETIME_FRACTION_POLV'>" + self.beam_config["PRECISETIME_FRACTION_POLV"] + "</precisetime_fraction_polv>"
    xml +=     "<precisetime_fraction_polh key='PRECISETIME_FRACTION_POLH'>" + self.beam_config["PRECISETIME_FRACTION_POLH"] + "</precisetime_fraction_polh>"
    xml +=     "<precisetime_uncertainty_polv key='PRECISETIME_UNCERTAINTY_POLV'>" + self.beam_config["PRECISETIME_UNCERTAINTY_POLV"] + "</precisetime_uncertainty_polv>"
    xml +=     "<precisetime_uncertainty_polh key='PRECISETIME_UNCERTAINTY_POLH'>" + self.beam_config["PRECISETIME_UNCERTAINTY_POLH"] + "</precisetime_uncertainty_polh>"
    xml +=     "<gnss_ktt_delta key='TFR_GNSS_KTT'>" + self.beam_config["TFR_GNSS_KTT"] + "</gnss_ktt_delta>"
    xml +=     "<nchan_per_stream key='NCHAN_PER_STREAM'>" + self.beam_config["NCHAN_PER_STREAM"] + "</nchan_per_stream>"
    xml +=   "</custom_parameters>"

    xml +=   "<processing_modes>"
    xml +=     "<fold key='PERFORM_FOLD'>" + self.beam_config["PERFORM_FOLD"] + "</fold>"
    xml +=     "<search key='PERFORM_SEARCH'>" + self.beam_config["PERFORM_SEARCH"] + "</search>"
    xml +=   "</processing_modes>"
    
    # processing parameters control the behaviour of PTUSE
    xml +=   "<fold_processing_parameters>"
    xml +=     "<output_nchannels key='FOLD_OUTNCHAN'>" + self.beam_config["OUTNCHAN"] + "</output_nchannels>"
    xml +=     "<dm key='FOLD_DM'>" + self.beam_config["DM"] + "</dm>"
    xml +=     "<output_nbins key='FOLD_OUTNBIN'>" + self.beam_config["OUTNBIN"] + "</output_nbins>"
    xml +=     "<output_tsubint key='FOLD_OUTTSUBINT'>" + self.beam_config["OUTTSUBINT"] + "</output_tsubint>"
    xml +=     "<output_npol key='FOLD_OUTNPOL'>" + self.beam_config["OUTNPOL"] + "</output_npol>"
    xml +=     "<mode key='MODE'>" + self.beam_config["MODE"] + "</mode>"
    xml +=   "</fold_processing_parameters>"

    xml +=   "<search_processing_parameters>"
    xml +=     "<output_nbits key='SEARCH_OUTNBIT'>" + self.beam_config["OUTNBIT"] + "</output_nbits>"
    xml +=     "<output_nchannels key='SEARCH_OUTNCHAN'>" + self.beam_config["OUTNCHAN"] + "</output_nchannels>"
    xml +=     "<dm key='SEARCH_DM'>" + self.beam_config["DM"] + "</dm>"
    xml +=     "<output_tdec key='SEARCH_OUTTDEC'>" + self.beam_config["OUTTDEC"] + "</output_tdec>"
    xml +=     "<output_npol key='OUTNPOL'>" + self.beam_config["OUTNPOL"] + "</output_npol>"
    xml +=   "</search_processing_parameters>"

    xml += "</obs_cmd>"

    return xml

  def get_xml_start_cmd (self):

    xml  = "<?xml version='1.0' encoding='ISO-8859-1'?>"
    xml += "<obs_cmd>"
    xml +=   "<command>start</command>"
    xml +=   "<beam_configuration>"
    xml +=     "<nbeam key='NBEAM'>1</nbeam>"
    xml +=     "<beam_state_0 key='BEAM_STATE_0' name='" + self.beam_name + "'>on</beam_state_0>"
    xml +=   "</beam_configuration>"
    xml +=   "<observation_parameters>"
    xml +=     "<utc_start key='UTC_START'>None</utc_start>"
    xml +=   "</observation_parameters>"
    xml += "</obs_cmd>"

    return xml

  def get_xml_stop_cmd (self):

    xml  = "<?xml version='1.0' encoding='ISO-8859-1'?>"
    xml += "<obs_cmd>"
    xml += "<command>stop</command>"
    xml +=   "<beam_configuration>"
    xml +=     "<nbeam key='NBEAM'>1</nbeam>"
    xml +=     "<beam_state_0 key='BEAM_STATE_0' name='" + self.beam_name + "'>on</beam_state_0>"
    xml +=   "</beam_configuration>"
    xml +=   "<observation_parameters>"
    xml +=     "<utc_stop key='UTC_STOP'>None</utc_stop>"
    xml +=   "</observation_parameters>"
    xml += "</obs_cmd>"

    return xml

  def update_lmc_sensors (self, host, xml):

    self.katcp._host_sensors["disk_size"].set_value (float(xml["lmc_reply"]["disk"]["size"]["#text"]))
    self.katcp._host_sensors["disk_available"].set_value (float(xml["lmc_reply"]["disk"]["available"]["#text"]))

    self.katcp._host_sensors["num_cores"].set_value (int(xml["lmc_reply"]["system_load"]["@ncore"]))
    self.katcp._host_sensors["load1"].set_value (float(xml["lmc_reply"]["system_load"]["load1"]))
    self.katcp._host_sensors["load5"].set_value (float(xml["lmc_reply"]["system_load"]["load5"]))
    self.katcp._host_sensors["load15"].set_value (float(xml["lmc_reply"]["system_load"]["load15"]))

    self.katcp._host_sensors["local_time_synced"].set_value (bool(xml["lmc_reply"]["local_time_synced"]))

    if not xml["lmc_reply"]["sensors"] == None:
      for sensor in xml["lmc_reply"]["sensors"]["metric"]:

        name = sensor["@name"]
        value = sensor["#text"]

        if name == "system_temp":
          self.katcp._host_sensors[name].set_value (float(value))
        elif (self.cpu_temp_pattern.match(name)):
          (cpu, junk) = name.split("_")
          self.katcp._host_sensors[name].set_value (float(value))
        elif (self.fan_speed_pattern.match(name)):
          self.katcp._host_sensors[name].set_value (float(value))
        elif (self.power_supply_pattern.match(name)):
          self.katcp._host_sensors[name].set_value (value == "ok")

  def update_repack_sensors (self, host, xml):

    for beam in xml["repack_state"].items():
      # each beam is a tuple with [0]=beam and [1]=XML
      beam_name = beam[1]["@name"]
      active    = beam[1]["@active"]

      self.log (3, "KATCPDaemon::update_repack_sensors beam="+beam_name+" active="+active)
      if active == "True":
        source = beam[1]["source"]["name"]["#text"]
        self.log (3, "KATCPDaemon::update_repack_sensors source="+source)

        start = beam[1]["observation"]["start"]["#text"]
        integrated = beam[1]["observation"]["integrated"]["#text"]
        snr = beam[1]["observation"]["snr"]
        self.log (3, "KATCPDaemon::update_repack_sensors start="+start+" length="+integrated+" snr="+snr)

        self.katcp._beam_sensors["observing"].set_value (1)
        self.katcp._beam_sensors["snr"].set_value (float(snr))
        self.katcp._beam_sensors["integrated"].set_value (float(integrated))

      else:
        self.katcp._beam_sensors["observing"].set_value (0)
        self.katcp._beam_sensors["snr"].set_value (0)
        self.katcp._beam_sensors["integrated"].set_value (0)

    return ("ok")

  def update_stat_sensors (self, host, xml):

    # hack for now
    self.katcp._beam_sensors["beamformer_stddev_polh"].set_value (0)
    self.katcp._beam_sensors["beamformer_stddev_polv"].set_value (0)

    return ("ok")

  def get_SNR (self, b):
    return ("ok", self.beam_state["SNR"])

  def get_power(self, b):
    return ("ok", self.beam_state["POWER"])

class KATCPBeamDaemon (KATCPDaemon, BeamBased):

  def __init__ (self, name, id):
    KATCPDaemon.__init__(self, name, str(id))
    BeamBased.__init__(self, str(id), self.cfg)

    self.beam_name = self.cfg["BEAM_"+str(id)]
    self.beam_state = {}
    self.beam_state["NAME"] = self.beam_name
    self.beam_state["SNR"] = 0
    self.beam_state["POWER"] = 0

    self.beam_config = {}
    self.beam_config["lock"] = threading.Lock()
    self.reset_beam_config ()

    # if each beam is used independently of the others
    self.tcs_host = self.cfg["TCS_INTERFACE_HOST_" + str(id)]
    self.tcs_port = self.cfg["TCS_INTERFACE_PORT_" + str(id)]

###############################################################################
#
if __name__ == "__main__":

  if len(sys.argv) != 2:
    print "ERROR: at most 1 command line argument expected"
    sys.exit(1)

  # this should come from command line argument
  beam_id = sys.argv[1]

  # if the beam_id is < 0, then there is a single KATCP 
  # for all beams, otherwise, 1 per beam

  if int(beam_id) == -1:
    print "ERROR: script not configured for server mode"
    sys.exit(1)

  script = KATCPBeamDaemon ("meerkat_katcp", int(beam_id))

  state = script.configure (DAEMONIZE, DL, "katcp", "katcp")
  if state != 0:
    sys.exit(state)

  server = []

  try:
    server_host = sockets.getHostNameShort()
    server_port = 5000 + int(beam_id)

    script.log(2, "__main__: KATCPServer(" +server_host+"," + str(server_port) + ")")
    server = KATCPServer (server_host, server_port, script)

    script.log(2, "__main__: script.set_katcp()")
    script.set_katcp(server)

    script.log(1, "STARTING SCRIPT")
   
    script.log(1, "quit_file="+str(script.quit_file))
    script.log(2, "__main__: server.start()")
    server.start()

    pubsub = PubSubThread (script, beam_id)
    script.log(2, "__main__: pubsub.set_beam_name()")
    pubsub.set_beam_name (script.beam_name)
    script.log(2, "__main__: pubsub.start()")
    pubsub.start()
    script.set_pubsub (pubsub)

    script.log(2, "__main__: script.main()")
    script.main (beam_id)
    script.log(2, "__main__: script.main() returned")

    script.log(2, "__main__: stopping server")
    if server.running():
      server.stop()
    server.join()

    script.log(2, "__main__: stopping pubsub_thread")
    pubsub.join()

  except:

    script.quit_event.set()

    script.log(-2, "exception caught: " + str(sys.exc_info()[0]))
    print '-'*60
    traceback.print_exc(file=sys.stdout)
    print '-'*60
    if server.running():
      server.stop()
    server.join()

  script.log(1, "STOPPING SCRIPT")
  script.conclude()
  sys.exit(0)
