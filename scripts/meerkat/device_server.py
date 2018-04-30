#!/usr/bin/env python

###############################################################################
#  
#     Copyright (C) 2015 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
# 
###############################################################################

from katcp import DeviceServer, Sensor, ProtocolFlags, AsyncReply
from katcp.kattypes import (Str, Int, Float, Bool, Timestamp, Discrete,
                            request, return_reply)

import os, threading, sys, socket, select, signal, traceback, xmltodict, re
import errno, json, time

from xmltodict import parse
from xml.parsers.expat import ExpatError

from spip.utils import sockets
from spip.utils import catalog

DAEMONIZE = True
DL = 2

##############################################################
# Actual KATCP server implementation
class KATCPServer (DeviceServer):

  VERSION_INFO = ("ptuse-api", 2, 0)
  BUILD_INFO = ("ptuse-implementation", 0, 1, "")

  # Optionally set the KATCP protocol version and features. Defaults to
  # the latest implemented version of KATCP, with all supported optional
  # features
  PROTOCOL_INFO = ProtocolFlags(5, 0, set([
    ProtocolFlags.MULTI_CLIENT,
    ProtocolFlags.MESSAGE_IDS,
  ]))

  def __init__ (self, server_host, server_port, script):
    self.script = script
    self._host_sensors = {}
    self._beam_sensors = {}
    self._data_product = {}
    self._data_product["id"] = "None"
    self._data_product["state"] = "None"

    self.data_product_res = []
    self.data_product_res.append(re.compile ("^[a-zA-Z]+_1"))
    self.data_product_res.append(re.compile ("^[a-zA-Z]+_2"))
    self.data_product_res.append(re.compile ("^[a-zA-Z]+_3"))
    self.data_product_res.append(re.compile ("^[a-zA-Z]+_4"))

    self.script.log(2, "KATCPServer::__init__ starting DeviceServer on " + server_host + ":" + str(server_port))
    DeviceServer.__init__(self, server_host, server_port)

  DEVICE_STATUSES = ["ok", "degraded", "fail"]

  def setup_sensors(self):
    """Setup server sensors."""
    self.script.log(2, "KATCPServer::setup_sensors()")

    self._device_status = Sensor.discrete("device-status",
      description="Status of entire system",
      params=self.DEVICE_STATUSES,
      default="ok")
    self.add_sensor(self._device_status)

    self._beam_name = Sensor.string("beam-name",
      description="name of configured beam",
      unit="",
      default="")
    self.add_sensor(self._beam_name)

    # setup host based sensors   
    self._host_name = Sensor.string("host-name",
      description="hostname of this server",
      unit="",
      default="")
    self.add_sensor(self._host_name)

    # GUI URL TODO remove hardcoding
    guis = [ { "title": "PTUSE Web Interface",
               "description": "Live Pulsar timing monitoring plots", 
               "href": self.script.cfg["SPIP_ADDRESS"] } ]
    encoded = json.dumps(guis)
    self._gui_urls = Sensor.string("gui-urls",
      description="PTUSE GUI URL",
      unit="",
      default=encoded)
    self.add_sensor(self._gui_urls) 
    self._gui_urls.set_value(encoded)

    # give LMC some time to prepare the socket
    time.sleep(5)

    self.script.log(1, "KATCPServer::setup_sensors lmc="+str(self.script.lmc))
    (host, port) = self.script.lmc.split(":")
    self.setup_sensors_host (host, port)

    self.script.log(2, "KATCPServer::setup_sensors beams="+str(self.script.beam))
    self.setup_sensors_beam (self.script.beam_name)

  # add sensors based on the reply from the specified host
  def setup_sensors_host (self, host, port):

    self.script.log(1, "KATCPServer::setup_sensors_host ("+host+","+port+")")
    sock = sockets.openSocket (DL, host, int(port), 1)

    if sock:
      self.script.log(2, "KATCPServer::setup_sensors_host sock.send(" + self.script.lmc_cmd + ")") 
      sock.send (self.script.lmc_cmd + "\r\n")
      lmc_reply = sock.recv (65536)
      sock.close()
      xml = xmltodict.parse(lmc_reply)
      self.script.log(2, "KATCPServer::setup_sensors_host sock.recv=" + str(xml))

      self._host_sensors = {}

      # Disk sensors
      self.script.log(2, "KATCPServer::setup_sensors_host configuring disk sensors")
      disk_prefix = host+".disk"
      self._host_sensors["disk_size"] = Sensor.float(disk_prefix+".size",
        description=host+": disk size",
        unit="MB",
        params=[8192,1e9],
        default=0)
      self._host_sensors["disk_available"] = Sensor.float(disk_prefix+".available",
        description=host+": disk available space",
        unit="MB",
        params=[1024,1e9],
        default=0)
      self.add_sensor(self._host_sensors["disk_size"])
      self.add_sensor(self._host_sensors["disk_available"])

      # Server Load sensors
      self.script.log(2, "KATCPServer::setup_sensors_host configuring load sensors")
      self._host_sensors["num_cores"] = Sensor.integer (host+".num_cores",
        description=host+": disk available space",
        unit="MB",
        params=[1,64],
        default=0)

      self._host_sensors["load1"] = Sensor.float(host+".load.1min",
        description=host+": 1 minute load ",
        unit="",
        default=0)

      self._host_sensors["load5"] = Sensor.float(host+".load.5min",
        description=host+": 5 minute load ",
        unit="",
        default=0)
      
      self._host_sensors["load15"] = Sensor.float(host+".load.15min",
        description=host+": 15 minute load ",
        unit="",
        default=0)

      self._host_sensors["local_time_synced"] = Sensor.boolean("local_time_synced",
        description=host+": NTP server synchronisation",
        unit="",
        default=0)

      self.add_sensor(self._host_sensors["num_cores"])
      self.add_sensor(self._host_sensors["num_cores"])
      self.add_sensor(self._host_sensors["load1"])
      self.add_sensor(self._host_sensors["load5"])
      self.add_sensor(self._host_sensors["load15"])
      self.add_sensor(self._host_sensors["local_time_synced"])

      cpu_temp_pattern  = re.compile("cpu[0-9]+_temp")
      fan_speed_pattern = re.compile("fan[0-9,a-z]+")
      power_supply_pattern = re.compile("ps[0-9]+_status")
        
      self.script.log(2, "KATCPServer::setup_sensors_host configuring other metrics")

      if not xml["lmc_reply"]["sensors"] == None:

        for sensor in xml["lmc_reply"]["sensors"]["metric"]:
          name = sensor["@name"]
          if name == "system_temp":
            self._host_sensors[name] = Sensor.float((host+".system_temp"),
              description=host+": system temperature",
              unit="C",
              params=[-20,150],
              default=0)
            self.add_sensor(self._host_sensors[name])

          if cpu_temp_pattern.match(name):
            (cpu, junk) = name.split("_")
            self._host_sensors[name] = Sensor.float((host+"." + name),
              description=host+": "+ cpu +" temperature",
              unit="C",
              params=[-20,150],
              default=0)
            self.add_sensor(self._host_sensors[name])

          if fan_speed_pattern.match(name):
            self._host_sensors[name] = Sensor.float((host+"." + name),
              description=host+": "+name+" speed",
              unit="RPM",
              params=[0,20000],
              default=0)
            self.add_sensor(self._host_sensors[name])

          if power_supply_pattern.match(name):
            self._host_sensors[name] = Sensor.boolean((host+"." + name),
              description=host+": "+name,
              unit="",
              default=0)
            self.add_sensor(self._host_sensors[name])

          # TODO consider adding power supply sensors: e.g.
          #   device-status-kronos1-powersupply1
          #   device-status-kronos1-powersupply2
          #   device-status-kronos2-powersupply1
          #   device-status-kronos2-powersupply2

          # TODO consider adding raid/disk sensors: e.g.
          #   device-status-<host>-raid
          #   device-status-<host>-raid-disk1
          #   device-status-<host>-raid-disk2

        self.script.log(2, "KATCPServer::setup_sensors_host done!")

      else:
        self.script.log(2, "KATCPServer::setup_sensors_host no sensors found")

    else:
      self.script.log(-2, "KATCPServer::setup_sensors_host: could not connect to LMC")

  # setup sensors for each beam
  def setup_sensors_beam (self, beam):

    b = str(beam)
    self._beam_sensors = {}

    self.script.log(2, "KATCPServer::setup_sensors_beam beam="+b)

    self._beam_sensors["observing"] = Sensor.boolean("observing",
      description="Beam " + b + " is observing",
      unit="",
      default=0)
    self.add_sensor(self._beam_sensors["observing"])

    self._beam_sensors["snr"] = Sensor.float("snr",
      description="SNR of Beam "+b,
      unit="",
      params=[0,1e9],
      default=0)
    self.add_sensor(self._beam_sensors["snr"])

    self._beam_sensors["beamformer_stddev_polh"] = Sensor.float("beamformer_stddev_polh",
      description="Standard deviation of beam voltages for pol H",
      unit="",
      params=[0,127],
      default=0)
    self.add_sensor(self._beam_sensors["beamformer_stddev_polh"])

    self._beam_sensors["beamformer_stddev_polv"] = Sensor.float("beamformer_stddev_polv",
      description="Standard deviation of beam voltages for pol V",
      unit="",
      params=[0,127],
      default=0)
    self.add_sensor(self._beam_sensors["beamformer_stddev_polv"])

    self._beam_sensors["integrated"] = Sensor.float("integrated",
      description="Length of integration for Beam "+b,
      unit="",
      default=0)
    self.add_sensor(self._beam_sensors["integrated"])

  @request()
  @return_reply(Str())
  def request_beam(self, req):
    """Return the configure beam name."""
    return ("ok", self._beam_name.value())

  @request()
  @return_reply(Str())
  def request_host_name(self, req):
    """Return the name of this server."""
    return ("ok", self._host_name.value())

  @request()
  @return_reply(Float())
  def request_snr(self, req):
    """Return the SNR for this beam."""
    return ("ok", self._beam_sensors["snr"].value())

  @request()
  @return_reply(Float())
  def request_beamformer_stddev_polh(self, req):
    """Return the standard deviation of the 8-bit power level of pol H."""
    return ("ok", self._beam_sensors["beamformer_stddev_polh"].value())

  @request()
  @return_reply(Float())
  def request_beamformer_stddev_polv(self, req):
    """Return the standard deviation of the 8-bit power level of pol V."""
    return ("ok", self._beam_sensors["beamformer_stddev_polv"].value())

  @request()
  @return_reply(Float())
  def request_local_time_synced (self, req):
    """Return the sychronisation with NTP time"""
    return ("ok", self._beam_sensors["local_time_synced"].value())

  @request(Float())
  @return_reply(Str())
  def request_sync_time (self, req, adc_sync_time):
    """Set the ADC_SYNC_TIME for beam of the data product."""
    if self._data_product["id"] == "None":
      return ("fail", "data product was not configured")
    self.script.beam_config["lock"].acquire()
    self.script.beam_config["ADC_SYNC_TIME"] = str(adc_sync_time)
    self.script.beam_config["lock"].release()
    return ("ok", "")

  @request(Str())
  @return_reply(Str())
  def request_proposal_id (self, req, proposal_id):
    """Set the PROPOSAL_ID for the data product."""
    if self._data_product["id"] == "None":
      return ("fail", "data product was not configured")
    self.script.beam_config["lock"].acquire()
    self.script.beam_config["PROPOSAL_ID"] = proposal_id
    self.script.beam_config["lock"].release()
    return ("ok", "")

  # ensure state changes work
  def change_state (self, command):

    state = self._data_product["state"]

    reply = "ok"
    message = ""

    if command == "configure":
      if state != "unconfigured":
        message = "received " + command + " command when in " + state + " state"
      else:
        new_state = "configured"

    elif command == "capture_init":
      if state != "configured":
        message = "received " + command + " command when in " + state + " state"
      else:
        new_state = "ready"

    elif command == "target_start":
      if state != "ready":
        message = "received " + command + " command when in " + state + " state"
      else:
        new_state = "recording"

    elif command == "target_stop":
      if state != "recording":
        message = "received " + command + " command when in " + state + " state"
      else:
        new_state = "ready"

    elif command == "capture_done":
      if state != "ready" and state != "configured":
        message = "received " + command + " command when in " + state + " state"
      else:
        new_state = "configured"

    elif command == "deconfigure":
      if state != "configured":
        message = "received " + command + " command when in " + state + " state"
      else:
        new_state = "unconfigured"

    if message == "":
      self.script.log (1, "change_state: " + self._data_product["state"] + " -> " + new_state)
      self._data_product["state"] = new_state
    else:
      self.script.log (-1, "change_state: " + message)
      reply = "fail"

    return (reply, message)


  @request(Str())
  @return_reply(Str())
  def request_target_start (self, req, target_name):
    """Commence data processing using target."""
    self.script.log (1, "request_target_start(" + target_name+")")
    if self._data_product["id"] == "None":
      return ("fail", "data product was not configured")

    self.script.log (1, "request_target_start ADC_SYNC_TIME=" + self.script.cam_config["ADC_SYNC_TIME"])

    self.script.beam_config["lock"].acquire()
    self.script.beam_config["TARGET"] = self.script.cam_config["TARGET"]
    if self.script.cam_config["ADC_SYNC_TIME"] != "0":
      self.script.beam_config["ADC_SYNC_TIME"] = self.script.cam_config["ADC_SYNC_TIME"]

    self.script.beam_config["NCHAN_PER_STREAM"] = self.script.cam_config["NCHAN_PER_STREAM"]
    self.script.beam_config["PRECISETIME_FRACTION_POLV"] = self.script.cam_config["PRECISETIME_FRACTION_POLV"]
    self.script.beam_config["PRECISETIME_FRACTION_POLH"] = self.script.cam_config["PRECISETIME_FRACTION_POLH"]
    self.script.beam_config["PRECISETIME_UNCERTAINTY_POLV"] = self.script.cam_config["PRECISETIME_UNCERTAINTY_POLV"]
    self.script.beam_config["PRECISETIME_UNCERTAINTY_POLH"] = self.script.cam_config["PRECISETIME_UNCERTAINTY_POLH"]
    self.script.beam_config["TFR_KTT_GNSS"] = self.script.cam_config["TFR_KTT_GNSS"]

    self.script.beam_config["OBSERVER"] = self.script.cam_config["OBSERVER"]
    self.script.beam_config["ANTENNAE"] = self.script.cam_config["ANTENNAE"]
    self.script.beam_config["SCHEDULE_BLOCK_ID"] = self.script.cam_config["SCHEDULE_BLOCK_ID"]
    self.script.beam_config["EXPERIMENT_ID"] = self.script.cam_config["EXPERIMENT_ID"]
    self.script.beam_config["DESCRIPTION"] = self.script.cam_config["DESCRIPTION"]
    self.script.beam_config["lock"].release()

    # check the pulsar specified is listed in the catalog
    (result, message) = self.test_pulsar_valid (target_name)
    if result != "ok":
      return (result, message)

    # check the ADC_SYNC_TIME is valid for this beam
    if self.script.beam_config["ADC_SYNC_TIME"] == "0":
      return ("fail", "ADC Synchronisation Time was not valid")

    # change the state 
    (result, message) = self.change_state ("target_start")
    if result != "ok":
      self.script.log (-1, "target_start: change_state failed: " + message)
      return (result, message)
  
    # set the pulsar name, this should include a check if the pulsar is in the catalog
    self.script.beam_config["lock"].acquire()
    self.script.beam_config["SOURCE"] = target_name
    self.script.beam_config["lock"].release()

    host = self.script.tcs_host
    port = self.script.tcs_port

    self.script.log (2, "request_target_start: opening socket to " + host + ":" + str(port))
    sock = sockets.openSocket (DL, host, int(port), 1)
    if sock:
      xml = self.script.get_xml_config()
      self.script.log (2, "request_target_start: get_xml_config=" + str(xml))
      sock.send(xml + "\r\n")
      reply = sock.recv (65536)
      self.script.log (2, "request_target_start: reply=" + str(reply))

      xml = self.script.get_xml_start_cmd()
      self.script.log (2, "request_target_start: get_xml_start_cmd=" + str(xml))
      sock.send(xml + "\r\n")
      reply = sock.recv (65536)
      self.script.log (2, "request_target_start: reply=" + str(reply))

      sock.close()
      return ("ok", "")
    else:
      return ("fail", "could not connect to TCS")

  @request()
  @return_reply(Str())
  def request_target_stop (self, req):
    """Cease data processing with target_name."""
    self.script.log (1, "request_target_stop()")
    return self.target_stop()

  def target_stop (self):

    if self._data_product["id"] == "None":
      return ("fail", "data product was not configured")

    # change the state 
    (result, message) = self.change_state ("target_stop")
    if result != "ok":
      self.script.log (-1, "target_stop: change_state failed: " + message)
      return (result, message)

    self.script.reset_beam_config ()

    host = self.script.tcs_host
    port = self.script.tcs_port
    sock = sockets.openSocket (DL, host, int(port), 1)
    if sock:
      xml = self.script.get_xml_stop_cmd ()
      sock.send(xml + "\r\n")
      reply = sock.recv (65536)
      sock.close()
      return ("ok", "")
    else:
      return ("fail", "could not connect to tcs[beam]")

  @request()
  @return_reply(Str())
  def request_capture_init (self, req):
    """Prepare the ingest process for data capture."""
    self.script.log (1, "request_capture_init()")

    # change the state 
    (result, message) = self.change_state ("capture_init")
    if result != "ok":
      self.script.log (-1, "capture_init: change_state failed: " + message)
      return (result, message)

    return ("ok", "")

  @request()
  @return_reply(Str())
  def request_capture_done(self, req):
    """Terminte the ingest process."""
    self.script.log (1, "request_capture_done()")
    return self.capture_done()

  def capture_done(self):

    # in case the observing was terminated early
    if self._data_product["state"] == "recording":
      (result, message) = self.target_stop ()

    # change the state
    (result, message) = self.change_state ("capture_done")
    if result != "ok":
      self.script.log (-1, "capture_done: change_state failed: " + message)
      return (result, message)

    return ("ok", "")

  @return_reply(Str())
  def request_configure(self, req, msg):
    """Prepare and configure for the reception of the data_product_id."""
    self.script.log (1, "request_configure: nargs= " + str(len(msg.arguments)) + " msg=" + str(msg))
    if len(msg.arguments) == 0:
      self.script.log (-1, "request_configure: no arguments provided")
      return ("ok", "configured data products: TBD")

    # the sub-array identifier
    data_product_id = msg.arguments[0]

    if len(msg.arguments) == 1:
      self.script.log (1, "request_configure: request for configuration of " + str(data_product_id))
      if data_product_id == self._data_product["id"]:
        configuration = str(data_product_id) + " " + \
                        str(self._data_product['antennas']) + " " + \
                        str(self._data_product['n_channels']) + " " + \
                        str(self._data_product['cbf_source']) + " " + \
                        str(self._data_product['proxy_name'])
        self.script.log (1, "request_configure: configuration of " + str(data_product_id) + "=" + configuration)
        return ("ok", configuration)
      else:
        self.script.log (-1, "request_configure: no configuration existed for " + str(data_product_id))
        return ("fail", "no configuration existed for " + str(data_product_id))

    if len(msg.arguments) == 5:
      # if the configuration for the specified data product matches extactly the 
      # previous specification for that data product, then no action is required
      self.script.log (1, "configure: configuring " + str(data_product_id))

      if data_product_id == self._data_product["id"] and \
          self._data_product['antennas'] == msg.arguments[1] and \
          self._data_product['n_channels'] == msg.arguments[2] and \
          self._data_product['cbf_source'] == str(msg.arguments[3]) and \
          self._data_product['proxy_name'] == str(msg.arguments[4]):
        response = "configuration for " + str(data_product_id) + " matched previous"
        self.script.log (1, "configure: " + response)
        return ("ok", response)

      # the data product requires configuration
      else:
        self.script.log (1, "configure: new data product " + data_product_id)

        # TODO decide what to do regarding preconfigured params (e.g. FREQ, BW) vs CAM supplied values

        # determine which sub-array we are matched against
        the_sub_array = -1
        for i in range(4):
          self.script.log (1, "configure: testing self.data_product_res[" + str(i) +"].match(" + data_product_id +")")
          if self.data_product_res[i].match (data_product_id):
            the_sub_array = i + 1

        if the_sub_array == -1:
          self.script.log (1, "configure: could not match subarray from " + data_product_id)
          return ("fail", "could not data product to sub array")

        antennas = msg.arguments[1]
        n_channels = msg.arguments[2]
        cbf_source = str(msg.arguments[3])
        streams = json.loads (msg.arguments[3])
        proxy_name = str(msg.arguments[4])

        self.script.log (2, "configure: streams="+str(streams))

        # check if the number of existing + new beams > available
        # (cfreq, bwd, nchan1) = self.script.cfg["SUBBAND_CONFIG_0"].split(":")
        # (cfreq, bwd, nchan2) = self.script.cfg["SUBBAND_CONFIG_1"].split(":")
        # nchan = int(nchan1) + int(nchan2)
        #if nchan != int(n_channels):
        #  self._data_product.pop(data_product_id, None)
        #  response = "PTUSE configured for " + str(nchan) + " channels"
        #  self.script.log (-1, "configure: " + response)
        #  return ("fail", response)

        self._data_product['id'] = data_product_id
        self._data_product['antennas'] = antennas
        self._data_product['n_channels'] = n_channels
        self._data_product['cbf_source'] = cbf_source
        self._data_product['streams'] = str(streams)
        self._data_product['proxy_name'] = proxy_name
        self._data_product['state'] = "unconfigured"

        # change the state
        (result, message) = self.change_state ("configure")
        if result != "ok":
          self.script.log (-1, "configure: change_state failed: " + message)
          return (result, message)

        # determine the CAM metadata server and update pubsub
        cam_server = "None"
        fengine_stream = "None"
        polh_stream = "None"
        polv_stream = "None"
      
        self.script.log (2, "configure: streams.keys()=" + str(streams.keys()))
        self.script.log (2, "configure: streams['cam.http'].keys()=" + str(streams['cam.http'].keys()))

        if 'cam.http' in streams.keys() and 'camdata' in streams['cam.http'].keys():
          cam_server = streams['cam.http']['camdata']
          self.script.log (2,"configure: cam_server="+str(cam_server))
        if 'cbf.antenna_channelised_voltage' in streams.keys():
          stream_name = streams['cbf.antenna_channelised_voltage'].keys()[0]
          fengine_stream = stream_name.split(".")[0]
          self.script.log (2,"configure: fengine_stream="+str(fengine_stream))
        if 'cbf.tied_array_channelised_voltage' in streams.keys() and \
          len(streams['cbf.tied_array_channelised_voltage'].keys()) == 2:
          polh_stream = streams['cbf.tied_array_channelised_voltage'].keys()[0]
          polv_stream = streams['cbf.tied_array_channelised_voltage'].keys()[1]
          self.script.log (2,"configure: polh_stream="+str(polh_stream))

        if cam_server != "None" and fengine_stream != "None" and polh_stream != "None":
          self.script.pubsub.update_cam (cam_server, fengine_stream, polh_stream, polv_stream, antennas)
        else:
          response = "Could not extract streams[cam.http][camdata]"
          self.script.log (1, "configure: cam_server=" + cam_server)
          self.script.log (1, "configure: fengine_stream=" + fengine_stream)
          self.script.log (1, "configure: polh_stream=" + polh_stream)
          self.script.log (-1, "configure: " + response)
          return ("fail", response)
        
        # restart the pubsub service
        self.script.log (1, "configure: restarting pubsub for new meta-data")
        self.script.pubsub.restart()

        # determine the X and Y tied array channelised voltage streams
        mcasts = {}
        ports = {}
        key = 'cbf.tied_array_channelised_voltage'
        if key in streams.keys():
          stream = 'i0.tied-array-channelised-voltage.0x'
          if stream in streams[key].keys():
            (mcast, port) = self.parseStreamAddress (streams[key][stream])
            mcasts['x'] = mcast
            ports['x'] = int(port)
          else:
            response = "Could not extract streams["+key+"]["+stream+"]"
            self.script.log (-1, "configure: " + response)
            return ("fail", response)

          stream = 'i0.tied-array-channelised-voltage.0y'
          if stream in streams[key].keys():
            (mcast, port) = self.parseStreamAddress (streams[key][stream])
            mcasts['y'] = mcast
            ports['y'] = int(port)
          else:
            response = "Could not extract streams["+key+"]["+stream+"]"
            self.script.log (-1, "configure: " + response)
            return ("fail", response)

        self.script.log (1, "configure: connecting to RECV instance to update configuration")

        for istream in range(int(self.script.cfg["NUM_STREAM"])):
          (host, beam_idx, subband) = self.script.cfg["STREAM_" + str(istream)].split(":")
          beam = self.script.cfg["BEAM_" + beam_idx]
          self.script.log (1, "configure: istream="+str(istream)+ " beam=" + beam + " script.beam_name=" + self.script.beam_name)
          if beam == self.script.beam_name:

            # reset ADC_SYNC_TIME on the beam
            self.script.beam_config["lock"].acquire()
            self.script.beam_config["ADC_SYNC_TIME"] = "0";
            self.script.beam_config["lock"].release()

            port = int(self.script.cfg["STREAM_RECV_PORT"]) + istream
            self.script.log (1, "configure: connecting to " + host + ":" + str(port))
            sock = sockets.openSocket (DL, host, port, 1)
            if sock:
              req =  "<?req version='1.0' encoding='ISO-8859-1'?>"
              req += "<recv_cmd>"
              req +=   "<command>configure</command>"
              req +=   "<params>"

              req +=     "<param key='DATA_MCAST_0'>" + mcasts['x'] + "</param>"
              req +=     "<param key='DATA_PORT_0'>" + str(ports['x']) + "</param>"
              req +=     "<param key='META_MCAST_0'>" + mcasts['x'] + "</param>"
              req +=     "<param key='META_PORT_0'>" + str(ports['x']) + "</param>"
              req +=     "<param key='DATA_MCAST_1'>" + mcasts['y'] + "</param>"
              req +=     "<param key='DATA_PORT_1'>" + str(ports['y']) + "</param>"
              req +=     "<param key='META_MCAST_1'>" + mcasts['y'] + "</param>"
              req +=     "<param key='META_PORT_1'>" + str(ports['y']) + "</param>"

              req +=   "</params>"
              req += "</recv_cmd>"

              self.script.log (1, "configure: sending XML req ["+req+"]")
              sock.send(req)
              self.script.log (1, "configure: send XML, receiving reply")
              recv_reply = sock.recv (65536)
              self.script.log (1, "configure: received " + recv_reply)
              sock.close()
            else:
              response = "configure: could not connect to stream " + str(istream) + " at " + host + ":" + str(port)
              self.script.log (-1, "configure: " + response)
              return ("fail", response)

      return ("ok", "data product " + str (data_product_id) + " configured")

    else:
      response = "expected 0, 1 or 5 arguments, received " + str(len(msg.arguments))
      self.script.log (-1, "configure: " + response)
      return ("fail", response)

  # parse address of from spead://AAA.BBB.CCC.DDD+NN:PORT into 
  def parseStreamAddress (self, stream):

    self.script.log (2, "parseStreamAddress: parsing " + stream)
    (prefix, spead_address) = stream.split("//")
    (mcast, port) = spead_address.split(":")
    self.script.log (2, "parseStreamAddress: parsed " + mcast + ":" + port)

    return (mcast, port)

  @return_reply(Str())
  def request_deconfigure(self, req, msg):
    """Deconfigure for the data_product."""

    # in case the observing was terminated early
    if self._data_product["state"] == "recording":
      (result, message) = self.target_stop ()

    if self._data_product["state"] == "ready":
      (result, message) = self.capture_done()

    data_product_id = self._data_product["id"]

    # check if the data product was previously configured
    if not data_product_id == self._data_product["id"]:
      response = str(data_product_id) + " did not match configured data product [" + self._data_product["id"] + "]"
      self.script.log (-1, "configure: " + response)
      return ("fail", response)

    # change the state
    (result, message) = self.change_state ("deconfigure")
    if result != "ok":
      self.script.log (-1, "deconfigure: change_state failed: " + message)
      return (result, message)

    for istream in range(int(self.script.cfg["NUM_STREAM"])):
      (host, beam_idx, subband) = self.script.cfg["STREAM_" + str(istream)].split(":")
      if self.script.beam_name == self.script.cfg["BEAM_" + beam_idx]:

        # reset ADC_SYNC_TIME on the beam
        self.script.beam_config["lock"].acquire()
        self.script.beam_config["ADC_SYNC_TIME"] = "0";
        self.script.beam_config["lock"].release()

        port = int(self.script.cfg["STREAM_RECV_PORT"]) + istream
        self.script.log (3, "configure: connecting to " + host + ":" + str(port))
        sock = sockets.openSocket (DL, host, port, 1)
        if sock:

          req =  "<?req version='1.0' encoding='ISO-8859-1'?>"
          req += "<recv_cmd>"
          req +=   "<command>deconfigure</command>"
          req += "</recv_cmd>"

          sock.send(req)
          recv_reply = sock.recv (65536)
          sock.close()

      # remove the data product
      self._data_product["id"] = "None"

    response = "data product " + str(data_product_id) + " deconfigured"
    self.script.log (1, "configure: " + response)
    return ("ok", response)

  @request(Int())
  @return_reply(Str())
  def request_output_channels (self, req, nchannels):
    """Set the number of output channels."""
    self.script.log (1, "request_output_channels: nchannels=" + str(nchannels))
    if not self.test_power_of_two (nchannels):
      self.script.log (-1, "request_output_channels: " + str(nchannels) + " not a power of two")
      return ("fail", "number of channels not a power of two")
    if nchannels < 64 or nchannels > 4096:
      self.script.log (-1, "request_output_channels: " + str(nchannels) + " not within range 64 - 4096")
      return ("fail", "number of channels not within range 64 - 4096")
    self.script.beam_config["lock"].acquire()
    self.script.beam_config["OUTNCHAN"] = str(nchannels)
    self.script.beam_config["lock"].release()
    return ("ok", "")

  @request(Int())
  @return_reply(Str())
  def request_output_bins(self, req, nbin):
    """Set the number of output phase bins."""
    if not self.test_power_of_two(nbin):
      self.script.log (-1, "request_output_bins: " + str(nbin) + " not a power of two")
      return ("fail", "nbin not a power of two")
    if nbin < 64 or nbin > 2048:
      self.script.log (-1, "request_output_bins: " + str(nbin) + " not within range 64 - 2048")
      return ("fail", "nbin not within range 64 - 2048")
    self.script.beam_config["lock"].acquire()
    self.script.beam_config["OUTNBIN"] = str(nbin)
    self.script.beam_config["lock"].release()
    return ("ok", "")

  @request(Int())
  @return_reply(Str())
  def request_output_tsubint (self, req, tsubint):
    """Set the length of output sub-integrations."""
    if tsubint < 10 or tsubint > 60:
      self.script.log (-1, "request_output_tsubint: " + str(tsubint) + " not within range 10 - 60")
      return ("fail", "length of output subints must be between 10 and 60 seconds")
    self.script.beam_config["lock"].acquire()
    self.script.beam_config["OUTTSUBINT"] = str(tsubint)
    self.script.beam_config["lock"].release()
    return ("ok", "")

  @request(Float())
  @return_reply(Str())
  def request_dispersion_measure (self, req, dm):
    """Set the value of dispersion measure to be removed"""
    if dm > 2000:
      self.script.log (-1, "request_dispersion_measure: " + str(dm) + " > 2000")
      return ("fail", "dm greater than limit of 2000")
    self.script.beam_config["lock"].acquire()
    self.script.beam_config["DM"] = str(dm)
    self.script.beam_config["lock"].release()
    return ("ok", "")

  @request(Float())
  @return_reply(Str())
  def request_calibration_freq(self, req, cal_freq):
    """Set the value of noise diode firing frequecny in Hz."""
    if cal_freq < 0 or cal_freq > 1000:
      return ("fail", "CAL freq not within range 0 - 1000")
    self.script.beam_config["lock"].acquire()
    self.script.beam_config["CALFREQ"] = str(cal_freq)
    if cal_freq == 0:
      self.script.beam_config["MODE"] = "PSR"
    else:
      self.script.beam_config["MODE"] = "CAL"
    self.script.beam_config["lock"].release()
    return ("ok", "")

  @request(Int())
  @return_reply(Str())
  def request_output_npol(self, req, outnpol):
    """Set the number of output pol parameters."""
    if outnpol != 1 and outnpol != 2 and outnpol != 3 and outnpol != 4:
      self.script.log (-1, "request_output_npol: " + str(outnpol) + " not 1, 2 or 4")
      return ("fail", "output npol must be between 1, 2 or 4")
    self.script.beam_config["lock"].acquire()
    self.script.beam_config["OUTNPOL"] = str(outnpol)
    self.script.beam_config["lock"].release()
    return ("ok", "")

  @request(Int())
  @return_reply(Str())
  def request_output_nbit(self, req, outnbit):
    """Set the number of bits per output sample."""
    if outnbit != 1 and outnbit != 2 and outnbit != 4 and outnbit != 8:
      self.script.log (-1, "request_output_nbit: " + str(outnbit) + " not 1, 2, 4 or 8")
      return ("fail", "output nbit must be between 1, 2, 4 or 8")
    self.script.beam_config["lock"].acquire()
    self.script.beam_config["OUTNBIT"] = str(outnbit)
    self.script.beam_config["lock"].release()
    return ("ok", "")

  @request(Int())
  @return_reply(Str())
  def request_output_tdec(self, req, outtdec):
    """Set the number of input samples integrated into 1 output sample."""
    if outtdec < 16 or outtdec > 131072:
      self.script.log (-1, "request_output_tdec: " + str(outtdec) + " not in range [16..131072]")
      return ("fail", "output tdec must be between 16 and 131072")
    self.script.beam_config["lock"].acquire()
    self.script.beam_config["OUTTDEC"] = str(outnbit)
    self.script.beam_config["lock"].release()
    return ("ok", "")

  @request()
  @return_reply(Str())
  def request_fold_mode (self, req):
    """Set the processing mode to produce folded archives."""
    self.script.beam_config["lock"].acquire()
    self.script.beam_config["PERFORM_FOLD"] = "1"
    self.script.beam_config["PERFORM_SEARCH"] = "0"
    self.script.beam_config["lock"].release()
    return ("ok", "")

  @request()
  @return_reply(Str())
  def request_search_mode (self, req):
    """Set the processing mode to produce filterbank data."""
    self.script.beam_config["lock"].acquire()
    self.script.beam_config["PERFORM_FOLD"] = "0"
    self.script.beam_config["PERFORM_SEARCH"] = "1"
    self.script.beam_config["lock"].release()
    return ("ok", "")

  # test if a number is a power of two
  def test_power_of_two (self, num):
    return num > 0 and not (num & (num - 1))

  # test whether the specified target exists in the pulsar catalog
  def test_pulsar_valid (self, target):

    self.script.log (2, "test_pulsar_valid: target='["+ target +"]")

    # remove the _R suffix
    if target.endswith('_R'):
      target = target[:-2]

    # check if the target matches the fluxcal.on file
    cmd = "grep " + target + " " + self.script.cfg["CONFIG_DIR"] + "/fluxcal.on | wc -l"
    rval, lines = self.script.system (cmd, 3)
    if rval == 0 and len(lines) == 1 and int(lines[0]) > 0:
      return ("ok", "")

    # check if the target matches the fluxcal.off file
    cmd = "grep " + target + " " + self.script.cfg["CONFIG_DIR"] + "/fluxcal.off | wc -l"
    rval, lines = self.script.system (cmd, 3)
    if rval == 0 and len(lines) == 1 and int(lines[0]) > 0:
      return ("ok", "")


    self.script.log (2, "test_pulsar_valid: get_psrcat_param (" + target + ", jname)")
    (reply, message) = self.get_psrcat_param (target, "jname")
    if reply != "ok":
      return (reply, message)

    self.script.log (2, "test_pulsar_valid: get_psrcat_param () reply=" + reply + " message=" + message)
    if message == target:
      return ("ok", "")
    else:
      return ("fail", "pulsar " + target + " did not exist in catalog")

  def get_psrcat_param (self, target, param):

    # remove the _R suffix
    if target.endswith('_R'):
      target = target[:-2]

    cmd = "psrcat -all " + target + " -c " + param + " -nohead -o short"
    rval, lines = self.script.system (cmd, 3)
    if rval != 0 or len(lines) <= 0:
      return ("fail", "could not use psrcat")

    if lines[0].startswith("WARNING"):
      return ("fail", "pulsar " + target_name + " did not exist in catalog")

    parts = lines[0].split()
    if len(parts) == 2 and parts[0] == "1":
      return ("ok", parts[1])

