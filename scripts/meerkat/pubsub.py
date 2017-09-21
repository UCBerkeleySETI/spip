#!/usr/bin/env python

###############################################################################
#  
#     Copyright (C) 2015 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
# 
###############################################################################

import logging
import tornado.gen
from katportalclient import KATPortalClient,SensorLookupError

import os, threading, sys, socket, select, signal, traceback
import errno, time, random

from spip.utils import sockets,times

DAEMONIZE = True
DL = 1

###############################################################
# PubSub daemon
class PubSubThread (threading.Thread):

  def __init__ (self, script, id):
    threading.Thread.__init__(self)
    self.script = script
   
    self.script.log(2, "PubSubThread.__init__()")

    self.curr_utc = times.getUTCTime()
    self.prev_utc = self.curr_utc

    self.cam_server    = self.script.cfg["CAM_ADDRESS"]

    self.fengine_stream = 'i0'
    self.polh_stream = 'i0.tied-array-channelised-voltage.0x'

    self.logger = logging.getLogger('pubsub') 
    self.logger.setLevel(logging.INFO)
    self.beam = -1
    self.sub_array = -1
    self.mappings = {}
    self.io_loop = []
    self.policy = "event-rate 5.0 300.0"
    self.title  = "ptuse_unconfigured"
    self.running = False

    self.restart_io_loop = True

  # get the KATCP sensor  names for the meta-data server
  @tornado.gen.coroutine
  def configure (self):

    self.mapping = {}

    sensors = {}
    sensors["TARGET"]            = {"comp": "cbf", "sensor": "target"}
    sensors["RA"]                = {"comp": "cbf", "sensor": "pos.request-base-ra"}
    sensors["BMAJ"]              = {"comp": "cbf", "sensor": "beam-major-axis"}
    sensors["BMIN"]              = {"comp": "cbf", "sensor": "beam-minor-axis"}
    sensors["BPA"]               = {"comp": "cbf", "sensor": "beam-position-axis"}
    sensors["NCHAN"]             = {"comp": "cbf", "sensor": self.fengine_stream + '.antenna-channelised-voltage-n-chans'}
    sensors["ADC_SYNC_TIME"]     = {"comp": "cbf", "sensor": self.fengine_stream + '.synchronisation-epoch'}
    sensors["ITRF"]              = {"comp": "sub", "sensor": "reference-location-itrf"}
    sensors["SCHEDULE_BLOCK_ID"] = {"comp": "sub", "sensor": "active-sbs"}
    sensors["FREQ"]              = {"comp": "sub", "sensor": 'streams.' + self.polh_stream + '.centre-frequency'}
    sensors["BW"]                = {"comp": "sub", "sensor": 'streams.' + self.polh_stream + '.bandwidth'}
    sensors["SIDEBAND"]          = {"comp": "sub", "sensor": 'streams.' + self.polh_stream + '.sideband'}
    sensors["EXPERIMENT_ID"]     = {"comp": "sub", "sensor": 'observation.script-experiment-id'}
    sensors["OBSERVER"]          = {"comp": "sub", "sensor": "observation.script-observer"}
    sensors["PROPOSAL_ID"]       = {"comp": "sub", "sensor": "observation.script-proposal-id"}

    # TODO CAM ICD mandates observation.script-proposal-id

    self.script.log(3, "PubSubThread::configure sensors=" + str(sensors))

    for key in sensors.keys():

      comp = sensors[key]["comp"]
      sens = sensors[key]["sensor"]
      try:
        name = yield self.ws_client.sensor_subarray_lookup (component=comp, sensor=sens, return_katcp_name=False)
        self.script.log(2, "PubSubThread::configure " + key + " -> " + name)

        self.script.log(2, "PubSubThread::configure mappings["+name+"]=" + key)
        self.mappings[name] = key 

        # instruct the web socket to sample this sensor
        try:
          self.script.log(2, "PubSubThread::configure set_sampling_strategy(" + name + ")")
          result = yield self.ws_client.set_sampling_strategy (self.title, name, self.policy)
          self.script.log(2, "PubSubThread::configure set_sampling_strategy result="+str(result))

        except Exception, e:
          print 'PubSubThread::configure failed to set sampling stratey: ', e

      except SensorLookupError as exc:
        self.script.log(0, "PubSubThread::configure failed to find " + comp + "." + sens)

    self.script.log(2, "PubSubThread::configure added sensors")

  # configure a new metadata server 
  def update_cam (self, server, fengine_stream, polh_stream):

    self.script.log(2, "PubSubThread::update_cam("+server+","+fengine_stream+","+polh_stream+")")
    # server name is configured with the following schema
    # http://{host}/api/client/{subarray_number}
    self.cam_server = server

    self.fengine_stream = fengine_stream
    self.polh_stream = polh_stream
    self.sub_array = server.split('/')[-1]

  # configure the pub/sub instance to 
  def set_beam_name (self, beam):
    self.script.log(2, "PubSubThread::set_beam_name (" + str(beam) + ")")
    self.beam = str(beam)
    self.script.log(2, "PubSubThread::set_beam_name done")

  def run (self):
    self.script.log(1, "PubSubThread::run starting while")
    while self.restart_io_loop:

      # open connection to CAM
      self.io_loop = tornado.ioloop.IOLoop.current()
      self.io_loop.add_callback (self.connect, self.logger)

      self.running = True
      self.restart_io_loop = False
      self.io_loop.start()
      self.running = False
      self.io_loop = []

      # unsubscribe and disconnect from CAM
      self.ws_client.unsubscribe(self.title)
      self.ws_client.disconnect()

    self.script.log(2, "PubSubThread::run exiting")

  def join (self):
    self.script.log(2, "PubSubThread::join self.stop()")
    self.stop()

  def stop (self):
    self.script.log(2, "PubSubThread::stop()")
    if self.running:
      self.script.log(2, "PubSubThread::stop io_loop.stop()")
      self.io_loop.stop()
    time.sleep(0.1)
    return

  def restart (self):
    # get the IO loop to restart on the call to stop()
    self.restart_io_loop = True
    if self.running:
      self.script.log(2, "PubSubThread::restart self.stop()")
      self.stop()
    return

  @tornado.gen.coroutine
  def connect (self, logger):

    self.script.log(2, "PubSubThread::connect(" + self.cam_server + ")")
    self.ws_client = KATPortalClient(self.cam_server, self.on_update_callback, logger=logger)
    self.script.log(2, "PubSubThread::connect self.ws_client.connect()")
    yield self.ws_client.connect()

    self.script.log(2, "PubSubThread::connect self.ws_client.subscribe(" + self.title + ")")
    result = yield self.ws_client.subscribe(self.title)
    self.script.log(2, "PubSubThread::connect self.ws_client.subscribe result="+str(result))

    self.configure()

  def on_update_callback (self, msg):

    self.curr_utc = times.getUTCTime()
    if times.diffUTCTimes(self.prev_utc, self.curr_utc) > 60:
      self.script.log(2, "PubSubThread::on_update_callback: heartbeat msg="+str(msg))
      self.prev_utc = self.curr_utc

    self.update_config (msg)

  def update_cam_config (self, key, name, value):
    if key in self.script.cam_config.keys():
      if self.script.cam_config[key] != value:
        self.script.log(1, key + "=" + value)
        self.script.log(2, "PubSubThread::update_cam_config " + key + "=" + value + " from " + name)
        self.script.cam_config[key] = value
      else:
        self.script.log(2, "PubSubThread::update_cam_config no update on " + key + "=" + value + " from " + name)
    else:
      self.script.log(1, key + "=" + value)
      self.script.log(2, "PubSubThread::update_cam_config " + key + "=" + value + " from " + name)
      self.script.cam_config[key] = value

  def update_config (self, msg):

    # ignore empty messages
    if msg == []: 
      return

    self.script.log(3, "PubSubThread::update_config msg="+str(msg))
    status = msg["msg_data"]["status"]
    value = msg["msg_data"]["value"]
    name = msg["msg_data"]["name"]

    #self.script.log(2, "PubSubThread::update_config " + name + "=" + str(value))
    #self.script.log(2, "PubSubThread::update_config name="+name+"  mapping.keys()=" + str(self.mappings.keys()))

    if name in self.mappings.keys():
      key = self.mappings[name]
      self.update_cam_config(key, name, str(value))
    else:
      self.script.log(2, "PubSubThread::update_config no match on " + name)

    self.script.log(3, "PubSubThread::update_config done")

