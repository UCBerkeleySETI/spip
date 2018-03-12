#!/usr/bin/env python

##############################################################################
#  
#     Copyright (C) 2015 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
# 
###############################################################################

#
# spip_lmc - 
#

import os, socket, threading, sys, errno, traceback, select, xmltodict, subprocess
from time import sleep
import spip_lmc_monitor as lmc_mon

from spip.daemons.bases import HostBased
from spip.daemons.daemon import Daemon
from spip.utils.sockets import getHostNameShort
from spip.config import Config
from spip.utils.core import system

DAEMONIZE = True
DL        = 1

#################################################################
#
# lmcThread
#
class lmcThread (threading.Thread):

  def __init__ (self, script, states):

    super(lmcThread, self).__init__()

    self.daemon_exit_wait = 10
    self.states = states
    self.parent = script
    self.daemons = {}
    self.ranks = []

    ranked_daemons = self.daemon_list
    for ranked_daemon in ranked_daemons:
      daemon, rank = ranked_daemon.split(":")
      if not rank in self.daemons:
        self.daemons[rank] = []
      self.daemons[rank].append(daemon)

    self.ranks = self.daemons.keys()
    self.ranks.sort()
    self.process_suffix = " " + str(self.id)
    self.file_suffix = "_" + str(self.id)

  def run (self):

    self.parent.log(2, self.prefix + "thread running")

    try:

      # start all of the daemons
      self.start_daemons (self.ranks)

      # monitor the daemons in each stream and 
      # process control commands specific to the stream
      while (not self.parent.quit_event.isSet()):

        # check if a reload has been requested
        if self.should_reload (): 
          self.parent.log(1, self.prefix + "reloading stream")
          self.stop_daemons (self.ranks)
          self.parent.log(1, self.prefix + "daemons stopped")
          self.start_daemons (self.ranks)
          self.parent.log(1, self.prefix + "daemons started")
          self.did_reload()
          self.parent.log(1, self.prefix + "stream reloaded")

        for rank in self.ranks:
          for daemon in self.daemons[rank]:
            cmd = "pgrep -f '^python " + self.parent.cfg["SCRIPTS_DIR"] + "/" + daemon + ".py" + self.process_suffix + "' | wc -l"
            rval, lines = self.parent.system (cmd, 2)
            self.states[daemon] = (rval == 0)
        counter = 5
        while (not self.parent.quit_event.isSet() and counter > 0):
          sleep(1)
          counter -= 1

      self.parent.log(1, self.prefix + "asking daemons to quit")
      self.stop_daemons (self.ranks)
      self.parent.log(1, self.prefix + "thread exiting")

    except:
      self.parent.quit_event.set()

      formatted_lines = traceback.format_exc().splitlines()
      self.parent.log(1, self.prefix + '-'*60)
      for line in formatted_lines:
        self.parent.log(1, self.prefix + line)
      self.parent.log(1, self.prefix + '-'*60)


  # start all the daemons in ranks (ascending)
  def start_daemons (self, ranks):

    self.parent.system_lock.acquire()
    to_sleep = 2

    for rank in ranks:

      if rank > 0:
        sleep(to_sleep)
        to_sleep = 0

      self.parent.log(1, self.prefix + "launching daemons of rank " + rank)
      for daemon in self.daemons[rank]:
        self.states[daemon] = False
        cmd = "python " + self.parent.cfg["SCRIPTS_DIR"] + "/" + daemon + ".py" + self.process_suffix
        self.parent.log(1, self.prefix + cmd)
        rval, lines = self.parent.system (cmd)
        if rval:
          for line in lines:
            self.parent.log(-2, self.prefix + line)
          self.parent.quit_event.set()
        else:
          for line in lines:
            self.parent.log(2, self.prefix + line)
      self.parent.log (1, self.prefix + "launched daemons of rank " + rank)

    self.parent.log(1, self.prefix + "launched all daemons")
    self.parent.system_lock.release ()

  # stop the daemons listed in ranks in reverse order
  def stop_daemons (self, ranks):

    for rank in ranks[::-1]:
      if rank == 0:
        self.parent.log (1, self.prefix + " sleep(5) for rank 0 daemons")
        sleep(5)

      for daemon in self.daemons[rank]:
        open(self.control_dir + "/" + daemon + self.file_suffix + ".quit", 'w').close()

      rank_timeout = self.daemon_exit_wait
      daemon_running = 1

      while daemon_running and rank_timeout > 0:
        daemon_running = 0
        self.parent.system_lock.acquire()
        for daemon in self.daemons[rank]:
          cmd = "pgrep -f '^python " + self.parent.cfg["SCRIPTS_DIR"] + "/" + daemon + ".py" + self.process_suffix + "'"
          rval, lines = self.parent.system (cmd, 3)
          if rval == 0 and len(lines) > 0:
            daemon_running = 1
            self.parent.log(2, self.prefix + "daemon " + daemon + " with rank " +
                        str(rank) + " still running")
        self.parent.system_lock.release()
        if daemon_running:
          self.parent.log(3, self.prefix + "daemons " + "with rank " + str(rank) +
                      " still running")
          sleep(1)

      # if any daemons in this rank timed out, hard kill them
      if rank_timeout == 0:
        self.parent.system_lock.acquire()
        for daemon in self.daemons[rank]:
          cmd = "pkill -f ''^python " + self.parent.cfg["SCRIPTS_DIR"] + "/" + daemon + ".py" + self.process_suffix + "'"
          rval, lines = self.parent.system (cmd, 3)
        self.parent.system_lock.release()

      # remove daemon.quit files for this rank
      for daemon in self.daemons[rank]:
        try:
          os.remove (self.control_dir + "/" + daemon + self.file_suffix + ".quit")
        except OSError:
          pass

#
# LMC implementation for client Streams
#
class streamThread (lmcThread):

  def __init__ (self, id, script, states):

    # configure the stream
    self.daemon_list = script.cfg["CLIENT_DAEMONS"].split()
    self.control_dir = script.cfg["CLIENT_CONTROL_DIR"]
    self.prefix = "streamThread["+str(id)+"] "
    self.id = id

    # launch the threads
    lmcThread.__init__(self, script, states)

  def should_reload (self):
    return self.parent.reload_streams[self.id]

  # mark this thread as reloaded 
  def did_reload (self):
    self.parent.reload_streams[self.id] = False

################################################################$
# 
# LMC implementation for Beam Threads
#
class beamThread (lmcThread):

  def __init__ (self, id, script, states):

    self.daemon_list = []
    self.control_dir = ""
    self.id = id
    self.prefix = "beamThread["+str(id)+"] "

    # configure the beam
    if "BEAM_DAEMONS" in script.cfg.keys():
      self.daemon_list = script.cfg["BEAM_DAEMONS"].split()
      self.control_dir = script.cfg["BEAM_CONTROL_DIR"]

    # launch the thread
    lmcThread.__init__(self, script, states)

  # check if this thread has been asked to reload
  def should_reload (self):
    return self.parent.reload_beams[self.id]

  # mark this thread as reloaded 
  def did_reload (self):
    self.parent.reload_beams[self.id] = False

################################################################$
# 
# LMC implementation for Server Threads
#
class serverThread (lmcThread):

  def __init__ (self, id, script, states):

    # configure the beam
    self.daemon_list = script.cfg["SERVER_DAEMONS"].split()
    self.control_dir = script.cfg["SERVER_CONTROL_DIR"]
    self.prefix = "serverThread "
    self.id = id

    # launch the thread
    lmcThread.__init__(self, script, states)

  def should_reload (self):
    return False

  def did_reload (self):
    return False

################################################################$
#
# 
#
class LMCDaemon (Daemon, HostBased):

  def __init__ (self, name, hostname):
    Daemon.__init__(self, name, hostname)
    HostBased.__init__(self, hostname, self.cfg)

  def main (self):

    control_thread = []
  
    # the threads for each instance of beams, streams and server    
    self.stream_threads = {}
    self.beam_threads = {}
    self.server_thread = []

    self.reload_beams = {}
    self.reload_streams = {}
    self.system_lock = threading.Lock()

    # find matching client streams for this host
    host_streams = []
    for istream in range(int(self.cfg["NUM_STREAM"])):
      (req_host, beam_id, subband_id) = Config.getStreamConfig (istream, self.cfg)
      if req_host == self.req_host and not istream in host_streams:
        host_streams.append(istream)

    # find matching client streams for this host
    host_beams = []
    for istream in range(int(self.cfg["NUM_STREAM"])):
      (req_host, beam_id, subband_id) = Config.getStreamConfig (istream, self.cfg)
      if req_host == self.req_host and not beam_id in host_beams:
        host_beams.append(beam_id)

    # find matching server stream
    host_servers = []
    if self.cfg["SERVER_HOST"] == self.req_host:
      host_servers.append(-1)

    server_daemon_states = {}
    stream_daemon_states = {}
    beam_daemon_states = {}

    for stream in host_servers:
      self.log(2, "main: server_thread["+str(stream)+"] = streamThread(-1)")
      server_daemon_states[stream] = {}
      server_thread = serverThread(stream, self, server_daemon_states[stream])
      self.log(2, "main: server_thread["+str(stream)+"].start()")
      server_thread.start()
      self.log(2, "main: server_thread["+str(stream)+"] started")

    sleep(1)

    # start a thread for each stream
    for stream in host_streams:
      stream_daemon_states[stream] = {}
      self.log(2, "main: stream_threads["+str(stream)+"] = streamThread ("+str(stream)+")")
      self.reload_streams[stream] = False
      self.stream_threads[stream] = streamThread (stream, self, stream_daemon_states[stream])
      self.log(2, "main: stream_threads["+str(stream)+"].start()")
      self.stream_threads[stream].start()
      self.log(2, "main: stream_thread["+str(stream)+"] started!")

    # start a thread for each beam
    for beam in host_beams:
      beam_daemon_states[beam] = {}
      self.log(2, "main: beam_threads["+str(beam)+"] = beamThread ("+str(beam)+")")
      self.reload_beams[beam] = False
      self.beam_threads[beam] = beamThread (beam, self, beam_daemon_states[beam])
      self.log(2, "main: beam_threads["+str(beam)+"].start()")
      self.beam_threads[beam].start()
      self.log(2, "main: beam_thread["+str(beam)+"] started!")

    # main thread
    if len(host_servers) > 0:
      disks_to_monitor = [self.cfg["SERVER_DIR"]]
    else:
      disks_to_monitor = [self.cfg["CLIENT_DIR"]]

    # create socket for LMC commands
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    bound = False
    while not bound:
      try:
        sock.bind((self.req_host, int(self.cfg["LMC_PORT"])))
        bound = True
      except socket.error, e:
        self.log(-1, "Could not bind to " + self.req_host + ":" + self.cfg["LMC_PORT"] + ", sleep(5)")
        sleep(5)
    sock.listen(5)

    can_read = [sock]
    can_write = []
    can_error = []
    timeout = 1
    hw_poll = 5
    counter = 0 

    sensors = {}

    # monitor / control loop
    while not self.quit_event.isSet():

      self.log(3, "Main Loop: counter="+str(counter))

      while (counter == 0):

        self.log(2, "Refreshing monitoring points")

        self.log(3, "main: getDiskCapacity ()")
        rval, disks = lmc_mon.getDiskCapacity (disks_to_monitor, DL)
        self.log(3, "main: " + str(disks))

        self.log(3, "main: getLoads()")
        rval, loads = lmc_mon.getLoads (DL)
        self.log(3, "main: " + str(loads))

        self.log(3, "main: getNTPSynced()")
        rval, ntp_sync = lmc_mon.getNTPSynced(DL)
        self.log(3, "main: " + str(ntp_sync))

        self.log(3, "main: getSMRBCapacity(" + str(host_streams)+ ")")
        rval, smrbs = lmc_mon.getSMRBCapacity (host_streams, self.quit_event, DL)
        self.log(3, "main: " + str(smrbs))

        self.log(3, "main: getIPMISensors()")
        rval, sensors = lmc_mon.getIPMISensors (DL)
        self.log(3, "main: " + str(sensors))

        counter = hw_poll

      self.log(3, "main: calling select len(can_read)="+str(len(can_read)))
      timeout = 1
      did_read = []
      did_write = []
      did_error = []

      try:
        did_read, did_write, did_error = select.select(can_read, can_write, can_error, timeout)
      except select.error as e:
        self.quit_event.set()
      else:
        self.log(3, "main: read="+str(len(did_read))+" write="+
                    str(len(did_write))+" error="+str(len(did_error)))

      if (len(did_read) > 0):
        for handle in did_read:
          if (handle == sock):
            (new_conn, addr) = sock.accept()
            self.log(2, "main: accept connection from "+repr(addr))
            # add the accepted connection to can_read
            can_read.append(new_conn)
            # new_conn.send("Welcome to the LMC interface\r\n")

          # an accepted connection must have generated some data
          else:
            try:
              raw = handle.recv(4096)
            except socket.error, e:
              if e.errno == errno.ECONNRESET:
                self.log(2, "main: closing connection")
                handle.close()
                for i, x in enumerate(can_read):
                  if (x == handle):
                    del can_read[i]
              else:
                raise e
            else: 
              message = raw.strip()
              self.log(2, "main: message='" + message+"'")

              if len(message) == 0:
                self.log(2, "main: closing connection")
                handle.close()
                for i, x in enumerate(can_read):
                  if (x == handle):
                    del can_read[i]

              else:
                xml = xmltodict.parse(message)

                command = xml["lmc_cmd"]["command"]

                if command == "reload_clients":
                  self.log(1, "Reloading streams")
                  for stream in host_streams:
                    self.reload_streams[stream] = True
                  self.log(1, "Reloading beams")
                  for beam in host_beams:
                    self.reload_beams[beam] = True
            
                  all_reloaded = False
                  while (not self.parent.quit_event.isSet() and not all_reloaded):
                    all_reloaded = True
                    for stream in host_streams:
                      if not self.reload_streams[stream]:
                        all_reloaded = False
                    for beam in host_beams:
                      if not self.reload_beams[beam]:
                        all_reloaded = False
                    if not all_reloaded:
                      self.log(1, "Waiting for clients to reload")
                      sleep(1)

                  self.log(1, "Clients reloaded")
                  response = "<lmc_reply>OK</lmc_reply>"

                if command == "daemon_status":
                  response = ""
                  response += "<lmc_reply>"

                  # state of the configure server
                  for stream in host_servers:
                    response += "<server id='" + str(stream) +"'>"
                    for daemon in server_daemon_states[stream].keys():
                      response += "<daemon name='" + daemon + "'>" + str(server_daemon_states[stream][daemon]) + "</daemon>"
                    response += "</server>"

                  # state of the configured streams
                  for stream in host_streams:
                    response += "<stream id='" + str(stream) +"'>"
                    for daemon in stream_daemon_states[stream].keys():
                      response += "<daemon name='" + daemon + "'>" + \
                                    str(stream_daemon_states[stream][daemon]) + \
                                  "</daemon>"
                    response += "</stream>"

                  # state of the configured beams
                  for beam in host_beams:
                    response += "<beam id='" + str(beam) +"'>"
                    for daemon in beam_daemon_states[beam].keys():
                      response += "<daemon name='" + daemon + "'>" + \
                                    str(beam_daemon_states[beam][daemon]) + \
                                  "</daemon>"
                    response += "</beam>"

                  response += "</lmc_reply>"

                elif command == "host_status":
                  response = "<lmc_reply>"

                  for disk in disks.keys():
                    percent_full = 1.0 - (float(disks[disk]["available"]) / float(disks[disk]["size"]))
                    response += "<disk mount='" + disk +"' percent_full='"+str(percent_full)+"'>"
                    response += "<size units='MB'>" + disks[disk]["size"] + "</size>"
                    response += "<used units='MB'>" + disks[disk]["used"] + "</used>"
                    response += "<available units='MB'>" + disks[disk]["available"] + "</available>"
                    response += "</disk>"

                  response += "<local_time_synced>"+str(ntp_sync)+"</local_time_synced>"

                  for stream in smrbs.keys():
                    for key in smrbs[stream].keys():
                      smrb = smrbs[stream][key]
                      response += "<smrb stream='" + str(stream) + "' key='" + str(key) + "'>"
                      response += "<header_block nbufs='"+str(smrb['hdr']['nbufs'])+"'>"+ str(smrb['hdr']['full'])+"</header_block>"
                      response += "<data_block nbufs='"+str(smrb['data']['nbufs'])+"'>"+ str(smrb['data']['full'])+"</data_block>"
                      response += "</smrb>"
                  
                  response += "<system_load ncore='"+loads["ncore"]+"'>"
                  response += "<load1>" + loads["1min"] + "</load1>"
                  response += "<load5>" + loads["5min"] + "</load5>"
                  response += "<load15>" + loads["15min"] + "</load15>"
                  response += "</system_load>"

                  response += "<sensors>"
                  for sensor in sensors.keys():
                    response += "<metric name='" + sensor + "' units='"+sensors[sensor]["units"]+"'>" + sensors[sensor]["value"] + "</metric>"
                  response += "</sensors>"
                  
                  response += "</lmc_reply>"

                else:
                  response = "<lmc_reply>OK</lmc_reply>"

                self.log(2, "-> " + response)

                handle.send(response + "\r\n")

      counter -= 1


    def conclude (self):

      self.quit_event.set()

      for stream in self.client_streams:
        self.client_threads[stream].join()

      if self.server_thread:
        self.server_thread.join()

      script.log(1, "STOPPING SCRIPT")
      Daemon.conclude (self)


###############################################################################
#
if __name__ == "__main__":

  if len(sys.argv) != 1:
    print "ERROR: 0 command line argument expected"
    sys.exit(1)

  # cfg = sys.argv[1]
  # cfg_dir = os.environ.get('SPIP_ROOT') + "/share"
  # cmd = "cp " + cfg_dir + "/" + cfg + "/*.cfg " + cfg_dir + "/"
  # system (cmd, False)  

  hostname = getHostNameShort()

  script = LMCDaemon ("spip_lmc", hostname)
  state = script.configure (DAEMONIZE, DL, "lmc", "lmc")
  if state != 0:
    sys.exit(state)

  script.log(1, "STARTING SCRIPT")

  try:

    script.main()

  except:
    script.quit_event.set()

    script.log(-2, "exception caught: " + str(sys.exc_info()[0]))
    print '-'*60
    traceback.print_exc(file=sys.stdout)
    print '-'*60

  script.conclude()
  sys.exit(0)


