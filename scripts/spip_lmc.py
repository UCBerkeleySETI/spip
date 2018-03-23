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
from spip.threads.reporting_thread import ReportingThread
from spip.utils.sockets import getHostNameShort
from spip.config import Config
from spip.utils.core import system

DAEMONIZE = True
DL = 1

#################################################################
#
# lmcThread
#
# Manages the start/stop of all configured Daemons for this LMC host
#
class lmcThread (threading.Thread):

  def __init__ (self, script, states):

    super(lmcThread, self).__init__()

    self.daemon_exit_wait = 10
    self.states = states
    self.parent = script
    self.sustain = True
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

  # ask the thread to terminate
  def conclude (self):
    self.sustain = False

  def run (self):

    try:

      # start all of the daemons
      self.start_daemons (self.ranks)

      # monitor the daemons in each stream and 
      # process control commands specific to the stream
      while self.sustain:

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
            rval, lines = self.parent.system (cmd, 3, True)
            self.states[daemon] = (rval == 0)
        counter = 5
        while (self.sustain and counter > 0):
          sleep(1)
          counter -= 1

      self.stop_daemons (self.ranks)

    except:
      self.parent.log(1, self.prefix + " Exception during RUN")
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

      self.parent.log(2, self.prefix + "launching daemons of rank " + rank)
      for daemon in self.daemons[rank]:
        self.states[daemon] = False
        cmd = "python " + self.parent.cfg["SCRIPTS_DIR"] + "/" + daemon + ".py" + self.process_suffix
        self.parent.log(2, self.prefix + cmd)
        rval, lines = self.parent.system (cmd)
        if rval:
          self.parent.log(0, "Launch failure: " + daemon + " asking all threads to quit")
          for line in lines:
            self.parent.log(-2, self.prefix + line)
          self.parent.quit_event.set()
        else:
          for line in lines:
            self.parent.log(2, self.prefix + line)
      self.parent.log (2, self.prefix + "launched daemons of rank " + rank)

    self.parent.log(1, self.prefix + "launched all daemons")
    self.parent.system_lock.release ()

  # stop the daemons listed in ranks in reverse order
  def stop_daemons (self, ranks):

    for rank in ranks[::-1]:
      if rank == 0:
        self.parent.log (2, self.prefix + " sleep(5) for rank 0 daemons")
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
          rval, lines = self.parent.system (cmd, 3, True)
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
      self.control_dir = script.cfg["CLIENT_CONTROL_DIR"]

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
    return 0

################################################################$
# 
# Provide XML monitoring service via a socket
#
class LMCReportingThread (ReportingThread):

  def __init__ (self, script):

    host = script.req_host
    port = int(script.cfg["LMC_PORT"])
    script.log (2, "LMCReportingThread::__init__ ReportingThread on " + host + ":" + str(port))
    ReportingThread.__init__(self, script, host, port)

    # allow 5 queued connections
    self.set_listening_slots (5)

  def run (self):

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    self.script.log (2, "LMCReportingThread listening on " + self.host + ":" + str(self.port))
    sock.bind((self.host, int(self.port)))
    self.script.log (3, "LMCReportingThread configuring number of listening slots to " + str(self.nlisten))
    sock.listen(self.nlisten)

    self.can_read = [sock]
    self.can_write = []
    self.can_error = []

    serve_requests = True

    while serve_requests:

      timeout = 1

      did_read = []
      did_write = []
      did_error = []

      try:
        # wait for some activity on the control socket
        self.script.log (3, "LMCReportingThread::run select len(can_read)=" + str(len(self.can_read)) + " can_read=" + str(self.can_read))
        did_read, did_write, did_error = select.select(self.can_read, self.can_write, self.can_error, timeout)
        self.script.log (3, "LMCReportingThread::run read="+str(len(did_read))+" write="+ str(len(did_write))+" error="+str(len(did_error)))

      except select.error as e:
        if e[0] == errno.EINTR:
          self.script.log(0, "SIGINT received during select, exiting")
          self.script.quit_event.set()
        else:
          raise

      if (len(did_read) > 0):
        self.script.log (3, "LMCReportingThread::run len(did_read)=" + str(len(did_read)))
        for handle in did_read:
          if (handle == sock):
            self.script.log (3, "LMCReportingThread::run accepting new connection")
            (new_conn, addr) = sock.accept()
            self.script.log (2, "LMCReportingThread: accept connection from "+repr(addr))

            # add the accepted connection to can_read
            self.can_read.append(new_conn)

          # an accepted connection must have generated some data
          else:
            self.script.log (3, "LMCReportingThread::run processing data on existing connection")
            try:
              self.process_message_on_handle (handle)

            except socket.error as e:
              self.script.log (3, "LMCReportingThread::run socket error on existing connection")
              if e.errno == errno.ECONNRESET:
                self.script.log (2, "LMCReportingThread closing connection")
                handle.close()
                self.script.log (3, "LMCReportingThread::run trying to delete socket from can_read")
                for i, x in enumerate(self.can_read):
                  if (x == handle):
                    self.script.log (3, "LMCReportingThread::run deleting can_read[" + str(i) + "]")
                    del self.can_read[i]
              else:
                raise

      # if we are asked to quit, close the listening socket and remove it from the list
      if self.script.quit_event.isSet():
        self.script.log (3, "LMCReportingThread::run script.quit_event.isSet() == True")
        for i, x in enumerate(self.can_read):
          if (x == sock):
            self.script.log (2, "LMCReportingThread::run closing listening socket at index[" + str(i) + "]")
            sock.close()
            del self.can_read[i]

        self.script.log (2, "LMCReportingThread::run concluding, len(can_read)=" + \
                         str(len(self.can_read)) + " can_read=" + str(self.can_read))
        if len(self.can_read) == 0:
          serve_requests = False

    self.script.log (2, "LMCReportingThread::run exiting")

  def parse_message (self, request):

    self.script.log (2, "LMCReportingThread::parse_message: " + str(request))

    command = request["lmc_cmd"]["command"]

    self.script.resource_lock.acquire()

    if command == "reload_clients":
      self.script.log(1, "Reloading streams")
      for stream in self.script.host_streams:
        self.sciprt.reload_streams[stream] = True
      self.script.log(1, "Reloading beams")
      for beam in self.script.host_beams:
        self.script.reload_beams[beam] = True

      all_reloaded = False
      while (not self.script.parent.quit_event.isSet() and not all_reloaded):
        all_reloaded = True
        for stream in self.script.host_streams:
          if not self.script.reload_streams[stream]:
            all_reloaded = False
        for beam in self.script.host_beams:
          if not self.script.reload_beams[beam]:
            all_reloaded = False
        if not all_reloaded:
          self.script.log(1, "Waiting for clients to reload")
          sleep(1)

      self.script.log(1, "Clients reloaded")
      response = "<lmc_reply>OK</lmc_reply>"

    if command == "daemon_status":
      response = ""
      response += "<lmc_reply>"

      # state of the configure server
      for stream in self.script.host_servers:
        response += "<server id='" + str(stream) +"'>"
        for daemon in self.script.server_daemon_states[stream].keys():
          response += "<daemon name='" + daemon + "'>" + str(self.script.server_daemon_states[stream][daemon]) + "</daemon>"
        response += "</server>"

      # state of the configured streams
      for stream in self.script.host_streams:
        response += "<stream id='" + str(stream) +"'>"
        for daemon in self.script.stream_daemon_states[stream].keys():
          response += "<daemon name='" + daemon + "'>" + \
                        str(self.script.stream_daemon_states[stream][daemon]) + \
                      "</daemon>"
        response += "</stream>"

      # state of the configured beams
      for beam in self.script.host_beams:
        response += "<beam id='" + str(beam) +"'>"
        for daemon in self.script.beam_daemon_states[beam].keys():
          response += "<daemon name='" + daemon + "'>" + \
                        str(self.script.beam_daemon_states[beam][daemon]) + \
                      "</daemon>"
        response += "</beam>"

      response += "</lmc_reply>"

    # reply with information about the host
    elif command == "host_status":

      response = "<lmc_reply>"

      for disk in self.script.disks.keys():
        d = self.script.disks[disk]
        percent_full = 1.0 - (float(d["available"]) / float(d["size"]))
        response += "<disk mount='" + disk +"' percent_full='"+str(percent_full)+"'>"
        response += "<size units='MB'>" + d["size"] + "</size>"
        response += "<used units='MB'>" + d["used"] + "</used>"
        response += "<available units='MB'>" + d["available"] + "</available>"
        response += "</disk>"

      response += "<local_time_synced>"+str(self.script.ntp_sync)+"</local_time_synced>"

      for stream in self.script.smrbs.keys():
        for key in self.script.smrbs[stream].keys():
          smrb = self.script.smrbs[stream][key]
          response += "<smrb stream='" + str(stream) + "' key='" + str(key) + "'>"
          response += "<header_block nbufs='"+str(smrb['hdr']['nbufs'])+"'>"+ str(smrb['hdr']['full'])+"</header_block>"
          response += "<data_block nbufs='"+str(smrb['data']['nbufs'])+"'>"+ str(smrb['data']['full'])+"</data_block>"
          response += "</smrb>"

      response += "<system_load ncore='" + self.script.loads["ncore"] + "'>"
      response += "<load1>" + self.script.loads["1min"] + "</load1>"
      response += "<load5>" + self.script.loads["5min"] + "</load5>"
      response += "<load15>" + self.script.loads["15min"] + "</load15>"
      response += "</system_load>"

      response += "<sensors>"
      for sensor in self.script.sensors.keys():
        s  = self.script.sensors[sensor]
        response += "<metric name='" + sensor + "' units='"+s["units"]+"'>" + s["value"] + "</metric>"
      response += "</sensors>"

      response += "</lmc_reply>"

    # be agreeable and just reply OK
    else:
      response = "<lmc_reply>OK</lmc_reply>"

    self.script.log(2, "-> " + response)
    response += "\r\n"

    self.script.resource_lock.release()

    return True, response


################################################################$
#
# 
#
class LMCDaemon (Daemon, HostBased):

  def __init__ (self, name, hostname):
    Daemon.__init__(self, name, hostname)
    HostBased.__init__(self, hostname, self.cfg)

    self.resource_lock = threading.Lock()

    self.resource_lock.acquire()

    self.disks = {}
    self.loads = { 'ncore': '0', '1min': '0', '5min': '0', '15min': '0' }
    self.ntp_sync = True
    self.smrbs = {}
    self.sensors = {}
    self.disks_to_monitor = []

  def gather_stats (self):

    self.log(3, "LMCDaemon::gather_stats getDiskCapacity ()")
    rval, self.disks = lmc_mon.getDiskCapacity (self.disks_to_monitor, DL)
    self.log(3, "LMCDaemon::gather_stats " + str(self.disks))

    self.log(3, "LMCDaemon::gather_stats getLoads()")
    rval, self.loads = lmc_mon.getLoads (DL)
    self.log(3, "LMCDaemon::gather_stats " + str(self.loads))

    self.log(3, "LMCDaemon::gather_stats getNTPSynced()")
    rval, self.ntp_sync = lmc_mon.getNTPSynced(DL)
    self.log(3, "LMCDaemon::gather_stats " + str(self.ntp_sync))

    self.log(3, "LMCDaemon::gather_stats getSMRBCapacity(" + str(self.host_streams)+ ")")
    rval, self.smrbs = lmc_mon.getSMRBCapacity (self.host_streams, self.quit_event, DL)
    self.log(3, "LMCDaemon::gather_stats " + str(self.smrbs))

    self.log(3, "LMCDaemon::gather_stats getIPMISensors()")
    rval, self.sensors = lmc_mon.getIPMISensors (DL)
    self.log(3, "LMCDaemon::gather_stats " + str(self.sensors))

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
    self.host_streams = []
    for istream in range(int(self.cfg["NUM_STREAM"])):
      (req_host, beam_id, subband_id) = Config.getStreamConfig (istream, self.cfg)
      if req_host == self.req_host and not istream in self.host_streams:
        self.host_streams.append(istream)

    # find matching client streams for this host
    self.host_beams = []
    for istream in range(int(self.cfg["NUM_STREAM"])):
      (req_host, beam_id, subband_id) = Config.getStreamConfig (istream, self.cfg)
      if req_host == self.req_host and not beam_id in self.host_beams:
        self.host_beams.append(beam_id)

    # find matching server stream
    self.host_servers = []
    if self.cfg["SERVER_HOST"] == self.req_host:
      self.host_servers.append(-1)

    self.server_daemon_states = {}
    self.stream_daemon_states = {}
    self.beam_daemon_states = {}

 
    # configure disk systems to monitor
    if len(self.host_servers) > 0:
      self.disks_to_monitor = [self.cfg["SERVER_DIR"]]
    else:
      self.disks_to_monitor = [self.cfg["CLIENT_DIR"]]

    # gather some initial statistics
    self.gather_stats()

    # start server thread
    for stream in self.host_servers:
      self.log(2, "main: server_thread["+str(stream)+"] = streamThread(-1)")
      self.server_daemon_states[stream] = {}
      self.server_thread = serverThread(stream, self, self.server_daemon_states[stream])
      self.log(2, "main: server_thread["+str(stream)+"].start()")
      self.server_thread.start()
      self.log(2, "main: server_thread["+str(stream)+"] started")

    sleep(5)
    self.log(1, "Server threads started: " + str(len(self.host_servers)))

    # start a thread for each stream
    for stream in self.host_streams:
      self.stream_daemon_states[stream] = {}
      self.log(2, "main: stream_threads["+str(stream)+"] = streamThread ("+str(stream)+")")
      self.reload_streams[stream] = False
      self.stream_threads[stream] = streamThread (stream, self, self.stream_daemon_states[stream])
      self.log(2, "main: stream_threads["+str(stream)+"].start()")
      self.stream_threads[stream].start()
      self.log(2, "main: stream_thread["+str(stream)+"] started!")

    self.log(1, "Stream threads started: " + str(len(self.host_streams)))

    # start a thread for each beam
    for beam in self.host_beams:
      self.beam_daemon_states[beam] = {}
      self.log(2, "main: beam_threads["+str(beam)+"] = beamThread ("+str(beam)+")")
      self.reload_beams[beam] = False
      self.beam_threads[beam] = beamThread (beam, self, self.beam_daemon_states[beam])
      self.log(2, "main: beam_threads["+str(beam)+"].start()")
      self.beam_threads[beam].start()
      self.log(2, "main: beam_thread["+str(beam)+"] started!")

    self.log(1, "Beam threads started: " + str(len(self.host_beams)))

    # main thread

    hw_poll = 5
    counter = 0 

    first_time = True

    self.log(2, "main: starting main loop quit_event=" + str(self.quit_event.isSet()))

    # control loop
    while not self.quit_event.isSet():

      # update the monitoring points
      if counter == 0:

        self.log(2, "main: refreshing monitoring points")
        if not first_time:
          self.resource_lock.acquire ()

        self.log(3, "main: self.gather_stats()")
        self.gather_stats()

        first_time = False
        self.resource_lock.release ()

        self.log(3, "main: monitoring points refreshed")

        counter = hw_poll

      if not self.quit_event.isSet():
        sleep(1)

      counter -= 1

    self.log(2, "main: quit_event set, asking lmcThreads to terminate")
    self.concludeThreads()

    self.log(2, "main: done")


  ########################################################################## 
  # 
  # instruct lmc worker threads to conclude
  #
  def concludeThreads (self):

    script.log(2, "LMCDaemon::concludeThreads()")

    for beam in self.beam_threads.keys():
      self.beam_threads[beam].conclude()
    for beam in self.beam_threads.keys():
      self.beam_threads[beam].join()

    for stream in self.stream_threads.keys():
      self.stream_threads[stream].conclude()
    for stream in self.stream_threads.keys():
      self.stream_threads[stream].join()

    if self.server_thread:
      self.server_thread.conclude()
      self.server_thread.join()

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

    reporting_thread = LMCReportingThread(script)
    reporting_thread.start()

    script.main()

    reporting_thread.join()

  except:

    print "ERROR: exception during script.main, script.quit_event.set()"
    script.quit_event.set()

    script.log(-2, "exception caught: " + str(sys.exc_info()[0]))
    print '-'*60
    traceback.print_exc(file=sys.stdout)
    print '-'*60

  script.conclude()
  sys.exit(0)


