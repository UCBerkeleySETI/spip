#!/usr/bin/env python

##############################################################################
#  
#     Copyright (C) 2015 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
# 
###############################################################################
#
#   configured SMRBs initialized on launch
#   monitoring of SMRBs in separate threads
#   SMRBs destroyed upon exit
#

import errno
import threading, sys, traceback, socket, select
from time import sleep

from json import dumps

from spip.daemons.bases import StreamBased
from spip.daemons.daemon import Daemon
from spip.log_socket import LogSocket
from spip.utils.core import system
from spip.utils import sockets


DAEMONIZE = True
DL        = 1

# Instansiate a data block in this stream
class smrbThread (threading.Thread):

  def __init__ (self, key, db_id, stream_id, numa_node, pipe, script):
    threading.Thread.__init__(self)
    self.key = key
    self.db_id = db_id 
    self.stream_id = stream_id 
    self.numa_node = numa_node
    self.pipe = pipe
    self.script = script

    # configuration for this data block
    self.nbufs = self.getBlockParam ("NBUFS")
    if self.nbufs == False:
      raise Exception ("SMRBDaemon BLOCK_NBUFS_" + str(self.db_id) + " was not defined")

    self.bufsz = self.getBlockParam ("BUFSZ")
    if self.nbufs == False:
      raise Exception ("SMRBDaemon BLOCK_BUFSZ_" + str(self.db_id) + " was not defined")

    self.nread = self.getBlockParam ("NREAD")
    self.page  = self.getBlockParam ("PAGE")
    self.gpuid = self.getBlockParam ("GPUID")

  def getBlockParam (self, param):

    key = "BLOCK_" + param + "_" + str(self.db_id) + "_" + str(self.stream_id)
    if key in self.script.cfg.keys():
      return self.script.cfg[key]

    key = "BLOCK_" + param + "_" + str(self.db_id)
    if key in self.script.cfg.keys():
      return self.script.cfg[key]

    return False


  def run (self):

    self.script.log(2, "smrbThread::run key=" + self.key + " nbufs=" + self.nbufs + \
                       " bufsz=" + self.bufsz + " nread=" + self.nread + \
                       " numa_node=" + self.numa_node + " page=" + self.page)

    # command to establish the data block
    cmd = "dada_db -k " + self.key + " -w -n " + self.nbufs + " -b " + self.bufsz

    # number of readers is optional
    if self.nread != False:
      cmd = cmd + " -r " + self.nread

    # numa node comes from the stream numa configuration
    cmd = cmd + " -c " + self.numa_node

    # page and lock the datablock
    if self.page != False:
      cmd += " -p -l"

    # put the data block resident in GPU memory
    if self.gpuid != False:
      cmd += " -g " + self.gpuid

    # run the command
    self.script.log (1, "START " + cmd)
    rval = self.script.system_piped (cmd, self.pipe, 2)
    self.script.log (1, "END   " + cmd)

# Monitor the state of all data blocks in this stream
class monThread (threading.Thread):

  def __init__ (self, keys, script):
    threading.Thread.__init__(self)
    self.keys = keys
    self.id = script.id
    self.quit_event = script.quit_event
    self.script = script
    self.poll = 5

  def run (self):

    can_read = []
    can_write = []
    can_error = []
    script = self.script

    try:
      script.log (2, "monThread launching")

      script.log(2, "monThread::run opening mon socket")
      sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
      port = SMRBDaemon.getDBMonPort(self.id)
      script.log(2, "monThread::run binding to localhost:" + str(port))
      sock.bind(("localhost", port))

      sock.listen(10)

      can_read.append(sock)
      timeout = 1

      while (not self.quit_event.isSet()): 

        smrb = {}

        for key in self.keys:

          # read the current state of the data block
          rval, hdr, data = self.getDBState(key)
          smrb[key] = {'hdr': hdr, 'data' : data}
     
        # from json
        serialized = dumps(smrb)
  
        script.log(3, "monThread::run calling select len(can_read)=" + 
                    str(len(can_read)))
        timeout = self.poll
        did_read, did_write, did_error = select.select(can_read, can_write, 
                                                       can_error, timeout)
        script.log(3, "monThread::run read="+str(len(did_read)) + 
                    " write="+str(len(did_write))+" error="+str(len(did_error)))

        if (len(did_read) > 0):
          for handle in did_read:
            if (handle == sock):
              (new_conn, addr) = sock.accept()
              script.log(3, "monThread::run accept connection from " + 
                          repr(addr))
              can_read.append(new_conn)

            else:

              try:  
                message = handle.recv(4096)
              except socket.error as e:
                if e.errno == errno.ECONNRESET:
                  script.log (1, "Connection reset")
                  handle.close()
                  for i, x in enumerate(can_read):
                    if (x == handle):
                      del can_read[i]
              else:
                message = message.strip()
                script.log(3, "monThread::run message='" + message+"'")
                if (len(message) == 0):
                  script.log(3, "monThread::run closing connection")
                  handle.close()
                  for i, x in enumerate(can_read):
                    if (x == handle):
                      del can_read[i]
                else:
                  if message == "smrb_status":
                    script.log (3, "monThread::run " + str(smrb))
                    handle.send(serialized)
              
      for i, x in enumerate(can_read):
        x.close()
        del can_read[i]

    except:
      self.quit_event.set()
      script.log(1, "monThread::run exception caught: " +
                  str(sys.exc_info()[0]))
      print '-'*60
      traceback.print_exc(file=sys.stdout)
      print '-'*60
      for i, x in enumerate(can_read):
        x.close()
        del can_read[i]

  def getDBState (self, key):
    cmd = "dada_dbmetric -k " + key
    rval, lines = system (cmd, False)
    if rval == 0:
      a = lines[0].split(',')
      dat = {'nbufs':a[0], 'full':a[1], 'clear':a[2], 'written':a[3],'read':a[4]}
      hdr = {'nbufs':a[5], 'full':a[6], 'clear':a[7], 'written':a[8],'read':a[9]}
      return 0, hdr, dat
    else:
      return 1, {}, {}



#
class SMRBDaemon(Daemon,StreamBased):

  def __init__ (self, name, id):
    Daemon.__init__(self, name, str(id))
    StreamBased.__init__(self, id, self.cfg)

  def main (self):

    self.log (2, "SMRBDaemon::main()")

    mon_thread = []
    smrb_threads = {}

    # get a list of data block ids
    db_ids = self.cfg["DATA_BLOCK_IDS"].split(" ")
    db_prefix = self.cfg["DATA_BLOCK_PREFIX"]
    num_stream = self.cfg["NUM_STREAM"]
    numa_node = self.cfg["STREAM_NUMA_" + self.id]
    db_keys = []

    # log socket for output from SMRB instances
    log_host = self.cfg["SERVER_HOST"]
    log_port = int(self.cfg["SERVER_LOG_PORT"])
    log_pipe = LogSocket ("smrb_src", "smrb_src", str(self.id), "stream",
                          log_host, log_port, int(DL))
    log_pipe.connect()

    for db_id in db_ids:

      db_key = self.getDBKey (db_prefix, self.id, num_stream, db_id)
      self.log (2, "SMRBDaemon::main db_key for " + db_id + " is " + db_key)

      # check if the datablock already exists
      cmd = "ipcs | grep 0x0000" + db_key + " | wc -l"
      rval, lines = self.system (cmd)
      if rval == 0 and len(lines) == 1 and lines[0] == "1":
        self.log (-1, "Data block with key " + db_key + " existed at launch")
        cmd = "dada_db -k " + db_key + " -d"
        rval, lines = self.system (cmd)
        if rval != 0:
          self.log (-2, "Could not destroy existing datablock")

    for db_id in db_ids:

      db_key = self.getDBKey (db_prefix, self.id, num_stream, db_id)

      # start a thread to create a data block in persistence mode
      self.log (2, "SMRBDaemon::main smrbThread("+db_key+","+db_id+","+numa_node+")")
      smrb_threads[db_id] = smrbThread (db_key, db_id, self.id, numa_node, log_pipe.sock, script)
      smrb_threads[db_id].start()

      # append this command to the list of binaries that will be terminated
      # on script shutdown
      self.binary_list.append ("dada_db -k " + db_key)
      db_keys.append(db_key)

    # wait 5 seconds for threads to initialize
    sleep(5)

    self.log (2, "SMRBDaemon::main starting monThread")
    # after creation, launch thread to monitor smrb, maintaining state
    mon_thread = monThread (db_keys, self)
    mon_thread.start()

    # wait for termination signal
    while (not self.quit_event.isSet()):
      sleep(1)

    # join the monitoring thread
    if mon_thread:
      self.log(2, "SMRBDaemon::main joining mon thread")
      mon_thread.join()

    # join the SMRB threads
    for db_id in db_ids:
      self.log(2, "SMRBDaemon::main joining smrb_threads["+db_id+"]")
      smrb_threads[db_id].join()

    log_pipe.close()


  @classmethod
  def getDBKey(self, inst_id, stream_id, num_stream, db_id):
    index = (int(db_id) * int(num_stream)) + int(stream_id)
    db_key = inst_id + "%03x" % (2 * index)
    return db_key

  @classmethod
  def getDBMonPort (self, stream_id):
    start_port = 20000
    return start_port + int(stream_id)

  @classmethod
  def waitForSMRB (self, db_key, script):

    # port of the SMRB daemon for this stream
    smrb_port = SMRBDaemon.getDBMonPort(script.id)

    # wait up to 30s for the SMRB to be created
    smrb_wait = 60

    smrb_exists = False
    while not smrb_exists and smrb_wait > 0 and not script.quit_event.isSet():

      script.debug("trying to open connection to SMRB")
      smrb_sock = sockets.openSocket (DL, "localhost", smrb_port, 1)
      if smrb_sock:
        smrb_sock.send ("smrb_status\r\n")
        junk = smrb_sock.recv (65536)
        smrb_sock.close()
        smrb_exists = True
      else:
        sleep (1)
        smrb_wait -= 1

    return smrb_exists


if __name__ == "__main__":

  if len(sys.argv) != 2:
    print "ERROR: 1 command line argument expected"
    sys.exit(1)

  # this should come from command line argument
  stream_id = sys.argv[1]

  script = SMRBDaemon ("spip_smrb", stream_id)
  script.cpu_list = "-1"
  state = script.configure (DAEMONIZE, DL, "smrb", "smrb")
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

  script.log(1, "STOPPING SCRIPT")
  script.conclude()
  sys.exit(0)

