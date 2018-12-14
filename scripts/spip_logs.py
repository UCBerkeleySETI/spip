#!/usr/bin/env python

##############################################################################
#  
#     Copyright (C) 2015 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
# 
###############################################################################

import socket, sys, traceback, re
from select import select
from xmltodict import parse
from xml.parsers.expat import ExpatError
from time import sleep

from spip.config import Config
from spip.daemons.bases import ServerBased,BeamBased
from spip.daemons.daemon import Daemon
from spip.utils import sockets,times

DAEMONIZE = True
DL        = 1

class LogsDaemon(Daemon):

  def __init__ (self, name, id):
    Daemon.__init__(self, name, str(id))
    self.timestamp_re = re.compile ("^\[\d\d\d\d-\d\d-\d\d-\d\d:\d\d:\d\d\.\d\d\d\]")

  # over ride the default log message for this daemon
  def log (self, level, message):
  
    if level <= DL:
      if self.id == "-1":
        file = self.log_dir + "/logs.log"
      else:
        file = self.log_dir + "/logs_" + str(self.id) + ".log"
      fptr = open(file, 'a')

      prefix = "[" + times.getCurrentTimeMS() + "] "
      line = prefix + message

      fptr.write(line + "\n")
      fptr.close()

  # override configure logs too
  def configureLogs (self, source, dest, type):
   self.log (2, "LogsDaemon::configureLogs socket logging disabled")

  def main (self):

    log_host = sockets.getHostNameShort()
    log_port = self.cfg["SERVER_LOG_PORT"]

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    self.log(2, "main: binding to " + log_host + ":" + log_port)
    sock.bind((log_host, int(log_port)))

    # allow up to 128 queued connections
    sock.listen (128)

    can_read = [sock]
    can_write = []
    can_error = []
    headers = {}
    line_buffers = {}
    timeout = 1

    # TODO change this over to nslookups based on configured servers in
    # configuration files
    allowed_hosts = ["127.0.0.1", "130.155.182.74", "130.155.182.28", "130.155.182.31", "130.155.182.32", "130.155.182.33", "130.155.182.34", "130.155.182.35", "130.155.182.36", "130.155.182.37", "130.155.182.38", "10.98.72.1", "10.98.72.2" ]

    keep_logging = True
    to_wait = 10

    while keep_logging:

      # if the script has been asked to quit
      if self.quit_event.isSet():

        # if only the listening socket and LMC socket remain
        if len(can_read) == 2:
          keep_logging = False

        # some logging clients are still attached, wait 10 seconds before quitting
        else:
          to_wait -= 1

        if to_wait <= 0: 
          keep_logging = False

      self.log(3, "main: calling select len(can_read)="+str(len(can_read)))
      timeout = 1
      did_read, did_write, did_error = select(can_read, can_write, can_error, timeout)
      self.log(3, "main: read="+str(len(did_read))+" write="+ \
                    str(len(did_write))+" error="+str(len(did_error)))

      if (len(did_read) > 0):
        for handle in did_read:
          if (handle == sock):
            (new_conn, addr) = sock.accept()
            ip, port = addr
            self.log(2, "main: accept connection from " + ip + ":" + str(port))

            if len(allowed_hosts) > 0 and not (ip in allowed_hosts):
              self.log(1, "main: rejecting connection from " + ip + " not in " + str(allowed_hosts))
              new_conn.close()

            else:

              # read header information about this logging stream
              header = new_conn.recv (4096)

              try: 
                xml = parse(header)
              except ExpatError as e:
                self.log(0, "main: accept: xml error [" + header + "]")
                new_conn.send ("<xml>Malformed XML message</xml>\r\n")
                new_conn.close()
              else:
                # add to list of open file handles
                can_read.append(new_conn)
                # return a single character to confirm header has been receive
                new_conn.send('\n')
                headers[new_conn] = xml
                line_buffers[new_conn] = ""

          else:
            message = handle.recv(4096)

            # if the socket has been closed
            if len(message) == 0:
              self.log(2, "commandThread: closing connection")
              if len(line_buffers[handle]) > 0:
                line = line_buffers[handle]
                self.processLine(line, headers[handle])
              handle.close()
              for i, x in enumerate(can_read):
                if (x == handle):
                  del can_read[i]
              del headers[handle]
              del line_buffers[handle]

            else:
              well_terminated = message.endswith('\n')
              line_buffers[handle] += message.rstrip('\n')
              lines = line_buffers[handle].split("\n")
              to_use = len(lines)

              if well_terminated:
                line_buffers[handle] = ""
              else:
                to_use -= 1

              for i in range(to_use):
                line = lines[i]
                self.log(3, "commandThread: line='" + line + "'")
                self.processLine (line, headers[handle])


  def processLine (self, line, header):

    source = header['log_stream']['source']
    dest   = header['log_stream']['dest']
    id     = header['log_stream']['id']['#text']

    # add a time stamp if it was missing
    if not self.timestamp_re.match (line):
      prefix = "[" + times.getCurrentTimeMS() + "] "
      line = prefix + line

    # determine if the log file needs a prefix
    id_prefix = ""
    try:
      intid = int(id)
      # id was an integer
      if intid >= 0:
        id_prefix = format(intid, "02d ")
      else:
        id_prefix = ""
    except ValueError:
      id_prefix = id + " "

    file = self.log_dir + "/" + dest + ".log"
    fptr = open(file, 'a')
    fptr.write (id_prefix + line.decode('utf-8', 'replace') + "\n")
    fptr.close()


class LogsServerDaemon (LogsDaemon, ServerBased):

  def __init__ (self, name):
    LogsDaemon.__init__(self,name, "-1")
    ServerBased.__init__(self, self.cfg)

class LogsBeamDaemon (LogsDaemon, BeamBased):

  def __init__ (self, name, id):
    LogsDaemon.__init__(self, name, str(id))
    BeamBased.__init__(self, str(id), self.cfg)

###############################################################################
#
# spip_logs: main
#
if __name__ == "__main__":

  if len(sys.argv) != 2:
    print "ERROR: 1 command line argument expected"
    sys.exit(1)

  # this should come from command line argument
  beam_id = sys.argv[1]
  
  # if the beam_id is < 0, then there is a single TCS for 
  # all beams, otherwise, 1 per beam
  if int(beam_id) == -1:
    script = LogsServerDaemon ("spip_logs")
  else:
    script = LogsBeamDaemon ("spip_logs", beam_id)

  state = script.configure (DAEMONIZE, DL, "logs", "logs")
  if state != 0:
    sys.exit(state)

  script.log(1, "STARTING SCRIPT")

  try:

    script.main ()

  except:

    script.quit_event.set()

    script.log(-2, "exception caught: " + str(sys.exc_info()[0]))
    print '-'*60
    traceback.print_exc(file=sys.stdout)
    print '-'*60

  script.log(1, "STOPPING SCRIPT")
  script.conclude()
  sys.exit(0)

