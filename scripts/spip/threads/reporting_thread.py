##############################################################################
#  
#     Copyright (C) 2015 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
# 
###############################################################################

from socketed_thread import SocketedThread

from xmltodict import parse
from xml.parsers.expat import ExpatError
import socket, errno

class ReportingThread (SocketedThread):

  def __init__ (self, script, host, port):
    SocketedThread.__init__(self, script, host, port)

  def run (self):
    SocketedThread.run(self)

  def process_message_on_handle (self, handle):

    # try to read from the socket
    try:
      raw = handle.recv(4096)
    except socket.error, e:
      if e.errno == errno.ECONNRESET:
        self.script.log(2, "ReportingThread::process_message_on_handle closing connection")
        handle.close()
        self.script.log(2, "ReportingThread::process_message_on_handle removing socket from can_read")
        for i, x in enumerate(self.can_read):
          if (x == handle):
            del self.can_read[i]

      else:
        raise e

    # the socket read was successful
    else:

      message = raw.strip()
      self.script.log (2, "ReportingThread::process_message_on_handle message="+str(message))

      if len(message) == 0:
        self.script.log (2, "ReportingThread::process_message_on_handle length=0, closing handle")
        handle.close()
        for i, x in enumerate(self.can_read):
          if (x == handle):
            del self.can_read[i]

      else:
        self.script.log (2, "ReportingThread::process_message_on_handle parsing message to XML")
        try:
          xml = parse(message)
        except ExpatError as e:
          handle.send ("<xml>Malformed XML message</xml>\r\n")
          handle.close()
          for i, x in enumerate(self.can_read):
            if (x == handle):
              del self.can_read[i]

        self.script.log(3, "<- " + str(xml))

        self.script.log (2, "ReportingThread::process_message_on_handle self.parse_mesasge(xml)")
        retain, reply = self.parse_message (xml)

        if retain:
          bytes_sent = handle.send (reply)
          self.script.log (2, "ReportingThread: sent " + str(bytes_sent) + " bytes")
        else:
          bytes_sent = handle.sendall (reply)
          self.script.log (2, "ReportingThread: sent " + str(bytes_sent) + " bytes")

          self.script.log (3, "ReportingThread: handle.shutdown()")
          handle.shutdown(socket.SHUT_RDWR)
          self.script.log (3, "ReportingThread: handle.close()")
          handle.close()
          for i, x in enumerate(self.can_read):
            if (x == handle):
                  del self.can_read[i]

