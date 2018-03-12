#!/usr/bin/env python

##############################################################################
#  
#     Copyright (C) 2015 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
# 
###############################################################################

from os import path, unlink
from time import sleep
import threading

class ControlThread(threading.Thread):

  def __init__(self, script):
    threading.Thread.__init__(self)
    self.script = script

  def keep_running (self):
    return (not path.exists(self.script.quit_file)) and (not self.script.quit_event.isSet())

  def run (self):
    self.script.log (1, "ControlThread: starting")
    self.script.log (2, "ControlThread: pid_file=" + self.script.pid_file)
    self.script.log (2, "ControlThread: quit_file=" + self.script.quit_file)
    #self.script.log (2, "ControlThread: reload_file=" + self.script.reload_file)

    while (self.keep_running()):
      sleep(1)

    self.script.log (2, "ControlThread::run keep_running == false")
    self.script.quit_event.set()

    # terminate any binaries that are currently running
    self.script.log (2, "ControlThread::run self.script.killBinaries()")
    self.script.killBinaries()

    if path.exists(self.script.quit_file):
      self.script.log (1, "ControlThread: quit request detected")
      unlink (self.script.quit_file)

    #if path.exists(self.script.reload_file):
    #  self.script.log (2, "ControlThread: reload request detected")
    self.script.log (2, "ControlThread: exiting")
